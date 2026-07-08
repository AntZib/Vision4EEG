"""BCI2a (4-class motor imagery) metadata loading and raw ingestion.

Official distribution (bbci.de/competition/iv/#dataset2a): one continuous GDF
file per subject per session -- A0{1-9}T.gdf (training) and A0{1-9}E.gdf
(evaluation). 22 EEG + 3 EOG channels, 250Hz, already bandpass-filtered
0.5-100Hz with a 50Hz notch by the original amplifier (Graz, Austria --
European mains, NOT 60Hz).

Trial timing (see the dataset description PDF): t=0 trial start, cue onset at
t=2s (annotation "769"/"770"/"771"/"772" for left_hand/right_hand/feet/tongue
in T files), motor imagery from cue onset to t=6s. We extract the TRIAL_SEC=4s
window starting at cue onset.

Evaluation files carry no class info themselves (cue annotation "783",
"unknown") -- true labels were released after the competition as a separate
per-session `A0{id}E.mat` ("classlabel", 1-4, matching CLASSES order), see
docs/datasets.md. This mirrors the well-established loading convention used by
braindecode's BCICompetition4Set2A.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.data.preprocessing import EEGProcessor, process_array, write_segment_parquet

log = logging.getLogger(__name__)

LINE_FREQ = 50.0  # European mains -- see module docstring
TRIAL_SEC = 4.0
FS = 128
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
CUE_EVENT_TO_CLASS = {"769": 0, "770": 1, "771": 2, "772": 3}
UNKNOWN_CUE_EVENT = "783"


def load_metadata(meta_csv: str) -> pd.DataFrame:
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=[
        "record_id", "label", "subset", "channels", "s3_data_file", "segment_duration_sec",
    ])
    df = df[df["segment_duration_sec"] >= TRIAL_SEC]
    return df.reset_index(drop=True)


def load_true_labels(mat_path: Path) -> np.ndarray:
    """A0{id}E.mat -- released after the competition. `classlabel` is a
    (n_trials,) array of ints in {1,2,3,4} (matching CLASSES order, 1-indexed)."""
    classlabel = loadmat(mat_path)["classlabel"].ravel()
    return classlabel.astype(int) - 1  # 1-indexed -> 0-indexed, matching CLASSES


def extract_trials(raw: mne.io.BaseRaw, is_eval: bool, true_labels: np.ndarray | None = None):
    """raw: an already filtered/resampled continuous session (see EEGProcessor),
    annotations intact. Returns a list of (label, window (T, C)) for every
    usable trial in this session, cut as [cue_onset, cue_onset + TRIAL_SEC].

    is_eval=True: every "unknown" cue (event 783) is one trial, in order,
    labeled from `true_labels[i]`. is_eval=False: every cue in {769,770,771,772}
    is one trial, labeled directly by the event code."""
    fs = raw.info["sfreq"]
    trial_len = int(round(TRIAL_SEC * fs))
    data = raw.get_data().T  # (T, C)

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    id_to_desc = {code: desc for desc, code in event_id.items()}

    trials = []
    eval_trial_idx = 0
    for sample, _, code in events:
        desc = id_to_desc.get(code)
        if is_eval:
            if desc != UNKNOWN_CUE_EVENT:
                continue
            if true_labels is None:
                raise ValueError("is_eval=True requires true_labels (see load_true_labels)")
            label = int(true_labels[eval_trial_idx])
            eval_trial_idx += 1
        else:
            if desc not in CUE_EVENT_TO_CLASS:
                continue
            label = CUE_EVENT_TO_CLASS[desc]

        window = data[sample:sample + trial_len]
        if window.shape[0] < trial_len:
            log.warning("Trial runs past the end of the recording, skipping")
            continue
        trials.append((label, window))

    return trials


def ingest_session(gdf_path: Path, subset: str, out_dir: Path, labels_dir: Path | None = None,
                    processor: EEGProcessor | None = None) -> list[dict]:
    """gdf_path: one A0{id}{T,E}.gdf session file (many trials). subset: 'train' for
    T files, 'test' for E files (E needs labels_dir -- see module docstring)."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    raw.pick(picks="eeg")  # drop the 3 EOG channels -- provided for artifact removal only, not classification

    processor = processor or EEGProcessor(line_freq=LINE_FREQ)
    raw = processor.process(raw)  # filters/resamples in place, keeps annotations aligned

    is_eval = subset == "test"
    true_labels = load_true_labels(Path(labels_dir) / f"{gdf_path.stem}.mat") if is_eval else None
    trials = extract_trials(raw, is_eval, true_labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (label, window) in enumerate(trials):
        record_id = f"{gdf_path.stem}_trial-{i}"
        seg_path = out_dir / f"{record_id}.parquet"
        write_segment_parquet(window, raw.ch_names, seg_path)
        rows.append({
            "record_id": record_id, "label": label, "subset": subset, "channels": raw.ch_names,
            "s3_data_file": str(seg_path), "segment_duration_sec": TRIAL_SEC,
        })
    return rows


def ingest_dataset(raw_dir: Path, out_dir: Path, labels_dir: Path | None = None,
                    processor: EEGProcessor | None = None) -> pd.DataFrame:
    """raw_dir: A0{1-9}T.gdf and (optionally) A0{1-9}E.gdf, as officially distributed.
    labels_dir: required if raw_dir contains any *E.gdf file (see module docstring)."""
    rows = []
    for gdf_path in sorted(Path(raw_dir).glob("*.gdf")):
        subset = "train" if gdf_path.stem.endswith("T") else "test"
        if subset == "test" and labels_dir is None:
            log.warning(f"Skipping {gdf_path.name}: evaluation session needs --labels-dir (true labels)")
            continue
        rows.extend(ingest_session(gdf_path, subset, Path(out_dir), labels_dir, processor))

    meta = pd.DataFrame(rows)
    meta.to_csv(Path(out_dir) / "metadata.csv", index=False)
    log.info(f"Ingested {len(meta)} trials -> {out_dir}/metadata.csv")
    return meta
