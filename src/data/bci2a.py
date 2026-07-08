"""
Loader for the BCI Competition IV 2a dataset.

The dataset contains one continuous GDF file per subject and session:
- A0{id}T.gdf: training data with class annotations
- A0{id}E.gdf: evaluation data without labels

The recordings contain 22 EEG channels and 3 EOG channels sampled at 250 Hz.
The EOG channels are removed since they are not used for classification.

Training trials are extracted from the motor imagery period:
cue onset + 4 seconds (t=2s to t=6s).

Evaluation labels are provided separately in A0{id}E.mat files.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.data.preprocessing import EEGProcessor, write_segment_parquet


log = logging.getLogger(__name__)

LINE_FREQ = 50.0
TRIAL_SEC = 4.0
FS = 128

CLASSES = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

# Event codes used in training files
CUE_EVENT_TO_CLASS = {
    "769": 0,
    "770": 1,
    "771": 2,
    "772": 3,
}

# Event code used in evaluation files
UNKNOWN_CUE_EVENT = "783"


def load_metadata(meta_csv: str) -> pd.DataFrame:
    """Load metadata file and keep only usable segments."""
    log.info(f"Reading metadata: {meta_csv}")

    df = pd.read_csv(
        meta_csv,
        usecols=[
            "record_id",
            "label",
            "subset",
            "channels",
            "data_file",
            "segment_duration_sec",
        ],
    )

    df = df[df["segment_duration_sec"] >= TRIAL_SEC]

    return df.reset_index(drop=True)


def load_true_labels(mat_path: Path) -> np.ndarray:
    """
    Load labels from the .mat files provided for evaluation sessions.

    The original labels are stored as integers from 1 to 4.
    They are converted to 0-3 to match the class indexing used here.
    """
    labels = loadmat(mat_path)["classlabel"].ravel()
    return labels.astype(int) - 1


def extract_trials(
    raw: mne.io.BaseRaw,
    is_eval: bool,
    true_labels: np.ndarray | None = None,
):
    """
    Extract motor imagery trials from a continuous recording.

    Returns:
        list of tuples:
            (class_id, EEG window with shape (time, channels))

    Training:
        labels are obtained from the cue events.

    Evaluation:
        trials are identified by the 783 event and labels are loaded
        from the corresponding .mat file.
    """

    fs = raw.info["sfreq"]
    trial_len = int(round(TRIAL_SEC * fs))

    # MNE stores data as (channels, samples), we use (samples, channels)
    data = raw.get_data().T

    events, event_id = mne.events_from_annotations(
        raw,
        verbose="ERROR",
    )

    id_to_desc = {
        code: desc
        for desc, code in event_id.items()
    }

    trials = []
    eval_trial_idx = 0

    for sample, _, code in events:

        desc = id_to_desc.get(code)

        if is_eval:
            if desc != UNKNOWN_CUE_EVENT:
                continue

            if true_labels is None:
                raise ValueError(
                    "Evaluation data requires labels from A0{id}E.mat"
                )

            label = int(true_labels[eval_trial_idx])
            eval_trial_idx += 1

        else:
            if desc not in CUE_EVENT_TO_CLASS:
                continue

            label = CUE_EVENT_TO_CLASS[desc]

        window = data[sample:sample + trial_len]

        if window.shape[0] < trial_len:
            log.warning(
                "Trial reaches the end of the recording, skipping it"
            )
            continue

        trials.append((label, window))

    return trials


def ingest_session(
    gdf_path: Path,
    subset: str,
    out_dir: Path,
    labels_dir: Path | None = None,
    processor: EEGProcessor | None = None,
) -> list[dict]:
    """
    Load one BCI2a recording and save extracted trials.

    gdf_path:
        Path to A0{id}T.gdf or A0{id}E.gdf.

    subset:
        "train" for T files, "test" for E files.
    """

    raw = mne.io.read_raw_gdf(
        gdf_path,
        preload=True,
        verbose="ERROR",
    )

    # Remove EOG channels
    raw.pick(picks="eeg")

    processor = processor or EEGProcessor(
        line_freq=LINE_FREQ
    )

    raw = processor.process(raw)

    is_eval = subset == "test"

    true_labels = None
    if is_eval:
        if labels_dir is None:
            raise ValueError(
                "Evaluation sessions require labels_dir"
            )

        true_labels = load_true_labels(
            labels_dir / f"{gdf_path.stem}.mat"
        )

    trials = extract_trials(
        raw,
        is_eval,
        true_labels,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for i, (label, window) in enumerate(trials):

        record_id = f"{gdf_path.stem}_trial-{i}"

        seg_path = out_dir / f"{record_id}.parquet"

        write_segment_parquet(
            window,
            raw.ch_names,
            seg_path,
        )

        rows.append(
            {
                "record_id": record_id,
                "label": label,
                "subset": subset,
                "channels": raw.ch_names,
                "data_file": str(seg_path),
                "segment_duration_sec": TRIAL_SEC,
            }
        )

    return rows


def ingest_dataset(
    raw_dir: Path,
    out_dir: Path,
    labels_dir: Path | None = None,
    processor: EEGProcessor | None = None,
) -> pd.DataFrame:
    """
    Process all BCI2a GDF files in a directory.

    Evaluation files are skipped if no label directory is provided.
    """

    rows = []

    for gdf_path in sorted(Path(raw_dir).glob("*.gdf")):

        subset = (
            "train"
            if gdf_path.stem.endswith("T")
            else "test"
        )

        if subset == "test" and labels_dir is None:
            log.warning(
                f"Skipping {gdf_path.name}: missing labels"
            )
            continue

        rows.extend(
            ingest_session(
                gdf_path,
                subset,
                Path(out_dir),
                labels_dir,
                processor,
            )
        )

    meta = pd.DataFrame(rows)

    meta.to_csv(
        Path(out_dir) / "metadata.csv",
        index=False,
    )

    log.info(
        f"Saved metadata for {len(meta)} trials"
    )

    return meta