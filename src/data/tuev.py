"""TUEV (6-class event classification) metadata and raw ingestion.

Official distribution (isip.piconepress.com/projects/tuh_eeg): one continuous
EDF file per recording, laid out as root/{train,eval}/**/*.edf, each with a
matching `.rec` file next to it (same basename) -- a header-less CSV with
columns (channel_index, start_sec, end_sec, label_code). label_code is 1-6:

    1=spsw, 2=gped, 3=pled, 4=eyem, 5=artf, 6=bckg

(channel_index names which channel the event was detected on -- kept as
metadata only, the label applies to the whole multi-channel window, not just
that one channel).

Each row of the .rec file is one ~1s-long annotated event; we extract it as a
[start-PAD_SEC, end+PAD_SEC] window (5s total for a 1s event) from the
filtered/resampled continuous recording -- one window = one labeled example,
NOT a grid of fixed-size windows over the whole recording. This matches the
convention used by BIOT/LaBraM/CBraMod (see
github.com/ycq091044/BIOT/blob/main/datasets/TUEV/process.py, function
BuildEvents). Unlike that reference script, events within PAD_SEC of the
recording's start/end are dropped rather than padded by wrapping the signal
around on itself -- simpler, and avoids stitching in unrelated signal as fake
context.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.data.preprocessing import EEGProcessor, process_array, write_segment_parquet

log = logging.getLogger(__name__)

FS = 128
PAD_SEC = 2.0
CLASSES = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]


def load_metadata(meta_csv: str) -> pd.DataFrame:
    """One row per labeled event/window (not per recording)."""
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=[
        "record_id", "event_id", "label", "subset", "channels", "s3_data_file", "segment_duration_sec",
    ])
    return df.reset_index(drop=True)


def load_annotations(rec_path: str) -> pd.DataFrame:
    """A TUEV .rec file: header-less CSV, columns (channel, start, stop, label_code)."""
    arr = np.genfromtxt(rec_path, delimiter=",")
    if arr.ndim == 1:  # a single-event .rec loads as a 1D array
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr, columns=["channel", "start", "stop", "label_code"])


def extract_events(signal: np.ndarray, fs: float, annots: pd.DataFrame,
                    max_bckg: int | None = None) -> list[tuple[str, np.ndarray]]:
    """signal: (T, C), already filtered/resampled. Returns [(label, window (T', C)), ...],
    one per usable row of `annots` (see load_annotations) -- dropped if too close to the
    recording's start/end (see module docstring), capped for "bckg" via max_bckg."""
    events = []
    n_bckg = 0
    for row in annots.itertuples(index=False):
        label = CLASSES[int(row.label_code) - 1]
        start_sample = int(round((row.start - PAD_SEC) * fs))
        end_sample = int(round((row.stop + PAD_SEC) * fs))
        if start_sample < 0 or end_sample > signal.shape[0]:
            continue  # too close to the recording's start/end, see module docstring

        if label == "bckg":
            if max_bckg is not None and n_bckg >= max_bckg:
                continue
            n_bckg += 1

        events.append((label, signal[start_sample:end_sample]))
    return events


def ingest_recording(edf_path: Path, rec_path: Path, subset: str, out_dir: Path,
                      max_bckg: int | None = None, processor: EEGProcessor | None = None) -> list[dict]:
    """Read one raw continuous TUEV EDF + its matching .rec annotations, filter/
    resample to 128Hz, and write one parquet per annotated event."""
    record_id = edf_path.stem
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    signal, channels, fs = process_array(raw.get_data().T, raw.ch_names, raw.info["sfreq"], processor)
    annots = load_annotations(rec_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (label, window) in enumerate(extract_events(signal, fs, annots, max_bckg)):
        event_id = f"{record_id}_event-{i}"
        seg_path = out_dir / f"{event_id}.parquet"
        write_segment_parquet(window, channels, seg_path)
        rows.append({
            "record_id": record_id, "event_id": event_id, "label": label, "subset": subset,
            "channels": channels, "s3_data_file": str(seg_path), "segment_duration_sec": window.shape[0] / fs,
        })
    return rows


def ingest_dataset(raw_dir: Path, out_dir: Path, max_bckg_per_recording: int | None = 20,
                    processor: EEGProcessor | None = None) -> pd.DataFrame:
    """raw_dir: the official TUH EEG Events Corpus layout -- root/{train,eval}/**/*.edf,
    each with a matching *.rec next to it (see module docstring and docs/datasets.md).
    Writes out_dir/metadata.csv."""
    all_rows = []
    for edf_path in sorted(Path(raw_dir).rglob("*.edf")):
        rec_path = edf_path.with_suffix(".rec")
        if not rec_path.exists():
            log.warning(f"Skipping {edf_path}: no matching .rec file")
            continue
        subset = "test" if "eval" in edf_path.parts else "train"
        all_rows.extend(ingest_recording(edf_path, rec_path, subset, Path(out_dir), max_bckg_per_recording, processor))

    meta = pd.DataFrame(all_rows)
    meta.to_csv(Path(out_dir) / "metadata.csv", index=False)
    n_recordings = meta["record_id"].nunique() if len(meta) else 0
    log.info(f"Ingested {n_recordings} recordings -> {len(meta)} events at {out_dir}/metadata.csv")
    return meta
