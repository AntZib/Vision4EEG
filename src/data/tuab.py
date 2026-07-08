"""
Loader for the TUAB dataset (normal vs abnormal EEG classification).

TUAB provides continuous EDF recordings organized by split and label:

root/
├── train/
│   ├── normal/
│   └── abnormal/
└── eval/
    ├── normal/
    └── abnormal/

The label and subset can be inferred directly from the directory structure.
No additional label file is required.

This module preprocesses raw recordings, splits them into fixed-length
segments, and stores them as parquet files. Each metadata row corresponds to
one segment extracted from a recording.
"""

import logging
from pathlib import Path

import mne
import pandas as pd

from src.data.preprocessing import (
    EEGProcessor,
    chunk_signal,
    process_array,
    write_segment_parquet,
)


log = logging.getLogger(__name__)

WINDOW_SEC = 30
FS = 128

LABEL_MAP = {
    "normal": 0,
    "abnormal": 1,
}

CLASSES = [
    "normal",
    "abnormal",
]


# Some models require a fixed channel dimension.
# TUAB recordings may have different channel orders or missing channels.
# Missing channels are handled by zero-padding in the model preprocessing.
CANONICAL_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT9", "FT10", "T7", "C3", "CZ", "C4",
    "T8", "P7", "P3", "PZ", "P4", "P8",
    "O1", "O2", "A1", "A2",
]


def load_metadata(meta_csv: str) -> pd.DataFrame:
    """
    Load segment-level metadata.

    One row corresponds to one extracted EEG segment.
    """

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

    df = df[df["segment_duration_sec"] >= WINDOW_SEC]
    df = df[df["label"].isin(CLASSES)]

    return df.reset_index(drop=True)


def load_metadata_per_recording(meta_csv: str) -> pd.DataFrame:
    """
    Load metadata with one row per recording.

    Useful when using features already computed at the recording level.
    """

    log.info(f"Reading metadata: {meta_csv}")

    df = pd.read_csv(
        meta_csv,
        usecols=[
            "record_id",
            "label",
            "subset",
        ],
    )

    df = df[df["label"].isin(CLASSES)]

    return (
        df.drop_duplicates(subset=["record_id"])
        .reset_index(drop=True)
    )


def ingest_recording(
    edf_path: Path,
    label: str,
    subset: str,
    out_dir: Path,
    chunk_sec: float = 300.0,
    processor: EEGProcessor | None = None,
) -> list[dict]:
    """
    Load one TUAB EDF recording.

    The recording is:
        1. loaded with MNE
        2. preprocessed
        3. split into fixed-size segments
        4. saved as parquet files

    Returns metadata entries for all generated segments.
    """

    record_id = edf_path.stem

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose="ERROR",
    )

    signal, channels, fs = process_array(
        raw.get_data().T,
        raw.ch_names,
        raw.info["sfreq"],
        processor,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for start_sec, end_sec, chunk in chunk_signal(
        signal,
        fs,
        chunk_sec,
    ):

        seg_path = (
            out_dir
            / f"{record_id}_start_sec-{int(start_sec)}.parquet"
        )

        write_segment_parquet(
            chunk,
            channels,
            seg_path,
        )

        rows.append(
            {
                "record_id": record_id,
                "label": label,
                "subset": subset,
                "channels": channels,
                "data_file": str(seg_path),
                "segment_start_sec": start_sec,
                "segment_end_sec": end_sec,
                "segment_duration_sec": end_sec - start_sec,
            }
        )

    return rows


def ingest_dataset(
    raw_dir: Path,
    out_dir: Path,
    chunk_sec: float = 300.0,
    processor: EEGProcessor | None = None,
) -> pd.DataFrame:
    """
    Process a TUAB directory.

    The label and split are inferred from the EDF path:
        train/eval -> subset
        normal/abnormal -> label

    Saves the generated metadata to metadata.csv.
    """

    all_rows = []

    for edf_path in sorted(Path(raw_dir).rglob("*.edf")):

        parts = set(edf_path.parts)

        if "abnormal" in parts:
            label = "abnormal"

        elif "normal" in parts:
            label = "normal"

        else:
            log.warning(
                f"Skipping {edf_path}: unable to infer label"
            )
            continue

        subset = (
            "test"
            if "eval" in parts
            else "train"
        )

        all_rows.extend(
            ingest_recording(
                edf_path,
                label,
                subset,
                Path(out_dir),
                chunk_sec,
                processor,
            )
        )

    meta = pd.DataFrame(all_rows)

    meta.to_csv(
        Path(out_dir) / "metadata.csv",
        index=False,
    )

    n_recordings = (
        meta["record_id"].nunique()
        if len(meta)
        else 0
    )

    log.info(
        f"Ingested {n_recordings} recordings -> "
        f"{len(meta)} segments at {out_dir}/metadata.csv"
    )

    return meta