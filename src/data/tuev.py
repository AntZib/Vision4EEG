"""
Loader for the TUEV dataset (EEG event classification).

TUEV contains continuous EDF recordings with event annotations stored in
matching .rec files.

Dataset structure:

root/
├── train/
└── eval/

Each EDF file has a corresponding annotation file:

recording.edf
recording.rec

The .rec files contain:
    channel_index, start_time, end_time, label_code

The label describes the type of EEG event:
    1 - spsw
    2 - gped
    3 - pled
    4 - eyem
    5 - artf
    6 - bckg

Each annotated event is extracted as one training example. The extracted
window contains the event plus surrounding context (PAD_SEC before and after).

Unlike a sliding-window approach, examples are created only from annotated
events.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.data.preprocessing import (
    EEGProcessor,
    process_array,
    write_segment_parquet,
)


log = logging.getLogger(__name__)

FS = 128
PAD_SEC = 2.0

CLASSES = [
    "spsw",
    "gped",
    "pled",
    "eyem",
    "artf",
    "bckg",
]


def load_metadata(meta_csv: str) -> pd.DataFrame:
    """
    Load event-level metadata.

    Each row corresponds to one annotated EEG event.
    """

    log.info(f"Reading metadata: {meta_csv}")

    df = pd.read_csv(
        meta_csv,
        usecols=[
            "record_id",
            "event_id",
            "label",
            "subset",
            "channels",
            "data_file",
            "segment_duration_sec",
        ],
    )

    return df.reset_index(drop=True)


def load_annotations(rec_path: str) -> pd.DataFrame:
    """
    Load annotations from a TUEV .rec file.

    The file has no header and contains:
        channel, start, stop, label_code
    """

    arr = np.genfromtxt(
        rec_path,
        delimiter=",",
    )

    # Single-event files are loaded as a 1D array by numpy
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return pd.DataFrame(
        arr,
        columns=[
            "channel",
            "start",
            "stop",
            "label_code",
        ],
    )


def extract_events(
    signal: np.ndarray,
    fs: float,
    annots: pd.DataFrame,
    max_bckg: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Extract event windows from a continuous EEG signal.

    Args:
        signal:
            EEG array with shape (time, channels).
        fs:
            Sampling frequency.
        annots:
            DataFrame containing event annotations.
        max_bckg:
            Maximum number of background events to keep.

    Returns:
        List of:
            (label, EEG window)

    Events too close to the beginning or end of the recording are discarded
    because there is not enough surrounding context.
    """

    events = []
    n_bckg = 0

    for row in annots.itertuples(index=False):

        label = CLASSES[int(row.label_code) - 1]

        start_sample = int(
            round((row.start - PAD_SEC) * fs)
        )

        end_sample = int(
            round((row.stop + PAD_SEC) * fs)
        )

        if start_sample < 0 or end_sample > signal.shape[0]:
            continue

        if label == "bckg":

            if max_bckg is not None and n_bckg >= max_bckg:
                continue

            n_bckg += 1

        events.append(
            (
                label,
                signal[start_sample:end_sample],
            )
        )

    return events


def ingest_recording(
    edf_path: Path,
    rec_path: Path,
    subset: str,
    out_dir: Path,
    max_bckg: int | None = None,
    processor: EEGProcessor | None = None,
) -> list[dict]:
    """
    Process one TUEV EDF recording and its annotations.

    The recording is filtered/resampled and each annotated event is saved
    separately as a parquet file.
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

    annots = load_annotations(rec_path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    events = extract_events(
        signal,
        fs,
        annots,
        max_bckg,
    )

    for i, (label, window) in enumerate(events):

        event_id = f"{record_id}_event-{i}"

        seg_path = (
            out_dir
            / f"{event_id}.parquet"
        )

        write_segment_parquet(
            window,
            channels,
            seg_path,
        )

        rows.append(
            {
                "record_id": record_id,
                "event_id": event_id,
                "label": label,
                "subset": subset,
                "channels": channels,
                "data_file": str(seg_path),
                "segment_duration_sec": window.shape[0] / fs,
            }
        )

    return rows


def ingest_dataset(
    raw_dir: Path,
    out_dir: Path,
    max_bckg_per_recording: int | None = 20,
    processor: EEGProcessor | None = None,
) -> pd.DataFrame:
    """
    Process a complete TUEV dataset.

    EDF files are searched recursively. Each EDF must have a matching REC file.
    """

    all_rows = []

    for edf_path in sorted(Path(raw_dir).rglob("*.edf")):

        rec_path = edf_path.with_suffix(".rec")

        if not rec_path.exists():

            log.warning(
                f"Skipping {edf_path}: missing annotation file"
            )

            continue

        subset = (
            "test"
            if "eval" in edf_path.parts
            else "train"
        )

        all_rows.extend(
            ingest_recording(
                edf_path,
                rec_path,
                subset,
                Path(out_dir),
                max_bckg_per_recording,
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
        f"{len(meta)} events at {out_dir}/metadata.csv"
    )

    return meta

