"""BCI2a (4-class motor imagery) metadata loading and raw ingestion.

Trials are already exactly TRIAL_SEC long (segment_duration_sec == 5.0 for every
row) -- one window = one whole trial, no chunking like TUAB's 30s-out-of-300s.
"""

import logging
from pathlib import Path

import mne
import pandas as pd

from src.data.preprocessing import EEGProcessor, process_array, write_segment_parquet

log = logging.getLogger(__name__)

TRIAL_SEC = 5.0
FS = 128
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


def load_metadata(meta_csv: str) -> pd.DataFrame:
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=[
        "record_id", "label", "subset", "channels", "s3_data_file", "segment_duration_sec",
    ])
    df = df[df["segment_duration_sec"] >= TRIAL_SEC]
    return df.reset_index(drop=True)


def ingest_trial(edf_path: Path, out_dir: Path, processor: EEGProcessor | None = None) -> dict:
    """One EDF per trial, already exactly TRIAL_SEC long. The trial's class label
    is stored in the EDF header's subject_info.his_id (a class name from CLASSES),
    and the split (train/test) is encoded in the filename as `session-{n}{train,test}`."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    label_name = raw.info["subject_info"]["his_id"]
    subset = "train" if "train" in edf_path.stem else "test"
    record_id = edf_path.stem

    signal, channels, fs = process_array(raw.get_data().T, raw.ch_names, raw.info["sfreq"], processor)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / f"{record_id}.parquet"
    write_segment_parquet(signal, channels, seg_path)

    return {
        "record_id": record_id, "label": CLASSES.index(label_name), "subset": subset, "channels": channels,
        "s3_data_file": str(seg_path), "segment_duration_sec": signal.shape[0] / fs,
    }


def ingest_dataset(raw_dir: Path, out_dir: Path, processor: EEGProcessor | None = None) -> pd.DataFrame:
    """raw_dir: one EDF file per trial (see ingest_trial). Writes out_dir/metadata.csv."""
    rows = [ingest_trial(edf_path, Path(out_dir), processor) for edf_path in sorted(Path(raw_dir).glob("*.edf"))]
    meta = pd.DataFrame(rows)
    meta.to_csv(Path(out_dir) / "metadata.csv", index=False)
    log.info(f"Ingested {len(meta)} trials -> {out_dir}/metadata.csv")
    return meta
