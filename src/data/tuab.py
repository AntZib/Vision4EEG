"""TUAB (normal/abnormal, binary) metadata loading and raw ingestion.

Segment-level metadata: each row is a 300s recording segment. Windows are cut
out of segments downstream (see src/topomap/generation.segment_to_windows).
"""

import logging
from pathlib import Path

import mne
import pandas as pd

from src.data.preprocessing import EEGProcessor, chunk_signal, process_array, write_segment_parquet

log = logging.getLogger(__name__)

WINDOW_SEC = 30
FS = 128
LABEL_MAP = {"normal": 0, "abnormal": 1}
CLASSES = ["normal", "abnormal"]

# Fixed canonical channel order for baselines that need a consistent feature shape
# across segments (TUAB segments don't all list channels in the same order/count).
# Missing channels are zero-padded; see src/models/baselines.py.
CANONICAL_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT9", "FT10", "T7", "C3", "CZ", "C4",
    "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2", "A1", "A2",
]


def load_metadata(meta_csv: str) -> pd.DataFrame:
    """Segment-level metadata, one row per 300s recording segment."""
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=[
        "datalakeID", "label", "subset", "channels", "s3_data_file", "segment_duration_sec",
    ])
    df = df[df["segment_duration_sec"] >= WINDOW_SEC]
    df = df[df["label"].isin(["normal", "abnormal"])]
    return df.reset_index(drop=True)


def load_metadata_per_recording(meta_csv: str) -> pd.DataFrame:
    """One row per unique recording (datalakeID) -- for feature sources that are already
    aggregated per recording, e.g. a pre-built embedding cache keyed by datalakeID."""
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=["datalakeID", "label", "subset"])
    df = df[df["label"].isin(["normal", "abnormal"])]
    return df.drop_duplicates(subset=["datalakeID"]).reset_index(drop=True)


def ingest_recording(edf_path: Path, record_id: str, label: str, subset: str, out_dir: Path,
                      chunk_sec: float = 300.0, processor: EEGProcessor | None = None) -> list[dict]:
    """Read one raw continuous TUAB EDF, filter/resample it to 128Hz, cut it into
    chunk_sec-long segments, and write each as a local parquet. Returns one
    metadata row (dict) per segment, in the same schema as load_metadata expects."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    signal, channels, fs = process_array(raw.get_data().T, raw.ch_names, raw.info["sfreq"], processor)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start_sec, end_sec, chunk in chunk_signal(signal, fs, chunk_sec):
        seg_path = out_dir / f"{record_id}_start_sec-{int(start_sec)}.parquet"
        write_segment_parquet(chunk, channels, seg_path)
        rows.append({
            "datalakeID": record_id, "label": label, "subset": subset, "channels": channels,
            "s3_data_file": str(seg_path), "segment_start_sec": start_sec, "segment_end_sec": end_sec,
            "segment_duration_sec": end_sec - start_sec,
        })
    return rows


def ingest_dataset(raw_dir: Path, labels_csv: Path, out_dir: Path, chunk_sec: float = 300.0,
                    processor: EEGProcessor | None = None) -> pd.DataFrame:
    """raw_dir: one EDF file per recording, named `{record_id}.edf`.
    labels_csv: columns (record_id, label, subset) -- label in {normal, abnormal}, since
    that information isn't encoded in the raw EDF itself. Writes out_dir/metadata.csv."""
    labels = pd.read_csv(labels_csv)
    all_rows = []
    for row in labels.itertuples(index=False):
        edf_path = Path(raw_dir) / f"{row.record_id}.edf"
        if not edf_path.exists():
            log.warning(f"Skipping {row.record_id}: {edf_path} not found")
            continue
        all_rows.extend(ingest_recording(edf_path, row.record_id, row.label, row.subset, Path(out_dir),
                                          chunk_sec, processor))

    meta = pd.DataFrame(all_rows)
    meta.to_csv(Path(out_dir) / "metadata.csv", index=False)
    log.info(f"Ingested {len(labels)} recordings -> {len(meta)} segments at {out_dir}/metadata.csv")
    return meta
