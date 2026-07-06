"""TUEV (6-class event classification) metadata and annotation loading.

TUEV convention (matches BIOT/LaBraM/CBraMod preprocessing): windows are 5s, one per
labeled interval, NOT a majority-vote sliding grid. Each TUAB-style 300s segment
parquet is chunked into non-overlapping 5s windows; a window is kept only if it falls
ENTIRELY within a single annotated interval (boundary-straddling windows are dropped,
same as the literature convention -- avoids ambiguous/mixed-label windows).

Labels come from a per-RECORDING annotation parquet (path in the "label" column of the
segment metadata, shared across all 300s chunks of that recording), with columns
(start, stop, label) in absolute recording-time seconds.
"""

import logging
from pathlib import Path

import mne
import pandas as pd

from src.data.preprocessing import EEGProcessor, chunk_signal, process_array, write_segment_parquet
from src.data.utils import get_s3fs

log = logging.getLogger(__name__)

WINDOW_SEC = 5.0
FS = 128
CLASSES = ["bckg", "artf", "eyem", "spsw", "gped", "pled"]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

_ANNOT_CACHE: dict = {}


def load_metadata(meta_csv: str) -> pd.DataFrame:
    log.info(f"Reading metadata: {meta_csv}")
    df = pd.read_csv(meta_csv, usecols=[
        "datalakeID", "label", "subset", "channels", "s3_data_file",
        "segment_start_sec", "segment_end_sec", "segment_duration_sec",
    ])
    df = df[df["segment_duration_sec"] >= WINDOW_SEC]
    return df.reset_index(drop=True)


def load_annotations(annot_path: str) -> pd.DataFrame:
    """One (start, stop, label) dataframe per recording, covering its full duration. Cached
    since every 300s chunk of the same recording points at the same annotation file."""
    if annot_path not in _ANNOT_CACHE:
        if annot_path.startswith("s3://"):
            with get_s3fs().open(annot_path.replace("s3://", ""), "rb") as f:
                _ANNOT_CACHE[annot_path] = pd.read_parquet(f)
        else:
            _ANNOT_CACHE[annot_path] = pd.read_parquet(annot_path)
    return _ANNOT_CACHE[annot_path]


def label_for_window(annots: pd.DataFrame, abs_start: float, abs_end: float) -> str | None:
    """Return the label of the single annotated interval that fully contains
    [abs_start, abs_end), or None if the window straddles a boundary / isn't covered."""
    covering = annots[(annots["start"] <= abs_start) & (annots["stop"] >= abs_end)]
    if len(covering) != 1:
        return None
    return covering.iloc[0]["label"]


def ingest_recording(edf_path: Path, record_id: str, annot_path: Path, subset: str, out_dir: Path,
                      chunk_sec: float = 300.0, processor: EEGProcessor | None = None) -> list[dict]:
    """Read one raw continuous TUEV EDF, filter/resample it to 128Hz, cut it into
    chunk_sec-long segments, and write each as a local parquet. annot_path: a local
    (start, stop, label) parquet for the whole recording (see load_annotations) --
    referenced, not copied, since every chunk of this recording shares the same one."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    signal, channels, fs = process_array(raw.get_data().T, raw.ch_names, raw.info["sfreq"], processor)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start_sec, end_sec, chunk in chunk_signal(signal, fs, chunk_sec):
        seg_path = out_dir / f"{record_id}_start_sec-{int(start_sec)}.parquet"
        write_segment_parquet(chunk, channels, seg_path)
        rows.append({
            "datalakeID": record_id, "label": str(annot_path), "subset": subset, "channels": channels,
            "s3_data_file": str(seg_path), "segment_start_sec": start_sec, "segment_end_sec": end_sec,
            "segment_duration_sec": end_sec - start_sec,
        })
    return rows


def ingest_dataset(raw_dir: Path, annotations_dir: Path, labels_csv: Path, out_dir: Path,
                    chunk_sec: float = 300.0, processor: EEGProcessor | None = None) -> pd.DataFrame:
    """raw_dir: one EDF file per recording, named `{record_id}.edf`.
    annotations_dir: one (start, stop, label) parquet per recording, named `{record_id}.parquet`.
    labels_csv: columns (record_id, subset) -- TUEV has no whole-recording label (events are
    labeled per interval, see annotations), only a train/test split to carry over.
    Writes out_dir/metadata.csv."""
    labels = pd.read_csv(labels_csv)
    all_rows = []
    for row in labels.itertuples(index=False):
        edf_path = Path(raw_dir) / f"{row.record_id}.edf"
        annot_path = Path(annotations_dir) / f"{row.record_id}.parquet"
        if not edf_path.exists() or not annot_path.exists():
            log.warning(f"Skipping {row.record_id}: missing {edf_path} or {annot_path}")
            continue
        all_rows.extend(ingest_recording(edf_path, row.record_id, annot_path, row.subset, Path(out_dir),
                                          chunk_sec, processor))

    meta = pd.DataFrame(all_rows)
    meta.to_csv(Path(out_dir) / "metadata.csv", index=False)
    log.info(f"Ingested {len(labels)} recordings -> {len(meta)} segments at {out_dir}/metadata.csv")
    return meta
