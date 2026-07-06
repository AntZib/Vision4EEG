#!/usr/bin/env python
"""Turn raw EDF recordings into the 128Hz segment parquets + metadata.csv that
every other script in this repo consumes (see src/data/{tuab,bci2a,tuev}.py).

No external ETL/cloud dependency -- everything runs locally with mne. See
docs/datasets.md for where to obtain each dataset and how to lay out --raw-dir.

Usage:
    python scripts/preprocess_raw.py --dataset bci2a --raw-dir data/raw/bci2a --out-dir data/processed/bci2a
    python scripts/preprocess_raw.py --dataset tuab --raw-dir data/raw/tuab --labels-csv data/raw/tuab_labels.csv --out-dir data/processed/tuab
    python scripts/preprocess_raw.py --dataset tuev --raw-dir data/raw/tuev/edf --annotations-dir data/raw/tuev/annots --labels-csv data/raw/tuev_labels.csv --out-dir data/processed/tuev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import bci2a as bci2a_data
from src.data import tuab as tuab_data
from src.data import tuev as tuev_data
from src.data.preprocessing import EEGProcessor
from src.utils.logging import setup_logging

log = setup_logging("preprocess_raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tuab", "bci2a", "tuev"])
    ap.add_argument("--raw-dir", required=True, help="directory of raw EDF files (see docstring per dataset)")
    ap.add_argument("--annotations-dir", default=None, help="TUEV only -- per-recording (start,stop,label) parquets")
    ap.add_argument("--labels-csv", default=None,
                     help="TUAB: columns (record_id, label, subset). TUEV: columns (record_id, subset). "
                          "Not needed for BCI2a (label/subset are in the EDF header/filename).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-sec", type=float, default=300.0, help="TUAB/TUEV only -- segment length before windowing")
    ap.add_argument("--line-freq", type=float, default=60.0, help="powerline frequency to notch out (60 US, 50 EU)")
    ap.add_argument("--highpass-hz", type=float, default=0.5)
    ap.add_argument("--no-car", action="store_true", help="skip common average re-referencing")
    args = ap.parse_args()

    processor = EEGProcessor(line_freq=args.line_freq, highpass_hz=args.highpass_hz, apply_car=not args.no_car)
    out_dir = Path(args.out_dir)

    if args.dataset == "bci2a":
        meta = bci2a_data.ingest_dataset(Path(args.raw_dir), out_dir)
    elif args.dataset == "tuab":
        if not args.labels_csv:
            ap.error("--labels-csv is required for --dataset tuab")
        meta = tuab_data.ingest_dataset(Path(args.raw_dir), Path(args.labels_csv), out_dir, chunk_sec=args.chunk_sec)
    else:  # tuev
        if not args.labels_csv or not args.annotations_dir:
            ap.error("--labels-csv and --annotations-dir are required for --dataset tuev")
        meta = tuev_data.ingest_dataset(Path(args.raw_dir), Path(args.annotations_dir), Path(args.labels_csv),
                                         out_dir, chunk_sec=args.chunk_sec)

    log.info(f"Done: {len(meta)} segments written to {out_dir}/metadata.csv")


if __name__ == "__main__":
    main()
