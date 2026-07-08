#!/usr/bin/env python
"""Turn raw EDF/GDF recordings into the 128Hz segment parquets + metadata.csv
that every other script in this repo consumes (see src/data/{tuab,bci2a,tuev}.py).

No external ETL/cloud dependency -- everything runs locally with mne. Each
dataset's --raw-dir is expected in its official distribution layout (folder
structure, file naming) -- see docs/datasets.md for exactly what that means
and where to obtain each dataset.

Usage:
    python scripts/preprocess_raw.py --dataset tuab --raw-dir data/raw/tuab --out-dir data/processed/tuab
    python scripts/preprocess_raw.py --dataset bci2a --raw-dir data/raw/bci2a --labels-dir data/raw/bci2a_true_labels --out-dir data/processed/bci2a
    python scripts/preprocess_raw.py --dataset tuev --raw-dir data/raw/tuev --out-dir data/processed/tuev --max-bckg-per-recording 20
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
    ap.add_argument("--raw-dir", required=True,
                     help="dataset in its official distribution layout -- see docs/datasets.md")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels-dir", default=None,
                     help="BCI2a only -- true labels for *E.gdf evaluation sessions (training sessions are "
                          "labeled directly from their own GDF cue events, no dir needed for those)")
    ap.add_argument("--max-bckg-per-recording", type=int, default=20,
                     help="TUEV only -- cap on 'bckg' events kept per recording (bckg otherwise dominates hugely)")
    ap.add_argument("--chunk-sec", type=float, default=300.0, help="TUAB only -- segment length before windowing")
    ap.add_argument("--line-freq", type=float, default=None,
                     help="powerline frequency to notch out -- defaults to 50Hz for bci2a (Europe), 60Hz otherwise (US)")
    ap.add_argument("--highpass-hz", type=float, default=0.5)
    ap.add_argument("--no-car", action="store_true", help="skip common average re-referencing")
    args = ap.parse_args()

    default_line_freq = 50.0 if args.dataset == "bci2a" else 60.0
    line_freq = args.line_freq if args.line_freq is not None else default_line_freq
    processor = EEGProcessor(line_freq=line_freq, highpass_hz=args.highpass_hz, apply_car=not args.no_car)
    out_dir = Path(args.out_dir)

    if args.dataset == "bci2a":
        labels_dir = Path(args.labels_dir) if args.labels_dir else None
        meta = bci2a_data.ingest_dataset(Path(args.raw_dir), out_dir, labels_dir, processor)
    elif args.dataset == "tuab":
        meta = tuab_data.ingest_dataset(Path(args.raw_dir), out_dir, chunk_sec=args.chunk_sec, processor=processor)
    else:  # tuev
        meta = tuev_data.ingest_dataset(Path(args.raw_dir), out_dir,
                                         max_bckg_per_recording=args.max_bckg_per_recording, processor=processor)

    log.info(f"Done: {len(meta)} segments written to {out_dir}/metadata.csv")


if __name__ == "__main__":
    main()
