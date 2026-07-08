#!/usr/bin/env python
"""Precompute TUEV topomap-RGB images once and cache them locally. Each row of
--meta-csv is already exactly one padded event window (see src/data/tuev.py --
ingestion cuts and bckg-caps events, this script only renders each one as a
topomap image), so unlike TUAB/BCI2a there's no further windowing here.

--meta-csv is produced by scripts/preprocess_raw.py --dataset tuev.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.tuev import CLASSES, FS, load_metadata
from src.data.utils import parse_channels, read_segment_array
from src.topomap.bands import BAND_DEFS
from src.topomap.cache import allocate_images, finalize_split
from src.topomap.generation import build_topomap_geometry, window_to_image
from src.utils.logging import setup_logging

log = setup_logging("extract_embeddings_tuev")

META_CSV = "data/processed/tuev/metadata.csv"
LOCAL_DIR = Path("data/cache/tuev")
BANDS = {name: BAND_DEFS[name] for name in ["delta", "theta", "alpha"]}
LIMIT = None  # set to a small int for a smoke test


def build_split(df, split_name: str, limit: int | None):
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    total_expected = len(df)  # one window per event, exactly
    log.info(f"[{split_name}] {len(df)} events")

    images = allocate_images(LOCAL_DIR, split_name, total_expected)
    geom_cache: dict = {}
    labels, groups = [], []
    idx = 0
    for n_seen, row in enumerate(df.itertuples(index=False)):
        channels = parse_channels(row.channels)
        geom = geom_cache.setdefault(tuple(channels), build_topomap_geometry(channels))
        try:
            window = read_segment_array(row.s3_data_file)
        except Exception as e:
            log.warning(f"Skipping {row.s3_data_file}: {e}")
            continue

        img = window_to_image(window, geom, BANDS, fs=FS)
        images[idx] = (img * 255.0).astype(np.uint8)
        labels.append(CLASSES.index(row.label))
        groups.append(row.record_id)
        idx += 1

        if (n_seen + 1) % 200 == 0:
            log.info(f"[{split_name}] {n_seen + 1}/{len(df)} events, {idx} windows so far")

    labels_arr = np.array(labels, dtype=np.int64)
    class_counts = {CLASSES[c]: int((labels_arr == c).sum()) for c in range(len(CLASSES))}
    finalize_split(LOCAL_DIR, split_name, images, labels, groups, total_expected, BANDS,
                    {"classes": CLASSES, "class_counts": class_counts})


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata(META_CSV)
    subsets = sorted(df["subset"].unique())
    has_val = "validation" in subsets
    if not has_val:
        log.warning(f"subsets found: {subsets} -- no validation split, carve one from train "
                    "(see src.probing.cross_val.carve_validation_by_group)")

    splits = [("train", "train"), ("test", "test")] + ([("val", "validation")] if has_val else [])
    for split_name, subset_value in splits:
        build_split(df[df["subset"] == subset_value], split_name, LIMIT)
    log.info(f"Cache complete: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
