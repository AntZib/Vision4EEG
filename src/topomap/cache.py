
"""
Utilities for reading and writing the topomap image cache.

The cache format is shared between the scripts generating topomap images and
the scripts training/evaluating models on them.

Each split contains:

    <split>_images.dat
        uint8 memmap containing images:
        (n_windows, IMG_SIZE, IMG_SIZE, 3)

    <split>_labels.npy
        Integer label for each window.

    <split>_groups.npy
        Recording/trial identifier associated with each window.

    <split>_meta.json
        Additional information about the cache such as image size and frequency
        bands.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.topomap.generation import IMG_SIZE


log = logging.getLogger(__name__)


def allocate_images(
    local_dir: Path,
    split_name: str,
    total_expected: int,
) -> np.memmap:
    """
    Create an empty memory-mapped image file.

    Using a memmap avoids loading the whole dataset into RAM when generating
    large image caches.
    """

    images_path = local_dir / f"{split_name}_images.dat"

    return np.memmap(
        images_path,
        dtype=np.uint8,
        mode="w+",
        shape=(
            total_expected,
            IMG_SIZE,
            IMG_SIZE,
            3,
        ),
    )


def finalize_split(
    local_dir: Path,
    split_name: str,
    images: np.memmap,
    labels: list,
    groups: list,
    total_expected: int,
    bands: dict,
    extra_meta: dict | None = None,
) -> int:
    """
    Save metadata and labels after image generation is complete.

    The image file is flushed to disk and the number of generated windows is
    stored separately from the initially allocated size.
    """

    images.flush()

    n_windows = len(labels)

    np.save(
        local_dir / f"{split_name}_labels.npy",
        np.array(labels, dtype=np.int64),
    )

    np.save(
        local_dir / f"{split_name}_groups.npy",
        np.array(groups, dtype=object),
    )

    meta = {
        "n_windows": n_windows,
        "total_allocated": total_expected,
        "img_size": IMG_SIZE,
        "bands": list(bands.keys()),
        **(extra_meta or {}),
    }

    (
        local_dir / f"{split_name}_meta.json"
    ).write_text(
        json.dumps(meta)
    )

    images_path = (
        local_dir
        / f"{split_name}_images.dat"
    )

    log.info(
        f"[{split_name}] done: {n_windows} windows "
        f"cached at {images_path} "
        f"({images_path.stat().st_size / 1e9:.3f} GB)"
    )

    return n_windows


def load_split(
    cache_dir: Path,
    split_name: str,
    limit: int | None = None,
):
    """
    Load one dataset split from the cache.

    Returns:
        images:
            Memory-mapped image array.
        labels:
            Labels for each window.
        groups:
            Recording/trial identifiers.
        idx:
            Indices selected after applying an optional limit.

    The limit is applied by recording rather than by individual windows to avoid
    keeping only part of a recording.
    """

    meta = json.loads(
        (
            cache_dir
            / f"{split_name}_meta.json"
        ).read_text()
    )

    n_windows = meta["n_windows"]

    images = np.memmap(
        cache_dir / f"{split_name}_images.dat",
        dtype=np.uint8,
        mode="r",
        shape=(
            meta["total_allocated"],
            IMG_SIZE,
            IMG_SIZE,
            3,
        ),
    )[:n_windows]

    labels = np.load(
        cache_dir / f"{split_name}_labels.npy"
    )

    groups = np.load(
        cache_dir / f"{split_name}_groups.npy",
        allow_pickle=True,
    )

    idx = np.arange(n_windows)

    if limit:

        # Select complete recordings instead of random windows.
        selected_groups = (
            pd.Series(groups)
            .drop_duplicates()
            .sample(
                n=min(
                    limit,
                    len(set(groups)),
                ),
                random_state=42,
            )
        )

        idx = idx[
            np.isin(
                groups,
                selected_groups.to_numpy(),
            )
        ]

    return images, labels, groups, idx
