"""
Tests for EEG to topomap image conversion.

Uses a small set of real 10-20 channel names (montage-matchable) and random
signals, so no real EEG data is needed to run these tests.
"""

import numpy as np
import pytest

from src.topomap.ablations import shuffle_pixels
from src.topomap.bands import BAND_DEFS, band_power_per_channel
from src.topomap.generation import (
    IMG_SIZE,
    build_topomap_geometry,
    segment_to_windows,
    window_to_image,
)

CHANNELS = ["FP1", "FP2", "C3", "C4", "CZ", "O1", "O2", "PZ"]


def test_build_topomap_geometry_matches_all_channels():
    geom = build_topomap_geometry(CHANNELS)

    assert len(geom["matched_idx"]) == len(CHANNELS)
    assert geom["xy"].shape == (len(CHANNELS), 2)
    assert geom["mask"].shape == (IMG_SIZE, IMG_SIZE)


def test_build_topomap_geometry_rejects_too_few_channels():
    # Only 2 channels can be matched to montage positions, need at least 4.
    with pytest.raises(ValueError):
        build_topomap_geometry(["FP1", "FP2"])


def test_band_power_per_channel_shape():
    rng = np.random.default_rng(0)

    # 2 seconds of signal at 128Hz.
    signal = rng.normal(size=(256, len(CHANNELS))).astype(np.float32)
    powers = band_power_per_channel(signal, BAND_DEFS, fs=128)

    assert set(powers.keys()) == set(BAND_DEFS.keys())

    for band_power in powers.values():
        assert band_power.shape == (len(CHANNELS),)


def test_window_to_image_shape_and_range():
    geom = build_topomap_geometry(CHANNELS)
    rng = np.random.default_rng(0)

    # 30 seconds of signal at 128Hz.
    window = rng.normal(size=(30 * 128, len(CHANNELS))).astype(np.float32)
    bands = {
        "delta": BAND_DEFS["delta"],
        "theta": BAND_DEFS["theta"],
        "alpha": BAND_DEFS["alpha"],
    }

    image = window_to_image(window, geom, bands, fs=128)

    assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_segment_to_windows_drops_remainder():
    # 10 seconds of signal cut into 3-second windows -> 3 full windows,
    # the last 1-second remainder is dropped.
    signal = np.zeros((1000, 2), dtype=np.float32)

    windows = segment_to_windows(signal, window_sec=3.0, fs=100.0)

    assert len(windows) == 3
    assert all(window.shape == (300, 2) for window in windows)


def test_shuffle_pixels_preserves_value_distribution():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(8, 8, 3)).astype(np.float32)

    shuffled = shuffle_pixels(image, rng)

    assert shuffled.shape == image.shape
    np.testing.assert_allclose(
        np.sort(image.reshape(-1)),
        np.sort(shuffled.reshape(-1)),
    )
