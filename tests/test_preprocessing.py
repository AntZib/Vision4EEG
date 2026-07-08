"""
Tests for the raw EEG preprocessing pipeline.

These tests use synthetic signals only, so they can run without any real
EEG data available.
"""

import numpy as np

from src.data.preprocessing import (
    EEGProcessor,
    chunk_signal,
    process_array,
    write_segment_parquet,
)
from src.data.utils import read_segment_array


def synthetic_signal(n_samples=1280, n_channels=4, fs=256.0):
    """
    Generate a fake EEG-like signal: a 10Hz sine wave plus noise, one
    channel repeated with independent noise per channel.
    """

    rng = np.random.default_rng(0)

    t = np.arange(n_samples) / fs
    sine = np.sin(2 * np.pi * 10 * t)[:, None]
    noise = rng.normal(scale=0.1, size=(n_samples, n_channels))

    return (sine + noise).astype(np.float32)


def test_process_array_resamples_to_target_fs():
    # 5 seconds of signal at 256Hz should come back at 128Hz.
    signal = synthetic_signal(n_samples=1280, fs=256.0)

    processed, channels, fs = process_array(
        signal,
        ["C3", "C4", "Cz", "Fz"],
        fs=256.0,
    )

    assert fs == 128
    assert processed.shape == (640, 4)
    assert channels == ["C3", "C4", "Cz", "Fz"]
    assert not np.isnan(processed).any()


def test_process_array_no_resample_when_already_target_fs():
    signal = synthetic_signal(n_samples=640, fs=128.0)

    processed, _, fs = process_array(
        signal,
        ["C3", "C4", "Cz", "Fz"],
        fs=128.0,
    )

    assert fs == 128
    assert processed.shape[0] == 640


def test_car_reference_zeros_channel_mean():
    """
    Common average reference subtracts the mean across channels, so at
    every timepoint the channels should sum to roughly zero.
    """

    signal = synthetic_signal(n_samples=640, n_channels=4, fs=128.0)
    processor = EEGProcessor(target_fs=128, apply_car=True)

    processed, _, _ = process_array(
        signal,
        ["C3", "C4", "Cz", "Fz"],
        fs=128.0,
        processor=processor,
    )

    assert np.allclose(processed.mean(axis=1), 0.0, atol=1e-6)


def test_chunk_signal_keeps_short_final_chunk():
    # 10 seconds of signal, cut into 3-second chunks -> 3 full chunks and
    # one shorter 1-second chunk at the end, not dropped.
    signal = np.zeros((1000, 2), dtype=np.float32)

    chunks = chunk_signal(signal, fs=100.0, chunk_sec=3.0)
    durations = [round(end - start, 2) for start, end, _ in chunks]

    assert durations == [3.0, 3.0, 3.0, 1.0]
    assert sum(chunk.shape[0] for _, _, chunk in chunks) == 1000


def test_write_segment_parquet_roundtrip(tmp_path):
    signal = synthetic_signal(n_samples=128, n_channels=3, fs=128.0)
    channels = ["C3", "C4", "Cz"]
    path = tmp_path / "segment.parquet"

    write_segment_parquet(signal, channels, path)
    reloaded = read_segment_array(str(path))

    assert reloaded.shape == signal.shape
    np.testing.assert_allclose(reloaded, signal, atol=1e-5)
