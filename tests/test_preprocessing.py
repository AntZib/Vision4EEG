import numpy as np

from src.data.preprocessing import EEGProcessor, chunk_signal, process_array, write_segment_parquet
from src.data.utils import read_segment_array


def synthetic_signal(n_samples=1280, n_channels=4, fs=256.0):
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / fs
    sine = np.sin(2 * np.pi * 10 * t)[:, None]  # 10Hz alpha-ish component
    noise = rng.normal(scale=0.1, size=(n_samples, n_channels))
    return (sine + noise).astype(np.float32)


def test_process_array_resamples_to_target_fs():
    signal = synthetic_signal(n_samples=1280, fs=256.0)  # 5s @ 256Hz
    processed, channels, fs = process_array(signal, ["C3", "C4", "Cz", "Fz"], fs=256.0)

    assert fs == 128
    assert processed.shape == (640, 4)  # 5s @ 128Hz
    assert channels == ["C3", "C4", "Cz", "Fz"]
    assert not np.isnan(processed).any()


def test_process_array_no_resample_when_already_target_fs():
    signal = synthetic_signal(n_samples=640, fs=128.0)
    processed, _, fs = process_array(signal, ["C3", "C4", "Cz", "Fz"], fs=128.0)
    assert fs == 128
    assert processed.shape[0] == 640


def test_car_reference_zeros_channel_mean():
    signal = synthetic_signal(n_samples=640, n_channels=4, fs=128.0)
    processor = EEGProcessor(target_fs=128, apply_car=True)
    processed, _, _ = process_array(signal, ["C3", "C4", "Cz", "Fz"], fs=128.0, processor=processor)
    # Common average reference: at every timepoint, channels sum to ~0.
    assert np.allclose(processed.mean(axis=1), 0.0, atol=1e-6)


def test_chunk_signal_keeps_short_final_chunk():
    signal = np.zeros((1000, 2), dtype=np.float32)
    chunks = chunk_signal(signal, fs=100.0, chunk_sec=3.0)  # 10s signal, 3s chunks
    durations = [round(end - start, 2) for start, end, _ in chunks]
    assert durations == [3.0, 3.0, 3.0, 1.0]
    assert sum(c.shape[0] for _, _, c in chunks) == 1000


def test_write_segment_parquet_roundtrip(tmp_path):
    signal = synthetic_signal(n_samples=128, n_channels=3, fs=128.0)
    channels = ["C3", "C4", "Cz"]
    path = tmp_path / "segment.parquet"
    write_segment_parquet(signal, channels, path)

    back = read_segment_array(str(path))
    assert back.shape == signal.shape
    np.testing.assert_allclose(back, signal, atol=1e-5)
