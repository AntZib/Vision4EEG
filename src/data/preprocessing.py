"""Raw EEG -> filtered/resampled/segmented parquet, from scratch.

No external ETL dependency -- this is the one place a raw continuous recording
(any native sampling rate, standard 10-20/10-05 channel names) is turned into the
128Hz segment parquets that every other module in this repo (src/topomap,
src/data/{tuab,bci2a,tuev}.py) consumes. See scripts/preprocess_raw.py for the
per-dataset CLI that walks a raw-data directory and calls this module.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

mne.set_log_level("ERROR")

TARGET_FS = 128


@dataclass
class EEGProcessor:
    """notch -> high-pass -> resample -> common average reference, in that order.

    Notch first (removes powerline harmonics before they alias down during
    resampling), high-pass before resampling (cheaper at the higher native rate),
    CAR last (needs the final, clean, already-referenced-per-channel signal)."""
    target_fs: float = TARGET_FS
    line_freq: float = 60.0
    highpass_hz: float = 0.5
    apply_car: bool = True

    def process(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        nyquist = raw.info["sfreq"] / 2
        harmonics = np.arange(self.line_freq, nyquist, self.line_freq)
        if len(harmonics):
            raw.notch_filter(harmonics, verbose="ERROR")
        raw.filter(l_freq=self.highpass_hz, h_freq=None, verbose="ERROR")
        if raw.info["sfreq"] != self.target_fs:
            raw.resample(self.target_fs, verbose="ERROR")
        if self.apply_car:
            raw.set_eeg_reference("average", verbose="ERROR")
        return raw


def build_raw(signal: np.ndarray, channel_names: list[str], fs: float) -> mne.io.RawArray:
    """signal: (T, C) in volts-ish scale (mne assumes volts for EEG; if the source
    is already in microvolts, scale by 1e-6 before calling this)."""
    info = mne.create_info(ch_names=list(channel_names), sfreq=fs, ch_types="eeg")
    return mne.io.RawArray(signal.T, info, verbose="ERROR")


def process_array(signal: np.ndarray, channel_names: list[str], fs: float,
                   processor: EEGProcessor | None = None) -> tuple[np.ndarray, list[str], float]:
    """signal: (T, C). Returns (processed_signal (T', C), channel_names, new_fs)."""
    processor = processor or EEGProcessor()
    raw = build_raw(signal, channel_names, fs)
    raw = processor.process(raw)
    return raw.get_data().T, raw.ch_names, raw.info["sfreq"]


def chunk_signal(signal: np.ndarray, fs: float, chunk_sec: float) -> list[tuple[float, float, np.ndarray]]:
    """signal: (T, C). Split into non-overlapping chunks of `chunk_sec`, keeping a
    shorter final chunk instead of dropping the remainder. Returns
    [(start_sec, end_sec, chunk_array), ...]."""
    chunk_len = int(round(chunk_sec * fs))
    n = signal.shape[0]
    chunks = []
    for start in range(0, n, chunk_len):
        end = min(start + chunk_len, n)
        chunks.append((start / fs, end / fs, signal[start:end]))
    return chunks


def write_segment_parquet(signal: np.ndarray, channel_names: list[str], path: Path) -> None:
    """signal: (T, C). Same column-per-channel format read by src/data/utils.read_segment_array."""
    pd.DataFrame(signal, columns=channel_names).to_parquet(path)
