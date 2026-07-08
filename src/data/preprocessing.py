
"""
EEG preprocessing utilities.

This module converts raw continuous EEG recordings into the format used by the
rest of the project.

The preprocessing pipeline is:
    raw EEG -> filtering -> resampling -> optional CAR -> saved segments

The output format is a parquet file with one column per EEG channel and one row
per time sample.

Dataset-specific loaders (TUAB, TUEV, BCI2a) use these functions to apply the
same preprocessing pipeline before training models.
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
    """
    EEG preprocessing configuration.

    Processing order:
        1. notch filter
        2. high-pass filter
        3. resampling
        4. common average reference (optional)

    The order is kept fixed to avoid applying CAR on unprocessed signals and
    to remove line noise before downsampling.
    """

    target_fs: float = TARGET_FS
    line_freq: float = 60.0
    highpass_hz: float = 0.5
    apply_car: bool = True

    def process(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply preprocessing to a MNE Raw object."""

        nyquist = raw.info["sfreq"] / 2

        # Remove power line harmonics (50/60 Hz depending on the dataset)
        harmonics = np.arange(
            self.line_freq,
            nyquist,
            self.line_freq,
        )

        if len(harmonics):
            raw.notch_filter(
                harmonics,
                verbose="ERROR",
            )

        # Remove slow drifts
        raw.filter(
            l_freq=self.highpass_hz,
            h_freq=None,
            verbose="ERROR",
        )

        # Reduce sampling frequency if needed
        if raw.info["sfreq"] != self.target_fs:
            raw.resample(
                self.target_fs,
                verbose="ERROR",
            )

        # Reference channels to their average
        if self.apply_car:
            raw.set_eeg_reference(
                "average",
                verbose="ERROR",
            )

        return raw


def build_raw(
    signal: np.ndarray,
    channel_names: list[str],
    fs: float,
) -> mne.io.RawArray:
    """
    Create a MNE Raw object from an EEG array.

    Args:
        signal:
            EEG signal with shape (time, channels).
        channel_names:
            List of channel names.
        fs:
            Sampling frequency.

    MNE expects EEG values in volts. If the input is in microvolts,
    convert it using a factor of 1e-6 before calling this function.
    """

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=fs,
        ch_types="eeg",
    )

    return mne.io.RawArray(
        signal.T,
        info,
        verbose="ERROR",
    )


def process_array(
    signal: np.ndarray,
    channel_names: list[str],
    fs: float,
    processor: EEGProcessor | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """
    Apply the EEG preprocessing pipeline to a numpy array.

    Input:
        signal shape: (time, channels)

    Returns:
        processed_signal:
            Preprocessed signal with shape (time, channels)
        channel_names:
            Channel names after processing
        sampling_frequency:
            New sampling frequency
    """

    processor = processor or EEGProcessor()

    raw = build_raw(
        signal,
        channel_names,
        fs,
    )

    raw = processor.process(raw)

    return (
        raw.get_data().T,
        raw.ch_names,
        raw.info["sfreq"],
    )


def chunk_signal(
    signal: np.ndarray,
    fs: float,
    chunk_sec: float,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Split a continuous signal into consecutive chunks.

    The last chunk is kept even if it is shorter than `chunk_sec`.

    Returns:
        List of:
            (start_time, end_time, chunk)
    """

    chunk_len = int(round(chunk_sec * fs))

    chunks = []

    for start in range(0, signal.shape[0], chunk_len):
        end = min(
            start + chunk_len,
            signal.shape[0],
        )

        chunks.append(
            (
                start / fs,
                end / fs,
                signal[start:end],
            )
        )

    return chunks


def write_segment_parquet(
    signal: np.ndarray,
    channel_names: list[str],
    path: Path,
) -> None:
    """
    Save an EEG segment as parquet.

    The parquet format stores one EEG channel per column and one time point per
    row, which is the format expected by the dataset loaders.
    """

    pd.DataFrame(
        signal,
        columns=channel_names,
    ).to_parquet(path)

