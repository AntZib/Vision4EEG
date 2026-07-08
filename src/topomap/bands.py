"""
Frequency bands and EEG band power computation.

This module contains common EEG frequency ranges and utilities to compute
spectral power for each channel.

Band power is estimated using Welch's method and converted to log-power, which
is commonly used for EEG features because it reduces the impact of large power
variations.
"""

import numpy as np
from scipy.signal import welch


BAND_DEFS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),

    # Frequency ranges commonly used for motor imagery.
    # Mu and beta rhythms are related to ERD/ERS effects.
    "mu": (8.0, 12.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
}


def band_power_per_channel(
    subframe: np.ndarray,
    bands: dict,
    fs: int,
) -> dict:
    """
    Compute spectral power in different frequency bands.

    Args:
        subframe:
            EEG signal with shape (time, channels).
        bands:
            Dictionary mapping band names to frequency ranges.
        fs:
            Sampling frequency.

    Returns:
        Dictionary:
            band_name -> log power for each channel (channels,)

    The power spectral density is estimated independently for each channel
    using Welch's method.
    """

    freqs, psd = welch(
        subframe,
        fs=fs,
        nperseg=min(
            fs * 2,
            subframe.shape[0],
        ),
        axis=0,
    )

    powers = {}

    for name, (low, high) in bands.items():

        mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        powers[name] = np.log(
            psd[mask].mean(axis=0)
            + 1e-12
        )

    return powers
