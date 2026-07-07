"""Frequency band definitions and per-channel band-power computation."""

import numpy as np
from scipy.signal import welch

BAND_DEFS = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
    # Motor-imagery-relevant split (mu/beta ERD-ERS), used for BCI2a instead of the
    # TUAB-tuned delta/theta/alpha triplet.
    "mu": (8.0, 12.0), "low_beta": (13.0, 20.0), "high_beta": (20.0, 30.0),
}


def band_power_per_channel(subframe: np.ndarray, bands: dict, fs: int) -> dict:
    """subframe: (T, C). Returns {band_name: (C,) log-power array}."""
    freqs, psd = welch(subframe, fs=fs, nperseg=min(fs * 2, subframe.shape[0]), axis=0)  # (F, C)
    powers = {}
    for name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = np.log(psd[band_mask].mean(axis=0) + 1e-12)
    return powers
