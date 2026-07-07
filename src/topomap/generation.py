"""EEG-to-topomap-RGB image generation.

The expensive part of the whole pipeline is EEG -> topomap (Welch PSD + Delaunay/
CloughTocher interpolation per window), not the backbone forward pass. This module
is the single place that logic lives -- callers precompute images once (see
scripts/extract_embeddings.py) and cache the result; probe scripts read from that
cache instead of recomputing images every time a different backbone is tried.
"""

import logging

import mne
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay

from src.topomap.bands import band_power_per_channel

log = logging.getLogger(__name__)

IMG_SIZE = 224  # multiple of 14 (DINOv2 patch size)


def build_topomap_geometry(channels: list[str], img_size: int = IMG_SIZE) -> dict:
    """Project channel 3D head positions to 2D scalp coordinates and precompute
    the interpolation grid (same projection convention as EEG topomaps: azimuthal
    equidistant from the vertex)."""
    montage = mne.channels.make_standard_montage("standard_1005")
    pos3d = montage.get_positions()["ch_pos"]  # {name: (x, y, z)} in head coords
    lookup = {name.upper(): name for name in pos3d}

    matched_idx, xy = [], []
    for i, ch in enumerate(channels):
        key = lookup.get(ch.upper())
        if key is None:
            continue
        x, y, z = pos3d[key]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar angle from vertex
        phi = np.arctan2(y, x)
        radius = theta / (np.pi / 2)  # scalp edge (theta=pi/2) -> radius 1
        xy.append((radius * np.cos(phi), radius * np.sin(phi)))
        matched_idx.append(i)

    if len(matched_idx) < 4:
        raise ValueError(f"Only {len(matched_idx)} channels matched a montage position, need >=4.")

    xy = np.asarray(xy, dtype=np.float32)

    lin = np.linspace(-1.0, 1.0, img_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(lin, lin)
    mask = (grid_x**2 + grid_y**2) <= 1.0

    dropped = len(channels) - len(matched_idx)
    if dropped:
        log.warning(f"build_topomap_geometry: {dropped} channel(s) had no montage position and were dropped.")

    # Electrode positions (and therefore the Delaunay triangulation) are identical
    # for every segment of a given dataset -- computing it once per unique channel
    # set and reusing it is what makes a full-dataset run tractable.
    tri = Delaunay(xy)

    return {
        "matched_idx": np.array(matched_idx, dtype=np.int64),
        "xy": xy, "tri": tri, "grid_x": grid_x, "grid_y": grid_y, "mask": mask,
    }


def render_topomap(values: np.ndarray, geom: dict) -> np.ndarray:
    """Interpolate per-channel scalar `values` onto the scalp grid. Returns (img_size, img_size) float32 in [0, 1]."""
    interp = CloughTocher2DInterpolator(geom["tri"], values, fill_value=np.nan)
    img = interp(geom["grid_x"], geom["grid_y"])
    img = np.nan_to_num(img, nan=np.nanmin(img) if np.isfinite(img).any() else 0.0)
    img[~geom["mask"]] = img[geom["mask"]].min() if geom["mask"].any() else 0.0

    lo, hi = np.percentile(img[geom["mask"]], [1, 99])
    img = np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return img.astype(np.float32)


def window_to_image(window: np.ndarray, geom: dict, bands: dict, fs: int) -> np.ndarray:
    """window: (window_sec*fs, C). Returns (IMG_SIZE, IMG_SIZE, 3) float32 in [0, 1] --
    one topomap per band, computed over the full window, stacked as RGB."""
    sub = window[:, geom["matched_idx"]]
    powers = band_power_per_channel(sub, bands, fs=fs)
    image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for b, band_name in enumerate(bands):
        image[:, :, b] = render_topomap(powers[band_name], geom)
    return image


def segment_to_windows(signal: np.ndarray, window_sec: float, fs: int) -> list[np.ndarray]:
    """signal: (T, C). Returns list of (window_sec*fs, C) non-overlapping windows."""
    win_len = int(window_sec * fs)
    n_windows = signal.shape[0] // win_len
    return [signal[i * win_len:(i + 1) * win_len, :] for i in range(n_windows)]
