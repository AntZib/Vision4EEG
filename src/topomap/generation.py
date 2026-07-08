"""
EEG to topomap image conversion.

This module contains the main logic used to transform EEG segments into RGB
images that can be processed by computer vision models.

The conversion steps are:

    EEG window
        -> frequency band power per channel
        -> interpolation on a scalp grid
        -> RGB topomap image

The generated images are expensive to compute because they require spectral
estimation and spatial interpolation. They are therefore usually generated once
and stored in a cache for later model training.
"""

import logging

import mne
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay

from src.topomap.bands import band_power_per_channel


log = logging.getLogger(__name__)


IMG_SIZE = 224


def build_topomap_geometry(
    channels: list[str],
    img_size: int = IMG_SIZE,
) -> dict:
    """
    Compute the spatial information needed to generate topomaps.

    EEG electrodes are projected from their 3D head coordinates to a 2D scalp
    representation. The interpolation grid and triangulation are computed once
    and reused for all windows having the same channel configuration.

    Returns:
        Dictionary containing:
            - matched channel indices
            - 2D electrode positions
            - Delaunay triangulation
            - interpolation grid
            - scalp mask
    """

    montage = mne.channels.make_standard_montage(
        "standard_1005"
    )

    positions = montage.get_positions()["ch_pos"]

    channel_lookup = {
        name.upper(): name
        for name in positions
    }

    matched_idx = []
    xy = []

    for i, channel in enumerate(channels):

        montage_name = channel_lookup.get(
            channel.upper()
        )

        if montage_name is None:
            continue

        x, y, z = positions[montage_name]

        radius = np.sqrt(
            x**2 + y**2 + z**2
        )

        theta = np.arccos(
            np.clip(
                z / radius,
                -1.0,
                1.0,
            )
        )

        phi = np.arctan2(y, x)

        # Normalize the projection so the scalp border has radius 1
        scalp_radius = theta / (np.pi / 2)

        xy.append(
            (
                scalp_radius * np.cos(phi),
                scalp_radius * np.sin(phi),
            )
        )

        matched_idx.append(i)

    if len(matched_idx) < 4:
        raise ValueError(
            f"Only {len(matched_idx)} channels matched montage positions"
        )

    xy = np.asarray(
        xy,
        dtype=np.float32,
    )

    values = np.linspace(
        -1,
        1,
        img_size,
        dtype=np.float32,
    )

    grid_x, grid_y = np.meshgrid(
        values,
        values,
    )

    mask = (
        grid_x**2 + grid_y**2
        <= 1.0
    )

    dropped = len(channels) - len(matched_idx)

    if dropped:
        log.warning(
            f"{dropped} channel(s) have no montage position and were ignored"
        )

    return {
        "matched_idx": np.array(
            matched_idx,
            dtype=np.int64,
        ),
        "xy": xy,
        "tri": Delaunay(xy),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "mask": mask,
    }


def render_topomap(
    values: np.ndarray,
    geom: dict,
) -> np.ndarray:
    """
    Interpolate channel values onto the scalp image grid.

    Returns:
        2D float32 image normalized between 0 and 1.
    """

    interpolator = CloughTocher2DInterpolator(
        geom["tri"],
        values,
        fill_value=np.nan,
    )

    image = interpolator(
        geom["grid_x"],
        geom["grid_y"],
    )

    if np.isfinite(image).any():
        replacement = np.nanmin(image)
    else:
        replacement = 0.0

    image = np.nan_to_num(
        image,
        nan=replacement,
    )

    if geom["mask"].any():
        image[~geom["mask"]] = image[
            geom["mask"]
        ].min()

    low, high = np.percentile(
        image[geom["mask"]],
        [1, 99],
    )

    image = np.clip(
        (image - low) / max(high - low, 1e-8),
        0,
        1,
    )

    return image.astype(np.float32)


def window_to_image(
    window: np.ndarray,
    geom: dict,
    bands: dict,
    fs: int,
) -> np.ndarray:
    """
    Convert one EEG window into an RGB topomap.

    Each RGB channel corresponds to one frequency band.
    The band power is computed over the complete window.
    
    Args:
        window:
            EEG signal with shape (time, channels).

    Returns:
        Image with shape (IMG_SIZE, IMG_SIZE, 3).
    """

    window = window[
        :,
        geom["matched_idx"],
    ]

    powers = band_power_per_channel(
        window,
        bands,
        fs,
    )

    image = np.zeros(
        (
            IMG_SIZE,
            IMG_SIZE,
            3,
        ),
        dtype=np.float32,
    )

    for i, band_name in enumerate(bands):
        image[:, :, i] = render_topomap(
            powers[band_name],
            geom,
        )

    return image


def segment_to_windows(
    signal: np.ndarray,
    window_sec: float,
    fs: int,
) -> list[np.ndarray]:
    """
    Split a continuous EEG segment into non-overlapping windows.

    Any remaining samples shorter than one full window are discarded.
    """

    window_length = int(
        window_sec * fs
    )

    n_windows = (
        signal.shape[0]
        // window_length
    )

    return [
        signal[
            i * window_length:
            (i + 1) * window_length,
            :
        ]
        for i in range(n_windows)
    ]
