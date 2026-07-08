"""
Small image perturbations used to test what information a model relies on.

These functions are meant for sanity checks. For example, shuffling pixels
keeps the same pixel values but removes the spatial organization of the image,
allowing us to check whether a model uses spatial structure or only global
image statistics.
"""

import numpy as np


def shuffle_pixels(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly shuffle pixel locations in an image.

    Args:
        image:
            Image with shape (height, width, channels).
        rng:
            Random number generator used for the permutation.

    Returns:
        Image with the same shape and pixel values, but with spatial structure
        destroyed.

    This keeps the RGB value distribution unchanged while removing information
    carried by neighboring pixels and spatial arrangement.
    """

    h, w, c = image.shape

    permutation = rng.permutation(h * w)

    shuffled = (
        image.reshape(h * w, c)[permutation]
        .reshape(h, w, c)
    )

    return shuffled
