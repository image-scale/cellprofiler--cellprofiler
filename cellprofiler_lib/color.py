"""
Color conversion functions for image processing.

This module provides functions for converting between color spaces
and manipulating color channels in images.
"""
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from skimage import color


def rgb_to_grayscale(image: NDArray) -> NDArray:
    """
    Convert an RGB image to grayscale.

    Uses luminance weights: 0.2125 * R + 0.7154 * G + 0.0721 * B

    Parameters
    ----------
    image : ndarray
        RGB image with shape (H, W, 3).

    Returns
    -------
    ndarray
        Grayscale image with shape (H, W).
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got {image.ndim}D")

    if image.shape[2] == 4:
        # RGBA - first convert to RGB
        image = rgba_to_rgb(image)
    elif image.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {image.shape[2]}")

    return color.rgb2gray(image)


def rgba_to_rgb(image: NDArray) -> NDArray:
    """
    Convert an RGBA image to RGB by dropping the alpha channel.

    Parameters
    ----------
    image : ndarray
        RGBA image with shape (H, W, 4).

    Returns
    -------
    ndarray
        RGB image with shape (H, W, 3).
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {image.shape}")

    return image[:, :, :3]


def rgba_to_grayscale(image: NDArray) -> NDArray:
    """
    Convert an RGBA image to grayscale.

    Parameters
    ----------
    image : ndarray
        RGBA image with shape (H, W, 4).

    Returns
    -------
    ndarray
        Grayscale image with shape (H, W).
    """
    rgb = rgba_to_rgb(image)
    return rgb_to_grayscale(rgb)


def split_channels(image: NDArray) -> Tuple[NDArray, ...]:
    """
    Split a color image into separate channel arrays.

    Parameters
    ----------
    image : ndarray
        Color image with shape (H, W, C).

    Returns
    -------
    tuple of ndarray
        Individual channel arrays, each with shape (H, W).
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got {image.ndim}D")

    return tuple(image[:, :, i] for i in range(image.shape[2]))


def combine_channels(*channels: NDArray) -> NDArray:
    """
    Combine separate channel arrays into a color image.

    Parameters
    ----------
    *channels : ndarray
        Individual channel arrays, each with shape (H, W).

    Returns
    -------
    ndarray
        Color image with shape (H, W, C).
    """
    if len(channels) == 0:
        raise ValueError("At least one channel is required")

    # Validate all channels have same shape
    shape = channels[0].shape
    for i, ch in enumerate(channels[1:], 1):
        if ch.shape != shape:
            raise ValueError(f"Channel {i} has different shape than channel 0")

    return np.stack(channels, axis=-1)


def grayscale_to_rgb(image: NDArray) -> NDArray:
    """
    Convert a grayscale image to RGB by replicating the channel.

    Parameters
    ----------
    image : ndarray
        Grayscale image with shape (H, W).

    Returns
    -------
    ndarray
        RGB image with shape (H, W, 3).
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {image.ndim}D")

    return np.stack([image, image, image], axis=-1)


def normalize_intensity(
    image: NDArray,
    lower: float = 0.0,
    upper: float = 1.0
) -> NDArray:
    """
    Normalize image intensities to a specified range.

    Parameters
    ----------
    image : ndarray
        Input image.
    lower : float
        Lower bound of output range.
    upper : float
        Upper bound of output range.

    Returns
    -------
    ndarray
        Normalized image.
    """
    image = np.asarray(image, dtype=np.float64)

    img_min = image.min()
    img_max = image.max()

    if img_max == img_min:
        # Constant image - return mid-range value
        return np.full_like(image, (lower + upper) / 2)

    # Scale to [0, 1] then to [lower, upper]
    normalized = (image - img_min) / (img_max - img_min)
    return normalized * (upper - lower) + lower


def stretch_intensity(image: NDArray) -> NDArray:
    """
    Stretch image intensities to fill the 0-1 range.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        Intensity-stretched image with values in [0, 1].
    """
    return normalize_intensity(image, 0.0, 1.0)


def invert_image(image: NDArray) -> NDArray:
    """
    Invert image intensities.

    For images in [0, 1], computes 1 - image.
    For images in [0, 255], computes 255 - image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        Inverted image.
    """
    image = np.asarray(image)

    if image.max() > 1.0:
        # Assume 0-255 range
        return 255 - image
    else:
        return 1.0 - image


def adjust_gamma(image: NDArray, gamma: float = 1.0) -> NDArray:
    """
    Apply gamma correction to an image.

    Parameters
    ----------
    image : ndarray
        Input image with values in [0, 1].
    gamma : float
        Gamma value. Values > 1 darken, values < 1 brighten.

    Returns
    -------
    ndarray
        Gamma-corrected image.
    """
    image = np.asarray(image, dtype=np.float64)
    return np.power(image, gamma)


def extract_channel(image: NDArray, channel: int) -> NDArray:
    """
    Extract a single channel from a color image.

    Parameters
    ----------
    image : ndarray
        Color image with shape (H, W, C).
    channel : int
        Channel index to extract (0-indexed).

    Returns
    -------
    ndarray
        Single channel image with shape (H, W).
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got {image.ndim}D")

    if channel < 0 or channel >= image.shape[2]:
        raise ValueError(
            f"Channel {channel} out of range for image with {image.shape[2]} channels"
        )

    return image[:, :, channel]


def replace_channel(
    image: NDArray,
    channel: int,
    data: NDArray
) -> NDArray:
    """
    Replace a single channel in a color image.

    Parameters
    ----------
    image : ndarray
        Color image with shape (H, W, C).
    channel : int
        Channel index to replace (0-indexed).
    data : ndarray
        New channel data with shape (H, W).

    Returns
    -------
    ndarray
        Image with the channel replaced.
    """
    image = np.asarray(image).copy()
    data = np.asarray(data)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got {image.ndim}D")

    if data.shape != image.shape[:2]:
        raise ValueError(
            f"Channel data shape {data.shape} doesn't match image shape {image.shape[:2]}"
        )

    if channel < 0 or channel >= image.shape[2]:
        raise ValueError(
            f"Channel {channel} out of range for image with {image.shape[2]} channels"
        )

    image[:, :, channel] = data
    return image


def to_float(image: NDArray) -> NDArray:
    """
    Convert image to float representation in [0, 1].

    Parameters
    ----------
    image : ndarray
        Input image (integer or float).

    Returns
    -------
    ndarray
        Float image with values in [0, 1].
    """
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.integer):
        # Integer type - normalize by max value for dtype
        max_val = np.iinfo(image.dtype).max
        return image.astype(np.float64) / max_val
    else:
        return image.astype(np.float64)


def to_uint8(image: NDArray) -> NDArray:
    """
    Convert image to uint8 representation in [0, 255].

    Parameters
    ----------
    image : ndarray
        Input image with values in [0, 1].

    Returns
    -------
    ndarray
        uint8 image with values in [0, 255].
    """
    image = np.asarray(image, dtype=np.float64)

    # Clip to [0, 1] and scale to [0, 255]
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)
