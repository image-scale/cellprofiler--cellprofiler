"""
Edge enhancement functions for image processing.

This module provides edge detection filters including Sobel, Prewitt,
Canny, and Laplacian of Gaussian for detecting edges in grayscale images.
"""
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import feature


class EdgeDirection(Enum):
    """Direction for edge detection."""
    ALL = "all"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def sobel_filter(
    image: NDArray,
    mask: Optional[NDArray] = None,
    direction: EdgeDirection = EdgeDirection.ALL
) -> NDArray:
    """
    Detect edges using the Sobel filter.

    The Sobel operator computes the gradient of the image intensity,
    emphasizing regions with high spatial frequency (edges).

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    mask : ndarray, optional
        Binary mask specifying regions to process.
    direction : EdgeDirection
        Edge direction to detect: ALL, HORIZONTAL, or VERTICAL.

    Returns
    -------
    ndarray
        Edge magnitude image (same shape as input).
    """
    original_dtype = image.dtype
    image_float = np.asarray(image, dtype=np.float64)

    if direction == EdgeDirection.HORIZONTAL:
        # Horizontal edges (vertical gradient)
        edges = ndimage.sobel(image_float, axis=0)
    elif direction == EdgeDirection.VERTICAL:
        # Vertical edges (horizontal gradient)
        edges = ndimage.sobel(image_float, axis=1)
    else:
        # All directions: magnitude of gradient
        sx = ndimage.sobel(image_float, axis=0)
        sy = ndimage.sobel(image_float, axis=1)
        edges = np.hypot(sx, sy)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        edges = np.where(mask, edges, 0)

    return edges.astype(original_dtype)


def prewitt_filter(
    image: NDArray,
    mask: Optional[NDArray] = None,
    direction: EdgeDirection = EdgeDirection.ALL
) -> NDArray:
    """
    Detect edges using the Prewitt filter.

    The Prewitt operator is similar to Sobel but uses different
    kernel weights for gradient computation.

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    mask : ndarray, optional
        Binary mask specifying regions to process.
    direction : EdgeDirection
        Edge direction to detect: ALL, HORIZONTAL, or VERTICAL.

    Returns
    -------
    ndarray
        Edge magnitude image (same shape as input).
    """
    image = np.asarray(image, dtype=np.float64)

    if direction == EdgeDirection.HORIZONTAL:
        edges = ndimage.prewitt(image, axis=0)
    elif direction == EdgeDirection.VERTICAL:
        edges = ndimage.prewitt(image, axis=1)
    else:
        px = ndimage.prewitt(image, axis=0)
        py = ndimage.prewitt(image, axis=1)
        edges = np.hypot(px, py)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        edges = np.where(mask, edges, 0)

    return edges.astype(image.dtype)


def laplacian_of_gaussian(
    image: NDArray,
    sigma: float = 2.0,
    mask: Optional[NDArray] = None
) -> NDArray:
    """
    Detect edges using Laplacian of Gaussian (LoG) filter.

    The LoG filter smooths the image with a Gaussian and then
    computes the Laplacian to find zero-crossings at edges.

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    sigma : float
        Standard deviation for Gaussian smoothing.
    mask : ndarray, optional
        Binary mask specifying regions to process.

    Returns
    -------
    ndarray
        LoG filtered image (same shape as input).
    """
    image = np.asarray(image, dtype=np.float64)

    # Apply Gaussian smoothing then Laplacian
    smoothed = ndimage.gaussian_filter(image, sigma=sigma)
    edges = ndimage.laplace(smoothed)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        edges = np.where(mask, edges, 0)

    return edges.astype(image.dtype)


def canny_edge_detector(
    image: NDArray,
    mask: Optional[NDArray] = None,
    sigma: float = 1.0,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    use_quantiles: bool = False
) -> NDArray:
    """
    Detect edges using the Canny edge detector.

    The Canny detector uses multi-stage algorithm:
    1. Gaussian smoothing
    2. Gradient calculation
    3. Non-maximum suppression
    4. Double thresholding with hysteresis

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    mask : ndarray, optional
        Binary mask specifying regions to process.
    sigma : float
        Standard deviation for Gaussian smoothing.
    low_threshold : float, optional
        Lower threshold for hysteresis. If None, computed automatically.
    high_threshold : float, optional
        Upper threshold for hysteresis. If None, computed automatically.
    use_quantiles : bool
        If True, interpret thresholds as quantiles (0-1 range).

    Returns
    -------
    ndarray
        Binary edge map (bool dtype, same shape as input).
    """
    image = np.asarray(image, dtype=np.float64)

    # Handle automatic threshold selection
    if low_threshold is None and high_threshold is None:
        # Use default thresholds based on Otsu's method
        low_threshold = 0.1
        high_threshold = 0.2
        use_quantiles = True

    # Apply Canny edge detection
    edges = feature.canny(
        image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        mask=mask,
        use_quantiles=use_quantiles
    )

    return edges


def compute_gradient_magnitude(image: NDArray, sigma: float = 0.0) -> NDArray:
    """
    Compute gradient magnitude of an image.

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    sigma : float
        Standard deviation for optional Gaussian smoothing before gradient.

    Returns
    -------
    ndarray
        Gradient magnitude image.
    """
    image = np.asarray(image, dtype=np.float64)

    if sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=sigma)

    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)

    return np.hypot(gx, gy)


def compute_gradient_direction(image: NDArray, sigma: float = 0.0) -> NDArray:
    """
    Compute gradient direction of an image.

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    sigma : float
        Standard deviation for optional Gaussian smoothing before gradient.

    Returns
    -------
    ndarray
        Gradient direction in radians (-pi to pi).
    """
    image = np.asarray(image, dtype=np.float64)

    if sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=sigma)

    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)

    return np.arctan2(gy, gx)


def scharr_filter(
    image: NDArray,
    mask: Optional[NDArray] = None,
    direction: EdgeDirection = EdgeDirection.ALL
) -> NDArray:
    """
    Detect edges using the Scharr filter.

    The Scharr operator provides better rotation invariance than Sobel.

    Parameters
    ----------
    image : ndarray
        Input 2D grayscale image.
    mask : ndarray, optional
        Binary mask specifying regions to process.
    direction : EdgeDirection
        Edge direction to detect: ALL, HORIZONTAL, or VERTICAL.

    Returns
    -------
    ndarray
        Edge magnitude image (same shape as input).
    """
    image = np.asarray(image, dtype=np.float64)

    # Scharr kernels for x and y directions
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]]) / 16.0

    scharr_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]]) / 16.0

    if direction == EdgeDirection.HORIZONTAL:
        edges = ndimage.convolve(image, scharr_y)
    elif direction == EdgeDirection.VERTICAL:
        edges = ndimage.convolve(image, scharr_x)
    else:
        sx = ndimage.convolve(image, scharr_x)
        sy = ndimage.convolve(image, scharr_y)
        edges = np.hypot(sx, sy)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        edges = np.where(mask, edges, 0)

    return edges.astype(image.dtype)
