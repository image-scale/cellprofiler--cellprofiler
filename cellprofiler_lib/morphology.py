"""
Morphological operations for image processing.

This module provides morphological operations including erosion, dilation,
opening, closing, skeletonization, and gradient for binary and grayscale images.
"""
from enum import Enum
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from skimage import morphology


class StructuringElementShape(Enum):
    """Shapes for structuring elements."""
    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    BALL = "ball"
    CUBE = "cube"
    OCTAHEDRON = "octahedron"


def create_structuring_element(
    shape: StructuringElementShape,
    size: int,
    is_3d: bool = False
) -> NDArray:
    """
    Create a structuring element for morphological operations.

    Parameters
    ----------
    shape : StructuringElementShape
        Shape of the structuring element.
    size : int
        Size (radius for disk/ball, half-width for square/cube).
    is_3d : bool
        If True, create a 3D structuring element.

    Returns
    -------
    ndarray
        The structuring element (binary array).
    """
    if is_3d:
        if shape == StructuringElementShape.BALL:
            return morphology.ball(size)
        elif shape == StructuringElementShape.CUBE:
            return morphology.cube(size * 2 + 1)
        elif shape == StructuringElementShape.OCTAHEDRON:
            return morphology.octahedron(size)
        elif shape == StructuringElementShape.DISK:
            # For 2D shape on 3D, use disk
            return morphology.ball(size)
        else:
            # Default to ball for 3D
            return morphology.ball(size)
    else:
        if shape == StructuringElementShape.DISK:
            return morphology.disk(size)
        elif shape == StructuringElementShape.SQUARE:
            return morphology.square(size * 2 + 1)
        elif shape == StructuringElementShape.DIAMOND:
            return morphology.diamond(size)
        else:
            # Default to disk for 2D
            return morphology.disk(size)


def erode(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Erode an image with a structuring element.

    Erosion shrinks bright regions and expands dark regions.

    Parameters
    ----------
    image : ndarray
        Input image (binary or grayscale).
    selem : ndarray, optional
        Structuring element. If None, one is created from shape and size.
    shape : StructuringElementShape
        Shape of structuring element if selem is None.
    size : int
        Size of structuring element if selem is None.

    Returns
    -------
    ndarray
        Eroded image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    if image.dtype == bool:
        return morphology.binary_erosion(image, selem)
    else:
        return morphology.erosion(image, selem)


def dilate(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Dilate an image with a structuring element.

    Dilation expands bright regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Input image (binary or grayscale).
    selem : ndarray, optional
        Structuring element. If None, one is created from shape and size.
    shape : StructuringElementShape
        Shape of structuring element if selem is None.
    size : int
        Size of structuring element if selem is None.

    Returns
    -------
    ndarray
        Dilated image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    if image.dtype == bool:
        return morphology.binary_dilation(image, selem)
    else:
        return morphology.dilation(image, selem)


def opening(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Perform morphological opening (erosion followed by dilation).

    Opening removes small bright spots and smooths object boundaries.

    Parameters
    ----------
    image : ndarray
        Input image (binary or grayscale).
    selem : ndarray, optional
        Structuring element. If None, one is created from shape and size.
    shape : StructuringElementShape
        Shape of structuring element if selem is None.
    size : int
        Size of structuring element if selem is None.

    Returns
    -------
    ndarray
        Opened image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    if image.dtype == bool:
        return morphology.binary_opening(image, selem)
    else:
        return morphology.opening(image, selem)


def closing(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Perform morphological closing (dilation followed by erosion).

    Closing fills small holes and connects nearby objects.

    Parameters
    ----------
    image : ndarray
        Input image (binary or grayscale).
    selem : ndarray, optional
        Structuring element. If None, one is created from shape and size.
    shape : StructuringElementShape
        Shape of structuring element if selem is None.
    size : int
        Size of structuring element if selem is None.

    Returns
    -------
    ndarray
        Closed image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    if image.dtype == bool:
        return morphology.binary_closing(image, selem)
    else:
        return morphology.closing(image, selem)


def skeletonize(image: NDArray) -> NDArray:
    """
    Reduce binary objects to single-pixel-wide skeletons.

    Parameters
    ----------
    image : ndarray
        Input binary image.

    Returns
    -------
    ndarray
        Skeletonized binary image.
    """
    if image.ndim == 3:
        return morphology.skeletonize_3d(image)
    else:
        return morphology.skeletonize(image.astype(bool))


def morphological_gradient(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Compute the morphological gradient (difference between dilation and erosion).

    The gradient highlights edges and boundaries in the image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    selem : ndarray, optional
        Structuring element. If None, one is created from shape and size.
    shape : StructuringElementShape
        Shape of structuring element if selem is None.
    size : int
        Size of structuring element if selem is None.

    Returns
    -------
    ndarray
        Gradient image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    dilated = morphology.dilation(image, selem)
    eroded = morphology.erosion(image, selem)

    return dilated - eroded


def white_tophat(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Compute white top-hat (image minus opening).

    Extracts small bright spots from dark background.

    Parameters
    ----------
    image : ndarray
        Input grayscale image.
    selem : ndarray, optional
        Structuring element.
    shape : StructuringElementShape
        Shape if selem is None.
    size : int
        Size if selem is None.

    Returns
    -------
    ndarray
        White top-hat image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    return morphology.white_tophat(image, selem)


def black_tophat(
    image: NDArray,
    selem: Optional[NDArray] = None,
    shape: StructuringElementShape = StructuringElementShape.DISK,
    size: int = 1
) -> NDArray:
    """
    Compute black top-hat (closing minus image).

    Extracts small dark spots from bright background.

    Parameters
    ----------
    image : ndarray
        Input grayscale image.
    selem : ndarray, optional
        Structuring element.
    shape : StructuringElementShape
        Shape if selem is None.
    size : int
        Size if selem is None.

    Returns
    -------
    ndarray
        Black top-hat image.
    """
    if selem is None:
        is_3d = image.ndim == 3
        selem = create_structuring_element(shape, size, is_3d)

    return morphology.black_tophat(image, selem)


def fill_holes(image: NDArray) -> NDArray:
    """
    Fill holes in binary objects.

    A hole is a dark region surrounded entirely by the object.

    Parameters
    ----------
    image : ndarray
        Input binary image.

    Returns
    -------
    ndarray
        Binary image with holes filled.
    """
    from scipy import ndimage
    return ndimage.binary_fill_holes(image)


def remove_small_objects(
    image: NDArray,
    min_size: int = 64,
    connectivity: int = 1
) -> NDArray:
    """
    Remove small connected components from a binary image.

    Parameters
    ----------
    image : ndarray
        Input binary image.
    min_size : int
        Minimum size (in pixels) of objects to keep.
    connectivity : int
        Connectivity to use for determining connected components.

    Returns
    -------
    ndarray
        Binary image with small objects removed.
    """
    return morphology.remove_small_objects(
        image.astype(bool),
        min_size=min_size,
        connectivity=connectivity
    )


def remove_small_holes(
    image: NDArray,
    area_threshold: int = 64,
    connectivity: int = 1
) -> NDArray:
    """
    Remove small holes from a binary image.

    Parameters
    ----------
    image : ndarray
        Input binary image.
    area_threshold : int
        Maximum area (in pixels) of holes to fill.
    connectivity : int
        Connectivity to use for determining holes.

    Returns
    -------
    ndarray
        Binary image with small holes filled.
    """
    return morphology.remove_small_holes(
        image.astype(bool),
        area_threshold=area_threshold,
        connectivity=connectivity
    )
