"""
Image transformation functions.

This module provides functions for geometric transformations of images,
including resize, crop, flip, rotate, pad, translate, and affine transformations.
"""
from enum import Enum
from typing import Optional, Tuple, Union, Sequence
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage
import skimage.transform


class InterpolationOrder(Enum):
    """Interpolation methods for image transformations."""
    NEAREST = 0
    BILINEAR = 1
    BIQUADRATIC = 2
    BICUBIC = 3
    BIQUARTIC = 4
    BIQUINTIC = 5


class PaddingMode(Enum):
    """Padding modes for image transformations."""
    CONSTANT = "constant"
    EDGE = "edge"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"


def resize(
    image: NDArray,
    output_shape: Tuple[int, ...],
    order: InterpolationOrder = InterpolationOrder.BILINEAR,
    preserve_range: bool = True,
    anti_aliasing: bool = True,
) -> NDArray:
    """Resize an image to the specified dimensions.

    Parameters
    ----------
    image : NDArray
        Input image (2D or 3D).
    output_shape : Tuple[int, ...]
        Desired output shape (height, width) or (depth, height, width).
    order : InterpolationOrder
        Interpolation method.
    preserve_range : bool
        Whether to preserve the original range of values.
    anti_aliasing : bool
        Whether to apply anti-aliasing filter when downsampling.

    Returns
    -------
    NDArray
        Resized image.
    """
    image = np.asarray(image)
    original_dtype = image.dtype

    # Handle color images (add channel dimension to output_shape)
    if image.ndim == 3 and len(output_shape) == 2:
        # 2D color image
        output_shape = output_shape + (image.shape[2],)
    elif image.ndim == 4 and len(output_shape) == 3:
        # 3D color image
        output_shape = output_shape + (image.shape[3],)

    result = skimage.transform.resize(
        image,
        output_shape,
        order=order.value,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
    )

    if preserve_range and np.issubdtype(original_dtype, np.integer):
        result = np.round(result).astype(original_dtype)

    return result


def crop(
    image: NDArray,
    top: int,
    left: int,
    height: int,
    width: int,
    depth_start: Optional[int] = None,
    depth_size: Optional[int] = None,
) -> NDArray:
    """Crop an image to a specified rectangular region.

    Parameters
    ----------
    image : NDArray
        Input image (2D or 3D).
    top : int
        Top row coordinate.
    left : int
        Left column coordinate.
    height : int
        Height of crop region.
    width : int
        Width of crop region.
    depth_start : int, optional
        Starting depth for 3D images.
    depth_size : int, optional
        Depth size for 3D images.

    Returns
    -------
    NDArray
        Cropped image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image[top:top + height, left:left + width].copy()
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            # 2D color image (H, W, C)
            return image[top:top + height, left:left + width, :].copy()
        else:
            # 3D grayscale image (D, H, W)
            if depth_start is not None and depth_size is not None:
                return image[
                    depth_start:depth_start + depth_size,
                    top:top + height,
                    left:left + width
                ].copy()
            else:
                return image[:, top:top + height, left:left + width].copy()
    elif image.ndim == 4:
        # 3D color image (D, H, W, C)
        if depth_start is not None and depth_size is not None:
            return image[
                depth_start:depth_start + depth_size,
                top:top + height,
                left:left + width,
                :
            ].copy()
        else:
            return image[:, top:top + height, left:left + width, :].copy()
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D image, got {image.ndim}D")


def crop_center(
    image: NDArray,
    crop_height: int,
    crop_width: int,
) -> NDArray:
    """Crop an image from the center.

    Parameters
    ----------
    image : NDArray
        Input image (2D or 3D).
    crop_height : int
        Height of crop region.
    crop_width : int
        Width of crop region.

    Returns
    -------
    NDArray
        Center-cropped image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            h, w = image.shape[:2]
        else:
            _, h, w = image.shape
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")

    top = (h - crop_height) // 2
    left = (w - crop_width) // 2

    return crop(image, top, left, crop_height, crop_width)


def flip_horizontal(image: NDArray) -> NDArray:
    """Flip an image horizontally (left-right).

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    NDArray
        Horizontally flipped image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return np.fliplr(image).copy()
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            # 2D color image
            return np.fliplr(image).copy()
        else:
            # 3D grayscale image
            return np.flip(image, axis=2).copy()
    elif image.ndim == 4:
        # 3D color image
        return np.flip(image, axis=2).copy()
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")


def flip_vertical(image: NDArray) -> NDArray:
    """Flip an image vertically (up-down).

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    NDArray
        Vertically flipped image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return np.flipud(image).copy()
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            # 2D color image
            return np.flipud(image).copy()
        else:
            # 3D grayscale image
            return np.flip(image, axis=1).copy()
    elif image.ndim == 4:
        # 3D color image
        return np.flip(image, axis=1).copy()
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")


def flip_depth(image: NDArray) -> NDArray:
    """Flip a 3D image along the depth axis.

    Parameters
    ----------
    image : NDArray
        Input 3D image.

    Returns
    -------
    NDArray
        Depth-flipped image.
    """
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[2] > 4:
        # 3D grayscale
        return np.flip(image, axis=0).copy()
    elif image.ndim == 4:
        # 3D color
        return np.flip(image, axis=0).copy()
    else:
        raise ValueError("flip_depth requires a 3D image")


def rotate(
    image: NDArray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    order: InterpolationOrder = InterpolationOrder.BILINEAR,
    fill_value: float = 0.0,
    resize_output: bool = False,
) -> NDArray:
    """Rotate an image by a specified angle.

    Parameters
    ----------
    image : NDArray
        Input image (2D or 2D color).
    angle : float
        Rotation angle in degrees (counter-clockwise).
    center : Tuple[float, float], optional
        Center of rotation (row, col). Defaults to image center.
    order : InterpolationOrder
        Interpolation method.
    fill_value : float
        Value for pixels outside the image boundaries.
    resize_output : bool
        If True, resize output to fit entire rotated image.

    Returns
    -------
    NDArray
        Rotated image.
    """
    image = np.asarray(image)
    original_dtype = image.dtype

    result = skimage.transform.rotate(
        image.astype(np.float64),
        angle,
        center=center,
        order=order.value,
        cval=fill_value,
        resize=resize_output,
        preserve_range=True,
    )

    if np.issubdtype(original_dtype, np.integer):
        result = np.round(result).astype(original_dtype)
    else:
        result = result.astype(original_dtype)

    return result


def rotate_90(image: NDArray, k: int = 1) -> NDArray:
    """Rotate an image by 90 degrees.

    Parameters
    ----------
    image : NDArray
        Input image.
    k : int
        Number of 90-degree rotations (positive = counter-clockwise).

    Returns
    -------
    NDArray
        Rotated image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return np.rot90(image, k).copy()
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            # 2D color
            return np.rot90(image, k, axes=(0, 1)).copy()
        else:
            # 3D grayscale
            return np.rot90(image, k, axes=(1, 2)).copy()
    elif image.ndim == 4:
        # 3D color
        return np.rot90(image, k, axes=(1, 2)).copy()
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")


def pad(
    image: NDArray,
    pad_width: Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_value: float = 0.0,
) -> NDArray:
    """Pad an image with specified values or mode.

    Parameters
    ----------
    image : NDArray
        Input image.
    pad_width : int or tuple
        Padding width. If int, applies to all sides.
        If tuple of 2, (before, after) for all axes.
        If tuple of tuples, ((before_0, after_0), (before_1, after_1), ...).
    mode : PaddingMode
        Padding mode.
    constant_value : float
        Value for constant padding.

    Returns
    -------
    NDArray
        Padded image.
    """
    image = np.asarray(image)

    kwargs = {}
    if mode == PaddingMode.CONSTANT:
        kwargs["constant_values"] = constant_value

    return np.pad(
        image,
        pad_width,
        mode=mode.value,
        **kwargs,
    )


def translate(
    image: NDArray,
    shift: Tuple[float, ...],
    order: InterpolationOrder = InterpolationOrder.BILINEAR,
    fill_value: float = 0.0,
) -> NDArray:
    """Translate (shift) an image by specified offsets.

    Parameters
    ----------
    image : NDArray
        Input image.
    shift : Tuple[float, ...]
        Shift amounts for each axis (row_shift, col_shift) for 2D,
        or (depth_shift, row_shift, col_shift) for 3D.
    order : InterpolationOrder
        Interpolation method.
    fill_value : float
        Value for pixels outside the image boundaries.

    Returns
    -------
    NDArray
        Translated image.
    """
    image = np.asarray(image)
    original_dtype = image.dtype

    # For color images, only shift spatial dimensions
    if image.ndim == 3 and image.shape[2] <= 4:
        # 2D color - shift is (row, col), need to add 0 for channel axis
        shift = tuple(shift) + (0,)
    elif image.ndim == 4:
        # 3D color - add 0 for channel axis
        shift = tuple(shift) + (0,)

    result = scipy.ndimage.shift(
        image.astype(np.float64),
        shift,
        order=order.value,
        cval=fill_value,
    )

    if np.issubdtype(original_dtype, np.integer):
        result = np.round(result).astype(original_dtype)
    else:
        result = result.astype(original_dtype)

    return result


def affine_transform(
    image: NDArray,
    matrix: NDArray,
    output_shape: Optional[Tuple[int, ...]] = None,
    order: InterpolationOrder = InterpolationOrder.BILINEAR,
    fill_value: float = 0.0,
) -> NDArray:
    """Apply an affine transformation to an image.

    Parameters
    ----------
    image : NDArray
        Input image (2D).
    matrix : NDArray
        Transformation matrix (2x3 or 3x3).
    output_shape : Tuple[int, ...], optional
        Output image shape. Defaults to input shape.
    order : InterpolationOrder
        Interpolation method.
    fill_value : float
        Value for pixels outside the image boundaries.

    Returns
    -------
    NDArray
        Transformed image.
    """
    image = np.asarray(image)
    matrix = np.asarray(matrix)
    original_dtype = image.dtype

    if matrix.shape == (2, 3):
        # Convert 2x3 to 3x3
        matrix = np.vstack([matrix, [0, 0, 1]])
    elif matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 2x3 or 3x3, got {matrix.shape}")

    if output_shape is None:
        output_shape = image.shape[:2]

    # Create skimage AffineTransform
    tform = skimage.transform.AffineTransform(matrix=matrix)

    result = skimage.transform.warp(
        image.astype(np.float64),
        tform.inverse,
        output_shape=output_shape,
        order=order.value,
        cval=fill_value,
        preserve_range=True,
    )

    if np.issubdtype(original_dtype, np.integer):
        result = np.round(result).astype(original_dtype)
    else:
        result = result.astype(original_dtype)

    return result


def scale(
    image: NDArray,
    scale_factor: Union[float, Tuple[float, float]],
    order: InterpolationOrder = InterpolationOrder.BILINEAR,
) -> NDArray:
    """Scale an image by a factor.

    Parameters
    ----------
    image : NDArray
        Input image.
    scale_factor : float or Tuple[float, float]
        Scale factor. If float, scales uniformly.
        If tuple, (row_scale, col_scale).
    order : InterpolationOrder
        Interpolation method.

    Returns
    -------
    NDArray
        Scaled image.
    """
    image = np.asarray(image)

    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)

    if image.ndim == 2:
        new_shape = (
            int(image.shape[0] * scale_factor[0]),
            int(image.shape[1] * scale_factor[1]),
        )
    elif image.ndim == 3 and image.shape[2] <= 4:
        # 2D color
        new_shape = (
            int(image.shape[0] * scale_factor[0]),
            int(image.shape[1] * scale_factor[1]),
        )
    else:
        raise ValueError("Scale only supports 2D images")

    return resize(image, new_shape, order=order)


def transpose(image: NDArray) -> NDArray:
    """Transpose an image (swap rows and columns).

    Parameters
    ----------
    image : NDArray
        Input image.

    Returns
    -------
    NDArray
        Transposed image.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image.T.copy()
    elif image.ndim == 3:
        if image.shape[2] <= 4:
            # 2D color - swap H and W
            return np.transpose(image, (1, 0, 2)).copy()
        else:
            # 3D grayscale - swap H and W
            return np.transpose(image, (0, 2, 1)).copy()
    elif image.ndim == 4:
        # 3D color - swap H and W
        return np.transpose(image, (0, 2, 1, 3)).copy()
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")


def rescale_intensity(
    image: NDArray,
    in_range: Optional[Tuple[float, float]] = None,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> NDArray:
    """Rescale image intensity to a specified range.

    Parameters
    ----------
    image : NDArray
        Input image.
    in_range : Tuple[float, float], optional
        Input intensity range. Defaults to (min, max) of image.
    out_range : Tuple[float, float]
        Output intensity range.

    Returns
    -------
    NDArray
        Rescaled image.
    """
    image = np.asarray(image, dtype=np.float64)

    if in_range is None:
        in_min, in_max = image.min(), image.max()
    else:
        in_min, in_max = in_range

    out_min, out_max = out_range

    if in_max == in_min:
        return np.full_like(image, (out_min + out_max) / 2)

    scale = (out_max - out_min) / (in_max - in_min)
    return (image - in_min) * scale + out_min
