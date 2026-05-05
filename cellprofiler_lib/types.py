"""
Image type definitions and validation for the CellProfiler library.

This module provides type annotations and validation functions for image arrays,
supporting 2D/3D images in grayscale, color, binary, and label formats.
"""
from typing import Any, Union, Optional, Annotated, get_origin, get_args
import numpy as np
from numpy.typing import NDArray


class ImageValidationError(ValueError):
    """Exception raised when an image array fails validation."""

    def __init__(self, message: str):
        self.message = f"Image Validation Error: {message}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def validate_image(
    image: NDArray[Any],
    *,
    is_3d: bool = False,
    is_color: bool = False,
    dtype: Optional[Union[type, tuple]] = None,
    min_channels: int = 2
) -> NDArray[Any]:
    """
    Validate that a numpy array conforms to expected image constraints.

    Parameters
    ----------
    image : ndarray
        The image array to validate.
    is_3d : bool
        If True, expect a 3D volumetric image (z, y, x) or (z, y, x, c).
    is_color : bool
        If True, expect a multi-channel (color) image.
    dtype : type or tuple of types, optional
        Expected dtype(s) for the image.
    min_channels : int
        Minimum number of spatial dimensions (z for 3D images).

    Returns
    -------
    ndarray
        The validated image (unchanged if valid).

    Raises
    ------
    ImageValidationError
        If the image fails any validation check.
    """
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"Expected ndarray, got {type(image).__name__}")

    # Check dtype if specified
    if dtype is not None:
        if isinstance(dtype, tuple):
            if image.dtype not in dtype:
                dtype_names = ', '.join(str(d) for d in dtype)
                raise ImageValidationError(
                    f"Expected dtype in ({dtype_names}), got {image.dtype}"
                )
        else:
            # Handle Union types
            origin = get_origin(dtype)
            if origin is Union:
                allowed_types = get_args(dtype)
                if image.dtype not in allowed_types:
                    raise ImageValidationError(
                        f"Expected dtype in {allowed_types}, got {image.dtype}"
                    )
            elif image.dtype != dtype:
                raise ImageValidationError(
                    f"Expected dtype {dtype}, got {image.dtype}"
                )

    # Determine expected dimensions
    if is_3d:
        if is_color:
            expected_ndim = 4  # (z, y, x, c)
            if image.ndim != expected_ndim:
                raise ImageValidationError(
                    f"Expected 4D array (z, y, x, c), got {image.ndim}D"
                )
            if image.shape[0] < min_channels:
                raise ImageValidationError(
                    f"Expected at least {min_channels} z slices, got {image.shape[0]}"
                )
        else:
            if image.ndim < 3 or image.ndim > 4:
                raise ImageValidationError(
                    f"Expected 3D or 4D array (z, y, x, [c]), got {image.ndim}D"
                )
            if image.ndim == 3 and image.shape[0] < min_channels:
                raise ImageValidationError(
                    f"Expected at least {min_channels} z slices, got {image.shape[0]}"
                )
    else:
        if is_color:
            expected_ndim = 3  # (y, x, c)
            if image.ndim != expected_ndim:
                raise ImageValidationError(
                    f"Expected 3D array (y, x, c), got {image.ndim}D"
                )
        else:
            expected_ndim = 2  # (y, x)
            if image.ndim != expected_ndim:
                raise ImageValidationError(
                    f"Expected 2D array (y, x), got {image.ndim}D"
                )

    return image


def validate_2d_grayscale(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 2D grayscale image (y, x) with float32 or float64 dtype."""
    return validate_image(
        image,
        is_3d=False,
        is_color=False,
        dtype=(np.float32, np.float64)
    )


def validate_2d_color(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 2D color image (y, x, c) with float32 or float64 dtype."""
    return validate_image(
        image,
        is_3d=False,
        is_color=True,
        dtype=(np.float32, np.float64)
    )


def validate_3d_grayscale(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 3D grayscale image (z, y, x) with float32 or float64 dtype."""
    return validate_image(
        image,
        is_3d=True,
        is_color=False,
        dtype=(np.float32, np.float64)
    )


def validate_3d_color(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 3D color image (z, y, x, c) with float32 or float64 dtype."""
    return validate_image(
        image,
        is_3d=True,
        is_color=True,
        dtype=(np.float32, np.float64)
    )


def validate_binary(image: NDArray[Any], is_3d: bool = False) -> NDArray[Any]:
    """Validate a binary image with bool dtype."""
    return validate_image(
        image,
        is_3d=is_3d,
        is_color=False,
        dtype=(np.bool_,)
    )


def validate_2d_binary(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 2D binary image (y, x) with bool dtype."""
    return validate_binary(image, is_3d=False)


def validate_3d_binary(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 3D binary image (z, y, x) with bool dtype."""
    return validate_binary(image, is_3d=True)


def validate_labels(image: NDArray[Any], is_3d: bool = False) -> NDArray[Any]:
    """Validate a label image with integer dtype."""
    valid_dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    return validate_image(
        image,
        is_3d=is_3d,
        is_color=False,
        dtype=valid_dtypes
    )


def validate_2d_labels(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 2D label image (y, x) with integer dtype."""
    return validate_labels(image, is_3d=False)


def validate_3d_labels(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 3D label image (z, y, x) with integer dtype."""
    return validate_labels(image, is_3d=True)


def validate_mask(image: NDArray[Any], is_3d: bool = False) -> NDArray[Any]:
    """Validate a mask image (accepts both bool and grayscale dtypes)."""
    valid_dtypes = (np.bool_, np.float32, np.float64)
    return validate_image(
        image,
        is_3d=is_3d,
        is_color=False,
        dtype=valid_dtypes
    )


def validate_2d_mask(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 2D mask (y, x) with bool or float dtype."""
    return validate_mask(image, is_3d=False)


def validate_3d_mask(image: NDArray[Any]) -> NDArray[Any]:
    """Validate a 3D mask (z, y, x) with bool or float dtype."""
    return validate_mask(image, is_3d=True)


# Type aliases for documentation and type hinting
Grayscale2D = NDArray[np.floating]
Grayscale3D = NDArray[np.floating]
Color2D = NDArray[np.floating]
Color3D = NDArray[np.floating]
Binary2D = NDArray[np.bool_]
Binary3D = NDArray[np.bool_]
Labels2D = NDArray[np.integer]
Labels3D = NDArray[np.integer]
Mask2D = NDArray[Union[np.bool_, np.floating]]
Mask3D = NDArray[Union[np.bool_, np.floating]]
StructuringElement = NDArray[np.uint8]
