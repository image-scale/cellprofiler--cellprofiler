"""
Tests for image type validation functions.
"""
import pytest
import numpy as np

from cellprofiler_lib.types import (
    ImageValidationError,
    validate_image,
    validate_2d_grayscale,
    validate_2d_color,
    validate_3d_grayscale,
    validate_3d_color,
    validate_2d_binary,
    validate_3d_binary,
    validate_2d_labels,
    validate_3d_labels,
    validate_2d_mask,
    validate_3d_mask,
)


class TestImageValidationError:
    """Tests for the ImageValidationError exception class."""

    def test_error_message_includes_prefix(self):
        """Error message should include a descriptive prefix."""
        error = ImageValidationError("test message")
        assert "Image Validation Error:" in str(error)
        assert "test message" in str(error)

    def test_error_is_value_error(self):
        """ImageValidationError should be a subclass of ValueError."""
        error = ImageValidationError("test")
        assert isinstance(error, ValueError)


class TestValidate2DGrayscale:
    """Tests for 2D grayscale image validation."""

    def test_valid_float32_image(self):
        """A (H, W) float32 array should pass validation."""
        img = np.ones((10, 20), dtype=np.float32)
        result = validate_2d_grayscale(img)
        np.testing.assert_array_equal(result, img)

    def test_valid_float64_image(self):
        """A (H, W) float64 array should pass validation."""
        img = np.ones((10, 20), dtype=np.float64)
        result = validate_2d_grayscale(img)
        np.testing.assert_array_equal(result, img)

    def test_wrong_dimensions_raises_error(self):
        """A 3D array should fail 2D grayscale validation."""
        img = np.ones((10, 20, 3), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_grayscale(img)
        assert "Expected 2D" in str(exc_info.value)

    def test_wrong_dtype_raises_error(self):
        """An integer dtype should fail float validation."""
        img = np.ones((10, 20), dtype=np.int32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_grayscale(img)
        assert "Expected dtype" in str(exc_info.value)


class TestValidate2DColor:
    """Tests for 2D color image validation."""

    def test_valid_rgb_image(self):
        """A (H, W, 3) float32 array should pass validation."""
        img = np.ones((10, 20, 3), dtype=np.float32)
        result = validate_2d_color(img)
        np.testing.assert_array_equal(result, img)

    def test_valid_rgba_image(self):
        """A (H, W, 4) float32 array should pass validation."""
        img = np.ones((10, 20, 4), dtype=np.float32)
        result = validate_2d_color(img)
        np.testing.assert_array_equal(result, img)

    def test_2d_array_raises_error(self):
        """A 2D array should fail color image validation."""
        img = np.ones((10, 20), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_color(img)
        assert "Expected 3D" in str(exc_info.value)

    def test_wrong_dtype_raises_error(self):
        """A uint8 array should fail float validation."""
        img = np.ones((10, 20, 3), dtype=np.uint8)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_color(img)
        assert "Expected dtype" in str(exc_info.value)


class TestValidate3DGrayscale:
    """Tests for 3D grayscale image validation."""

    def test_valid_3d_grayscale(self):
        """A (Z, H, W) float32 array should pass validation."""
        img = np.ones((5, 10, 20), dtype=np.float32)
        result = validate_3d_grayscale(img)
        np.testing.assert_array_equal(result, img)

    def test_single_slice_raises_error(self):
        """A 3D array with only 1 z-slice should fail validation."""
        img = np.ones((1, 10, 20), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_3d_grayscale(img)
        assert "z slices" in str(exc_info.value)

    def test_2d_array_raises_error(self):
        """A 2D array should fail 3D validation."""
        img = np.ones((10, 20), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_3d_grayscale(img)
        assert "3D" in str(exc_info.value)


class TestValidate3DColor:
    """Tests for 3D color image validation."""

    def test_valid_3d_color(self):
        """A (Z, H, W, C) float32 array should pass validation."""
        img = np.ones((5, 10, 20, 3), dtype=np.float32)
        result = validate_3d_color(img)
        np.testing.assert_array_equal(result, img)

    def test_wrong_dimensions_raises_error(self):
        """A 3D array should fail 4D color validation."""
        img = np.ones((5, 10, 20), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_3d_color(img)
        assert "Expected 4D" in str(exc_info.value)


class TestValidate2DBinary:
    """Tests for 2D binary image validation."""

    def test_valid_binary_image(self):
        """A (H, W) bool array should pass validation."""
        img = np.ones((10, 20), dtype=np.bool_)
        result = validate_2d_binary(img)
        np.testing.assert_array_equal(result, img)

    def test_integer_array_raises_error(self):
        """An integer array should fail binary validation."""
        img = np.ones((10, 20), dtype=np.int32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_binary(img)
        assert "Expected dtype" in str(exc_info.value)


class TestValidate3DBinary:
    """Tests for 3D binary image validation."""

    def test_valid_3d_binary(self):
        """A (Z, H, W) bool array should pass validation."""
        img = np.ones((5, 10, 20), dtype=np.bool_)
        result = validate_3d_binary(img)
        np.testing.assert_array_equal(result, img)


class TestValidate2DLabels:
    """Tests for 2D label image validation."""

    def test_valid_int32_labels(self):
        """A (H, W) int32 array should pass validation."""
        img = np.zeros((10, 20), dtype=np.int32)
        img[2:5, 3:8] = 1
        img[6:9, 10:15] = 2
        result = validate_2d_labels(img)
        np.testing.assert_array_equal(result, img)

    def test_valid_uint16_labels(self):
        """A (H, W) uint16 array should pass validation."""
        img = np.zeros((10, 20), dtype=np.uint16)
        result = validate_2d_labels(img)
        np.testing.assert_array_equal(result, img)

    def test_float_array_raises_error(self):
        """A float array should fail label validation."""
        img = np.ones((10, 20), dtype=np.float32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_labels(img)
        assert "Expected dtype" in str(exc_info.value)


class TestValidate3DLabels:
    """Tests for 3D label image validation."""

    def test_valid_3d_labels(self):
        """A (Z, H, W) int32 array should pass validation."""
        img = np.zeros((5, 10, 20), dtype=np.int32)
        result = validate_3d_labels(img)
        np.testing.assert_array_equal(result, img)


class TestValidate2DMask:
    """Tests for 2D mask validation."""

    def test_valid_bool_mask(self):
        """A (H, W) bool array should pass mask validation."""
        mask = np.ones((10, 20), dtype=np.bool_)
        result = validate_2d_mask(mask)
        np.testing.assert_array_equal(result, mask)

    def test_valid_float_mask(self):
        """A (H, W) float array should pass mask validation."""
        mask = np.ones((10, 20), dtype=np.float32)
        result = validate_2d_mask(mask)
        np.testing.assert_array_equal(result, mask)

    def test_integer_mask_raises_error(self):
        """An integer array should fail mask validation."""
        mask = np.ones((10, 20), dtype=np.int32)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_2d_mask(mask)
        assert "Expected dtype" in str(exc_info.value)


class TestValidate3DMask:
    """Tests for 3D mask validation."""

    def test_valid_3d_mask(self):
        """A (Z, H, W) bool array should pass mask validation."""
        mask = np.ones((5, 10, 20), dtype=np.bool_)
        result = validate_3d_mask(mask)
        np.testing.assert_array_equal(result, mask)


class TestValidateImage:
    """Tests for the generic validate_image function."""

    def test_non_array_raises_error(self):
        """A non-ndarray should raise an error."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image([1, 2, 3])
        assert "Expected ndarray" in str(exc_info.value)

    def test_returns_valid_image_unchanged(self):
        """Valid images should be returned unchanged."""
        img = np.ones((10, 20), dtype=np.float32)
        result = validate_image(img, dtype=(np.float32,))
        assert result is img
