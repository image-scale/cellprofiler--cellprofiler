"""
Tests for image transformation functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cellprofiler_lib.transforms import (
    InterpolationOrder,
    PaddingMode,
    resize,
    crop,
    crop_center,
    flip_horizontal,
    flip_vertical,
    flip_depth,
    rotate,
    rotate_90,
    pad,
    translate,
    affine_transform,
    scale,
    transpose,
    rescale_intensity,
)


def create_test_image():
    """Create a simple 2D test image."""
    image = np.zeros((20, 20), dtype=np.float64)
    image[5:15, 5:15] = 1.0
    return image


def create_color_image():
    """Create a simple 2D color test image."""
    image = np.zeros((20, 20, 3), dtype=np.float64)
    image[5:15, 5:15, 0] = 1.0  # Red square
    image[5:15, 5:15, 1] = 0.5  # Green
    image[5:15, 5:15, 2] = 0.25  # Blue
    return image


def create_3d_image():
    """Create a simple 3D test image."""
    image = np.zeros((10, 20, 20), dtype=np.float64)
    image[3:7, 5:15, 5:15] = 1.0
    return image


def create_asymmetric_image():
    """Create an asymmetric image for testing orientation."""
    image = np.zeros((20, 30), dtype=np.float64)
    image[5:10, 5:25] = 1.0  # Horizontal bar
    image[5:15, 5:10] = 0.5  # Add an L shape
    return image


class TestResize:
    """Tests for resize function."""

    def test_resize_upscale(self):
        """Resize should upscale correctly."""
        image = create_test_image()
        resized = resize(image, (40, 40))

        assert resized.shape == (40, 40)

    def test_resize_downscale(self):
        """Resize should downscale correctly."""
        image = create_test_image()
        resized = resize(image, (10, 10))

        assert resized.shape == (10, 10)

    def test_resize_color_image(self):
        """Resize should handle color images."""
        image = create_color_image()
        resized = resize(image, (10, 10))

        assert resized.shape == (10, 10, 3)

    def test_resize_preserves_range(self):
        """Resize should preserve intensity range."""
        image = create_test_image()
        resized = resize(image, (40, 40), preserve_range=True)

        assert resized.min() >= 0
        assert resized.max() <= 1.0

    def test_resize_nearest_interpolation(self):
        """Resize with nearest neighbor interpolation."""
        image = create_test_image()
        resized = resize(image, (40, 40), order=InterpolationOrder.NEAREST)

        assert resized.shape == (40, 40)

    def test_resize_integer_dtype(self):
        """Resize should preserve integer dtype when possible."""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[2:8, 2:8] = 255

        resized = resize(image, (20, 20), preserve_range=True)

        assert resized.dtype == np.uint8


class TestCrop:
    """Tests for crop function."""

    def test_crop_basic(self):
        """Basic crop should work correctly."""
        image = create_test_image()
        cropped = crop(image, top=5, left=5, height=10, width=10)

        assert cropped.shape == (10, 10)
        # Cropped region should be all 1s
        assert_allclose(cropped, 1.0)

    def test_crop_color_image(self):
        """Crop should handle color images."""
        image = create_color_image()
        cropped = crop(image, top=5, left=5, height=10, width=10)

        assert cropped.shape == (10, 10, 3)

    def test_crop_3d_image(self):
        """Crop should handle 3D images."""
        image = create_3d_image()
        cropped = crop(image, top=5, left=5, height=10, width=10)

        assert cropped.shape == (10, 10, 10)

    def test_crop_preserves_values(self):
        """Crop should preserve exact pixel values."""
        image = np.arange(100).reshape(10, 10).astype(np.float64)
        cropped = crop(image, top=2, left=3, height=4, width=5)

        expected = image[2:6, 3:8]
        assert_array_equal(cropped, expected)


class TestCropCenter:
    """Tests for center cropping."""

    def test_crop_center_basic(self):
        """Center crop should be centered."""
        image = create_test_image()
        cropped = crop_center(image, 10, 10)

        assert cropped.shape == (10, 10)

    def test_crop_center_preserves_center(self):
        """Center crop should preserve center content."""
        image = np.zeros((20, 20), dtype=np.float64)
        image[9:11, 9:11] = 1.0  # 2x2 center pixel

        cropped = crop_center(image, 4, 4)
        # Center should still have the 1.0 pixels
        assert cropped[1:3, 1:3].sum() == 4.0


class TestFlipHorizontal:
    """Tests for horizontal flip."""

    def test_flip_horizontal_2d(self):
        """Horizontal flip should reverse columns."""
        image = create_asymmetric_image()
        flipped = flip_horizontal(image)

        assert flipped.shape == image.shape
        # First column should be last column of original
        assert_array_equal(flipped[:, 0], image[:, -1])

    def test_flip_horizontal_twice_returns_original(self):
        """Flipping twice should return original."""
        image = create_asymmetric_image()
        flipped_twice = flip_horizontal(flip_horizontal(image))

        assert_array_equal(flipped_twice, image)

    def test_flip_horizontal_color(self):
        """Horizontal flip should work on color images."""
        image = create_color_image()
        flipped = flip_horizontal(image)

        assert flipped.shape == image.shape


class TestFlipVertical:
    """Tests for vertical flip."""

    def test_flip_vertical_2d(self):
        """Vertical flip should reverse rows."""
        image = create_asymmetric_image()
        flipped = flip_vertical(image)

        assert flipped.shape == image.shape
        # First row should be last row of original
        assert_array_equal(flipped[0, :], image[-1, :])

    def test_flip_vertical_twice_returns_original(self):
        """Flipping twice should return original."""
        image = create_asymmetric_image()
        flipped_twice = flip_vertical(flip_vertical(image))

        assert_array_equal(flipped_twice, image)


class TestFlipDepth:
    """Tests for depth flip."""

    def test_flip_depth_3d(self):
        """Depth flip should reverse z-axis."""
        image = create_3d_image()
        flipped = flip_depth(image)

        assert flipped.shape == image.shape
        # First slice should be last slice of original
        assert_array_equal(flipped[0, :, :], image[-1, :, :])

    def test_flip_depth_2d_raises(self):
        """Depth flip should raise for 2D images."""
        image = create_test_image()

        with pytest.raises(ValueError):
            flip_depth(image)


class TestRotate:
    """Tests for rotation."""

    def test_rotate_90_degrees(self):
        """Rotation by 90 degrees should work."""
        image = create_asymmetric_image()
        rotated = rotate(image, 90, resize_output=True)

        # Shape should be swapped (approximately)
        assert rotated.shape[0] >= image.shape[1] - 2
        assert rotated.shape[1] >= image.shape[0] - 2

    def test_rotate_180_degrees(self):
        """Rotation by 180 degrees should flip both axes."""
        image = create_asymmetric_image()
        rotated = rotate(image, 180)

        # 180 rotation is equivalent to flipping both ways
        expected = np.flipud(np.fliplr(image))
        assert_allclose(rotated, expected, atol=0.1)

    def test_rotate_360_returns_original(self):
        """Rotation by 360 degrees should return original."""
        image = create_test_image()
        rotated = rotate(image, 360)

        assert_allclose(rotated, image, atol=0.01)

    def test_rotate_preserves_dtype(self):
        """Rotation should preserve integer dtype."""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[2:8, 2:8] = 255

        rotated = rotate(image, 45)

        assert rotated.dtype == np.uint8


class TestRotate90:
    """Tests for 90-degree rotation."""

    def test_rotate_90_once(self):
        """Single 90-degree rotation."""
        image = create_asymmetric_image()
        rotated = rotate_90(image, k=1)

        # Shape should swap
        assert rotated.shape == (image.shape[1], image.shape[0])

    def test_rotate_90_four_times_returns_original(self):
        """Four 90-degree rotations should return original."""
        image = create_asymmetric_image()
        rotated = rotate_90(image, k=4)

        assert_array_equal(rotated, image)

    def test_rotate_90_color(self):
        """90-degree rotation should work on color images."""
        image = create_color_image()
        rotated = rotate_90(image)

        assert rotated.shape == (image.shape[1], image.shape[0], 3)


class TestPad:
    """Tests for padding."""

    def test_pad_constant(self):
        """Constant padding should add zeros."""
        image = create_test_image()
        padded = pad(image, 5, mode=PaddingMode.CONSTANT, constant_value=0)

        assert padded.shape == (30, 30)
        # Edges should be zero
        assert padded[0, 0] == 0
        assert padded[-1, -1] == 0

    def test_pad_reflect(self):
        """Reflect padding should mirror edges."""
        image = np.arange(9).reshape(3, 3).astype(np.float64)
        padded = pad(image, 1, mode=PaddingMode.REFLECT)

        assert padded.shape == (5, 5)
        # Check reflection
        assert padded[0, 1] == image[1, 0]  # Reflected from second row

    def test_pad_asymmetric(self):
        """Asymmetric padding."""
        image = create_test_image()
        padded = pad(image, ((2, 3), (4, 5)))

        assert padded.shape == (25, 29)

    def test_pad_preserves_original(self):
        """Padding should preserve original content."""
        image = create_test_image()
        padded = pad(image, 5)

        # Check that original content is preserved
        assert_array_equal(padded[5:25, 5:25], image)


class TestTranslate:
    """Tests for translation."""

    def test_translate_right(self):
        """Translation should shift content."""
        image = np.zeros((10, 10), dtype=np.float64)
        image[4:6, 4:6] = 1.0

        translated = translate(image, (0, 2))

        # Content should have moved right
        assert_allclose(translated[4:6, 6:8], 1.0, atol=0.1)

    def test_translate_down(self):
        """Translation down should shift rows."""
        image = np.zeros((10, 10), dtype=np.float64)
        image[4:6, 4:6] = 1.0

        translated = translate(image, (2, 0))

        # Content should have moved down
        assert_allclose(translated[6:8, 4:6], 1.0, atol=0.1)

    def test_translate_fill_value(self):
        """Translation should use fill value."""
        image = np.ones((10, 10), dtype=np.float64)

        translated = translate(image, (5, 5), fill_value=-1.0)

        # Top-left should be filled
        assert translated[0, 0] == -1.0


class TestAffineTransform:
    """Tests for affine transformation."""

    def test_affine_identity(self):
        """Identity matrix should not change image."""
        image = create_test_image()
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        transformed = affine_transform(image, matrix)

        assert_allclose(transformed, image, atol=0.01)

    def test_affine_translation(self):
        """Affine translation should shift image."""
        image = np.zeros((10, 10), dtype=np.float64)
        image[4:6, 4:6] = 1.0

        # Translation matrix (shift by 2 in x)
        matrix = np.array([
            [1, 0, 2],
            [0, 1, 0],
            [0, 0, 1]
        ])

        transformed = affine_transform(image, matrix)

        # Content should have moved
        assert transformed.shape == image.shape

    def test_affine_2x3_matrix(self):
        """Should accept 2x3 matrix."""
        image = create_test_image()
        matrix = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        transformed = affine_transform(image, matrix)

        assert transformed.shape == image.shape


class TestScale:
    """Tests for scaling."""

    def test_scale_up(self):
        """Scaling up should increase size."""
        image = create_test_image()
        scaled = scale(image, 2.0)

        assert scaled.shape == (40, 40)

    def test_scale_down(self):
        """Scaling down should decrease size."""
        image = create_test_image()
        scaled = scale(image, 0.5)

        assert scaled.shape == (10, 10)

    def test_scale_asymmetric(self):
        """Asymmetric scaling."""
        image = create_test_image()
        scaled = scale(image, (2.0, 0.5))

        assert scaled.shape == (40, 10)


class TestTranspose:
    """Tests for transpose."""

    def test_transpose_2d(self):
        """Transpose should swap axes."""
        image = create_asymmetric_image()
        transposed = transpose(image)

        assert transposed.shape == (image.shape[1], image.shape[0])

    def test_transpose_twice_returns_original(self):
        """Double transpose should return original."""
        image = create_asymmetric_image()
        transposed_twice = transpose(transpose(image))

        assert_array_equal(transposed_twice, image)

    def test_transpose_color(self):
        """Transpose should work on color images."""
        image = create_color_image()
        transposed = transpose(image)

        assert transposed.shape == (image.shape[1], image.shape[0], 3)


class TestRescaleIntensity:
    """Tests for intensity rescaling."""

    def test_rescale_to_0_1(self):
        """Rescale to [0, 1] range."""
        image = np.array([[0, 100], [200, 255]], dtype=np.float64)
        rescaled = rescale_intensity(image, out_range=(0.0, 1.0))

        assert_allclose(rescaled.min(), 0.0)
        assert_allclose(rescaled.max(), 1.0)

    def test_rescale_custom_range(self):
        """Rescale to custom range."""
        image = np.array([[0, 1]], dtype=np.float64)
        rescaled = rescale_intensity(image, out_range=(-1.0, 1.0))

        assert_allclose(rescaled.min(), -1.0)
        assert_allclose(rescaled.max(), 1.0)

    def test_rescale_with_input_range(self):
        """Rescale with specified input range."""
        image = np.array([[50, 100]], dtype=np.float64)
        rescaled = rescale_intensity(image, in_range=(0, 100), out_range=(0.0, 1.0))

        assert_allclose(rescaled[0, 0], 0.5)
        assert_allclose(rescaled[0, 1], 1.0)

    def test_rescale_constant_image(self):
        """Rescale constant image."""
        image = np.ones((5, 5), dtype=np.float64) * 0.5
        rescaled = rescale_intensity(image, out_range=(0.0, 1.0))

        # Should return middle value
        assert_allclose(rescaled, 0.5)


class Test3DSupport:
    """Tests for 3D image support."""

    def test_flip_3d(self):
        """Flips should work on 3D images."""
        image = create_3d_image()

        h_flipped = flip_horizontal(image)
        v_flipped = flip_vertical(image)
        d_flipped = flip_depth(image)

        assert h_flipped.shape == image.shape
        assert v_flipped.shape == image.shape
        assert d_flipped.shape == image.shape

    def test_crop_3d(self):
        """Crop should work on 3D images."""
        image = create_3d_image()
        cropped = crop(image, top=5, left=5, height=10, width=10)

        assert cropped.shape == (10, 10, 10)

    def test_pad_3d(self):
        """Padding should work on 3D images."""
        image = create_3d_image()
        padded = pad(image, 2)

        assert padded.shape == (14, 24, 24)


class TestDtypePreservation:
    """Tests for dtype preservation."""

    def test_crop_preserves_dtype(self):
        """Crop should preserve dtype."""
        image = np.zeros((10, 10), dtype=np.uint16)
        image[2:8, 2:8] = 1000

        cropped = crop(image, 2, 2, 6, 6)

        assert cropped.dtype == np.uint16

    def test_flip_preserves_dtype(self):
        """Flip should preserve dtype."""
        image = np.zeros((10, 10), dtype=np.int32)
        image[5:, :] = 100

        flipped = flip_vertical(image)

        assert flipped.dtype == np.int32

    def test_pad_preserves_dtype(self):
        """Pad should preserve dtype."""
        image = np.zeros((10, 10), dtype=np.float32)

        padded = pad(image, 2)

        assert padded.dtype == np.float32
