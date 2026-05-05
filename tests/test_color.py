"""
Tests for color conversion functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from cellprofiler_lib.color import (
    rgb_to_grayscale,
    rgba_to_rgb,
    rgba_to_grayscale,
    split_channels,
    combine_channels,
    grayscale_to_rgb,
    normalize_intensity,
    stretch_intensity,
    invert_image,
    adjust_gamma,
    extract_channel,
    replace_channel,
    to_float,
    to_uint8,
)


def create_rgb_image():
    """Create a simple RGB test image."""
    image = np.zeros((10, 10, 3), dtype=np.float64)
    image[:, :, 0] = 1.0  # Red channel
    image[:, :, 1] = 0.5  # Green channel
    image[:, :, 2] = 0.25  # Blue channel
    return image


def create_rgba_image():
    """Create a simple RGBA test image."""
    image = np.zeros((10, 10, 4), dtype=np.float64)
    image[:, :, 0] = 1.0  # Red
    image[:, :, 1] = 0.5  # Green
    image[:, :, 2] = 0.25  # Blue
    image[:, :, 3] = 0.8  # Alpha
    return image


def create_grayscale_image():
    """Create a simple grayscale test image."""
    return np.linspace(0, 1, 100).reshape((10, 10)).astype(np.float64)


class TestRGBToGrayscale:
    """Tests for RGB to grayscale conversion."""

    def test_rgb_to_grayscale(self):
        """RGB image should convert to 2D grayscale."""
        rgb = create_rgb_image()
        gray = rgb_to_grayscale(rgb)

        assert gray.ndim == 2
        assert gray.shape == (10, 10)

    def test_grayscale_values(self):
        """Grayscale values should use luminance weights."""
        # Pure red pixel
        red = np.array([[[1.0, 0.0, 0.0]]])
        gray_red = rgb_to_grayscale(red)

        # Pure green pixel
        green = np.array([[[0.0, 1.0, 0.0]]])
        gray_green = rgb_to_grayscale(green)

        # Pure blue pixel
        blue = np.array([[[0.0, 0.0, 1.0]]])
        gray_blue = rgb_to_grayscale(blue)

        # Green should contribute most to luminance
        assert gray_green[0, 0] > gray_red[0, 0]
        assert gray_green[0, 0] > gray_blue[0, 0]

    def test_rgba_auto_converts(self):
        """RGBA image should automatically convert to grayscale."""
        rgba = create_rgba_image()
        gray = rgb_to_grayscale(rgba)

        assert gray.ndim == 2

    def test_wrong_dimensions_raises(self):
        """Non-3D image should raise error."""
        gray = create_grayscale_image()
        with pytest.raises(ValueError):
            rgb_to_grayscale(gray)


class TestRGBAToRGB:
    """Tests for RGBA to RGB conversion."""

    def test_drops_alpha_channel(self):
        """Alpha channel should be dropped."""
        rgba = create_rgba_image()
        rgb = rgba_to_rgb(rgba)

        assert rgb.shape == (10, 10, 3)
        assert_array_equal(rgb[:, :, 0], rgba[:, :, 0])
        assert_array_equal(rgb[:, :, 1], rgba[:, :, 1])
        assert_array_equal(rgb[:, :, 2], rgba[:, :, 2])

    def test_wrong_channels_raises(self):
        """Non-4-channel image should raise error."""
        rgb = create_rgb_image()
        with pytest.raises(ValueError):
            rgba_to_rgb(rgb)


class TestRGBAToGrayscale:
    """Tests for RGBA to grayscale conversion."""

    def test_rgba_to_grayscale(self):
        """RGBA image should convert to 2D grayscale."""
        rgba = create_rgba_image()
        gray = rgba_to_grayscale(rgba)

        assert gray.ndim == 2
        assert gray.shape == (10, 10)


class TestSplitChannels:
    """Tests for channel splitting."""

    def test_split_rgb(self):
        """RGB image should split into 3 channels."""
        rgb = create_rgb_image()
        r, g, b = split_channels(rgb)

        assert r.shape == (10, 10)
        assert g.shape == (10, 10)
        assert b.shape == (10, 10)
        assert_array_equal(r, rgb[:, :, 0])
        assert_array_equal(g, rgb[:, :, 1])
        assert_array_equal(b, rgb[:, :, 2])

    def test_split_rgba(self):
        """RGBA image should split into 4 channels."""
        rgba = create_rgba_image()
        channels = split_channels(rgba)

        assert len(channels) == 4

    def test_wrong_dimensions_raises(self):
        """2D image should raise error."""
        gray = create_grayscale_image()
        with pytest.raises(ValueError):
            split_channels(gray)


class TestCombineChannels:
    """Tests for channel combining."""

    def test_combine_rgb(self):
        """Three channels should combine into RGB image."""
        r = np.ones((10, 10), dtype=np.float64)
        g = np.ones((10, 10), dtype=np.float64) * 0.5
        b = np.ones((10, 10), dtype=np.float64) * 0.25

        rgb = combine_channels(r, g, b)

        assert rgb.shape == (10, 10, 3)
        assert_array_equal(rgb[:, :, 0], r)
        assert_array_equal(rgb[:, :, 1], g)
        assert_array_equal(rgb[:, :, 2], b)

    def test_combine_mismatched_shapes_raises(self):
        """Channels with different shapes should raise error."""
        r = np.ones((10, 10))
        g = np.ones((8, 8))

        with pytest.raises(ValueError):
            combine_channels(r, g)

    def test_combine_empty_raises(self):
        """No channels should raise error."""
        with pytest.raises(ValueError):
            combine_channels()


class TestGrayscaleToRGB:
    """Tests for grayscale to RGB conversion."""

    def test_grayscale_to_rgb(self):
        """Grayscale should convert to RGB by replicating."""
        gray = create_grayscale_image()
        rgb = grayscale_to_rgb(gray)

        assert rgb.shape == (10, 10, 3)
        assert_array_equal(rgb[:, :, 0], gray)
        assert_array_equal(rgb[:, :, 1], gray)
        assert_array_equal(rgb[:, :, 2], gray)

    def test_wrong_dimensions_raises(self):
        """Non-2D image should raise error."""
        rgb = create_rgb_image()
        with pytest.raises(ValueError):
            grayscale_to_rgb(rgb)


class TestNormalizeIntensity:
    """Tests for intensity normalization."""

    def test_normalize_to_0_1(self):
        """Normalizing to [0, 1] should scale properly."""
        image = np.array([[0, 100], [200, 255]])
        normalized = normalize_intensity(image, 0.0, 1.0)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_custom_range(self):
        """Normalizing to custom range should work."""
        image = np.array([[0, 100], [200, 255]])
        normalized = normalize_intensity(image, -1.0, 1.0)

        assert_allclose(normalized.min(), -1.0)
        assert_allclose(normalized.max(), 1.0)

    def test_constant_image(self):
        """Constant image should normalize to mid-range."""
        image = np.ones((5, 5)) * 0.5
        normalized = normalize_intensity(image, 0.0, 1.0)

        assert_allclose(normalized, 0.5)


class TestStretchIntensity:
    """Tests for intensity stretching."""

    def test_stretch(self):
        """Stretching should fill 0-1 range."""
        image = np.array([[0.2, 0.4], [0.6, 0.8]])
        stretched = stretch_intensity(image)

        assert_allclose(stretched.min(), 0.0)
        assert_allclose(stretched.max(), 1.0)


class TestInvertImage:
    """Tests for image inversion."""

    def test_invert_float(self):
        """Float image should invert correctly."""
        image = np.array([[0.0, 0.25], [0.5, 1.0]])
        inverted = invert_image(image)

        assert_allclose(inverted, [[1.0, 0.75], [0.5, 0.0]])

    def test_invert_uint8(self):
        """uint8 image should invert correctly."""
        image = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        inverted = invert_image(image)

        assert_array_equal(inverted, [[255, 155], [55, 0]])


class TestAdjustGamma:
    """Tests for gamma adjustment."""

    def test_gamma_1_unchanged(self):
        """Gamma=1 should leave image unchanged."""
        image = np.array([[0.25, 0.5], [0.75, 1.0]])
        adjusted = adjust_gamma(image, gamma=1.0)

        assert_allclose(adjusted, image)

    def test_gamma_greater_1_darkens(self):
        """Gamma > 1 should darken image."""
        image = np.array([[0.5]])
        adjusted = adjust_gamma(image, gamma=2.0)

        assert adjusted[0, 0] < image[0, 0]

    def test_gamma_less_1_brightens(self):
        """Gamma < 1 should brighten image."""
        image = np.array([[0.5]])
        adjusted = adjust_gamma(image, gamma=0.5)

        assert adjusted[0, 0] > image[0, 0]


class TestExtractChannel:
    """Tests for channel extraction."""

    def test_extract_red(self):
        """Should extract red channel."""
        rgb = create_rgb_image()
        red = extract_channel(rgb, 0)

        assert red.shape == (10, 10)
        assert_array_equal(red, rgb[:, :, 0])

    def test_extract_invalid_channel_raises(self):
        """Invalid channel index should raise error."""
        rgb = create_rgb_image()
        with pytest.raises(ValueError):
            extract_channel(rgb, 5)


class TestReplaceChannel:
    """Tests for channel replacement."""

    def test_replace_red(self):
        """Should replace red channel."""
        rgb = create_rgb_image()
        new_red = np.ones((10, 10)) * 0.5

        result = replace_channel(rgb, 0, new_red)

        assert_array_equal(result[:, :, 0], new_red)
        # Other channels unchanged
        assert_array_equal(result[:, :, 1], rgb[:, :, 1])
        assert_array_equal(result[:, :, 2], rgb[:, :, 2])

    def test_original_unchanged(self):
        """Original image should not be modified."""
        rgb = create_rgb_image()
        original_red = rgb[:, :, 0].copy()
        new_red = np.ones((10, 10)) * 0.5

        replace_channel(rgb, 0, new_red)

        assert_array_equal(rgb[:, :, 0], original_red)


class TestToFloat:
    """Tests for converting to float."""

    def test_uint8_to_float(self):
        """uint8 should convert to [0, 1] range."""
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        result = to_float(image)

        assert result.dtype == np.float64
        assert_allclose(result[0, 0], 0.0)
        assert_allclose(result[0, 2], 1.0)

    def test_float_unchanged(self):
        """Float input should just convert dtype."""
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = to_float(image)

        assert result.dtype == np.float64
        assert_allclose(result, image)


class TestToUint8:
    """Tests for converting to uint8."""

    def test_float_to_uint8(self):
        """Float [0, 1] should convert to [0, 255]."""
        image = np.array([[0.0, 0.5, 1.0]])
        result = to_uint8(image)

        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 1] == 127  # or 128
        assert result[0, 2] == 255

    def test_clips_out_of_range(self):
        """Values outside [0, 1] should be clipped."""
        image = np.array([[-0.5, 1.5]])
        result = to_uint8(image)

        assert result[0, 0] == 0
        assert result[0, 1] == 255
