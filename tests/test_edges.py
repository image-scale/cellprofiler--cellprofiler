"""
Tests for edge enhancement functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from cellprofiler_lib.edges import (
    EdgeDirection,
    sobel_filter,
    prewitt_filter,
    laplacian_of_gaussian,
    canny_edge_detector,
    compute_gradient_magnitude,
    compute_gradient_direction,
    scharr_filter,
)


def create_test_image_with_edge():
    """Create a test image with a clear vertical edge."""
    image = np.zeros((10, 10), dtype=np.float64)
    image[:, 5:] = 1.0  # Right half is bright
    return image


def create_test_image_with_horizontal_edge():
    """Create a test image with a clear horizontal edge."""
    image = np.zeros((10, 10), dtype=np.float64)
    image[5:, :] = 1.0  # Bottom half is bright
    return image


class TestSobelFilter:
    """Tests for Sobel edge detection."""

    def test_sobel_all_directions(self):
        """Sobel filter detects edges in all directions."""
        image = create_test_image_with_edge()
        edges = sobel_filter(image, direction=EdgeDirection.ALL)

        # Should detect strong edge near column 5
        assert edges.shape == image.shape
        assert edges[:, 5].mean() > edges[:, 0].mean()

    def test_sobel_vertical_direction(self):
        """Sobel filter detects vertical edges only."""
        image = create_test_image_with_edge()
        edges = sobel_filter(image, direction=EdgeDirection.VERTICAL)

        # Should detect vertical edge (horizontal gradient)
        assert edges.shape == image.shape
        assert edges[:, 5].mean() > 0

    def test_sobel_horizontal_direction(self):
        """Sobel filter detects horizontal edges only."""
        image = create_test_image_with_horizontal_edge()
        edges = sobel_filter(image, direction=EdgeDirection.HORIZONTAL)

        # Should detect horizontal edge (vertical gradient)
        assert edges.shape == image.shape
        assert edges[5, :].mean() > edges[0, :].mean()

    def test_sobel_with_mask(self):
        """Sobel filter respects mask."""
        image = create_test_image_with_edge()
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :5] = True  # Only process left half

        edges = sobel_filter(image, mask=mask)

        # Right half (outside mask) should be zero
        assert edges[:, 7].sum() == 0

    def test_sobel_preserves_dtype(self):
        """Sobel filter handles different dtypes."""
        image_f32 = np.ones((5, 5), dtype=np.float32)
        image_f64 = np.ones((5, 5), dtype=np.float64)

        edges_f32 = sobel_filter(image_f32)
        edges_f64 = sobel_filter(image_f64)

        assert edges_f32.dtype == np.float32
        assert edges_f64.dtype == np.float64


class TestPrewittFilter:
    """Tests for Prewitt edge detection."""

    def test_prewitt_all_directions(self):
        """Prewitt filter detects edges in all directions."""
        image = create_test_image_with_edge()
        edges = prewitt_filter(image, direction=EdgeDirection.ALL)

        assert edges.shape == image.shape
        assert edges[:, 5].mean() > edges[:, 0].mean()

    def test_prewitt_vertical_direction(self):
        """Prewitt filter detects vertical edges only."""
        image = create_test_image_with_edge()
        edges = prewitt_filter(image, direction=EdgeDirection.VERTICAL)

        assert edges.shape == image.shape

    def test_prewitt_with_mask(self):
        """Prewitt filter respects mask."""
        image = create_test_image_with_edge()
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :5] = True

        edges = prewitt_filter(image, mask=mask)
        assert edges[:, 7].sum() == 0


class TestLaplacianOfGaussian:
    """Tests for Laplacian of Gaussian edge detection."""

    def test_log_detects_edges(self):
        """LoG filter detects edges."""
        image = create_test_image_with_edge()
        edges = laplacian_of_gaussian(image, sigma=1.0)

        assert edges.shape == image.shape
        # LoG produces zero-crossings at edges
        assert np.abs(edges[:, 5]).mean() > np.abs(edges[:, 0]).mean()

    def test_log_sigma_affects_smoothing(self):
        """Higher sigma produces smoother output."""
        image = create_test_image_with_edge()

        edges_small_sigma = laplacian_of_gaussian(image, sigma=0.5)
        edges_large_sigma = laplacian_of_gaussian(image, sigma=3.0)

        # Larger sigma should produce smoother (lower max) edges
        assert np.abs(edges_large_sigma).max() < np.abs(edges_small_sigma).max()

    def test_log_with_mask(self):
        """LoG filter respects mask."""
        image = create_test_image_with_edge()
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :5] = True

        edges = laplacian_of_gaussian(image, mask=mask)
        assert edges[:, 7].sum() == 0


class TestCannyEdgeDetector:
    """Tests for Canny edge detection."""

    def test_canny_detects_edges(self):
        """Canny detector finds edges."""
        image = create_test_image_with_edge()
        edges = canny_edge_detector(image, sigma=1.0)

        assert edges.shape == image.shape
        assert edges.dtype == bool
        # Should detect edge near column 5
        assert edges[:, 4:6].sum() > 0

    def test_canny_with_auto_thresholds(self):
        """Canny works with automatic threshold selection."""
        image = create_test_image_with_edge()
        edges = canny_edge_detector(image)

        assert edges.dtype == bool
        assert edges.sum() > 0

    def test_canny_with_manual_thresholds(self):
        """Canny works with manual thresholds."""
        image = create_test_image_with_edge()
        edges = canny_edge_detector(
            image,
            low_threshold=0.1,
            high_threshold=0.3,
            use_quantiles=True
        )

        assert edges.dtype == bool

    def test_canny_with_mask(self):
        """Canny detector respects mask."""
        image = create_test_image_with_edge()
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :5] = True

        edges = canny_edge_detector(image, mask=mask)
        # Edges outside mask should be False
        assert edges[:, 8].sum() == 0


class TestGradientFunctions:
    """Tests for gradient computation functions."""

    def test_gradient_magnitude(self):
        """Gradient magnitude is computed correctly."""
        image = create_test_image_with_edge()
        magnitude = compute_gradient_magnitude(image)

        assert magnitude.shape == image.shape
        # Edge should have high magnitude
        assert magnitude[:, 5].mean() > magnitude[:, 0].mean()

    def test_gradient_magnitude_with_smoothing(self):
        """Gradient magnitude can use optional smoothing."""
        image = create_test_image_with_edge()
        mag_no_smooth = compute_gradient_magnitude(image, sigma=0)
        mag_smooth = compute_gradient_magnitude(image, sigma=2.0)

        # Smoothing should reduce max magnitude
        assert mag_smooth.max() <= mag_no_smooth.max()

    def test_gradient_direction(self):
        """Gradient direction is computed correctly."""
        image = create_test_image_with_edge()
        direction = compute_gradient_direction(image)

        assert direction.shape == image.shape
        # Direction should be in radians
        assert direction.min() >= -np.pi
        assert direction.max() <= np.pi


class TestScharrFilter:
    """Tests for Scharr edge detection."""

    def test_scharr_all_directions(self):
        """Scharr filter detects edges in all directions."""
        image = create_test_image_with_edge()
        edges = scharr_filter(image, direction=EdgeDirection.ALL)

        assert edges.shape == image.shape
        assert edges[:, 5].mean() > edges[:, 0].mean()

    def test_scharr_vertical_direction(self):
        """Scharr filter detects vertical edges only."""
        image = create_test_image_with_edge()
        edges = scharr_filter(image, direction=EdgeDirection.VERTICAL)

        assert edges.shape == image.shape

    def test_scharr_with_mask(self):
        """Scharr filter respects mask."""
        image = create_test_image_with_edge()
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :5] = True

        edges = scharr_filter(image, mask=mask)
        assert edges[:, 7].sum() == 0


class TestEdgeFilterProperties:
    """General tests for edge filter properties."""

    def test_all_filters_same_output_shape(self):
        """All edge filters produce same shape as input."""
        image = np.random.rand(20, 30).astype(np.float64)

        assert sobel_filter(image).shape == image.shape
        assert prewitt_filter(image).shape == image.shape
        assert laplacian_of_gaussian(image).shape == image.shape
        assert canny_edge_detector(image).shape == image.shape
        assert scharr_filter(image).shape == image.shape

    def test_uniform_image_has_no_edges(self):
        """Uniform image should produce minimal edge response."""
        image = np.ones((10, 10), dtype=np.float64) * 0.5

        # All gradient-based filters should give near-zero on uniform image
        assert np.abs(sobel_filter(image)).max() < 1e-10
        assert np.abs(prewitt_filter(image)).max() < 1e-10
        assert np.abs(scharr_filter(image)).max() < 1e-10
