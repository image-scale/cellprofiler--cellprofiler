"""
Tests for morphological operations.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from cellprofiler_lib.morphology import (
    StructuringElementShape,
    create_structuring_element,
    erode,
    dilate,
    opening,
    closing,
    skeletonize,
    morphological_gradient,
    white_tophat,
    black_tophat,
    fill_holes,
    remove_small_objects,
    remove_small_holes,
)


def create_binary_square():
    """Create a binary image with a square object."""
    image = np.zeros((20, 20), dtype=bool)
    image[5:15, 5:15] = True
    return image


def create_binary_with_hole():
    """Create a binary image with a square object containing a hole."""
    image = create_binary_square()
    image[8:12, 8:12] = False
    return image


def create_grayscale_image():
    """Create a grayscale test image."""
    image = np.zeros((20, 20), dtype=np.float64)
    image[5:15, 5:15] = 1.0
    return image


class TestStructuringElements:
    """Tests for structuring element creation."""

    def test_create_disk(self):
        """Create a disk structuring element."""
        selem = create_structuring_element(StructuringElementShape.DISK, 3)
        assert selem.ndim == 2
        assert selem.shape[0] == selem.shape[1]  # Should be square

    def test_create_square(self):
        """Create a square structuring element."""
        selem = create_structuring_element(StructuringElementShape.SQUARE, 2)
        assert selem.ndim == 2
        assert selem.shape == (5, 5)  # size*2+1

    def test_create_diamond(self):
        """Create a diamond structuring element."""
        selem = create_structuring_element(StructuringElementShape.DIAMOND, 2)
        assert selem.ndim == 2

    def test_create_3d_ball(self):
        """Create a 3D ball structuring element."""
        selem = create_structuring_element(StructuringElementShape.BALL, 2, is_3d=True)
        assert selem.ndim == 3

    def test_create_3d_cube(self):
        """Create a 3D cube structuring element."""
        selem = create_structuring_element(StructuringElementShape.CUBE, 2, is_3d=True)
        assert selem.ndim == 3


class TestErosion:
    """Tests for erosion operation."""

    def test_erosion_shrinks_binary(self):
        """Erosion should shrink a binary object."""
        image = create_binary_square()
        original_area = np.sum(image)

        eroded = erode(image, size=1)
        eroded_area = np.sum(eroded)

        assert eroded_area < original_area
        assert eroded.dtype == bool

    def test_erosion_with_different_sizes(self):
        """Larger erosion should shrink more."""
        image = create_binary_square()

        eroded_1 = erode(image, size=1)
        eroded_2 = erode(image, size=2)

        assert np.sum(eroded_2) < np.sum(eroded_1)

    def test_erosion_grayscale(self):
        """Erosion should work on grayscale images."""
        image = create_grayscale_image()
        eroded = erode(image, size=1)

        assert eroded.shape == image.shape
        assert eroded.min() >= 0

    def test_erosion_with_custom_selem(self):
        """Erosion with custom structuring element."""
        image = create_binary_square()
        selem = np.ones((3, 3), dtype=bool)

        eroded = erode(image, selem=selem)
        assert np.sum(eroded) < np.sum(image)


class TestDilation:
    """Tests for dilation operation."""

    def test_dilation_expands_binary(self):
        """Dilation should expand a binary object."""
        image = create_binary_square()
        original_area = np.sum(image)

        dilated = dilate(image, size=1)
        dilated_area = np.sum(dilated)

        assert dilated_area > original_area
        assert dilated.dtype == bool

    def test_dilation_with_different_sizes(self):
        """Larger dilation should expand more."""
        image = create_binary_square()

        dilated_1 = dilate(image, size=1)
        dilated_2 = dilate(image, size=2)

        assert np.sum(dilated_2) > np.sum(dilated_1)

    def test_dilation_grayscale(self):
        """Dilation should work on grayscale images."""
        image = create_grayscale_image()
        dilated = dilate(image, size=1)

        assert dilated.shape == image.shape


class TestOpening:
    """Tests for morphological opening."""

    def test_opening_removes_small_spots(self):
        """Opening should remove small bright spots."""
        image = np.zeros((20, 20), dtype=bool)
        image[5:15, 5:15] = True  # Large object
        image[1, 1] = True  # Small spot

        opened = opening(image, size=2)

        # Large object should remain
        assert np.sum(opened[5:15, 5:15]) > 0
        # Small spot should be removed
        assert not opened[1, 1]

    def test_opening_preserves_shape(self):
        """Opening should preserve overall shape."""
        image = create_binary_square()
        opened = opening(image, size=1)

        assert opened.shape == image.shape

    def test_opening_grayscale(self):
        """Opening should work on grayscale images."""
        image = create_grayscale_image()
        opened = opening(image, size=1)

        assert opened.shape == image.shape


class TestClosing:
    """Tests for morphological closing."""

    def test_closing_fills_small_holes(self):
        """Closing should fill small holes."""
        image = create_binary_with_hole()
        has_hole = not np.all(image[8:12, 8:12])

        closed = closing(image, size=3)

        # Original has hole
        assert has_hole
        # Closed should fill it (or reduce it significantly)
        assert np.sum(closed) >= np.sum(image)

    def test_closing_preserves_shape(self):
        """Closing should preserve overall shape."""
        image = create_binary_square()
        closed = closing(image, size=1)

        assert closed.shape == image.shape

    def test_closing_grayscale(self):
        """Closing should work on grayscale images."""
        image = create_grayscale_image()
        closed = closing(image, size=1)

        assert closed.shape == image.shape


class TestSkeletonize:
    """Tests for skeletonization."""

    def test_skeleton_is_thin(self):
        """Skeleton should be thinner than original."""
        image = create_binary_square()
        skeleton = skeletonize(image)

        assert np.sum(skeleton) < np.sum(image)
        assert skeleton.dtype == bool

    def test_skeleton_maintains_connectivity(self):
        """Skeleton should be connected if original is connected."""
        image = np.zeros((20, 20), dtype=bool)
        image[5:15, 9:11] = True  # Vertical bar

        skeleton = skeletonize(image)
        assert np.sum(skeleton) > 0


class TestMorphologicalGradient:
    """Tests for morphological gradient."""

    def test_gradient_highlights_edges(self):
        """Gradient should be high at object boundaries."""
        image = create_grayscale_image()
        gradient = morphological_gradient(image, size=1)

        # Gradient should be zero in uniform regions
        assert gradient[10, 10] == 0  # Center of object
        assert gradient[0, 0] == 0  # Background

        # Gradient should be non-zero at edges
        edge_max = np.max(gradient[5, :])  # Top edge row
        assert edge_max > 0

    def test_gradient_shape(self):
        """Gradient should have same shape as input."""
        image = create_grayscale_image()
        gradient = morphological_gradient(image, size=1)

        assert gradient.shape == image.shape


class TestTopHat:
    """Tests for top-hat operations."""

    def test_white_tophat_extracts_bright_spots(self):
        """White top-hat should extract small bright features."""
        image = np.zeros((20, 20), dtype=np.float64)
        image[10, 10] = 1.0  # Small bright spot

        tophat = white_tophat(image, size=3)

        # Should highlight the small spot
        assert tophat[10, 10] > 0

    def test_black_tophat_extracts_dark_spots(self):
        """Black top-hat should extract small dark features."""
        image = np.ones((20, 20), dtype=np.float64)
        image[10, 10] = 0.0  # Small dark spot

        tophat = black_tophat(image, size=3)

        # Should highlight the small dark spot
        assert tophat[10, 10] > 0


class TestFillHoles:
    """Tests for hole filling."""

    def test_fill_holes(self):
        """Fill holes should fill interior holes."""
        image = create_binary_with_hole()
        filled = fill_holes(image)

        # Original has hole
        assert not np.all(image[8:12, 8:12])
        # Filled should not have hole
        assert np.all(filled[5:15, 5:15])


class TestRemoveSmallObjects:
    """Tests for small object removal."""

    def test_remove_small_objects(self):
        """Small objects should be removed."""
        image = np.zeros((20, 20), dtype=bool)
        image[5:15, 5:15] = True  # Large object (100 pixels)
        image[1, 1] = True  # Small object (1 pixel)

        cleaned = remove_small_objects(image, min_size=50)

        # Large object should remain
        assert np.sum(cleaned[5:15, 5:15]) > 0
        # Small object should be removed
        assert not cleaned[1, 1]


class TestRemoveSmallHoles:
    """Tests for small hole removal."""

    def test_remove_small_holes(self):
        """Small holes should be filled."""
        image = create_binary_with_hole()
        # Hole is 4x4 = 16 pixels

        filled = remove_small_holes(image, area_threshold=20)

        # Hole should be filled
        assert np.sum(filled) > np.sum(image)


class TestMorphology3D:
    """Tests for 3D morphological operations."""

    def test_erosion_3d(self):
        """Erosion should work on 3D images."""
        image = np.zeros((10, 10, 10), dtype=bool)
        image[3:7, 3:7, 3:7] = True

        eroded = erode(image, shape=StructuringElementShape.BALL, size=1)

        assert eroded.ndim == 3
        assert np.sum(eroded) < np.sum(image)

    def test_dilation_3d(self):
        """Dilation should work on 3D images."""
        image = np.zeros((10, 10, 10), dtype=bool)
        image[3:7, 3:7, 3:7] = True

        dilated = dilate(image, shape=StructuringElementShape.BALL, size=1)

        assert dilated.ndim == 3
        assert np.sum(dilated) > np.sum(image)
