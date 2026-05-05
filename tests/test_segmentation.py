"""
Tests for segmentation format conversion functions.
"""
import pytest
import numpy as np

from cellprofiler_lib.segmentation import (
    SparseAxis,
    DenseAxis,
    validate_dense_matrix,
    validate_labels,
    validate_ijv,
    validate_sparse,
    indices_from_dense,
    convert_dense_to_sparse,
    convert_labels_to_dense,
    downsample_labels,
    indices_from_ijv,
    count_from_ijv,
    areas_from_ijv,
    convert_ijv_to_sparse,
    convert_sparse_to_ijv,
    convert_labels_to_ijv,
    convert_ijv_to_labels,
    convert_dense_to_labelset,
)


class TestValidateDenseMatrix:
    """Tests for dense matrix validation."""

    def test_valid_6d_dense(self):
        """A 6D array should pass validation."""
        dense = np.zeros((1, 1, 1, 1, 3, 3), dtype=np.int32)
        validate_dense_matrix(dense)  # Should not raise

    def test_wrong_dimensions_raises(self):
        """A non-6D array should fail validation."""
        dense = np.zeros((3, 3), dtype=np.int32)
        with pytest.raises(ValueError) as exc_info:
            validate_dense_matrix(dense)
        assert "6D" in str(exc_info.value)

    def test_non_array_raises(self):
        """A non-ndarray should fail validation."""
        with pytest.raises(ValueError) as exc_info:
            validate_dense_matrix([[1, 2], [3, 4]])
        assert "ndarray" in str(exc_info.value)


class TestValidateLabels:
    """Tests for label matrix validation."""

    def test_valid_2d_labels(self):
        """A 2D array should pass validation."""
        labels = np.zeros((10, 10), dtype=np.int32)
        validate_labels(labels)

    def test_valid_3d_labels(self):
        """A 3D array should pass validation."""
        labels = np.zeros((5, 10, 10), dtype=np.int32)
        validate_labels(labels)

    def test_wrong_dimensions_raises(self):
        """A 4D array should fail validation."""
        labels = np.zeros((2, 2, 2, 2), dtype=np.int32)
        with pytest.raises(ValueError) as exc_info:
            validate_labels(labels)
        assert "2D or 3D" in str(exc_info.value)


class TestValidateIJV:
    """Tests for IJV format validation."""

    def test_valid_ijv(self):
        """A (n, 3) array should pass validation."""
        ijv = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 2]], dtype=np.int32)
        validate_ijv(ijv)

    def test_wrong_columns_raises(self):
        """An array with != 3 columns should fail."""
        ijv = np.array([[0, 0], [0, 1]], dtype=np.int32)
        with pytest.raises(ValueError) as exc_info:
            validate_ijv(ijv)
        assert "3 columns" in str(exc_info.value)

    def test_wrong_dimensions_raises(self):
        """A 1D array should fail validation."""
        ijv = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError) as exc_info:
            validate_ijv(ijv)
        assert "2D" in str(exc_info.value)


class TestIndicesFromDense:
    """Tests for extracting label indices from dense matrices."""

    def test_empty_dense_returns_empty_indices(self):
        """All-zeros dense should return empty indices."""
        dense = np.zeros((1, 1, 1, 1, 3, 3), dtype=np.int32)
        indices = indices_from_dense(dense)
        assert len(indices) == 1
        assert len(indices[0]) == 0

    def test_non_overlapping_labels(self):
        """Non-overlapping labels should be extracted correctly."""
        dense = np.array([1, 1, 0, 2, 0, 0, 2, 0, 0]).reshape((1, 1, 1, 1, 3, 3))
        indices = indices_from_dense(dense)
        assert len(indices) == 1
        np.testing.assert_array_equal(indices[0], [1, 2])

    def test_overlapping_labels(self):
        """Overlapping labels in separate label_idx slices."""
        # First slice: label 1, second slice: label 2
        dense = np.array([
            [1, 0, 1, 0, 0, 0],  # label 1
            [0, 0, 2, 2, 0, 0]   # label 2 overlaps with label 1
        ]).reshape((2, 1, 1, 1, 3, 2))
        indices = indices_from_dense(dense)
        assert len(indices) == 2
        np.testing.assert_array_equal(indices[0], [1])
        np.testing.assert_array_equal(indices[1], [2])


class TestConvertDenseToSparse:
    """Tests for converting dense to sparse format."""

    def test_empty_dense_to_sparse(self):
        """Empty dense should produce empty sparse."""
        dense = np.zeros((1, 1, 1, 1, 3, 3), dtype=np.int32)
        sparse = convert_dense_to_sparse(dense)
        assert len(sparse) == 0

    def test_non_overlapping_to_sparse(self):
        """Dense with labels should convert to sparse with correct coords."""
        # Label 1 at (0,0), (0,1); Label 2 at (1,0), (2,0)
        dense = np.array([1, 1, 0, 2, 0, 0, 2, 0, 0]).reshape((1, 1, 1, 1, 3, 3))
        sparse = convert_dense_to_sparse(dense)

        # Check we have 4 non-zero pixels
        assert len(sparse) == 4

        # Verify coordinates and labels
        ys = sparse[SparseAxis.y.value]
        xs = sparse[SparseAxis.x.value]
        labels = sparse[SparseAxis.label.value]

        # Label 1 pixels
        label1_mask = labels == 1
        np.testing.assert_array_equal(sorted(ys[label1_mask]), [0, 0])
        np.testing.assert_array_equal(sorted(xs[label1_mask]), [0, 1])

        # Label 2 pixels
        label2_mask = labels == 2
        np.testing.assert_array_equal(sorted(ys[label2_mask]), [1, 2])

    def test_overlapping_to_sparse(self):
        """Overlapping labels should appear in sparse format."""
        dense = np.array([
            [[1, 0], [1, 0], [0, 0]],
            [[0, 0], [2, 2], [0, 0]]
        ]).reshape((2, 1, 1, 1, 3, 2))
        sparse = convert_dense_to_sparse(dense)

        # 2 pixels for label 1 + 2 pixels for label 2 = 4 total
        assert len(sparse) == 4


class TestConvertLabelsToDense:
    """Tests for converting labels to dense format."""

    def test_2d_labels_to_dense(self):
        """2D labels should expand to 6D dense."""
        labels = np.array([[1, 0], [2, 0]], dtype=np.int32)
        dense = convert_labels_to_dense(labels)

        assert dense.ndim == 6
        assert dense.shape == (1, 1, 1, 1, 2, 2)
        assert dense[0, 0, 0, 0, 0, 0] == 1
        assert dense[0, 0, 0, 0, 1, 0] == 2

    def test_3d_labels_to_dense(self):
        """3D labels should expand to 6D dense."""
        labels = np.zeros((3, 4, 5), dtype=np.int32)
        labels[1, 2, 3] = 1
        dense = convert_labels_to_dense(labels)

        assert dense.ndim == 6
        assert dense.shape == (1, 1, 1, 3, 4, 5)


class TestDownsampleLabels:
    """Tests for label dtype optimization."""

    def test_small_labels_use_int8(self):
        """Labels < 128 should use int8."""
        labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = downsample_labels(labels)
        assert result.dtype == np.int8

    def test_medium_labels_use_int16(self):
        """Labels < 32768 should use int16."""
        labels = np.array([[1, 200], [3, 400]], dtype=np.int32)
        result = downsample_labels(labels)
        assert result.dtype == np.int16

    def test_large_labels_use_int32(self):
        """Labels >= 32768 should use int32."""
        labels = np.array([[1, 40000]], dtype=np.int32)
        result = downsample_labels(labels)
        assert result.dtype == np.int32


class TestIJVOperations:
    """Tests for IJV format operations."""

    def test_indices_from_empty_ijv(self):
        """Empty IJV should return empty indices."""
        ijv = np.zeros((0, 3), dtype=np.int32)
        indices = indices_from_ijv(ijv)
        assert len(indices) == 0

    def test_indices_from_ijv(self):
        """IJV should return sequential indices 1 to max_label."""
        ijv = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 3],  # note: label 2 is missing
        ], dtype=np.int32)
        indices = indices_from_ijv(ijv)
        # Should return 1, 2, 3 even though label 2 has no pixels
        np.testing.assert_array_equal(indices, [1, 2, 3])

    def test_count_from_ijv(self):
        """Count should equal number of unique labels."""
        ijv = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 2],
        ], dtype=np.int32)
        count = count_from_ijv(ijv)
        assert count == 2

    def test_areas_from_ijv(self):
        """Areas should be pixel counts per label."""
        ijv = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],  # label 1: 3 pixels
            [1, 1, 2],  # label 2: 1 pixel
        ], dtype=np.int32)
        areas = areas_from_ijv(ijv)
        np.testing.assert_array_equal(areas, [3, 1])

    def test_areas_from_empty_ijv(self):
        """Empty IJV should return empty areas."""
        ijv = np.zeros((0, 3), dtype=np.int32)
        areas = areas_from_ijv(ijv)
        assert len(areas) == 0


class TestIJVConversions:
    """Tests for IJV conversion functions."""

    def test_ijv_to_sparse(self):
        """IJV should convert to sparse with correct fields."""
        ijv = np.array([
            [0, 0, 1],
            [0, 1, 2],
        ], dtype=np.int32)
        sparse = convert_ijv_to_sparse(ijv)

        assert len(sparse) == 2
        assert SparseAxis.y.value in sparse.dtype.names
        assert SparseAxis.x.value in sparse.dtype.names
        assert SparseAxis.label.value in sparse.dtype.names

    def test_sparse_to_ijv(self):
        """Sparse should convert back to IJV."""
        ijv_orig = np.array([
            [0, 0, 1],
            [0, 1, 2],
        ], dtype=np.int32)
        sparse = convert_ijv_to_sparse(ijv_orig)
        ijv_back = convert_sparse_to_ijv(sparse)

        np.testing.assert_array_equal(ijv_back, ijv_orig)

    def test_labels_to_ijv(self):
        """Labels should convert to IJV format."""
        labels = np.array([
            [1, 0, 0],
            [1, 2, 0],
            [0, 2, 0]
        ], dtype=np.int32)
        ijv = convert_labels_to_ijv(labels)

        # Should have 4 non-zero pixels
        assert len(ijv) == 4

        # Check all coords are present
        coords_labels = set(tuple(row) for row in ijv)
        expected = {(0, 0, 1), (1, 0, 1), (1, 1, 2), (2, 1, 2)}
        assert coords_labels == expected

    def test_ijv_to_labels(self):
        """IJV should convert back to labels."""
        ijv = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 2],
        ], dtype=np.int32)
        labels = convert_ijv_to_labels(ijv, shape=(2, 2))

        expected = np.array([[1, 1], [2, 0]], dtype=np.int32)
        np.testing.assert_array_equal(labels, expected)

    def test_empty_ijv_to_labels(self):
        """Empty IJV with shape should produce empty labels."""
        ijv = np.zeros((0, 3), dtype=np.int32)
        labels = convert_ijv_to_labels(ijv, shape=(3, 3))

        expected = np.zeros((3, 3), dtype=np.int32)
        np.testing.assert_array_equal(labels, expected)


class TestConvertDenseToLabelset:
    """Tests for dense to labelset conversion."""

    def test_single_label_idx(self):
        """Single label_idx should produce single-element list."""
        dense = np.array([1, 2, 0, 0, 0, 0]).reshape((1, 1, 1, 1, 2, 3))
        labelset = convert_dense_to_labelset(dense)

        assert len(labelset) == 1
        labels, indices = labelset[0]
        np.testing.assert_array_equal(indices, [1, 2])

    def test_multiple_label_idx(self):
        """Multiple label_idx should produce multiple tuples."""
        dense = np.array([
            [1, 0, 0, 0],
            [0, 0, 2, 0]
        ]).reshape((2, 1, 1, 1, 2, 2))
        labelset = convert_dense_to_labelset(dense)

        assert len(labelset) == 2
        _, idx0 = labelset[0]
        _, idx1 = labelset[1]
        np.testing.assert_array_equal(idx0, [1])
        np.testing.assert_array_equal(idx1, [2])
