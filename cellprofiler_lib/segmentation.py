"""
Segmentation format conversion functions for object label representations.

This module provides functions for converting between different representations
of segmented objects: dense label matrices, sparse coordinate lists, and IJV format.

Dense format: 6D array with shape (label_idx, c, t, z, y, x)
Sparse format: Structured array with fields for coordinates and label values
IJV format: 2D array with columns (row, col, label_value)
"""
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


class SparseAxis(Enum):
    """Axis names for sparse segmentation representation."""
    label = "label"
    c = "c"
    t = "t"
    z = "z"
    y = "y"
    x = "x"


class DenseAxis(Enum):
    """Axis indices for dense segmentation representation."""
    label_idx = 0
    c = 1
    t = 2
    z = 3
    y = 4
    x = 5


# Axis name tuples for iteration
SPARSE_AXIS_NAMES = tuple(ax.value for ax in SparseAxis)
SPATIAL_AXES = tuple(ax.value for ax in SparseAxis if ax != SparseAxis.label)
DENSE_AXIS_NAMES = tuple(ax.name for ax in DenseAxis)
DENSE_SHAPE_AXES = tuple(ax.name for ax in DenseAxis if ax != DenseAxis.label_idx)


def validate_dense_matrix(dense: NDArray) -> None:
    """
    Validate that an array is a proper dense label matrix.

    A dense matrix has 6 dimensions: (label_idx, c, t, z, y, x)
    The label_idx dimension allows for overlapping labels when > 1.

    Parameters
    ----------
    dense : ndarray
        The array to validate.

    Raises
    ------
    ValueError
        If the array is not a valid dense label matrix.
    """
    if not isinstance(dense, np.ndarray):
        raise ValueError(f"dense must be ndarray, got {type(dense).__name__}")

    expected_ndim = len(DENSE_AXIS_NAMES)
    if dense.ndim != expected_ndim:
        raise ValueError(
            f"dense must be {expected_ndim}D with shape {DENSE_AXIS_NAMES}, "
            f"got {dense.ndim}D"
        )


def validate_labels(labels: NDArray) -> None:
    """
    Validate that an array is a proper 2D or 3D label matrix.

    Labels are a simpler representation without the full dense structure,
    with shape (y, x) for 2D or (z, y, x) for 3D images.

    Parameters
    ----------
    labels : ndarray
        The label array to validate.

    Raises
    ------
    ValueError
        If the array is not valid.
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError(f"labels must be ndarray, got {type(labels).__name__}")

    if labels.ndim not in (2, 3):
        raise ValueError(f"labels must be 2D or 3D, got {labels.ndim}D")


def validate_ijv(ijv: NDArray) -> None:
    """
    Validate that an array is proper IJV format.

    IJV format is a 2D array with shape (n_pixels, 3) where columns are
    (row, col, label_value).

    Parameters
    ----------
    ijv : ndarray
        The IJV array to validate.

    Raises
    ------
    ValueError
        If the array is not valid IJV format.
    """
    if not isinstance(ijv, np.ndarray):
        raise ValueError(f"ijv must be ndarray, got {type(ijv).__name__}")

    if ijv.ndim != 2:
        raise ValueError(f"ijv must be 2D, got {ijv.ndim}D")

    if ijv.shape[1] != 3:
        raise ValueError(f"ijv must have 3 columns (i, j, v), got {ijv.shape[1]}")


def validate_sparse(sparse: np.recarray) -> None:
    """
    Validate that an array is a proper sparse label representation.

    Sparse format is a structured array (recarray) with named fields
    from the set {label, c, t, z, y, x}.

    Parameters
    ----------
    sparse : recarray
        The sparse array to validate.

    Raises
    ------
    ValueError
        If the array is not valid sparse format.
    """
    if not isinstance(sparse, (np.ndarray, np.recarray)):
        raise ValueError(f"sparse must be ndarray or recarray, got {type(sparse).__name__}")

    if sparse.ndim != 1:
        raise ValueError(f"sparse must be 1D, got {sparse.ndim}D")

    field_names = sparse.dtype.names
    if field_names is None:
        raise ValueError("sparse must have named dtype fields")

    valid_fields = set(SPARSE_AXIS_NAMES)
    field_set = set(field_names)

    if not field_set.issubset(valid_fields):
        invalid = field_set - valid_fields
        raise ValueError(f"sparse has invalid fields: {invalid}")


def indices_from_dense(dense: NDArray, validate: bool = True) -> List[NDArray]:
    """
    Extract unique label values from each label_idx slice of a dense matrix.

    Parameters
    ----------
    dense : ndarray
        A 6D dense label matrix.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    list of ndarray
        A list where each element is an array of unique non-zero labels
        present in that label_idx slice.
    """
    if validate:
        validate_dense_matrix(dense)

    indices = []
    for label_slice in dense:
        unique_vals = np.unique(label_slice)
        # Remove zero (background)
        if len(unique_vals) > 0 and unique_vals[0] == 0:
            unique_vals = unique_vals[1:]
        indices.append(unique_vals)

    return indices


def convert_dense_to_sparse(dense: NDArray, validate: bool = True) -> np.recarray:
    """
    Convert a dense label matrix to sparse representation.

    Parameters
    ----------
    dense : ndarray
        A 6D dense label matrix with shape (label_idx, c, t, z, y, x).
    validate : bool
        Whether to validate the input.

    Returns
    -------
    recarray
        A structured array with fields for coordinates and labels.
        Fields included depend on which dimensions have size > 1.
    """
    if validate:
        validate_dense_matrix(dense)

    full_shape = dense.shape
    label_dim_size = full_shape[DenseAxis.label_idx.value]

    # Get spatial shape (excluding label_idx)
    spatial_shape = tuple(
        full_shape[getattr(DenseAxis, name).value]
        for name in DENSE_SHAPE_AXES
    )

    # Determine which spatial axes are non-trivial (size > 1)
    axes_to_include = [
        name for name, size in zip(SPATIAL_AXES, spatial_shape)
        if size > 1
    ]

    # Squeeze the array but preserve label dimension for iteration
    compact = np.squeeze(dense)
    if label_dim_size == 1:
        compact = np.expand_dims(compact, axis=0)

    # Find all non-zero coordinates
    coords = np.where(compact != 0)
    labels = compact[coords]

    # Skip the label_idx dimension in coordinates
    coords = coords[1:]

    # Determine dtypes based on array sizes
    max_coord = max(compact.shape) if compact.size > 0 else 0
    coords_dtype = np.uint16 if max_coord < 2**16 else np.uint32

    if len(labels) > 0:
        max_label = np.max(labels)
        if max_label < 2**8:
            labels_dtype = np.uint8
        elif max_label < 2**16:
            labels_dtype = np.uint16
        else:
            labels_dtype = np.uint32
    else:
        labels_dtype = np.uint8

    # Build dtype for structured array
    dtype = [(axis, coords_dtype) for axis in axes_to_include]
    dtype.append((SparseAxis.label.value, labels_dtype))

    # Create the structured array
    sparse = np.core.records.fromarrays(
        list(coords) + [labels],
        dtype=dtype
    )

    return sparse


def convert_labels_to_dense(labels: NDArray, validate: bool = True) -> NDArray:
    """
    Convert a 2D or 3D label array to 6D dense format.

    Parameters
    ----------
    labels : ndarray
        A 2D (y, x) or 3D (z, y, x) label array.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        A 6D dense array with shape (1, 1, 1, z, y, x) or (1, 1, 1, 1, y, x).
    """
    if validate:
        validate_labels(labels)

    typed_labels = downsample_labels(labels, validate=False)

    if labels.ndim == 3:
        # 3D labels: add label_idx, c, t dimensions
        expand_axes = (
            DenseAxis.label_idx.value,
            DenseAxis.c.value,
            DenseAxis.t.value
        )
    else:
        # 2D labels: add label_idx, c, t, z dimensions
        expand_axes = (
            DenseAxis.label_idx.value,
            DenseAxis.c.value,
            DenseAxis.t.value,
            DenseAxis.z.value
        )

    return np.expand_dims(typed_labels, axis=expand_axes)


def downsample_labels(labels: NDArray, validate: bool = True) -> NDArray:
    """
    Convert labels to the smallest integer type that can hold all values.

    Parameters
    ----------
    labels : ndarray
        A label array.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        The label array with optimized dtype.
    """
    if validate:
        validate_labels(labels)

    max_label = np.max(labels) if labels.size > 0 else 0

    if max_label < 128:
        return labels.astype(np.int8)
    elif max_label < 32768:
        return labels.astype(np.int16)
    else:
        return labels.astype(np.int32)


def indices_from_ijv(ijv: NDArray, validate: bool = True) -> NDArray:
    """
    Get unique label indices from IJV format.

    Parameters
    ----------
    ijv : ndarray
        An (n, 3) IJV array.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        Array of unique label values (1 to max_label).
    """
    if validate:
        validate_ijv(ijv)

    if len(ijv) == 0:
        return np.zeros(0, dtype=np.int32)

    max_label = np.max(ijv[:, 2])
    return np.arange(1, max_label + 1, dtype=np.int32)


def count_from_ijv(ijv: NDArray, indices: Optional[NDArray] = None, validate: bool = True) -> int:
    """
    Count the number of unique labels in IJV format.

    Parameters
    ----------
    ijv : ndarray
        An (n, 3) IJV array.
    indices : ndarray, optional
        Pre-computed label indices.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    int
        Number of unique labels.
    """
    if validate:
        validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    return len(indices)


def areas_from_ijv(ijv: NDArray, indices: Optional[NDArray] = None, validate: bool = True) -> NDArray:
    """
    Compute the area (pixel count) of each label in IJV format.

    Parameters
    ----------
    ijv : ndarray
        An (n, 3) IJV array.
    indices : ndarray, optional
        Pre-computed label indices.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        Array of areas, one per label in indices order.
    """
    if validate:
        validate_ijv(ijv)

    if indices is None:
        indices = indices_from_ijv(ijv, validate=False)

    if len(indices) == 0:
        return np.zeros(0, dtype=int)

    # bincount gives count of each value; index with indices to get areas
    counts = np.bincount(ijv[:, 2])
    return counts[indices]


def convert_ijv_to_sparse(ijv: NDArray, validate: bool = True) -> np.recarray:
    """
    Convert IJV format to sparse representation.

    Parameters
    ----------
    ijv : ndarray
        An (n, 3) IJV array.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    recarray
        A structured array with y, x, and label fields.
    """
    if validate:
        validate_ijv(ijv)

    return np.core.records.fromarrays(
        (ijv[:, 0], ijv[:, 1], ijv[:, 2]),
        dtype=[
            (SparseAxis.y.value, ijv.dtype),
            (SparseAxis.x.value, ijv.dtype),
            (SparseAxis.label.value, ijv.dtype)
        ]
    )


def convert_sparse_to_ijv(sparse: np.recarray, validate: bool = True) -> NDArray:
    """
    Convert sparse representation to IJV format.

    Parameters
    ----------
    sparse : recarray
        A sparse structured array with y, x, label fields.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        An (n, 3) IJV array.
    """
    if validate:
        validate_sparse(sparse)

    return np.column_stack([
        sparse[SparseAxis.y.value],
        sparse[SparseAxis.x.value],
        sparse[SparseAxis.label.value]
    ])


def convert_labels_to_ijv(labels: NDArray, validate: bool = True) -> NDArray:
    """
    Convert a label matrix to IJV format.

    Parameters
    ----------
    labels : ndarray
        A 2D or 3D label array.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        An (n, 3) IJV array with (row, col, label) coordinates.
    """
    if validate:
        validate_labels(labels)

    dense = convert_labels_to_dense(labels, validate=False)
    sparse = convert_dense_to_sparse(dense, validate=False)
    ijv = convert_sparse_to_ijv(sparse, validate=False)

    return ijv


def convert_ijv_to_labels(
    ijv: NDArray,
    shape: Optional[Tuple[int, int]] = None,
    validate: bool = True
) -> NDArray:
    """
    Convert IJV format to a 2D label matrix.

    Parameters
    ----------
    ijv : ndarray
        An (n, 3) IJV array.
    shape : tuple of int, optional
        Output shape (height, width). If None, inferred from coordinates.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    ndarray
        A 2D label array.

    Notes
    -----
    If labels overlap at any coordinate, the last label value wins.
    """
    if validate:
        validate_ijv(ijv)

    if len(ijv) == 0:
        if shape is None:
            return np.zeros((0, 0), dtype=np.int32)
        return np.zeros(shape, dtype=np.int32)

    if shape is None:
        height = int(np.max(ijv[:, 0])) + 1
        width = int(np.max(ijv[:, 1])) + 1
        shape = (height, width)

    labels = np.zeros(shape, dtype=np.int32)
    labels[ijv[:, 0], ijv[:, 1]] = ijv[:, 2]

    return labels


def convert_dense_to_labelset(
    dense: NDArray,
    indices: Optional[List[NDArray]] = None,
    validate: bool = True
) -> List[Tuple[NDArray, NDArray]]:
    """
    Convert dense format to a list of (labels, indices) tuples.

    Each tuple corresponds to one slice of the label_idx dimension.

    Parameters
    ----------
    dense : ndarray
        A 6D dense label matrix.
    indices : list of ndarray, optional
        Pre-computed indices for each label_idx slice.
    validate : bool
        Whether to validate the input.

    Returns
    -------
    list of (ndarray, ndarray)
        List of (labels_array, unique_labels) tuples.
    """
    if validate:
        validate_dense_matrix(dense)

    if indices is None:
        indices = indices_from_dense(dense, validate=False)

    label_dim_size = dense.shape[DenseAxis.label_idx.value]
    squeezed = np.squeeze(dense)

    if label_dim_size == 1:
        return [(squeezed, indices[0])]

    return [(squeezed[i], indices[i]) for i in range(label_dim_size)]
