"""
Object size and shape measurement functions.

This module provides functions for measuring geometric properties of labeled
objects in segmented images, including area, perimeter, centroid, bounding box,
eccentricity, form factor, and more.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import skimage.measure


class ObjectMeasurements(NamedTuple):
    """Container for measurements of a single object."""
    label: int
    area: float
    perimeter: float
    centroid: Tuple[float, ...]
    bbox: Tuple[int, ...]
    eccentricity: float
    major_axis_length: float
    minor_axis_length: float
    orientation: float
    solidity: float
    extent: float
    form_factor: float


def measure_object_area(
    labels: NDArray,
) -> Dict[int, int]:
    """Compute the area (pixel count) for each labeled object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping object label to its area in pixels.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    areas = {}
    for label in unique_labels:
        areas[int(label)] = int(np.sum(labels == label))

    return areas


def measure_object_perimeter(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the perimeter (boundary length) for each labeled object.

    The perimeter is computed as the length of the object's boundary.
    For 3D images, this returns the surface area instead.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its perimeter.
    """
    labels = np.asarray(labels)

    if labels.ndim == 2:
        regions = skimage.measure.regionprops(labels)
        return {int(r.label): float(r.perimeter) for r in regions}
    elif labels.ndim == 3:
        # For 3D, compute surface area instead
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        perimeters = {}
        for label in unique_labels:
            binary = labels == label
            # Count boundary voxels (adjacent to non-object voxels)
            # This is a simplified approximation
            padded = np.pad(binary.astype(int), 1, mode='constant', constant_values=0)
            # Count faces between object and background
            faces = 0
            for axis in range(3):
                diff = np.abs(np.diff(padded, axis=axis))
                faces += np.sum(diff)
            perimeters[int(label)] = float(faces)

        return perimeters
    else:
        raise ValueError(f"Expected 2D or 3D labels, got {labels.ndim}D")


def measure_object_centroid(
    labels: NDArray,
) -> Dict[int, Tuple[float, ...]]:
    """Compute the centroid (center of mass) for each labeled object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, Tuple[float, ...]]
        Dictionary mapping object label to its centroid coordinates.
        For 2D: (row, col), for 3D: (z, row, col).
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    centroids = {}
    for r in regions:
        centroids[int(r.label)] = tuple(float(c) for c in r.centroid)

    return centroids


def measure_object_bounding_box(
    labels: NDArray,
) -> Dict[int, Tuple[int, ...]]:
    """Compute the bounding box for each labeled object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, Tuple[int, ...]]
        Dictionary mapping object label to its bounding box.
        For 2D: (min_row, min_col, max_row, max_col).
        For 3D: (min_z, min_row, min_col, max_z, max_row, max_col).
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    bboxes = {}
    for r in regions:
        bboxes[int(r.label)] = tuple(int(b) for b in r.bbox)

    return bboxes


def measure_object_eccentricity(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the eccentricity for each labeled object.

    Eccentricity measures how elongated an object is:
    - 0 = circle (or sphere for 3D)
    - approaching 1 = highly elongated

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its eccentricity.
    """
    labels = np.asarray(labels)

    if labels.ndim == 3:
        # For 3D, use equivalent 2D projection or return 0
        # skimage doesn't directly support 3D eccentricity
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        return {int(label): 0.0 for label in unique_labels}

    regions = skimage.measure.regionprops(labels)

    eccentricities = {}
    for r in regions:
        eccentricities[int(r.label)] = float(r.eccentricity)

    return eccentricities


def measure_object_major_axis_length(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the major axis length for each labeled object.

    The major axis is the length of the longest axis of the ellipse
    that has the same normalized second central moments as the object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its major axis length.
    """
    labels = np.asarray(labels)

    if labels.ndim == 3:
        regions = skimage.measure.regionprops(labels)
        lengths = {}
        for r in regions:
            # Use equivalent diameter as approximation for 3D
            lengths[int(r.label)] = float(r.equivalent_diameter_area)
        return lengths

    regions = skimage.measure.regionprops(labels)

    lengths = {}
    for r in regions:
        lengths[int(r.label)] = float(r.axis_major_length)

    return lengths


def measure_object_minor_axis_length(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the minor axis length for each labeled object.

    The minor axis is the length of the shortest axis of the ellipse
    that has the same normalized second central moments as the object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its minor axis length.
    """
    labels = np.asarray(labels)

    if labels.ndim == 3:
        # For 3D, approximate using volume-based calculation
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        lengths = {}
        for label in unique_labels:
            area = np.sum(labels == label)
            # Approximate as sphere diameter
            lengths[int(label)] = float(2 * (3 * area / (4 * np.pi)) ** (1/3))
        return lengths

    regions = skimage.measure.regionprops(labels)

    lengths = {}
    for r in regions:
        lengths[int(r.label)] = float(r.axis_minor_length)

    return lengths


def measure_object_orientation(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the orientation for each labeled object.

    Orientation is the angle between the major axis and the horizontal axis,
    in radians, ranging from -pi/2 to pi/2.

    Parameters
    ----------
    labels : NDArray
        Label image (2D only).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its orientation in radians.
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("Orientation measurement only supported for 2D images")

    regions = skimage.measure.regionprops(labels)

    orientations = {}
    for r in regions:
        orientations[int(r.label)] = float(r.orientation)

    return orientations


def measure_object_solidity(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the solidity for each labeled object.

    Solidity is the ratio of object area to convex hull area.
    A value of 1 indicates a fully convex object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its solidity.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    solidities = {}
    for r in regions:
        solidities[int(r.label)] = float(r.solidity)

    return solidities


def measure_object_extent(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the extent for each labeled object.

    Extent is the ratio of object area to bounding box area.
    A value of 1 indicates the object fills its bounding box.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its extent.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    extents = {}
    for r in regions:
        extents[int(r.label)] = float(r.extent)

    return extents


def measure_object_form_factor(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the form factor (circularity) for each labeled object.

    Form factor = 4 * pi * area / perimeter^2

    A circle has form factor of 1. More irregular shapes have lower values.
    Also known as circularity or isoperimetric quotient.

    Parameters
    ----------
    labels : NDArray
        Label image (2D only for accurate measurement).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its form factor.
    """
    labels = np.asarray(labels)

    areas = measure_object_area(labels)
    perimeters = measure_object_perimeter(labels)

    form_factors = {}
    for label in areas:
        area = areas[label]
        perimeter = perimeters.get(label, 0)

        if perimeter > 0:
            form_factors[label] = float(4 * np.pi * area / (perimeter ** 2))
        else:
            form_factors[label] = 0.0

    return form_factors


def measure_all_object_properties(
    labels: NDArray,
) -> Dict[int, ObjectMeasurements]:
    """Compute all object measurements for each labeled object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, ObjectMeasurements]
        Dictionary mapping object label to its complete measurements.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    measurements = {}

    for r in regions:
        label = int(r.label)

        # Common properties
        area = float(r.area)
        centroid = tuple(float(c) for c in r.centroid)
        bbox = tuple(int(b) for b in r.bbox)
        solidity = float(r.solidity)
        extent = float(r.extent)

        if labels.ndim == 2:
            # 2D-specific properties
            perimeter = float(r.perimeter)
            eccentricity = float(r.eccentricity)
            major_axis_length = float(r.axis_major_length)
            minor_axis_length = float(r.axis_minor_length)
            orientation = float(r.orientation)
        else:
            # 3D approximations
            # Count surface voxels for perimeter approximation
            binary = labels == label
            padded = np.pad(binary.astype(int), 1, mode='constant', constant_values=0)
            faces = 0
            for axis in range(3):
                diff = np.abs(np.diff(padded, axis=axis))
                faces += np.sum(diff)
            perimeter = float(faces)

            eccentricity = 0.0
            major_axis_length = float(r.equivalent_diameter_area)
            minor_axis_length = float(2 * (3 * area / (4 * np.pi)) ** (1/3))
            orientation = 0.0

        # Compute form factor
        if perimeter > 0:
            form_factor = float(4 * np.pi * area / (perimeter ** 2))
        else:
            form_factor = 0.0

        measurements[label] = ObjectMeasurements(
            label=label,
            area=area,
            perimeter=perimeter,
            centroid=centroid,
            bbox=bbox,
            eccentricity=eccentricity,
            major_axis_length=major_axis_length,
            minor_axis_length=minor_axis_length,
            orientation=orientation,
            solidity=solidity,
            extent=extent,
            form_factor=form_factor,
        )

    return measurements


def measure_object_equivalent_diameter(
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the equivalent diameter for each labeled object.

    The equivalent diameter is the diameter of a circle (2D) or sphere (3D)
    with the same area/volume as the object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its equivalent diameter.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    diameters = {}
    for r in regions:
        diameters[int(r.label)] = float(r.equivalent_diameter_area)

    return diameters


def measure_object_euler_number(
    labels: NDArray,
) -> Dict[int, int]:
    """Compute the Euler number for each labeled object.

    The Euler number is a topological invariant:
    - In 2D: Euler = 1 - holes
    - A solid object has Euler = 1

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping object label to its Euler number.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    euler_numbers = {}
    for r in regions:
        euler_numbers[int(r.label)] = int(r.euler_number)

    return euler_numbers


def measure_object_convex_area(
    labels: NDArray,
) -> Dict[int, int]:
    """Compute the convex hull area for each labeled object.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping object label to its convex hull area in pixels.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    convex_areas = {}
    for r in regions:
        convex_areas[int(r.label)] = int(r.area_convex)

    return convex_areas


def measure_object_filled_area(
    labels: NDArray,
) -> Dict[int, int]:
    """Compute the filled area for each labeled object.

    The filled area is the area of the object with all holes filled.

    Parameters
    ----------
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, int]
        Dictionary mapping object label to its filled area in pixels.
    """
    labels = np.asarray(labels)
    regions = skimage.measure.regionprops(labels)

    filled_areas = {}
    for r in regions:
        filled_areas[int(r.label)] = int(r.area_filled)

    return filled_areas
