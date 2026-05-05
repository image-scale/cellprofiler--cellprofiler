"""
Image intensity measurement functions.

This module provides functions for measuring intensity statistics of pixels
within labeled objects, including min, max, mean, median, standard deviation,
percentiles, and integrated intensity.
"""
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage


class IntensityMeasurements(NamedTuple):
    """Container for intensity measurements of a single object."""
    label: int
    min_intensity: float
    max_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    integrated_intensity: float
    mad_intensity: float  # Median absolute deviation


def measure_intensity_min(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the minimum intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its minimum intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        result[int(label)] = float(np.min(image[mask]))

    return result


def measure_intensity_max(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the maximum intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its maximum intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        result[int(label)] = float(np.max(image[mask]))

    return result


def measure_intensity_mean(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the mean intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its mean intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # Use scipy for efficiency
    means = scipy.ndimage.mean(image, labels, unique_labels)

    result = {}
    for i, label in enumerate(unique_labels):
        result[int(label)] = float(means[i])

    return result


def measure_intensity_median(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the median intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its median intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # Use scipy for efficiency
    medians = scipy.ndimage.median(image, labels, unique_labels)

    result = {}
    for i, label in enumerate(unique_labels):
        result[int(label)] = float(medians[i])

    return result


def measure_intensity_std(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the standard deviation of intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its intensity standard deviation.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # Use scipy for efficiency
    stds = scipy.ndimage.standard_deviation(image, labels, unique_labels)

    result = {}
    for i, label in enumerate(unique_labels):
        result[int(label)] = float(stds[i])

    return result


def measure_intensity_percentile(
    image: NDArray,
    labels: NDArray,
    percentile: float,
) -> Dict[int, float]:
    """Compute a specified percentile of intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.
    percentile : float
        Percentile to compute (0-100).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to the intensity at the specified percentile.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        result[int(label)] = float(np.percentile(image[mask], percentile))

    return result


def measure_intensity_percentiles(
    image: NDArray,
    labels: NDArray,
    percentiles: Sequence[float],
) -> Dict[int, Dict[float, float]]:
    """Compute multiple percentiles of intensity within each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.
    percentiles : Sequence[float]
        Percentiles to compute (each 0-100).

    Returns
    -------
    Dict[int, Dict[float, float]]
        Dictionary mapping object label to a dictionary of percentile -> value.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        values = image[mask]
        pct_values = np.percentile(values, percentiles)
        result[int(label)] = {
            float(p): float(v) for p, v in zip(percentiles, pct_values)
        }

    return result


def measure_integrated_intensity(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the integrated intensity (sum) within each labeled object.

    The integrated intensity is the sum of all pixel values within the object,
    which can be useful for quantifying total fluorescence.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its integrated intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    # Use scipy for efficiency
    sums = scipy.ndimage.sum(image, labels, unique_labels)

    result = {}
    for i, label in enumerate(unique_labels):
        result[int(label)] = float(sums[i])

    return result


def measure_intensity_mad(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, float]:
    """Compute the median absolute deviation of intensity within each labeled object.

    MAD = median(|x - median(x)|)

    This is a robust measure of variability.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its intensity MAD.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        values = image[mask]
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        result[int(label)] = float(mad)

    return result


def measure_all_intensity_properties(
    image: NDArray,
    labels: NDArray,
) -> Dict[int, IntensityMeasurements]:
    """Compute all intensity measurements for each labeled object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.

    Returns
    -------
    Dict[int, IntensityMeasurements]
        Dictionary mapping object label to its complete intensity measurements.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}
    for label in unique_labels:
        mask = labels == label
        values = image[mask]

        min_val = float(np.min(values))
        max_val = float(np.max(values))
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        std_val = float(np.std(values))
        integrated = float(np.sum(values))
        mad = float(np.median(np.abs(values - median_val)))

        result[int(label)] = IntensityMeasurements(
            label=int(label),
            min_intensity=min_val,
            max_intensity=max_val,
            mean_intensity=mean_val,
            median_intensity=median_val,
            std_intensity=std_val,
            integrated_intensity=integrated,
            mad_intensity=mad,
        )

    return result


def measure_intensity_in_channels(
    images: Sequence[NDArray],
    labels: NDArray,
    channel_names: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[str, IntensityMeasurements]]:
    """Compute intensity measurements for multiple image channels.

    Parameters
    ----------
    images : Sequence[NDArray]
        List of intensity images (one per channel).
    labels : NDArray
        Label image where each unique non-zero value represents a different object.
    channel_names : Sequence[str], optional
        Names for each channel. If None, uses "channel_0", "channel_1", etc.

    Returns
    -------
    Dict[int, Dict[str, IntensityMeasurements]]
        Dictionary mapping object label to a dictionary of channel name -> measurements.
    """
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(len(images))]

    if len(channel_names) != len(images):
        raise ValueError(
            f"Number of channel names ({len(channel_names)}) doesn't match "
            f"number of images ({len(images)})"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {int(label): {} for label in unique_labels}

    for channel_name, image in zip(channel_names, images):
        channel_measurements = measure_all_intensity_properties(image, labels)
        for label, measurements in channel_measurements.items():
            result[label][channel_name] = measurements

    return result


def measure_intensity_ratio(
    numerator_image: NDArray,
    denominator_image: NDArray,
    labels: NDArray,
    epsilon: float = 1e-10,
) -> Dict[int, float]:
    """Compute the ratio of mean intensities between two channels for each object.

    This is useful for ratiometric measurements like FRET ratios.

    Parameters
    ----------
    numerator_image : NDArray
        Intensity image for numerator.
    denominator_image : NDArray
        Intensity image for denominator.
    labels : NDArray
        Label image.
    epsilon : float
        Small value to add to denominator to avoid division by zero.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its intensity ratio.
    """
    numerator_means = measure_intensity_mean(numerator_image, labels)
    denominator_means = measure_intensity_mean(denominator_image, labels)

    result = {}
    for label in numerator_means:
        num = numerator_means[label]
        denom = denominator_means.get(label, 0) + epsilon
        result[label] = float(num / denom)

    return result


def measure_intensity_at_edge(
    image: NDArray,
    labels: NDArray,
    edge_width: int = 1,
) -> Dict[int, float]:
    """Compute the mean intensity at the edge (boundary) of each object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image.
    edge_width : int
        Width of edge to consider in pixels.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its edge intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}

    for label in unique_labels:
        # Create binary mask for this object
        mask = labels == label

        # Erode the mask to get interior
        eroded = scipy.ndimage.binary_erosion(mask, iterations=edge_width)

        # Edge is mask - eroded
        edge = mask & ~eroded

        if np.any(edge):
            result[int(label)] = float(np.mean(image[edge]))
        else:
            # Object too small for edge, use whole object
            result[int(label)] = float(np.mean(image[mask]))

    return result


def measure_intensity_at_center(
    image: NDArray,
    labels: NDArray,
    radius: int = 3,
) -> Dict[int, float]:
    """Compute the mean intensity at the center (centroid region) of each object.

    Parameters
    ----------
    image : NDArray
        Intensity image (grayscale).
    labels : NDArray
        Label image.
    radius : int
        Radius around centroid to consider.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping object label to its center intensity.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"Image shape {image.shape} doesn't match labels shape {labels.shape}"
        )

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    result = {}

    for label in unique_labels:
        mask = labels == label

        # Find centroid
        coords = np.argwhere(mask)
        centroid = np.mean(coords, axis=0).astype(int)

        # Create circular region around centroid
        if image.ndim == 2:
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            center_mask = (x - centroid[1])**2 + (y - centroid[0])**2 <= radius**2
        else:
            # 3D case
            z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
            center_mask = (
                (x - centroid[2])**2 +
                (y - centroid[1])**2 +
                (z - centroid[0])**2
            ) <= radius**2

        # Combine with object mask to stay within object
        center_mask = center_mask & mask

        if np.any(center_mask):
            result[int(label)] = float(np.mean(image[center_mask]))
        else:
            # Very small object, use centroid pixel
            result[int(label)] = float(image[tuple(centroid)])

    return result
