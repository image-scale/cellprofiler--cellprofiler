"""
Image overlap measurement functions.

This module provides functions for comparing segmentation results to ground truth,
computing metrics like precision, recall, F-score, Rand index, and Jaccard index.
"""
from typing import Dict, Optional, Tuple, NamedTuple
import numpy as np
from numpy.typing import NDArray
import scipy.sparse
import scipy.ndimage


class OverlapStatistics(NamedTuple):
    """Container for overlap statistics between two binary images."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f_score: float
    rand_index: float
    adjusted_rand_index: float


def compute_confusion_counts(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[int, int, int, int]:
    """Compute true positive, false positive, false negative, and true negative counts.

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image (True for foreground, False for background).
    test : NDArray
        Binary test/predicted image (True for foreground, False for background).
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation to specific regions.

    Returns
    -------
    Tuple[int, int, int, int]
        (true_positives, false_positives, false_negatives, true_negatives)
    """
    ground_truth = np.asarray(ground_truth, dtype=bool)
    test = np.asarray(test, dtype=bool)

    if ground_truth.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: ground_truth {ground_truth.shape} != test {test.shape}"
        )

    if mask is None:
        mask = np.ones(ground_truth.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ground_truth.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match image shape {ground_truth.shape}"
            )

    # Apply mask
    gt_masked = ground_truth & mask
    test_masked = test & mask

    # Compute confusion matrix components
    true_positives = np.sum(gt_masked & test_masked)
    false_positives = np.sum(~gt_masked & test_masked & mask)
    false_negatives = np.sum(gt_masked & ~test_masked & mask)
    true_negatives = np.sum(~gt_masked & ~test_masked & mask)

    return int(true_positives), int(false_positives), int(false_negatives), int(true_negatives)


def compute_precision(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute precision (positive predictive value).

    Precision = TP / (TP + FP)

    A precision of 1.0 means no false positives (all detected objects are real).

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Precision value between 0 and 1. Returns 1.0 if no positive predictions.
    """
    tp, fp, fn, tn = compute_confusion_counts(ground_truth, test, mask)

    if tp + fp == 0:
        return 1.0  # No positive predictions, precision is perfect by convention

    return float(tp) / float(tp + fp)


def compute_recall(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute recall (sensitivity, true positive rate).

    Recall = TP / (TP + FN)

    A recall of 1.0 means all ground truth objects were detected.

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Recall value between 0 and 1. Returns 1.0 if no ground truth positives.
    """
    tp, fp, fn, tn = compute_confusion_counts(ground_truth, test, mask)

    if tp + fn == 0:
        return 1.0  # No ground truth positives, recall is perfect by convention

    return float(tp) / float(tp + fn)


def compute_f_score(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
    beta: float = 1.0,
) -> float:
    """Compute F-score (F-measure), the harmonic mean of precision and recall.

    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    F1 score (beta=1) weights precision and recall equally.
    F2 score (beta=2) weights recall higher than precision.
    F0.5 score (beta=0.5) weights precision higher than recall.

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.
    beta : float, default=1.0
        Beta parameter controlling precision/recall tradeoff.

    Returns
    -------
    float
        F-score value between 0 and 1. Returns 0 if both precision and recall are 0.
    """
    precision = compute_precision(ground_truth, test, mask)
    recall = compute_recall(ground_truth, test, mask)

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_rand_index(
    labels1: NDArray,
    labels2: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute the Rand index measuring agreement between two labelings.

    The Rand index measures the similarity between two labelings by considering
    all pairs of pixels and counting pairs that are:
    - In the same cluster in both labelings (a)
    - In different clusters in both labelings (b)

    RI = (a + b) / (a + b + c + d)

    where c and d are disagreeing pairs.

    Parameters
    ----------
    labels1 : NDArray
        First label image (integer labels, 0 is background).
    labels2 : NDArray
        Second label image (integer labels, 0 is background).
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Rand index value between 0 and 1. Returns NaN if fewer than 2 pixels.
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    if labels1.shape != labels2.shape:
        raise ValueError(
            f"Shape mismatch: labels1 {labels1.shape} != labels2 {labels2.shape}"
        )

    if mask is None:
        mask = np.ones(labels1.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Flatten and apply mask
    l1 = labels1[mask].astype(np.uint32)
    l2 = labels2[mask].astype(np.uint32)

    n = len(l1)
    if n < 2:
        return np.nan

    # Build contingency matrix
    N_ij = scipy.sparse.coo_matrix(
        (np.ones(n), (l1, l2))
    ).toarray()

    def choose2(x):
        """Compute number of pairs = x * (x - 1) / 2"""
        return x * (x - 1) / 2

    # A = pairs in same set in both labelings
    A = np.sum(choose2(N_ij))

    # Row and column sums
    N_i = np.sum(N_ij, axis=1)
    N_j = np.sum(N_ij, axis=0)

    # C = pairs in same set in labels1 but different in labels2
    C = np.sum((N_i[:, np.newaxis] - N_ij) * N_ij) / 2

    # D = pairs in different sets in labels1 but same in labels2
    D = np.sum((N_j[np.newaxis, :] - N_ij) * N_ij) / 2

    # Total pairs
    total = choose2(n)

    # B = total - A - C - D (pairs in different sets in both)
    B = total - A - C - D

    rand_index = (A + B) / total

    return float(rand_index)


def compute_adjusted_rand_index(
    labels1: NDArray,
    labels2: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute the adjusted Rand index, correcting for chance agreement.

    The adjusted Rand index adjusts the Rand index to account for chance,
    making it useful when comparing labelings with different numbers of clusters.

    ARI = (RI - Expected RI) / (Max RI - Expected RI)

    Parameters
    ----------
    labels1 : NDArray
        First label image (integer labels, 0 is background).
    labels2 : NDArray
        Second label image (integer labels, 0 is background).
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Adjusted Rand index. Returns NaN if fewer than 2 pixels.
        Value of 1 indicates perfect agreement, 0 indicates random labeling.
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    if labels1.shape != labels2.shape:
        raise ValueError(
            f"Shape mismatch: labels1 {labels1.shape} != labels2 {labels2.shape}"
        )

    if mask is None:
        mask = np.ones(labels1.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Flatten and apply mask
    l1 = labels1[mask].astype(np.uint32)
    l2 = labels2[mask].astype(np.uint32)

    n = len(l1)
    if n < 2:
        return np.nan

    # Build contingency matrix
    N_ij = scipy.sparse.coo_matrix(
        (np.ones(n), (l1, l2))
    ).toarray()

    def choose2(x):
        """Compute number of pairs = x * (x - 1) / 2"""
        return x * (x - 1) / 2

    # A = pairs in same set in both labelings
    A = np.sum(choose2(N_ij))

    # Row and column sums
    N_i = np.sum(N_ij, axis=1)
    N_j = np.sum(N_ij, axis=0)

    # Total pairs
    total = choose2(n)

    # Expected index
    expected_index = np.sum(choose2(N_i)) * np.sum(choose2(N_j)) / total if total > 0 else 0

    # Max index
    max_index = (np.sum(choose2(N_i)) + np.sum(choose2(N_j))) / 2

    if max_index == expected_index:
        # Perfect agreement or degenerate case
        return 1.0 if A == max_index else 0.0

    adjusted_rand_index = (A - expected_index) / (max_index - expected_index)

    return float(adjusted_rand_index)


def compute_jaccard_index(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute the Jaccard index (Intersection over Union, IoU).

    Jaccard = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Jaccard index between 0 and 1. Returns 1.0 if both images are empty.
    """
    tp, fp, fn, tn = compute_confusion_counts(ground_truth, test, mask)

    if tp + fp + fn == 0:
        return 1.0  # Both empty, perfect overlap

    return float(tp) / float(tp + fp + fn)


def compute_jaccard_per_object(
    ground_truth_labels: NDArray,
    test_labels: NDArray,
) -> Dict[int, float]:
    """Compute Jaccard index for each object in the ground truth.

    For each ground truth object, finds the best matching test object
    and computes the Jaccard index (IoU) for that pair.

    Parameters
    ----------
    ground_truth_labels : NDArray
        Label image of ground truth objects (0 is background).
    test_labels : NDArray
        Label image of test/predicted objects (0 is background).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping ground truth label to its best Jaccard score.
    """
    ground_truth_labels = np.asarray(ground_truth_labels)
    test_labels = np.asarray(test_labels)

    if ground_truth_labels.shape != test_labels.shape:
        raise ValueError(
            f"Shape mismatch: ground_truth_labels {ground_truth_labels.shape} "
            f"!= test_labels {test_labels.shape}"
        )

    gt_labels = np.unique(ground_truth_labels)
    gt_labels = gt_labels[gt_labels != 0]  # Exclude background

    test_labels_unique = np.unique(test_labels)
    test_labels_unique = test_labels_unique[test_labels_unique != 0]

    jaccard_scores = {}

    for gt_label in gt_labels:
        gt_mask = ground_truth_labels == gt_label
        best_jaccard = 0.0

        # Find overlapping test objects
        overlapping_test_labels = np.unique(test_labels[gt_mask])
        overlapping_test_labels = overlapping_test_labels[overlapping_test_labels != 0]

        for test_label in overlapping_test_labels:
            test_mask = test_labels == test_label

            intersection = np.sum(gt_mask & test_mask)
            union = np.sum(gt_mask | test_mask)

            if union > 0:
                jaccard = float(intersection) / float(union)
                best_jaccard = max(best_jaccard, jaccard)

        jaccard_scores[int(gt_label)] = best_jaccard

    return jaccard_scores


def compute_overlap_statistics(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
    beta: float = 1.0,
) -> OverlapStatistics:
    """Compute comprehensive overlap statistics between two binary images.

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.
    beta : float, default=1.0
        Beta parameter for F-score calculation.

    Returns
    -------
    OverlapStatistics
        Named tuple containing all overlap metrics.
    """
    ground_truth = np.asarray(ground_truth, dtype=bool)
    test = np.asarray(test, dtype=bool)

    tp, fp, fn, tn = compute_confusion_counts(ground_truth, test, mask)
    precision = compute_precision(ground_truth, test, mask)
    recall = compute_recall(ground_truth, test, mask)
    f_score = compute_f_score(ground_truth, test, mask, beta)

    # For Rand index, convert binary to labels (background=0, foreground=1)
    gt_labels = ground_truth.astype(np.int32)
    test_as_labels = test.astype(np.int32)

    # Use scipy.ndimage.label to get connected components for Rand index
    gt_labeled, _ = scipy.ndimage.label(ground_truth)
    test_labeled, _ = scipy.ndimage.label(test)

    rand_index = compute_rand_index(gt_labeled, test_labeled, mask)
    adjusted_rand_index = compute_adjusted_rand_index(gt_labeled, test_labeled, mask)

    return OverlapStatistics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f_score=f_score,
        rand_index=rand_index,
        adjusted_rand_index=adjusted_rand_index,
    )


def compute_dice_coefficient(
    ground_truth: NDArray,
    test: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
) -> float:
    """Compute the Dice coefficient (Sørensen–Dice coefficient).

    Dice = 2 * |A ∩ B| / (|A| + |B|) = 2 * TP / (2 * TP + FP + FN)

    The Dice coefficient is equivalent to F1 score.

    Parameters
    ----------
    ground_truth : NDArray
        Binary ground truth image.
    test : NDArray
        Binary test/predicted image.
    mask : NDArray[np.bool_], optional
        Optional mask to limit computation.

    Returns
    -------
    float
        Dice coefficient between 0 and 1. Returns 1.0 if both images are empty.
    """
    tp, fp, fn, tn = compute_confusion_counts(ground_truth, test, mask)

    if 2 * tp + fp + fn == 0:
        return 1.0  # Both empty

    return float(2 * tp) / float(2 * tp + fp + fn)
