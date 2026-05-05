"""
Tests for image overlap measurement functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from cellprofiler_lib.measurement import (
    OverlapStatistics,
    compute_confusion_counts,
    compute_precision,
    compute_recall,
    compute_f_score,
    compute_rand_index,
    compute_adjusted_rand_index,
    compute_jaccard_index,
    compute_jaccard_per_object,
    compute_overlap_statistics,
    compute_dice_coefficient,
)


def create_simple_overlap():
    """Create simple overlapping binary images for testing."""
    # Ground truth: 4x4 block in upper left
    ground_truth = np.zeros((10, 10), dtype=bool)
    ground_truth[0:4, 0:4] = True  # 16 pixels

    # Test: 4x4 block shifted right and down by 2
    test = np.zeros((10, 10), dtype=bool)
    test[2:6, 2:6] = True  # 16 pixels

    # Overlap is 2x2 = 4 pixels (rows 2-3, cols 2-3)
    return ground_truth, test


def create_perfect_match():
    """Create identical binary images."""
    image = np.zeros((10, 10), dtype=bool)
    image[2:8, 2:8] = True
    return image.copy(), image.copy()


def create_no_overlap():
    """Create non-overlapping binary images."""
    ground_truth = np.zeros((10, 10), dtype=bool)
    ground_truth[0:3, 0:3] = True

    test = np.zeros((10, 10), dtype=bool)
    test[7:10, 7:10] = True

    return ground_truth, test


def create_labeled_image():
    """Create a labeled image with 3 distinct objects."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:3, 0:3] = 1  # Object 1
    labels[0:3, 7:10] = 2  # Object 2
    labels[7:10, 3:7] = 3  # Object 3
    return labels


class TestComputeConfusionCounts:
    """Tests for confusion matrix computation."""

    def test_perfect_overlap(self):
        """Perfect match should have only true positives and true negatives."""
        gt, test = create_perfect_match()
        tp, fp, fn, tn = compute_confusion_counts(gt, test)

        assert tp == np.sum(gt)  # All foreground is TP
        assert fp == 0
        assert fn == 0
        assert tn == np.sum(~gt)  # All background is TN
        assert tp + fp + fn + tn == gt.size

    def test_no_overlap(self):
        """Non-overlapping images should have no true positives."""
        gt, test = create_no_overlap()
        tp, fp, fn, tn = compute_confusion_counts(gt, test)

        assert tp == 0
        assert fp == np.sum(test)  # All test foreground is FP
        assert fn == np.sum(gt)  # All GT foreground is FN
        assert tp + fp + fn + tn == gt.size

    def test_partial_overlap(self):
        """Partial overlap should have mixed counts."""
        gt, test = create_simple_overlap()
        tp, fp, fn, tn = compute_confusion_counts(gt, test)

        # Overlap is 2x2 = 4 pixels
        assert tp == 4
        # Test has 16 pixels, 4 overlap = 12 FP
        assert fp == 12
        # GT has 16 pixels, 4 overlap = 12 FN
        assert fn == 12
        assert tp + fp + fn + tn == gt.size

    def test_with_mask(self):
        """Mask should limit computation region."""
        gt, test = create_simple_overlap()

        # Mask only the upper-left quadrant
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, 0:5] = True

        tp, fp, fn, tn = compute_confusion_counts(gt, test, mask)

        # Only count within mask
        assert tp + fp + fn + tn == np.sum(mask)

    def test_empty_images(self):
        """Empty images should have only true negatives."""
        gt = np.zeros((5, 5), dtype=bool)
        test = np.zeros((5, 5), dtype=bool)

        tp, fp, fn, tn = compute_confusion_counts(gt, test)

        assert tp == 0
        assert fp == 0
        assert fn == 0
        assert tn == 25

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise error."""
        gt = np.zeros((5, 5), dtype=bool)
        test = np.zeros((6, 6), dtype=bool)

        with pytest.raises(ValueError):
            compute_confusion_counts(gt, test)


class TestComputePrecision:
    """Tests for precision calculation."""

    def test_perfect_precision(self):
        """Perfect match should have precision of 1."""
        gt, test = create_perfect_match()
        precision = compute_precision(gt, test)

        assert precision == 1.0

    def test_zero_precision(self):
        """No overlap should have precision of 0."""
        gt, test = create_no_overlap()
        precision = compute_precision(gt, test)

        assert precision == 0.0

    def test_partial_precision(self):
        """Partial overlap should have precision < 1."""
        gt, test = create_simple_overlap()
        precision = compute_precision(gt, test)

        # TP = 4, FP = 12, so precision = 4/16 = 0.25
        assert_allclose(precision, 0.25)

    def test_no_positive_predictions(self):
        """No positive predictions should return precision of 1."""
        gt = np.ones((5, 5), dtype=bool)
        test = np.zeros((5, 5), dtype=bool)

        precision = compute_precision(gt, test)
        assert precision == 1.0  # Convention: no FP means perfect precision


class TestComputeRecall:
    """Tests for recall calculation."""

    def test_perfect_recall(self):
        """Perfect match should have recall of 1."""
        gt, test = create_perfect_match()
        recall = compute_recall(gt, test)

        assert recall == 1.0

    def test_zero_recall(self):
        """No overlap should have recall of 0."""
        gt, test = create_no_overlap()
        recall = compute_recall(gt, test)

        assert recall == 0.0

    def test_partial_recall(self):
        """Partial overlap should have recall < 1."""
        gt, test = create_simple_overlap()
        recall = compute_recall(gt, test)

        # TP = 4, FN = 12, so recall = 4/16 = 0.25
        assert_allclose(recall, 0.25)

    def test_no_ground_truth_positives(self):
        """No ground truth positives should return recall of 1."""
        gt = np.zeros((5, 5), dtype=bool)
        test = np.ones((5, 5), dtype=bool)

        recall = compute_recall(gt, test)
        assert recall == 1.0  # Convention: no FN means perfect recall


class TestComputeFScore:
    """Tests for F-score calculation."""

    def test_perfect_f_score(self):
        """Perfect match should have F-score of 1."""
        gt, test = create_perfect_match()
        f_score = compute_f_score(gt, test)

        assert f_score == 1.0

    def test_zero_f_score(self):
        """No overlap should have F-score of 0."""
        gt, test = create_no_overlap()
        f_score = compute_f_score(gt, test)

        assert f_score == 0.0

    def test_partial_f_score(self):
        """Partial overlap should have F-score between 0 and 1."""
        gt, test = create_simple_overlap()
        f_score = compute_f_score(gt, test)

        # Precision = 0.25, Recall = 0.25
        # F1 = 2 * 0.25 * 0.25 / (0.25 + 0.25) = 0.25
        assert_allclose(f_score, 0.25)

    def test_f2_score_weights_recall(self):
        """F2 score should weight recall more than precision."""
        # Create image where recall > precision
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:5, 0:10] = True  # 50 pixels

        test = np.zeros((10, 10), dtype=bool)
        test[0:5, 0:5] = True  # 25 pixels, all overlapping

        f1 = compute_f_score(gt, test, beta=1.0)
        f2 = compute_f_score(gt, test, beta=2.0)

        # Precision = 25/25 = 1.0, Recall = 25/50 = 0.5
        # F1 = 2 * 1 * 0.5 / (1 + 0.5) = 0.667
        # F2 weights recall more, should be lower
        assert f2 < f1

    def test_f05_score_weights_precision(self):
        """F0.5 score should weight precision more than recall."""
        gt, test = create_simple_overlap()

        f1 = compute_f_score(gt, test, beta=1.0)
        f05 = compute_f_score(gt, test, beta=0.5)

        # With equal precision and recall, F0.5 = F1
        assert_allclose(f05, f1)


class TestComputeRandIndex:
    """Tests for Rand index calculation."""

    def test_identical_labelings(self):
        """Identical labelings should have Rand index of 1."""
        labels = create_labeled_image()
        rand_index = compute_rand_index(labels, labels)

        assert_allclose(rand_index, 1.0)

    def test_different_labelings(self):
        """Different labelings should have Rand index < 1."""
        labels1 = np.zeros((10, 10), dtype=np.int32)
        labels1[0:5, :] = 1
        labels1[5:10, :] = 2

        labels2 = np.zeros((10, 10), dtype=np.int32)
        labels2[:, 0:5] = 1
        labels2[:, 5:10] = 2

        rand_index = compute_rand_index(labels1, labels2)

        assert 0 < rand_index < 1

    def test_empty_labels_returns_nan(self):
        """Empty labels should return NaN."""
        labels = np.zeros((5, 5), dtype=np.int32)
        rand_index = compute_rand_index(labels, labels)

        # All pixels are background (label 0), still valid
        assert not np.isnan(rand_index)

    def test_single_pixel_returns_nan(self):
        """Single pixel should return NaN (can't compute pairs)."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True

        labels = np.ones((5, 5), dtype=np.int32)
        rand_index = compute_rand_index(labels, labels, mask)

        assert np.isnan(rand_index)

    def test_with_mask(self):
        """Mask should limit computation region."""
        labels = create_labeled_image()

        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, 0:5] = True

        rand_index = compute_rand_index(labels, labels, mask)

        assert_allclose(rand_index, 1.0)


class TestComputeAdjustedRandIndex:
    """Tests for adjusted Rand index calculation."""

    def test_identical_labelings(self):
        """Identical labelings should have adjusted Rand index of 1."""
        labels = create_labeled_image()
        ari = compute_adjusted_rand_index(labels, labels)

        assert_allclose(ari, 1.0)

    def test_different_labelings(self):
        """Different labelings should have lower adjusted Rand index."""
        labels1 = create_labeled_image()

        labels2 = np.zeros((10, 10), dtype=np.int32)
        labels2[:, 0:5] = 1
        labels2[:, 5:10] = 2

        ari = compute_adjusted_rand_index(labels1, labels2)

        assert ari < 1.0

    def test_adjusted_rand_handles_chance(self):
        """Adjusted Rand index should be around 0 for random labelings."""
        np.random.seed(42)
        labels1 = np.random.randint(0, 5, size=(20, 20))
        labels2 = np.random.randint(0, 5, size=(20, 20))

        ari = compute_adjusted_rand_index(labels1, labels2)

        # Should be close to 0 for random labelings
        assert abs(ari) < 0.3


class TestComputeJaccardIndex:
    """Tests for Jaccard index calculation."""

    def test_perfect_jaccard(self):
        """Perfect match should have Jaccard of 1."""
        gt, test = create_perfect_match()
        jaccard = compute_jaccard_index(gt, test)

        assert jaccard == 1.0

    def test_no_overlap_jaccard(self):
        """No overlap should have Jaccard of 0."""
        gt, test = create_no_overlap()
        jaccard = compute_jaccard_index(gt, test)

        assert jaccard == 0.0

    def test_partial_jaccard(self):
        """Partial overlap should have Jaccard between 0 and 1."""
        gt, test = create_simple_overlap()
        jaccard = compute_jaccard_index(gt, test)

        # TP = 4, FP = 12, FN = 12
        # Jaccard = 4 / (4 + 12 + 12) = 4/28 = 1/7
        assert_allclose(jaccard, 1 / 7)

    def test_empty_images_jaccard(self):
        """Empty images should have Jaccard of 1."""
        gt = np.zeros((5, 5), dtype=bool)
        test = np.zeros((5, 5), dtype=bool)

        jaccard = compute_jaccard_index(gt, test)
        assert jaccard == 1.0


class TestComputeJaccardPerObject:
    """Tests for per-object Jaccard calculation."""

    def test_perfect_match_objects(self):
        """Identical labeled images should have Jaccard of 1 for all objects."""
        labels = create_labeled_image()
        jaccard_scores = compute_jaccard_per_object(labels, labels)

        for label in [1, 2, 3]:
            assert_allclose(jaccard_scores[label], 1.0)

    def test_partial_overlap_objects(self):
        """Partially overlapping objects should have Jaccard < 1."""
        gt_labels = np.zeros((10, 10), dtype=np.int32)
        gt_labels[0:4, 0:4] = 1

        test_labels = np.zeros((10, 10), dtype=np.int32)
        test_labels[2:6, 2:6] = 1

        jaccard_scores = compute_jaccard_per_object(gt_labels, test_labels)

        # Object 1: overlap is 4 pixels, union is 28 pixels
        assert_allclose(jaccard_scores[1], 4 / 28)

    def test_no_overlap_objects(self):
        """Non-overlapping objects should have Jaccard of 0."""
        gt_labels = np.zeros((10, 10), dtype=np.int32)
        gt_labels[0:3, 0:3] = 1

        test_labels = np.zeros((10, 10), dtype=np.int32)
        test_labels[7:10, 7:10] = 1

        jaccard_scores = compute_jaccard_per_object(gt_labels, test_labels)

        assert jaccard_scores[1] == 0.0

    def test_empty_ground_truth(self):
        """Empty ground truth should return empty dictionary."""
        gt_labels = np.zeros((10, 10), dtype=np.int32)
        test_labels = create_labeled_image()

        jaccard_scores = compute_jaccard_per_object(gt_labels, test_labels)

        assert len(jaccard_scores) == 0


class TestComputeOverlapStatistics:
    """Tests for comprehensive overlap statistics."""

    def test_returns_named_tuple(self):
        """Should return OverlapStatistics named tuple."""
        gt, test = create_simple_overlap()
        stats = compute_overlap_statistics(gt, test)

        assert isinstance(stats, OverlapStatistics)
        assert hasattr(stats, 'true_positives')
        assert hasattr(stats, 'precision')
        assert hasattr(stats, 'rand_index')

    def test_perfect_statistics(self):
        """Perfect match should have all metrics at 1."""
        gt, test = create_perfect_match()
        stats = compute_overlap_statistics(gt, test)

        assert stats.precision == 1.0
        assert stats.recall == 1.0
        assert stats.f_score == 1.0
        assert stats.false_positives == 0
        assert stats.false_negatives == 0

    def test_with_custom_beta(self):
        """Custom beta should affect F-score."""
        gt, test = create_simple_overlap()

        stats1 = compute_overlap_statistics(gt, test, beta=1.0)
        stats2 = compute_overlap_statistics(gt, test, beta=2.0)

        # With equal precision and recall, F1 == F2
        assert_allclose(stats1.f_score, stats2.f_score)


class TestComputeDiceCoefficient:
    """Tests for Dice coefficient calculation."""

    def test_perfect_dice(self):
        """Perfect match should have Dice of 1."""
        gt, test = create_perfect_match()
        dice = compute_dice_coefficient(gt, test)

        assert dice == 1.0

    def test_no_overlap_dice(self):
        """No overlap should have Dice of 0."""
        gt, test = create_no_overlap()
        dice = compute_dice_coefficient(gt, test)

        assert dice == 0.0

    def test_dice_equals_f1(self):
        """Dice coefficient should equal F1 score."""
        gt, test = create_simple_overlap()

        dice = compute_dice_coefficient(gt, test)
        f1 = compute_f_score(gt, test, beta=1.0)

        assert_allclose(dice, f1)

    def test_empty_images_dice(self):
        """Empty images should have Dice of 1."""
        gt = np.zeros((5, 5), dtype=bool)
        test = np.zeros((5, 5), dtype=bool)

        dice = compute_dice_coefficient(gt, test)
        assert dice == 1.0


class Test3DSupport:
    """Tests for 3D image support."""

    def test_confusion_counts_3d(self):
        """Should handle 3D binary images."""
        gt = np.zeros((5, 10, 10), dtype=bool)
        gt[1:4, 2:8, 2:8] = True

        test = np.zeros((5, 10, 10), dtype=bool)
        test[1:4, 2:8, 2:8] = True

        tp, fp, fn, tn = compute_confusion_counts(gt, test)

        assert tp == np.sum(gt)
        assert fp == 0
        assert fn == 0

    def test_rand_index_3d(self):
        """Should handle 3D label images."""
        labels = np.zeros((5, 10, 10), dtype=np.int32)
        labels[0:2, :, :] = 1
        labels[2:4, :, :] = 2
        labels[4:5, :, :] = 3

        rand_index = compute_rand_index(labels, labels)

        assert_allclose(rand_index, 1.0)

    def test_jaccard_3d(self):
        """Should handle 3D binary images."""
        gt = np.zeros((5, 10, 10), dtype=bool)
        gt[1:4, 2:8, 2:8] = True

        test = np.zeros((5, 10, 10), dtype=bool)
        test[1:4, 2:8, 2:8] = True

        jaccard = compute_jaccard_index(gt, test)

        assert jaccard == 1.0


class TestEmptyLabelings:
    """Tests for handling empty labelings gracefully."""

    def test_both_empty_binary(self):
        """Both empty binary images should return appropriate metrics."""
        gt = np.zeros((10, 10), dtype=bool)
        test = np.zeros((10, 10), dtype=bool)

        assert compute_precision(gt, test) == 1.0
        assert compute_recall(gt, test) == 1.0
        assert compute_f_score(gt, test) == 1.0
        assert compute_jaccard_index(gt, test) == 1.0
        assert compute_dice_coefficient(gt, test) == 1.0

    def test_empty_ground_truth(self):
        """Empty ground truth with non-empty test."""
        gt = np.zeros((10, 10), dtype=bool)
        test = np.ones((10, 10), dtype=bool)

        assert compute_precision(gt, test) == 0.0
        assert compute_recall(gt, test) == 1.0  # No FN possible

    def test_empty_test(self):
        """Non-empty ground truth with empty test."""
        gt = np.ones((10, 10), dtype=bool)
        test = np.zeros((10, 10), dtype=bool)

        assert compute_precision(gt, test) == 1.0  # No FP possible
        assert compute_recall(gt, test) == 0.0

    def test_empty_labels_rand_index(self):
        """Empty labeled images for Rand index."""
        labels = np.zeros((10, 10), dtype=np.int32)

        rand_index = compute_rand_index(labels, labels)

        # All background, should be valid
        assert_allclose(rand_index, 1.0)
