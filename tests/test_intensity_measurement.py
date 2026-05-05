"""
Tests for image intensity measurement functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cellprofiler_lib.intensity_measurement import (
    IntensityMeasurements,
    measure_intensity_min,
    measure_intensity_max,
    measure_intensity_mean,
    measure_intensity_median,
    measure_intensity_std,
    measure_intensity_percentile,
    measure_intensity_percentiles,
    measure_integrated_intensity,
    measure_intensity_mad,
    measure_all_intensity_properties,
    measure_intensity_in_channels,
    measure_intensity_ratio,
    measure_intensity_at_edge,
    measure_intensity_at_center,
)


def create_uniform_object():
    """Create a label image with a single object and uniform intensity."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1

    image = np.zeros((20, 20), dtype=np.float64)
    image[5:15, 5:15] = 0.5

    return image, labels


def create_gradient_object():
    """Create a label image with a single object with gradient intensity."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1

    image = np.zeros((20, 20), dtype=np.float64)
    # Create gradient from 0.0 to 1.0 within the object
    for i in range(10):
        image[5:15, 5+i] = i / 9.0

    return image, labels


def create_multiple_objects_with_intensity():
    """Create multiple objects with different intensities."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:8, 2:8] = 1  # Object 1
    labels[2:8, 22:28] = 2  # Object 2
    labels[22:28, 12:18] = 3  # Object 3

    image = np.zeros((30, 30), dtype=np.float64)
    image[2:8, 2:8] = 0.2  # Object 1: low intensity
    image[2:8, 22:28] = 0.5  # Object 2: medium intensity
    image[22:28, 12:18] = 0.8  # Object 3: high intensity

    return image, labels


def create_3d_object():
    """Create a 3D label image with intensity."""
    labels = np.zeros((10, 20, 20), dtype=np.int32)
    labels[2:8, 5:15, 5:15] = 1

    image = np.zeros((10, 20, 20), dtype=np.float64)
    image[2:8, 5:15, 5:15] = 0.5

    return image, labels


class TestMeasureIntensityMin:
    """Tests for minimum intensity measurement."""

    def test_uniform_object_min(self):
        """Uniform object should have min equal to its uniform value."""
        image, labels = create_uniform_object()
        mins = measure_intensity_min(image, labels)

        assert 1 in mins
        assert_allclose(mins[1], 0.5)

    def test_gradient_object_min(self):
        """Gradient object should have correct min."""
        image, labels = create_gradient_object()
        mins = measure_intensity_min(image, labels)

        assert 1 in mins
        assert_allclose(mins[1], 0.0)

    def test_multiple_objects_min(self):
        """Multiple objects should each have correct min."""
        image, labels = create_multiple_objects_with_intensity()
        mins = measure_intensity_min(image, labels)

        assert len(mins) == 3
        assert_allclose(mins[1], 0.2)
        assert_allclose(mins[2], 0.5)
        assert_allclose(mins[3], 0.8)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise error."""
        image = np.zeros((10, 10))
        labels = np.zeros((20, 20), dtype=np.int32)

        with pytest.raises(ValueError):
            measure_intensity_min(image, labels)


class TestMeasureIntensityMax:
    """Tests for maximum intensity measurement."""

    def test_uniform_object_max(self):
        """Uniform object should have max equal to its uniform value."""
        image, labels = create_uniform_object()
        maxs = measure_intensity_max(image, labels)

        assert 1 in maxs
        assert_allclose(maxs[1], 0.5)

    def test_gradient_object_max(self):
        """Gradient object should have correct max."""
        image, labels = create_gradient_object()
        maxs = measure_intensity_max(image, labels)

        assert 1 in maxs
        assert_allclose(maxs[1], 1.0)

    def test_multiple_objects_max(self):
        """Multiple objects should each have correct max."""
        image, labels = create_multiple_objects_with_intensity()
        maxs = measure_intensity_max(image, labels)

        assert len(maxs) == 3
        assert_allclose(maxs[1], 0.2)
        assert_allclose(maxs[2], 0.5)
        assert_allclose(maxs[3], 0.8)


class TestMeasureIntensityMean:
    """Tests for mean intensity measurement."""

    def test_uniform_object_mean(self):
        """Uniform object should have mean equal to its uniform value."""
        image, labels = create_uniform_object()
        means = measure_intensity_mean(image, labels)

        assert 1 in means
        assert_allclose(means[1], 0.5)

    def test_gradient_object_mean(self):
        """Gradient object should have correct mean."""
        image, labels = create_gradient_object()
        means = measure_intensity_mean(image, labels)

        assert 1 in means
        # Mean of 0, 0.111, 0.222, ..., 1.0 = 0.5
        assert_allclose(means[1], 0.5, atol=0.01)

    def test_multiple_objects_mean(self):
        """Multiple objects should each have correct mean."""
        image, labels = create_multiple_objects_with_intensity()
        means = measure_intensity_mean(image, labels)

        assert len(means) == 3
        assert_allclose(means[1], 0.2)
        assert_allclose(means[2], 0.5)
        assert_allclose(means[3], 0.8)


class TestMeasureIntensityMedian:
    """Tests for median intensity measurement."""

    def test_uniform_object_median(self):
        """Uniform object should have median equal to its uniform value."""
        image, labels = create_uniform_object()
        medians = measure_intensity_median(image, labels)

        assert 1 in medians
        assert_allclose(medians[1], 0.5)

    def test_gradient_object_median(self):
        """Gradient object should have correct median."""
        image, labels = create_gradient_object()
        medians = measure_intensity_median(image, labels)

        assert 1 in medians
        # For symmetric gradient, median should be near middle
        assert_allclose(medians[1], 0.5, atol=0.1)


class TestMeasureIntensityStd:
    """Tests for standard deviation measurement."""

    def test_uniform_object_zero_std(self):
        """Uniform object should have zero standard deviation."""
        image, labels = create_uniform_object()
        stds = measure_intensity_std(image, labels)

        assert 1 in stds
        assert_allclose(stds[1], 0.0, atol=1e-10)

    def test_gradient_object_nonzero_std(self):
        """Gradient object should have non-zero standard deviation."""
        image, labels = create_gradient_object()
        stds = measure_intensity_std(image, labels)

        assert 1 in stds
        assert stds[1] > 0


class TestMeasureIntensityPercentile:
    """Tests for percentile measurement."""

    def test_percentile_0(self):
        """0th percentile should equal minimum."""
        image, labels = create_gradient_object()

        mins = measure_intensity_min(image, labels)
        pct0 = measure_intensity_percentile(image, labels, 0)

        assert_allclose(pct0[1], mins[1])

    def test_percentile_100(self):
        """100th percentile should equal maximum."""
        image, labels = create_gradient_object()

        maxs = measure_intensity_max(image, labels)
        pct100 = measure_intensity_percentile(image, labels, 100)

        assert_allclose(pct100[1], maxs[1])

    def test_percentile_50(self):
        """50th percentile should equal median."""
        image, labels = create_gradient_object()

        medians = measure_intensity_median(image, labels)
        pct50 = measure_intensity_percentile(image, labels, 50)

        assert_allclose(pct50[1], medians[1], atol=0.01)

    def test_invalid_percentile_raises(self):
        """Invalid percentile should raise error."""
        image, labels = create_uniform_object()

        with pytest.raises(ValueError):
            measure_intensity_percentile(image, labels, 150)


class TestMeasureIntensityPercentiles:
    """Tests for multiple percentiles measurement."""

    def test_multiple_percentiles(self):
        """Should compute multiple percentiles at once."""
        image, labels = create_gradient_object()
        percentiles = [25, 50, 75]

        result = measure_intensity_percentiles(image, labels, percentiles)

        assert 1 in result
        assert 25.0 in result[1]
        assert 50.0 in result[1]
        assert 75.0 in result[1]

    def test_percentiles_ordered(self):
        """Percentiles should be ordered correctly."""
        image, labels = create_gradient_object()
        percentiles = [25, 50, 75]

        result = measure_intensity_percentiles(image, labels, percentiles)

        assert result[1][25.0] <= result[1][50.0] <= result[1][75.0]


class TestMeasureIntegratedIntensity:
    """Tests for integrated intensity measurement."""

    def test_uniform_object_integrated(self):
        """Integrated intensity should be sum of all pixel values."""
        image, labels = create_uniform_object()
        integrated = measure_integrated_intensity(image, labels)

        assert 1 in integrated
        # 10x10 = 100 pixels, each with value 0.5
        assert_allclose(integrated[1], 50.0)

    def test_multiple_objects_integrated(self):
        """Multiple objects should each have correct integrated intensity."""
        image, labels = create_multiple_objects_with_intensity()
        integrated = measure_integrated_intensity(image, labels)

        assert len(integrated) == 3
        # Each object is 6x6 = 36 pixels
        assert_allclose(integrated[1], 36 * 0.2)
        assert_allclose(integrated[2], 36 * 0.5)
        assert_allclose(integrated[3], 36 * 0.8)


class TestMeasureIntensityMAD:
    """Tests for median absolute deviation measurement."""

    def test_uniform_object_zero_mad(self):
        """Uniform object should have zero MAD."""
        image, labels = create_uniform_object()
        mads = measure_intensity_mad(image, labels)

        assert 1 in mads
        assert_allclose(mads[1], 0.0, atol=1e-10)

    def test_gradient_object_nonzero_mad(self):
        """Gradient object should have non-zero MAD."""
        image, labels = create_gradient_object()
        mads = measure_intensity_mad(image, labels)

        assert 1 in mads
        assert mads[1] > 0


class TestMeasureAllIntensityProperties:
    """Tests for comprehensive intensity measurement."""

    def test_returns_named_tuple(self):
        """Should return dictionary of IntensityMeasurements."""
        image, labels = create_uniform_object()
        measurements = measure_all_intensity_properties(image, labels)

        assert 1 in measurements
        assert isinstance(measurements[1], IntensityMeasurements)

    def test_all_properties_present(self):
        """All properties should be computed."""
        image, labels = create_uniform_object()
        measurements = measure_all_intensity_properties(image, labels)

        m = measurements[1]
        assert m.label == 1
        assert_allclose(m.min_intensity, 0.5)
        assert_allclose(m.max_intensity, 0.5)
        assert_allclose(m.mean_intensity, 0.5)
        assert_allclose(m.median_intensity, 0.5)
        assert_allclose(m.std_intensity, 0.0, atol=1e-10)
        assert_allclose(m.integrated_intensity, 50.0)
        assert_allclose(m.mad_intensity, 0.0, atol=1e-10)

    def test_multiple_objects_all_properties(self):
        """Multiple objects should all have measurements."""
        image, labels = create_multiple_objects_with_intensity()
        measurements = measure_all_intensity_properties(image, labels)

        assert len(measurements) == 3
        for label in [1, 2, 3]:
            assert label in measurements


class TestMeasureIntensityInChannels:
    """Tests for multi-channel intensity measurement."""

    def test_two_channels(self):
        """Should measure intensity in multiple channels."""
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1

        channel1 = np.zeros((20, 20), dtype=np.float64)
        channel1[5:15, 5:15] = 0.3

        channel2 = np.zeros((20, 20), dtype=np.float64)
        channel2[5:15, 5:15] = 0.7

        result = measure_intensity_in_channels(
            [channel1, channel2],
            labels,
            channel_names=["DAPI", "GFP"]
        )

        assert 1 in result
        assert "DAPI" in result[1]
        assert "GFP" in result[1]
        assert_allclose(result[1]["DAPI"].mean_intensity, 0.3)
        assert_allclose(result[1]["GFP"].mean_intensity, 0.7)

    def test_default_channel_names(self):
        """Should use default channel names if not provided."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:8, 2:8] = 1

        channel1 = np.ones((10, 10), dtype=np.float64) * 0.5
        channel2 = np.ones((10, 10), dtype=np.float64) * 0.5

        result = measure_intensity_in_channels([channel1, channel2], labels)

        assert 1 in result
        assert "channel_0" in result[1]
        assert "channel_1" in result[1]


class TestMeasureIntensityRatio:
    """Tests for intensity ratio measurement."""

    def test_simple_ratio(self):
        """Should compute correct ratio."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:8, 2:8] = 1

        numerator = np.ones((10, 10), dtype=np.float64) * 0.6
        denominator = np.ones((10, 10), dtype=np.float64) * 0.3

        ratios = measure_intensity_ratio(numerator, denominator, labels)

        assert 1 in ratios
        assert_allclose(ratios[1], 2.0, atol=0.01)

    def test_ratio_with_zero_denominator(self):
        """Should handle near-zero denominator with epsilon."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:8, 2:8] = 1

        numerator = np.ones((10, 10), dtype=np.float64) * 0.5
        denominator = np.zeros((10, 10), dtype=np.float64)

        ratios = measure_intensity_ratio(numerator, denominator, labels)

        assert 1 in ratios
        # Should be large but finite due to epsilon
        assert ratios[1] > 0


class TestMeasureIntensityAtEdge:
    """Tests for edge intensity measurement."""

    def test_edge_intensity(self):
        """Should measure intensity at object boundary."""
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1

        # Create image with higher intensity at edge
        image = np.zeros((20, 20), dtype=np.float64)
        image[5:15, 5:15] = 0.2  # Interior
        image[5, 5:15] = 1.0  # Top edge
        image[14, 5:15] = 1.0  # Bottom edge
        image[5:15, 5] = 1.0  # Left edge
        image[5:15, 14] = 1.0  # Right edge

        edge_intensity = measure_intensity_at_edge(image, labels, edge_width=1)

        assert 1 in edge_intensity
        # Edge should have higher intensity than center
        center_intensity = 0.2
        assert edge_intensity[1] > center_intensity

    def test_small_object_edge(self):
        """Should handle very small objects."""
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[10, 10] = 1  # Single pixel

        image = np.zeros((20, 20), dtype=np.float64)
        image[10, 10] = 0.5

        edge_intensity = measure_intensity_at_edge(image, labels)

        assert 1 in edge_intensity
        assert_allclose(edge_intensity[1], 0.5)


class TestMeasureIntensityAtCenter:
    """Tests for center intensity measurement."""

    def test_center_intensity(self):
        """Should measure intensity at object center."""
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1

        # Create image with higher intensity at center
        image = np.zeros((20, 20), dtype=np.float64)
        image[5:15, 5:15] = 0.2  # Object area
        image[8:12, 8:12] = 1.0  # Center region

        center_intensity = measure_intensity_at_center(image, labels, radius=2)

        assert 1 in center_intensity
        # Center should have higher intensity
        assert center_intensity[1] > 0.2


class Test3DSupport:
    """Tests for 3D image support."""

    def test_intensity_measurements_3d(self):
        """All measurements should work on 3D images."""
        image, labels = create_3d_object()

        assert measure_intensity_min(image, labels)[1] >= 0
        assert measure_intensity_max(image, labels)[1] >= 0
        assert_allclose(measure_intensity_mean(image, labels)[1], 0.5)
        assert_allclose(measure_intensity_median(image, labels)[1], 0.5)
        assert measure_intensity_std(image, labels)[1] >= 0
        assert measure_integrated_intensity(image, labels)[1] > 0

    def test_all_properties_3d(self):
        """Should measure all properties in 3D."""
        image, labels = create_3d_object()
        measurements = measure_all_intensity_properties(image, labels)

        assert 1 in measurements
        assert measurements[1].mean_intensity > 0


class TestEmptyLabels:
    """Tests for handling empty label images."""

    def test_empty_returns_empty_dict(self):
        """All measurement functions should return empty dict for empty labels."""
        image = np.ones((10, 10), dtype=np.float64)
        labels = np.zeros((10, 10), dtype=np.int32)

        assert measure_intensity_min(image, labels) == {}
        assert measure_intensity_max(image, labels) == {}
        assert measure_intensity_mean(image, labels) == {}
        assert measure_intensity_median(image, labels) == {}
        assert measure_intensity_std(image, labels) == {}
        assert measure_integrated_intensity(image, labels) == {}
        assert measure_all_intensity_properties(image, labels) == {}
