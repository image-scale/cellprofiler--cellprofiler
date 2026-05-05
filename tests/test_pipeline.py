"""
Tests for pipeline framework.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cellprofiler_lib.pipeline import (
    ModuleCategory,
    Workspace,
    Module,
    Pipeline,
    ThresholdModule,
    IdentifyObjectsModule,
    MeasureObjectSizeModule,
    MeasureIntensityModule,
    GaussianSmoothModule,
    RescaleIntensityModule,
    InvertModule,
    CropModule,
)


def create_test_image():
    """Create a simple test image with objects."""
    image = np.zeros((50, 50), dtype=np.float64)
    image[10:20, 10:20] = 0.8  # Object 1
    image[10:20, 30:40] = 0.9  # Object 2
    image[30:40, 20:30] = 0.7  # Object 3
    return image


class TestWorkspace:
    """Tests for Workspace class."""

    def test_set_get_image(self):
        """Should store and retrieve images."""
        ws = Workspace()
        image = np.ones((10, 10))

        ws.set_image("test", image)
        retrieved = ws.get_image("test")

        assert_array_equal(retrieved, image)

    def test_get_missing_image_raises(self):
        """Should raise KeyError for missing image."""
        ws = Workspace()

        with pytest.raises(KeyError):
            ws.get_image("nonexistent")

    def test_set_get_objects(self):
        """Should store and retrieve objects."""
        ws = Workspace()
        labels = np.array([[0, 1], [1, 2]], dtype=np.int32)

        ws.set_objects("cells", labels)
        retrieved = ws.get_objects("cells")

        assert_array_equal(retrieved, labels)

    def test_add_get_measurement(self):
        """Should store and retrieve measurements."""
        ws = Workspace()
        areas = {1: 100, 2: 200}

        ws.add_measurement("cells", "area", areas)
        retrieved = ws.get_measurement("cells", "area")

        assert retrieved == areas

    def test_get_missing_measurement_raises(self):
        """Should raise KeyError for missing measurement."""
        ws = Workspace()

        with pytest.raises(KeyError):
            ws.get_measurement("cells", "area")

    def test_clear(self):
        """Should clear all data."""
        ws = Workspace()
        ws.set_image("test", np.ones((5, 5)))
        ws.set_objects("cells", np.ones((5, 5), dtype=np.int32))
        ws.add_measurement("cells", "area", {1: 100})

        ws.clear()

        assert len(ws.images) == 0
        assert len(ws.objects) == 0
        assert len(ws.measurements) == 0


class TestModule:
    """Tests for Module base class."""

    def test_module_name(self):
        """Module should have default name from class."""
        module = ThresholdModule()
        assert module.name == "ThresholdModule"

    def test_module_custom_name(self):
        """Module should accept custom name."""
        module = ThresholdModule(name="MyThreshold")
        assert module.name == "MyThreshold"

    def test_module_enabled_by_default(self):
        """Module should be enabled by default."""
        module = ThresholdModule()
        assert module.enabled is True

    def test_module_can_be_disabled(self):
        """Module can be disabled."""
        module = ThresholdModule()
        module.enabled = False
        assert module.enabled is False

    def test_module_has_category(self):
        """Module should have a category."""
        threshold = ThresholdModule()
        identify = IdentifyObjectsModule()
        measure = MeasureObjectSizeModule()

        assert threshold.category == ModuleCategory.IMAGE_PROCESSING
        assert identify.category == ModuleCategory.SEGMENTATION
        assert measure.category == ModuleCategory.MEASUREMENT


class TestPipeline:
    """Tests for Pipeline class."""

    def test_create_empty_pipeline(self):
        """Should create empty pipeline."""
        pipeline = Pipeline()
        assert len(pipeline.modules) == 0

    def test_create_pipeline_with_modules(self):
        """Should create pipeline with initial modules."""
        modules = [ThresholdModule(), IdentifyObjectsModule()]
        pipeline = Pipeline(modules)

        assert len(pipeline.modules) == 2

    def test_add_module(self):
        """Should add module to pipeline."""
        pipeline = Pipeline()
        pipeline.add_module(ThresholdModule())

        assert len(pipeline.modules) == 1

    def test_add_module_at_index(self):
        """Should insert module at specified index."""
        pipeline = Pipeline([ThresholdModule(), MeasureObjectSizeModule()])
        pipeline.add_module(IdentifyObjectsModule(), index=1)

        assert len(pipeline.modules) == 3
        assert isinstance(pipeline.modules[1], IdentifyObjectsModule)

    def test_remove_module_by_index(self):
        """Should remove module by index."""
        pipeline = Pipeline([ThresholdModule(), IdentifyObjectsModule()])
        removed = pipeline.remove_module(0)

        assert len(pipeline.modules) == 1
        assert isinstance(removed, ThresholdModule)

    def test_remove_module_by_reference(self):
        """Should remove module by reference."""
        threshold = ThresholdModule()
        pipeline = Pipeline([threshold, IdentifyObjectsModule()])
        pipeline.remove_module(threshold)

        assert len(pipeline.modules) == 1

    def test_move_module(self):
        """Should move module to new position."""
        m1 = ThresholdModule()
        m2 = IdentifyObjectsModule()
        m3 = MeasureObjectSizeModule()
        pipeline = Pipeline([m1, m2, m3])

        pipeline.move_module(2, 0)

        assert pipeline.modules[0] is m3
        assert pipeline.modules[1] is m1
        assert pipeline.modules[2] is m2

    def test_run_pipeline(self):
        """Should execute all modules."""
        image = create_test_image()

        pipeline = Pipeline([
            ThresholdModule(threshold=0.5),
            IdentifyObjectsModule(),
        ])

        workspace = pipeline.run(image)

        assert "binary" in workspace.images
        assert "objects" in workspace.objects

    def test_run_pipeline_skips_disabled_modules(self):
        """Should skip disabled modules."""
        image = create_test_image()

        threshold = ThresholdModule(threshold=0.5)
        threshold.enabled = False

        pipeline = Pipeline([threshold])
        workspace = pipeline.run(image)

        assert "binary" not in workspace.images

    def test_validate_pipeline(self):
        """Should validate all modules."""
        pipeline = Pipeline([
            ThresholdModule(threshold=1.5),  # Invalid threshold
        ])

        errors = pipeline.validate()

        assert len(errors) > 0
        assert "Threshold" in errors[0]

    def test_get_all_measurements(self):
        """Should list all measurements from all modules."""
        pipeline = Pipeline([
            ThresholdModule(),
            IdentifyObjectsModule(),
            MeasureObjectSizeModule(),
            MeasureIntensityModule(),
        ])

        measurements = pipeline.get_all_measurements()

        assert "MeasureObjectSizeModule" in measurements
        assert "area" in measurements["MeasureObjectSizeModule"]
        assert "MeasureIntensityModule" in measurements
        assert "mean_intensity" in measurements["MeasureIntensityModule"]


class TestThresholdModule:
    """Tests for ThresholdModule."""

    def test_threshold_above(self):
        """Should threshold above value."""
        ws = Workspace()
        ws.set_image("input", np.array([[0.3, 0.7], [0.5, 0.9]]))

        module = ThresholdModule(threshold=0.5, above=True)
        module.run(ws)

        binary = ws.get_image("binary")
        expected = np.array([[False, True], [False, True]])
        assert_array_equal(binary, expected)

    def test_threshold_below(self):
        """Should threshold below value."""
        ws = Workspace()
        ws.set_image("input", np.array([[0.3, 0.7], [0.5, 0.9]]))

        module = ThresholdModule(threshold=0.5, above=False)
        module.run(ws)

        binary = ws.get_image("binary")
        expected = np.array([[True, False], [False, False]])
        assert_array_equal(binary, expected)

    def test_validate_threshold_range(self):
        """Should validate threshold is in range."""
        module = ThresholdModule(threshold=1.5)
        errors = module.validate_settings()

        assert len(errors) > 0


class TestIdentifyObjectsModule:
    """Tests for IdentifyObjectsModule."""

    def test_identify_objects(self):
        """Should identify connected objects."""
        ws = Workspace()
        binary = np.zeros((10, 10), dtype=bool)
        binary[1:3, 1:3] = True  # Object 1
        binary[5:8, 5:8] = True  # Object 2
        ws.set_image("binary", binary)

        module = IdentifyObjectsModule()
        module.run(ws)

        labels = ws.get_objects("objects")
        assert np.max(labels) == 2

    def test_identify_with_min_size(self):
        """Should filter by minimum size."""
        ws = Workspace()
        binary = np.zeros((20, 20), dtype=bool)
        binary[1:3, 1:3] = True  # Small object (4 pixels)
        binary[10:18, 10:18] = True  # Large object (64 pixels)
        ws.set_image("binary", binary)

        module = IdentifyObjectsModule(min_size=10)
        module.run(ws)

        labels = ws.get_objects("objects")
        assert np.max(labels) == 1  # Only large object remains


class TestMeasureObjectSizeModule:
    """Tests for MeasureObjectSizeModule."""

    def test_measure_areas(self):
        """Should measure object areas."""
        ws = Workspace()
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[1:3, 1:3] = 1  # 4 pixels
        labels[5:8, 5:8] = 2  # 9 pixels
        ws.set_objects("objects", labels)

        module = MeasureObjectSizeModule()
        module.run(ws)

        areas = ws.get_measurement("objects", "area")
        assert areas[1] == 4
        assert areas[2] == 9

    def test_measure_centroids(self):
        """Should measure object centroids."""
        ws = Workspace()
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[0:2, 0:2] = 1
        ws.set_objects("objects", labels)

        module = MeasureObjectSizeModule()
        module.run(ws)

        centroids = ws.get_measurement("objects", "centroid")
        assert 1 in centroids
        assert_allclose(centroids[1], (0.5, 0.5))


class TestMeasureIntensityModule:
    """Tests for MeasureIntensityModule."""

    def test_measure_mean_intensity(self):
        """Should measure mean intensity."""
        ws = Workspace()
        image = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([[1, 1], [1, 1]], dtype=np.int32)
        ws.set_image("input", image)
        ws.set_objects("objects", labels)

        module = MeasureIntensityModule()
        module.run(ws)

        means = ws.get_measurement("objects", "mean_intensity")
        assert_allclose(means[1], 0.5)

    def test_measure_integrated_intensity(self):
        """Should measure integrated (sum) intensity."""
        ws = Workspace()
        image = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([[1, 1], [1, 1]], dtype=np.int32)
        ws.set_image("input", image)
        ws.set_objects("objects", labels)

        module = MeasureIntensityModule()
        module.run(ws)

        integrated = ws.get_measurement("objects", "integrated_intensity")
        assert_allclose(integrated[1], 2.0)


class TestGaussianSmoothModule:
    """Tests for GaussianSmoothModule."""

    def test_smooth_image(self):
        """Should smooth image."""
        ws = Workspace()
        image = np.zeros((20, 20))
        image[10, 10] = 1.0  # Single bright pixel
        ws.set_image("input", image)

        module = GaussianSmoothModule(sigma=2.0)
        module.run(ws)

        smoothed = ws.get_image("smoothed")
        # Peak should be reduced
        assert smoothed.max() < image.max()
        # Sum should be preserved (approximately)
        assert_allclose(smoothed.sum(), image.sum(), atol=0.1)

    def test_validate_sigma(self):
        """Should validate positive sigma."""
        module = GaussianSmoothModule(sigma=-1.0)
        errors = module.validate_settings()

        assert len(errors) > 0


class TestRescaleIntensityModule:
    """Tests for RescaleIntensityModule."""

    def test_rescale_to_0_1(self):
        """Should rescale to 0-1 range."""
        ws = Workspace()
        image = np.array([[0, 100], [200, 255]])
        ws.set_image("input", image)

        module = RescaleIntensityModule()
        module.run(ws)

        rescaled = ws.get_image("rescaled")
        assert_allclose(rescaled.min(), 0.0)
        assert_allclose(rescaled.max(), 1.0)


class TestInvertModule:
    """Tests for InvertModule."""

    def test_invert_float_image(self):
        """Should invert float image."""
        ws = Workspace()
        image = np.array([[0.0, 0.25], [0.5, 1.0]])
        ws.set_image("input", image)

        module = InvertModule()
        module.run(ws)

        inverted = ws.get_image("inverted")
        expected = np.array([[1.0, 0.75], [0.5, 0.0]])
        assert_allclose(inverted, expected)


class TestCropModule:
    """Tests for CropModule."""

    def test_crop_image(self):
        """Should crop image."""
        ws = Workspace()
        image = np.arange(100).reshape(10, 10).astype(np.float64)
        ws.set_image("input", image)

        module = CropModule(top=2, left=3, height=4, width=5)
        module.run(ws)

        cropped = ws.get_image("cropped")
        assert cropped.shape == (4, 5)
        assert_array_equal(cropped, image[2:6, 3:8])


class TestFullPipeline:
    """Integration tests for full pipelines."""

    def test_typical_analysis_pipeline(self):
        """Test a typical image analysis pipeline."""
        # Create test image with objects
        image = create_test_image()

        # Build pipeline
        pipeline = Pipeline([
            RescaleIntensityModule(input_image="input", output_image="normalized"),
            ThresholdModule(
                input_image="normalized",
                output_image="binary",
                threshold=0.5
            ),
            IdentifyObjectsModule(input_image="binary", output_objects="cells"),
            MeasureObjectSizeModule(input_objects="cells"),
            MeasureIntensityModule(input_image="normalized", input_objects="cells"),
        ])

        # Validate
        errors = pipeline.validate()
        assert len(errors) == 0

        # Run
        workspace = pipeline.run(image)

        # Check results
        assert "cells" in workspace.objects
        assert "area" in workspace.measurements.get("cells", {})
        assert "mean_intensity" in workspace.measurements.get("cells", {})

        # Should have detected 3 objects
        labels = workspace.get_objects("cells")
        assert np.max(labels) == 3

    def test_pipeline_with_preprocessing(self):
        """Test pipeline with preprocessing steps."""
        image = np.random.rand(50, 50) * 0.3
        # Add bright spot for object
        image[20:30, 20:30] += 0.5

        pipeline = Pipeline([
            GaussianSmoothModule(
                input_image="input",
                output_image="smoothed",
                sigma=1.0
            ),
            ThresholdModule(
                input_image="smoothed",
                output_image="binary",
                threshold=0.4
            ),
            IdentifyObjectsModule(
                input_image="binary",
                output_objects="objects",
                min_size=20
            ),
        ])

        workspace = pipeline.run(image)

        labels = workspace.get_objects("objects")
        assert np.max(labels) >= 1  # Should find at least one object

    def test_pipeline_clears_workspace_between_runs(self):
        """Pipeline should clear workspace between runs."""
        image1 = np.ones((10, 10)) * 0.8
        image2 = np.ones((10, 10)) * 0.2

        pipeline = Pipeline([
            ThresholdModule(threshold=0.5),
        ])

        # First run
        ws1 = pipeline.run(image1)
        binary1 = ws1.get_image("binary").copy()

        # Second run
        ws2 = pipeline.run(image2)
        binary2 = ws2.get_image("binary")

        # Results should be different
        assert not np.array_equal(binary1, binary2)
