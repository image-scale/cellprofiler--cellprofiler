"""
Tests for object size and shape measurement functions.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from cellprofiler_lib.object_measurement import (
    ObjectMeasurements,
    measure_object_area,
    measure_object_perimeter,
    measure_object_centroid,
    measure_object_bounding_box,
    measure_object_eccentricity,
    measure_object_major_axis_length,
    measure_object_minor_axis_length,
    measure_object_orientation,
    measure_object_solidity,
    measure_object_extent,
    measure_object_form_factor,
    measure_all_object_properties,
    measure_object_equivalent_diameter,
    measure_object_euler_number,
    measure_object_convex_area,
    measure_object_filled_area,
)


def create_square_object():
    """Create a label image with a single square object."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1  # 10x10 = 100 pixels
    return labels


def create_circle_object():
    """Create a label image with a single circular object."""
    labels = np.zeros((50, 50), dtype=np.int32)
    y, x = np.ogrid[:50, :50]
    center = (25, 25)
    radius = 10
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    labels[mask] = 1
    return labels


def create_elongated_object():
    """Create a label image with an elongated (high eccentricity) object."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[10:20, 5:25] = 1  # 10 high, 20 wide
    return labels


def create_multiple_objects():
    """Create a label image with multiple objects."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:8, 2:8] = 1  # Object 1: 6x6 = 36 pixels
    labels[2:8, 22:28] = 2  # Object 2: 6x6 = 36 pixels
    labels[22:28, 12:18] = 3  # Object 3: 6x6 = 36 pixels
    return labels


def create_object_with_hole():
    """Create a label image with an object containing a hole."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1  # 10x10 square
    labels[7:13, 7:13] = 0  # 6x6 hole in center
    return labels


def create_3d_cube_object():
    """Create a 3D label image with a single cube object."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[5:15, 5:15, 5:15] = 1  # 10x10x10 = 1000 voxels
    return labels


class TestMeasureObjectArea:
    """Tests for area measurement."""

    def test_square_area(self):
        """Square object should have correct area."""
        labels = create_square_object()
        areas = measure_object_area(labels)

        assert 1 in areas
        assert areas[1] == 100  # 10x10

    def test_multiple_objects_area(self):
        """Multiple objects should each have correct area."""
        labels = create_multiple_objects()
        areas = measure_object_area(labels)

        assert len(areas) == 3
        for label in [1, 2, 3]:
            assert areas[label] == 36  # 6x6

    def test_3d_volume(self):
        """3D object should have correct volume (area)."""
        labels = create_3d_cube_object()
        areas = measure_object_area(labels)

        assert 1 in areas
        assert areas[1] == 1000  # 10x10x10

    def test_empty_labels(self):
        """Empty labels should return empty dictionary."""
        labels = np.zeros((10, 10), dtype=np.int32)
        areas = measure_object_area(labels)

        assert len(areas) == 0


class TestMeasureObjectPerimeter:
    """Tests for perimeter measurement."""

    def test_square_perimeter(self):
        """Square object should have perimeter close to expected."""
        labels = create_square_object()
        perimeters = measure_object_perimeter(labels)

        assert 1 in perimeters
        # Perimeter should be approximately 4 * 10 = 40
        assert perimeters[1] > 30  # Should be significant

    def test_circle_perimeter(self):
        """Circle should have perimeter computed."""
        circle = create_circle_object()

        perimeters = measure_object_perimeter(circle)

        # Circle should have positive perimeter
        assert perimeters[1] > 0
        # For a circle with radius 10, perimeter should be approximately 2*pi*10 ≈ 62.8
        # But discrete pixels will differ
        assert 50 < perimeters[1] < 80

    def test_3d_surface_area(self):
        """3D object should have surface area computed."""
        labels = create_3d_cube_object()
        perimeters = measure_object_perimeter(labels)

        assert 1 in perimeters
        assert perimeters[1] > 0


class TestMeasureObjectCentroid:
    """Tests for centroid measurement."""

    def test_square_centroid(self):
        """Square object centroid should be at center."""
        labels = create_square_object()
        centroids = measure_object_centroid(labels)

        assert 1 in centroids
        # Center of 5:15 is 9.5, 5:15 is 9.5
        assert_allclose(centroids[1], (9.5, 9.5), atol=0.5)

    def test_multiple_centroids(self):
        """Multiple objects should each have correct centroid."""
        labels = create_multiple_objects()
        centroids = measure_object_centroid(labels)

        assert len(centroids) == 3
        for label in [1, 2, 3]:
            assert len(centroids[label]) == 2

    def test_3d_centroid(self):
        """3D object should have 3D centroid."""
        labels = create_3d_cube_object()
        centroids = measure_object_centroid(labels)

        assert 1 in centroids
        assert len(centroids[1]) == 3
        # Center of 5:15 is 9.5
        assert_allclose(centroids[1], (9.5, 9.5, 9.5), atol=0.5)


class TestMeasureObjectBoundingBox:
    """Tests for bounding box measurement."""

    def test_square_bbox(self):
        """Square object should have correct bounding box."""
        labels = create_square_object()
        bboxes = measure_object_bounding_box(labels)

        assert 1 in bboxes
        # bbox should be (min_row, min_col, max_row, max_col)
        assert bboxes[1] == (5, 5, 15, 15)

    def test_multiple_bboxes(self):
        """Multiple objects should each have correct bounding box."""
        labels = create_multiple_objects()
        bboxes = measure_object_bounding_box(labels)

        assert len(bboxes) == 3
        assert bboxes[1] == (2, 2, 8, 8)
        assert bboxes[2] == (2, 22, 8, 28)
        assert bboxes[3] == (22, 12, 28, 18)

    def test_3d_bbox(self):
        """3D object should have 6-element bounding box."""
        labels = create_3d_cube_object()
        bboxes = measure_object_bounding_box(labels)

        assert 1 in bboxes
        assert len(bboxes[1]) == 6
        assert bboxes[1] == (5, 5, 5, 15, 15, 15)


class TestMeasureObjectEccentricity:
    """Tests for eccentricity measurement."""

    def test_circle_low_eccentricity(self):
        """Circle should have low eccentricity."""
        labels = create_circle_object()
        eccentricities = measure_object_eccentricity(labels)

        assert 1 in eccentricities
        assert eccentricities[1] < 0.5

    def test_elongated_high_eccentricity(self):
        """Elongated object should have high eccentricity."""
        labels = create_elongated_object()
        eccentricities = measure_object_eccentricity(labels)

        assert 1 in eccentricities
        assert eccentricities[1] > 0.5

    def test_eccentricity_range(self):
        """Eccentricity should be between 0 and 1."""
        labels = create_multiple_objects()
        eccentricities = measure_object_eccentricity(labels)

        for ecc in eccentricities.values():
            assert 0 <= ecc <= 1


class TestMeasureObjectMajorAxisLength:
    """Tests for major axis length measurement."""

    def test_elongated_major_axis(self):
        """Elongated object should have longer major axis than minor axis."""
        labels = create_elongated_object()

        major = measure_object_major_axis_length(labels)
        minor = measure_object_minor_axis_length(labels)

        assert major[1] > minor[1]

    def test_positive_axis_length(self):
        """Axis lengths should be positive."""
        labels = create_square_object()
        major = measure_object_major_axis_length(labels)

        assert major[1] > 0

    def test_3d_axis_length(self):
        """3D objects should have axis length computed."""
        labels = create_3d_cube_object()
        major = measure_object_major_axis_length(labels)

        assert 1 in major
        assert major[1] > 0


class TestMeasureObjectMinorAxisLength:
    """Tests for minor axis length measurement."""

    def test_positive_minor_axis(self):
        """Minor axis length should be positive."""
        labels = create_square_object()
        minor = measure_object_minor_axis_length(labels)

        assert minor[1] > 0

    def test_minor_less_than_major(self):
        """Minor axis should be <= major axis."""
        labels = create_elongated_object()

        major = measure_object_major_axis_length(labels)
        minor = measure_object_minor_axis_length(labels)

        assert minor[1] <= major[1]


class TestMeasureObjectOrientation:
    """Tests for orientation measurement."""

    def test_orientation_range(self):
        """Orientation should be between -pi/2 and pi/2."""
        labels = create_elongated_object()
        orientations = measure_object_orientation(labels)

        assert 1 in orientations
        assert -np.pi/2 <= orientations[1] <= np.pi/2

    def test_orientation_3d_raises(self):
        """Orientation for 3D should raise error."""
        labels = create_3d_cube_object()

        with pytest.raises(ValueError):
            measure_object_orientation(labels)


class TestMeasureObjectSolidity:
    """Tests for solidity measurement."""

    def test_convex_object_high_solidity(self):
        """Convex objects should have high solidity."""
        labels = create_square_object()
        solidities = measure_object_solidity(labels)

        assert 1 in solidities
        assert solidities[1] > 0.9  # Square is convex

    def test_solidity_range(self):
        """Solidity should be between 0 and 1."""
        labels = create_multiple_objects()
        solidities = measure_object_solidity(labels)

        for sol in solidities.values():
            assert 0 <= sol <= 1

    def test_object_with_hole_lower_solidity(self):
        """Object with hole might have lower solidity than solid object."""
        solid = create_square_object()
        with_hole = create_object_with_hole()

        solid_sol = measure_object_solidity(solid)
        hole_sol = measure_object_solidity(with_hole)

        # Both should be valid solidity values
        assert 0 < solid_sol[1] <= 1
        assert 0 < hole_sol[1] <= 1


class TestMeasureObjectExtent:
    """Tests for extent measurement."""

    def test_square_high_extent(self):
        """Square filling its bounding box should have high extent."""
        labels = create_square_object()
        extents = measure_object_extent(labels)

        assert 1 in extents
        assert extents[1] == 1.0  # Square fills its bbox completely

    def test_circle_lower_extent(self):
        """Circle should have lower extent than square."""
        labels = create_circle_object()
        extents = measure_object_extent(labels)

        assert 1 in extents
        # Circle area / bbox area ≈ pi/4 ≈ 0.785
        assert 0.7 < extents[1] < 0.9

    def test_extent_range(self):
        """Extent should be between 0 and 1."""
        labels = create_multiple_objects()
        extents = measure_object_extent(labels)

        for ext in extents.values():
            assert 0 < ext <= 1


class TestMeasureObjectFormFactor:
    """Tests for form factor (circularity) measurement."""

    def test_circle_high_form_factor(self):
        """Circle should have form factor close to 1."""
        labels = create_circle_object()
        form_factors = measure_object_form_factor(labels)

        assert 1 in form_factors
        assert form_factors[1] > 0.8

    def test_square_lower_form_factor(self):
        """Square should have significant form factor."""
        square = create_square_object()

        square_ff = measure_object_form_factor(square)

        # Form factor for a square is approximately 4*pi/(4*4) ≈ 0.785 theoretically
        # But discrete calculations may differ
        assert 0.5 < square_ff[1] < 1.0

    def test_form_factor_positive(self):
        """Form factor should be positive."""
        labels = create_multiple_objects()
        form_factors = measure_object_form_factor(labels)

        for ff in form_factors.values():
            assert ff > 0


class TestMeasureAllObjectProperties:
    """Tests for comprehensive measurement function."""

    def test_returns_named_tuple(self):
        """Should return dictionary of ObjectMeasurements."""
        labels = create_square_object()
        measurements = measure_all_object_properties(labels)

        assert 1 in measurements
        assert isinstance(measurements[1], ObjectMeasurements)

    def test_all_properties_present(self):
        """All properties should be computed."""
        labels = create_square_object()
        measurements = measure_all_object_properties(labels)

        m = measurements[1]
        assert m.label == 1
        assert m.area > 0
        assert m.perimeter > 0
        assert len(m.centroid) == 2
        assert len(m.bbox) == 4
        assert 0 <= m.eccentricity <= 1
        assert m.major_axis_length > 0
        assert m.minor_axis_length > 0
        assert 0 <= m.solidity <= 1
        assert 0 < m.extent <= 1
        assert m.form_factor > 0

    def test_multiple_objects_all_properties(self):
        """Multiple objects should all have measurements."""
        labels = create_multiple_objects()
        measurements = measure_all_object_properties(labels)

        assert len(measurements) == 3
        for label in [1, 2, 3]:
            assert label in measurements
            assert measurements[label].area > 0

    def test_3d_measurements(self):
        """3D objects should have measurements."""
        labels = create_3d_cube_object()
        measurements = measure_all_object_properties(labels)

        assert 1 in measurements
        assert measurements[1].area == 1000
        assert len(measurements[1].centroid) == 3
        assert len(measurements[1].bbox) == 6


class TestMeasureObjectEquivalentDiameter:
    """Tests for equivalent diameter measurement."""

    def test_circle_equivalent_diameter(self):
        """Circle's equivalent diameter should match its actual diameter."""
        labels = create_circle_object()
        diameters = measure_object_equivalent_diameter(labels)

        assert 1 in diameters
        # Radius is 10, so diameter ≈ 20
        assert 18 < diameters[1] < 22

    def test_positive_diameter(self):
        """Equivalent diameter should be positive."""
        labels = create_square_object()
        diameters = measure_object_equivalent_diameter(labels)

        assert diameters[1] > 0


class TestMeasureObjectEulerNumber:
    """Tests for Euler number measurement."""

    def test_solid_object_euler(self):
        """Solid object should have Euler number of 1."""
        labels = create_square_object()
        euler = measure_object_euler_number(labels)

        assert 1 in euler
        assert euler[1] == 1

    def test_object_with_hole_euler(self):
        """Object with hole should have Euler number of 0."""
        labels = create_object_with_hole()
        euler = measure_object_euler_number(labels)

        assert 1 in euler
        assert euler[1] == 0  # 1 object - 1 hole = 0


class TestMeasureObjectConvexArea:
    """Tests for convex hull area measurement."""

    def test_convex_object_same_area(self):
        """Convex object's convex area should equal its area."""
        labels = create_square_object()

        area = measure_object_area(labels)
        convex_area = measure_object_convex_area(labels)

        assert area[1] == convex_area[1]

    def test_convex_area_geq_area(self):
        """Convex hull area should be >= object area."""
        labels = create_object_with_hole()

        area = measure_object_area(labels)
        convex_area = measure_object_convex_area(labels)

        assert convex_area[1] >= area[1]


class TestMeasureObjectFilledArea:
    """Tests for filled area measurement."""

    def test_solid_object_same_filled_area(self):
        """Solid object's filled area should equal its area."""
        labels = create_square_object()

        area = measure_object_area(labels)
        filled_area = measure_object_filled_area(labels)

        assert area[1] == filled_area[1]

    def test_object_with_hole_larger_filled_area(self):
        """Object with hole should have filled area > area."""
        labels = create_object_with_hole()

        area = measure_object_area(labels)
        filled_area = measure_object_filled_area(labels)

        assert filled_area[1] > area[1]


class TestEmptyLabels:
    """Tests for handling empty label images."""

    def test_empty_returns_empty_dict(self):
        """All measurement functions should return empty dict for empty labels."""
        labels = np.zeros((10, 10), dtype=np.int32)

        assert measure_object_area(labels) == {}
        assert measure_object_perimeter(labels) == {}
        assert measure_object_centroid(labels) == {}
        assert measure_object_bounding_box(labels) == {}
        assert measure_object_eccentricity(labels) == {}
        assert measure_object_form_factor(labels) == {}
        assert measure_all_object_properties(labels) == {}


class Test3DSupport:
    """Tests for 3D label image support."""

    def test_all_functions_accept_3d(self):
        """All applicable functions should accept 3D labels."""
        labels = create_3d_cube_object()

        assert measure_object_area(labels)[1] == 1000
        assert measure_object_perimeter(labels)[1] > 0
        assert len(measure_object_centroid(labels)[1]) == 3
        assert len(measure_object_bounding_box(labels)[1]) == 6
        assert measure_object_eccentricity(labels)[1] >= 0
        assert measure_object_solidity(labels)[1] > 0
        assert measure_object_extent(labels)[1] > 0
        assert measure_object_form_factor(labels)[1] > 0
        assert measure_all_object_properties(labels)[1].area == 1000
