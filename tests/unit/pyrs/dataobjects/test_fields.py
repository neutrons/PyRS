# Standard and third party libraries
from collections import namedtuple
import numpy as np
import os
import pytest
import random
# PyRs libraries
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import DEFAULT_POINT_RESOLUTION
from pyrs.dataobjects.fields import (aggregate_scalar_field_samples, fuse_scalar_field_samples,
                                     ScalarFieldSample, StrainField, StressField, stack_scalar_field_samples,
                                     generateParameterField)
from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.peaks import PeakCollection  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

SampleMock = namedtuple('SampleMock', 'name values errors x y z')


@pytest.fixture(scope='module')
def field_cube_regular():
    r"""
    A volume, surface, and linear scan in a cube of side 4, with a linear intensity function.
    The sample points represent a regular 3D lattice
    """
    def intensity(x, y, z):
        return x + y + z

    def assert_checks(field):
        all_finite = np.all(np.isfinite(field.values))
        assert all_finite, 'some values are not finite'
        assert np.allclose(field.values, [intensity(*r) for r in list(field.coordinates)])

    values_returned = {'assert checks': assert_checks, 'edge length': 4}
    coordinates = np.transpose(np.mgrid[0:3:4j, 0:3:4j, 0:3:4j], (1, 2, 3, 0)).reshape(-1, 3)  # shape = (64, 3)
    values = np.array([intensity(*xyz) for xyz in coordinates])
    errors = 0.1 * values
    vx, vy, vz = list(coordinates.T)
    values_returned['volume scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    # Surface scan perpendicular to vz
    coordinates_surface = np.copy(coordinates)
    coordinates_surface[:, 2] = 0.0  # each coordinate is now repeated four times
    values = np.array([intensity(*xyz) for xyz in coordinates_surface])
    errors = 0.1 * values
    vx, vy, vz = list(coordinates_surface.T)
    values_returned['surface scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    # Linear scan along vx
    coordinates_linear = np.copy(coordinates_surface)
    coordinates_linear[:, 1] = 0.0  # each coordinate is now repeated 16 times
    values = np.array([intensity(*xyz) for xyz in coordinates_linear])
    errors = 0.1 * values
    vx, vy, vz = list(coordinates_linear.T)
    values_returned['linear scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    return values_returned


@pytest.fixture(scope='module')
def field_cube_with_vacancies():
    r"""
    A volume scan in a cube of side 6, with a linear intensity function.
    The cube contains 6^3 = 216 points, of which 4^3 = 64 are interior points. We randomly delete
    half the interior points. Thus, the extents of the cube are still 0 and 6.
    """
    def intensity(x, y, z):
        return x + y + z

    def assert_checks(field):
        all_finite = np.all(np.isfinite(field.values))
        assert all_finite, 'some values are not finite'
        assert np.allclose(field.values, [intensity(*r) for r in list(field.coordinates)])

    values_returned = {'assert checks': assert_checks, 'edge length': 6}
    indexes_interior = list(range(64))  # a counter for the sample points not in the surface of the cube
    random.shuffle(indexes_interior)  # we will randomly select half of these indexes as vacancies
    indexes_vacancy = indexes_interior[::2]  # flag every other interior point index as a vacancy
    index_interior = 0  # start the counter for the interior points
    coordinates = list()
    for vx in range(0, 6):
        vx_is_interior = 0 < vx < 5  # vx is not on the surface of the cube
        for vy in range(0, 6):
            vy_is_interior = 0 < vy < 5  # vx and vy are not on the surface of the cube
            for vz in range(0, 6):
                xyz = [vx, vy, vz]
                if vx_is_interior and vy_is_interior and 0 < vz < 5:
                    if index_interior not in indexes_vacancy:
                        coordinates.append(xyz)
                    index_interior += 1
                else:
                    coordinates.append(xyz)  # no vacancies on the surface of the cube
    coordinates = np.array(coordinates)  # shape = (216 - 32, 3)
    assert len(coordinates) == 6 ** 3 - 32
    vx, vy, vz = list(coordinates.transpose())
    values = np.array([intensity(*r) for r in coordinates])
    errors = 0.1 * values
    values_returned['volume scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    return values_returned


@pytest.fixture(scope='module')
def field_surface_with_vacancies():
    r"""
    A surface scan in a square of side 10, with a linear intensity function.
    The cube contains 10^2 = 100 points, of which 8^2 = 64 are interior points. We randomly delete
    half the interior points. Thus, the extents of the square are still 0 and 10.
    """
    def intensity(x, y, z):
        return x + y + z

    def assert_checks(field):
        all_finite = np.all(np.isfinite(field.values))
        assert all_finite, 'some values are not finite'
        assert np.allclose(field.values, [intensity(*r) for r in list(field.coordinates)])

    values_returned = {'assert checks': assert_checks, 'edge length': 10}
    indexes_interior = list(range(64))  # a counter for the sample points not in the surface of the cube
    random.shuffle(indexes_interior)  # we will randomly select half of these indexes as vacancies
    indexes_vacancy = indexes_interior[::2]  # flag every other interior point index as a vacancy
    index_interior = 0  # start the counter for the interior points
    coordinates = list()
    for vx in range(0, 10):
        vx_is_interior = 0 < vx < 9  # vx is not on the perimeter of the square
        for vy in range(0, 10):
            vz = 0  # declare a surface scan perpendicular to the vz-axis
            xyz = [vx, vy, vz]
            if vx_is_interior and 0 < vy < 9:
                if index_interior not in indexes_vacancy:
                    coordinates.append(xyz)
                index_interior += 1
            else:
                coordinates.append(xyz)  # no vacancies on the perimeter of the square
    coordinates = np.array(coordinates)  # shape = (100 - 32, 3)
    assert len(coordinates) == 10 ** 2 - 32
    vx, vy, vz = list(coordinates.transpose())
    values = np.array([intensity(*r) for r in coordinates])
    errors = 0.1 * values
    values_returned['surface scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    return values_returned


@pytest.fixture(scope='module')
def field_linear_with_vacancies():
    r"""
    A linear scan in a line of side 100, with a linear intensity function.
    The line contains 100 points, of which 98 interior points. We randomly delete
    10 interior points. Thus, the extents of the line are still 0 and 100.
    """
    def intensity(x, y, z):
        return x + y + z

    def assert_checks(field):
        all_finite = np.all(np.isfinite(field.values))
        assert all_finite, 'some values are not finite'
        assert np.allclose(field.values, [intensity(*r) for r in list(field.coordinates)])

    values_returned = {'assert checks': assert_checks, 'edge length': 10}
    indexes_interior = list(range(98))  # a counter for the sample points not in the edges of the line
    random.shuffle(indexes_interior)  # we will randomly select 10 of these indexes as vacancies
    indexes_vacancy = indexes_interior[0: 10]  # flag the first 10 indexes as vacancies
    index_interior = 0  # start the counter for the interior points
    coordinates = list()
    for vx in range(0, 100):
        vy, vz = 0, 0  # declare a linear scan along the vx-axis
        xyz = [vx, vy, vz]
        if 0 < vx < 99:  # vx is not on the edge of the linear scan
            if index_interior not in indexes_vacancy:
                coordinates.append(xyz)
            index_interior += 1
        else:
            coordinates.append(xyz)  # no vacancies on the perimeter of the square
    coordinates = np.array(coordinates)  # shape = (90, 3)
    assert len(coordinates) == 100 - 10
    vx, vy, vz = list(coordinates.transpose())
    values = np.array([intensity(*r) for r in coordinates])
    errors = 0.1 * values
    values_returned['linear scan'] = ScalarFieldSample('stress', values, errors, vx, vy, vz)
    return values_returned


class TestScalarFieldSample:

    sample1 = SampleMock('lattice',
                         [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090],  # values
                         [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                         [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000],  # x
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                         )

    # The last three points of sample1 overlaps with the first three points of sample1
    sample2 = SampleMock('lattice',
                         [1.071, 1.081, 1.091, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],  # values
                         [0.008, 0.008, 0.008, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # errors
                         [7.009, 8.001, 9.005, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00],  # x
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                         )

    sample3 = SampleMock('strain',
                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # values
                         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # errors
                         [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],  # x
                         [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5],  # y
                         [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5],  # z
                         )

    sample4 = SampleMock('strain',
                         [float('nan'), 0.1, 0.2, float('nan'), float('nan'), 0.5, 0.6, float('nan')],  # values
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # errors
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # x
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # y
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # z
                         )

    # Points 2 and 3 overlap
    sample5 = SampleMock('lattice',
                         [float('nan'), 1.0, 1.0, 2.0, 3.0, float('nan'), 0.1, 1.1, 2.1, 3.1, 4.1],
                         [0.0, 0.10, 0.11, 0.2, 0.3, 0.4, 0.1, 0.1, 0.2, 0.3, 0.4],
                         [0.0, 1.000, 1.001, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],  # x
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # z
                         )

    def test_init(self):
        assert ScalarFieldSample(*TestScalarFieldSample.sample1)
        assert ScalarFieldSample(*TestScalarFieldSample.sample2)
        assert ScalarFieldSample(*TestScalarFieldSample.sample3)
        sample_bad = list(TestScalarFieldSample.sample1)
        sample_bad[1] = [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080]  # one less value
        with pytest.raises(AssertionError):
            ScalarFieldSample(*sample_bad)

    def test_len(self):
        assert len(ScalarFieldSample(*TestScalarFieldSample.sample1)) == 10

    def test_values(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.values, TestScalarFieldSample.sample1.values)

    def test_errors(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.errors, TestScalarFieldSample.sample1.errors)

    def test_point_list(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.point_list.vx, TestScalarFieldSample.sample1.x)

    def test_coordinates(self):
        sample = list(TestScalarFieldSample.sample1)
        field = ScalarFieldSample(*sample)
        np.testing.assert_equal(field.coordinates, np.array(sample[3:]).transpose())

    def test_x(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.x, TestScalarFieldSample.sample1.x)

    def test_y(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.y, TestScalarFieldSample.sample1.y)

    def test_z(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.z, TestScalarFieldSample.sample1.z)

    def test_isfinite(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample4).isfinite
        for attribute in ('values', 'errors', 'x', 'y', 'z'):
            assert getattr(field, attribute) == pytest.approx([0.1, 0.2, 0.5, 0.6])

    def test_interpolated_sample_regular(self, field_cube_regular):
        r"""
        Test with an input regular grid. No interpolation should be necessary because
        the regular grid built using the extents of the field coincide with the input
        sample points
        """
        # A volumetric sample in a cube of side 4, with an linear intensity function
        field = field_cube_regular['volume scan']
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == field_cube_regular['edge length'] ** 3  # sample list spans a cube
        assert interpolated.point_list.is_equal_within_resolution(field.point_list,
                                                                  resolution=DEFAULT_POINT_RESOLUTION)
        field_cube_regular['assert checks'](interpolated)
        # A surface scan
        field = field_cube_regular['surface scan']  # 64 points, each point repeated once
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == field_cube_regular['edge length'] ** 2  # sample list spans a square
        # `field` has 64 points, each point repeated four times. `interpolated` has 16 points, each is unique
        assert interpolated.point_list.is_equal_within_resolution(field.point_list,
                                                                  resolution=DEFAULT_POINT_RESOLUTION)
        field_cube_regular['assert checks'](interpolated)
        # A linear scan
        field = field_cube_regular['linear scan']  # 64 points, each point is repeated 16 times
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == field_cube_regular['edge length']  # sample list spans a line of sampled points
        # `field` has 64 points, each point repeated 16 times. `interpolated` has 4 points, each is unique
        assert interpolated.point_list.is_equal_within_resolution(field.point_list,
                                                                  resolution=DEFAULT_POINT_RESOLUTION)
        field_cube_regular['assert checks'](interpolated)

    def test_interpolated_sample_cube_vacancies(self, field_cube_with_vacancies):
        r"""
        Test with an input regular grid spanning a cube, where some interior points are missing.
        `field` has 32 vacancies, thus a total of 6^3 - 32 points. Interpolated, on the other hand, has no
        vacancies. The reason lies in how the extents are calculated. See the example below for a square of
        side four on the [vx, vy] surface (vz=0 here) with two internal vacancies:
         o o o o
         o x o o
         o o x o
         o o o o
        the list of vx, vy, and vz have missing coordinates [1, 1, 0] and [2, 2, 0]:
        vx = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]  # 14 points, two (interior) points missing
        vy = [0, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 3]  # 14 points, two (interior) points missing
        vz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        When calculating the extents, we look separately to each list, and find out the number of unique coordinates
        unique vx = [0, 1, 2, 3]  # there are three instances of vx==1 (same for vx==2), so they are captured
        unique vy = [0, 1, 2, 3]  # same scenario than that of vx
        unique vz = [0]
        Extents for vx are:
             minimum = 0
             maximum = 3
             step =(maximum - minimum) / (unique_count -1) = (3 - 0) / (4 - 1) = 1
        When calculating the grid spanned by these extents, we have a grid from zero to 3 with four points,
        and same for vy and vz (let's generalize here of a cube, not a square). Thus, the interpolated grid
        has no vacancies.
        This situation will happen only when the number of vacancies is sufficiently small that all values
        of vx, vy, and vz are captured.
        """
        field = field_cube_with_vacancies['volume scan']
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == 6 ** 3
        field_cube_with_vacancies['assert checks'](interpolated)

    def test_interpolated_sample_surface_vacancies(self, field_surface_with_vacancies):
        r"""
        Test with an input regular grid spanning a square, where some interior points are missing.
                Test with an input regular grid spanning a cube, where some interior points are missing.
        `field` has 32 vacancies, thus a total of 6^3 - 32 points. Interpolated, on the other hand, has no
        vacancies. The reason lies in how the extents are calculated. See the example below for a square of
        side four on the [vx, vy] surface (vz=0 here) with two internal vacancies:
         o o o o
         o x o o
         o o x o
         o o o o
        the list of vx, vy, and vz have missing coordinates [1, 1, 0] and [2, 2, 0]:
        vx = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]  # 14 points, two (interior) points missing
        vy = [0, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 3]  # 14 points, two (interior) points missing
        vz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        When calculating the extents, we look separately to each list, and find out the number of unique coordinates
        unique vx = [0, 1, 2, 3]  # there are three instances of vx==1 (same for vx==2), so they are captured
        unique vy = [0, 1, 2, 3]  # same scenario than that of vx
        unique vz = [0]
        Extents for vx are:
             minimum = 0
             maximum = 3
             step =(maximum - minimum) / (unique_count -1) = (3 - 0) / (4 - 1) = 1
        When calculating the grid spanned by these extents, we have a grid from zero to 3 with four points,
        and same for vy. Thus, the interpolated grid has no vacancies.
        This situation will happen only when the number of vacancies is sufficiently small that all values
        of vx, and vy are captured.
        """
        field = field_surface_with_vacancies['surface scan']
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == 10 ** 2
        field_surface_with_vacancies['assert checks'](interpolated)

    def test_interpolated_sample_linear_vacancies(self, field_linear_with_vacancies):
        r"""
        Test with an input regular grid spanning a square, where some interior points are missing.
        """
        field = field_linear_with_vacancies['linear scan']
        interpolated = field.interpolated_sample(method='linear', keep_nan=True, resolution=DEFAULT_POINT_RESOLUTION,
                                                 criterion='min_error')
        assert len(interpolated) == 90
        field_linear_with_vacancies['assert checks'](interpolated)

    def test_extract(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        target_indexes = range(0, 10, 2)
        selection = field.extract(target_indexes)
        assert selection.name == 'lattice'
        np.testing.assert_equal(selection.values, [1.000, 1.020, 1.040, 1.060, 1.080])
        np.testing.assert_equal(selection.errors, [0.000, 0.002, 0.004, 0.006, 0.008])
        np.testing.assert_equal(selection.x, [0.000, 2.000, 4.000, 6.000, 8.000])

    def test_aggregate(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample2 = ScalarFieldSample(*TestScalarFieldSample.sample2)
        sample = sample1.aggregate(sample2)
        # index 9 of aggregate sample corresponds to the last point of sample1
        # index 10 of aggregate sample corresponds to the first point of sample2
        np.testing.assert_equal(sample.values[9: 11], [1.090, 1.071])
        np.testing.assert_equal(sample.errors[9: 11], [0.009, 0.008])
        np.testing.assert_equal(sample.x[9: 11], [9.000, 7.009])

    def test_intersection(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.intersection(ScalarFieldSample(*TestScalarFieldSample.sample2))
        assert len(sample) == 6  # three points from sample1 and three points from sample2
        assert sample.name == 'lattice'
        np.testing.assert_equal(sample.values, [1.070, 1.080, 1.090, 1.071, 1.081, 1.091])
        np.testing.assert_equal(sample.errors, [0.007, 0.008, 0.009, 0.008, 0.008, 0.008])
        np.testing.assert_equal(sample.x, [7.000, 8.000, 9.000, 7.009, 8.001, 9.005])

    def test_coalesce(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.aggregate(ScalarFieldSample(*TestScalarFieldSample.sample2))
        sample = sample.coalesce(criterion='min_error')
        assert len(sample) == 17  # discard the last point from sample1 and the first two points from sample2
        assert sample.name == 'lattice'
        # index 6 of aggregate sample corresponds to index 6 of sample1
        # index 11 of aggregate sample corresponds to index 3 of sample2
        np.testing.assert_equal(sample.values[6: 11], [1.060, 1.070, 1.080, 1.091, 1.10])
        np.testing.assert_equal(sample.errors[6: 11], [0.006, 0.007, 0.008, 0.008, 0.0])
        np.testing.assert_equal(sample.x[6: 11], [6.000, 7.000, 8.000, 9.005, 10.00])

    def test_fuse(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.fuse(ScalarFieldSample(*TestScalarFieldSample.sample2), criterion='min_error')
        assert len(sample) == 17  # discard the last point from sample1 and the first two points from sample2
        assert sample.name == 'lattice'
        # index 6 of aggregate sample corresponds to index 6 of sample1
        # index 11 of aggregate sample corresponds to index 3 of sample2
        np.testing.assert_equal(sample.values[6: 11], [1.060, 1.070, 1.080, 1.091, 1.10])
        np.testing.assert_equal(sample.errors[6: 11], [0.006, 0.007, 0.008, 0.008, 0.0])
        np.testing.assert_equal(sample.x[6: 11], [6.000, 7.000, 8.000, 9.005, 10.00])

    def test_export(self):
        # Create a scalar field
        xyz = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
        xyz = np.vstack(np.meshgrid(*xyz)).reshape(3, -1)  # shape = (3, 1000)
        signal, errors = np.arange(0, 1000, 1, dtype=float), np.zeros(1000, dtype=float)
        sample = ScalarFieldSample('strain', signal, errors, *xyz)

        # Test export to MDHistoWorkspace
        workspace = sample.export(form='MDHistoWorkspace', name='strain1', units='mm')
        assert workspace.name() == 'strain1'

        # Test export to CSV file
        with pytest.raises(NotImplementedError):
            sample.export(form='CSV', file='/tmp/csv.txt')

    def test_to_md(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample3)
        assert field
        histo = field.to_md_histo_workspace('sample3', interpolate=False)
        assert histo
        assert histo.id() == 'MDHistoWorkspace'

        for i, (min_value, max_value) in enumerate(((0.0, 0.5), (1.0, 1.5), (2.0, 2.5))):
            dimension = histo.getDimension(i)
            assert dimension.getUnits() == 'meter'
            # adding half a bin each direction since values from mdhisto are boundaries and constructor uses centers
            assert dimension.getMinimum() == min_value - .25
            assert dimension.getMaximum() == max_value + .25
            assert dimension.getNBins() == 2

        np.testing.assert_equal(histo.getSignalArray().ravel(), self.sample3.values, err_msg='Signal')
        np.testing.assert_equal(histo.getErrorSquaredArray().ravel(), np.square(self.sample3.errors), err_msg='Errors')

        # clean up
        histo.delete()


@pytest.fixture(scope='module')
def strain_field_samples(test_data_dir):
    r"""
    A number of StrainField objects from mock and real data
    """
    sample_fields = {}
    #####
    # The first sample has 2 points in each direction
    #####
    subruns = np.arange(1, 9, dtype=int)

    # create the test peak collection - d-refernce is 1 to make checks easier
    # uncertainties are all zero
    peaks_array = np.zeros(subruns.size, dtype=get_parameter_dtype('gaussian', 'Linear'))
    peaks_array['PeakCentre'][:] = 180.  # position of two-theta in degrees
    peaks_error = np.zeros(subruns.size, dtype=get_parameter_dtype('gaussian', 'Linear'))
    peak_collection = PeakCollection('dummy', 'gaussian', 'linear', wavelength=2.,
                                     d_reference=1., d_reference_error=0.)
    peak_collection.set_peak_fitting_values(subruns, peaks_array, parameter_errors=peaks_error,
                                            fit_costs=np.zeros(subruns.size, dtype=float))

    # create the test workspace - only sample logs are needed
    workspace = HidraWorkspace()
    workspace.set_sub_runs(subruns)
    # arbitray points in space
    workspace.set_sample_log('vx', subruns, np.arange(1, 9, dtype=int))
    workspace.set_sample_log('vy', subruns, np.arange(11, 19, dtype=int))
    workspace.set_sample_log('vz', subruns, np.arange(21, 29, dtype=int))

    # call the function
    strain = StrainField(hidraworkspace=workspace, peak_collection=peak_collection)

    # test the result
    assert strain
    assert len(strain) == subruns.size
    assert strain.peak_collection == peak_collection
    np.testing.assert_almost_equal(strain.values, 0.)
    np.testing.assert_equal(strain.errors, np.zeros(subruns.size, dtype=float))
    sample_fields['strain with two points per direction'] = strain

    #####
    # Create StrainField samples from two files and different peak tags
    #####
    # TODO: substitute/fix HB2B_1628.h5 with other data, because reported vx, vy, and vz are all 'nan'
    # filename_tags_pairs = [('HB2B_1320.h5', ('', 'peak0')), ('HB2B_1628.h5', ('peak0', 'peak1', 'peak2'))]
    filename_tags_pairs = [('HB2B_1320.h5', ('', 'peak0'))]
    for filename, tags in filename_tags_pairs:
        file_path = os.path.join(test_data_dir, filename)
        prefix = filename.split('.')[0] + '_'
        for tag in tags:
            sample_fields[prefix + tag] = StrainField(filename=file_path, peak_tag=tag)

    return sample_fields


class TestStrainField:

    def test_peak_collection(self, strain_field_samples):
        strain = strain_field_samples['strain with two points per direction']
        assert isinstance(strain.peak_collection, PeakCollection)
        # TODO: test the RuntimeError when the strain is a composite

    def test_peak_collections(self, strain_field_samples):
        strain = strain_field_samples['strain with two points per direction']
        assert len(strain.peak_collections) == 1
        assert isinstance(strain.peak_collections[0], PeakCollection)

    def test_coordinates(self, strain_field_samples):
        strain = strain_field_samples['strain with two points per direction']
        coordinates = np.array([[1., 11., 21.], [2., 12., 22.], [3., 13., 23.], [4., 14., 24.],
                                [5., 15., 25.], [6., 16., 26.], [7., 17., 27.], [8., 18., 28.]])
        assert np.allclose(strain.coordinates, coordinates)

    def test_fuse_with(self, strain_field_samples):
        strain1 = strain_field_samples['HB2B_1320_peak0']
        strain2 = strain_field_samples['strain with two points per direction']
        strain = strain1.fuse_with(strain2)
        with pytest.raises(RuntimeError) as exception_info:
            strain.peak_collection
        assert 'more than one peak collection' in str(exception_info.value)
        assert strain.peak_collections == [strain1.peak_collection, strain2.peak_collection]
        assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))

        with pytest.raises(RuntimeError) as exception_info:
            strain1.fuse_with(strain1)  # fusing a scan with itself should raise a runtime error
        assert 'both contain scan' in str(exception_info.value)

    def test_add(self, strain_field_samples):
        strain1 = strain_field_samples['HB2B_1320_peak0']
        strain2 = strain_field_samples['strain with two points per direction']
        strain = strain1 + strain2
        with pytest.raises(RuntimeError) as exception_info:
            strain.peak_collection
        assert 'more than one peak collection' in str(exception_info.value)
        assert strain.peak_collections == [strain1.peak_collection, strain2.peak_collection]
        assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))

    def test_create_strain_field_from_file_no_peaks(self, test_data_dir):
        # this project file doesn't have peaks in it
        file_path = os.path.join(test_data_dir, 'HB2B_1060_first3_subruns.h5')
        try:
            _ = StrainField(file_path)  # noqa F841
            assert False, 'Should not be able to read ' + file_path
        except IOError:
            pass  # this is what should happen

    def test_fuse_strains(self, strain_field_samples, allclose_with_sorting):
        # TODO HB2B_1320_peak0 and HB2B_1320_ are the same scan. We need two different scans
        strain1 = strain_field_samples['HB2B_1320_peak0']
        strain2 = strain_field_samples['HB2B_1320_']
        strain3 = strain_field_samples['strain with two points per direction']
        # Use fuse_strains().
        strain_fused = StrainField.fuse_strains(strain1, strain2, strain3, resolution=DEFAULT_POINT_RESOLUTION,
                                                criterion='min_error')
        # the sum should give the same, since we passed default resolution and criterion options
        strain_sum = strain1 + strain2 + strain3
        for strain in (strain_fused, strain_sum):
            assert len(strain) == 312 + 8  # strain1 and strain2 give strain1 because they contain the same data
            assert strain.peak_collections == [s.peak_collection for s in (strain1, strain2, strain3)]
            values = np.concatenate((strain1.values, strain3.values))  # again, no strain2 because it's the same as strain1
            assert allclose_with_sorting(strain.values, values)

    def test_stack_strains(self, strain_field_samples, allclose_with_sorting):
        strain1 = strain_field_samples['HB2B_1320_peak0']
        strain2 = strain_field_samples['HB2B_1320_']
        # Stack two strains having the same evaluation points.
        strain1_stacked, strain2_stacked = strain1 * strain2  # default resolution and stacking mode
        for strain in (strain1_stacked, strain2_stacked):
            assert len(strain) == len(strain1)
            assert bool(np.all(np.isfinite(strain.values))) is True  # all points are common to strain1 and strain2
        # Stack two strains having completely different evaluation points.
        strain3 = strain_field_samples['strain with two points per direction']
        strain2_stacked, strain3_stacked = strain2 * strain3  # default resolution and stacking mode
        # The common list of points is the sum of the points from each strain
        for strain in (strain2_stacked, strain3_stacked):
            assert len(strain) == len(strain2) + len(strain3)
        # There's no common point that is common to both strain2 and strain3
        # Each stacked strain only have finite measurements on points coming from the un-stacked strain
        for strain_stacked, strain in ((strain2_stacked, strain2), (strain3_stacked, strain3)):
            finite_measurements_count = len(np.where(np.isfinite(strain_stacked.values))[0])
            assert finite_measurements_count == len(strain)
        # The points evaluated as 'nan' must come from the other scan
        for strain_stacked, strain_other in ((strain2_stacked, strain3), (strain3_stacked, strain2)):
            nan_measurements_count = len(np.where(np.isnan(strain_stacked.values))[0])
            assert nan_measurements_count == len(strain_other)

    def test_fuse_and_stack_strains(self, strain_field_samples, allclose_with_sorting):
        # TODO HB2B_1320_peak0 and HB2B_1320_ are the same scan. We need two different scans
        strain1 = strain_field_samples['HB2B_1320_peak0']
        strain2 = strain_field_samples['HB2B_1320_']
        strain3 = strain_field_samples['strain with two points per direction']
        strain1_stacked, strain23_stacked = strain1 * (strain2 + strain3)  # default resolution and stacking mode
        # Check number of points with finite strains measuments
        for strain_stacked in (strain1_stacked, strain23_stacked):
            assert len(strain_stacked) == len(strain2) + len(strain3)
        for strain_stacked, finite_count, nan_count in zip((strain1_stacked, strain23_stacked), (312, 320), (8, 0)):
            finite_measurements_count = len(np.where(np.isfinite(strain_stacked.values))[0])
            assert finite_measurements_count == finite_count
            nan_measurements_count = len(np.where(np.isnan(strain_stacked.values))[0])
            assert nan_measurements_count == nan_count
        # Check peak collections carry-over
        assert strain1_stacked.peak_collection == strain1.peak_collection
        assert strain23_stacked.peak_collections == [strain2.peak_collection, strain3.peak_collection]


def test_generateParameterField(test_data_dir):
    file_path = os.path.join(test_data_dir, 'HB2B_1320.h5')

    source_project = HidraProjectFile(file_path, mode=HidraProjectFileMode.READONLY)
    workspace = HidraWorkspace(file_path)
    workspace.load_hidra_project(source_project, False, False)
    x = workspace.get_sample_log_values('vx')
    y = workspace.get_sample_log_values('vy')
    z = workspace.get_sample_log_values('vz')

    peak = source_project.read_peak_parameters('peak0')
    expected_values, expected_errors = peak.get_effective_params()

    # thorough testing of Center
    center = generateParameterField('Center', workspace, peak)
    assert isinstance(center, ScalarFieldSample)
    assert center.name == 'Center'
    np.testing.assert_equal(center.x, x)
    np.testing.assert_equal(center.y, y)
    np.testing.assert_equal(center.z, z)
    np.testing.assert_equal(center.values, expected_values['Center'])
    np.testing.assert_equal(center.errors, expected_errors['Center'])

    center_md = center.to_md_histo_workspace()
    dim_x = center_md.getXDimension()
    assert dim_x.getNBins() == 18
    np.testing.assert_almost_equal(dim_x.getMinimum(), -31.765, decimal=5)
    np.testing.assert_almost_equal(dim_x.getMaximum(), 31.765, decimal=5)
    dim_y = center_md.getYDimension()
    assert dim_y.getNBins() == 6
    np.testing.assert_almost_equal(dim_y.getMinimum(), -7.2, decimal=5)
    np.testing.assert_almost_equal(dim_y.getMaximum(), 7.2, decimal=5)
    dim_z = center_md.getZDimension()
    assert dim_z.getNBins() == 3
    np.testing.assert_almost_equal(dim_z.getMinimum(), -15)
    np.testing.assert_almost_equal(dim_z.getMaximum(), 15)
    signal = center_md.getSignalArray()
    np.testing.assert_almost_equal(signal.min(), 89.94377, decimal=5)
    np.testing.assert_almost_equal(signal.max(), 90.15296, decimal=5)
    errors = center_md.getErrorSquaredArray()
    np.testing.assert_almost_equal(errors.min(), 0.02019792)
    np.testing.assert_almost_equal(errors.max(), 0.57951983)

    # quick test the other peak paramters
    for param in ('Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'):
        field = generateParameterField(param, workspace, peak)
        assert isinstance(field, ScalarFieldSample)
        assert field.name == param
        np.testing.assert_equal(field.x, x)
        np.testing.assert_equal(field.y, y)
        np.testing.assert_equal(field.z, z)
        np.testing.assert_equal(field.values, expected_values[param])
        np.testing.assert_equal(field.errors, expected_errors[param])

    # quick test the d spacing
    for param in ('dspacing_center', 'd_reference'):
        field = generateParameterField(param, workspace, peak)
        assert isinstance(field, ScalarFieldSample)
        assert field.name == param
        np.testing.assert_equal(field.x, x)
        np.testing.assert_equal(field.y, y)
        np.testing.assert_equal(field.z, z)
        expected_values, expected_errors = getattr(peak, f'get_{param}')()
        np.testing.assert_equal(field.values, expected_values)
        np.testing.assert_equal(field.errors, expected_errors)


@pytest.fixture(scope='module')
def strains_for_stress_field_1():
    X = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
    Y = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    Z = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    # selected to make terms drop out

    sample11 = ScalarFieldSample('strain',
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                 X, Y, Z)
    sample22 = ScalarFieldSample('strain',
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                 X, Y, Z)
    sample33 = ScalarFieldSample('strain',
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                 [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                 X, Y, Z)
    return sample11, sample22, sample33


@pytest.fixture(scope='module')
def stress_samples(strains_for_stress_field_1):
    POISSON = 1. / 3.  # makes nu / (1 - 2*nu) == 1
    YOUNG = 1 + POISSON  # makes E / (1 + nu) == 1
    sample11, sample22, sample33 = strains_for_stress_field_1
    return {'stress diagonal': StressField(sample11, sample22, sample33, YOUNG, POISSON)}


class TestStressField:

    def test_point_list(self, strains_for_stress_field_1):
        r"""Test point_list property"""
        vx = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
        vy = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        vz = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        coordinates = np.array([vx, vy, vz]).T
        field = StressField(*strains_for_stress_field_1, 1.0, 1.0)
        np.allclose(field.point_list.coordinates, coordinates)

    def test_youngs_modulus(self, strains_for_stress_field_1):
        r"""Test poisson_ratio property"""
        youngs_modulus = random.random()
        field = StressField(*strains_for_stress_field_1, youngs_modulus, 1.0)
        assert field.youngs_modulus == pytest.approx(youngs_modulus)

    def test_poisson_ratio(self, strains_for_stress_field_1):
        r"""Test poisson_ratio property"""
        poisson_ratio = random.random()
        field = StressField(*strains_for_stress_field_1, 1.0, poisson_ratio)
        assert field.poisson_ratio == pytest.approx(poisson_ratio)

    def test_create_stress_field(self):
        X = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
        Y = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        Z = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        # selected to make terms drop out

        sample11 = ScalarFieldSample('strain',
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                     X, Y, Z)
        sample22 = ScalarFieldSample('strain',
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                     X, Y, Z)
        sample33 = ScalarFieldSample('strain',
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],  # values
                                     [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                                     X, Y, Z)

        POISSON = 1. / 3.  # makes nu / (1 - 2*nu) == 1
        YOUNG = 1 + POISSON  # makes E / (1 + nu) == 1

        # test diagonal calculation
        diagonal = StressField(sample11, sample22, sample33, YOUNG, POISSON)
        assert diagonal
        # check strains
        assert diagonal.get_strain11 == sample11
        assert diagonal.get_strain22 == sample22
        assert diagonal.get_strain33 == sample33
        # check coordinates
        np.testing.assert_equal(diagonal.point_list.vx, X)
        np.testing.assert_equal(diagonal.point_list.vy, Y)
        np.testing.assert_equal(diagonal.point_list.vz, Z)
        # check values
        second = (sample11.values + sample22.values + sample33.values)
        diagonal.select('11')
        np.testing.assert_allclose(diagonal.values, sample11.values + second)
        diagonal.select('22')
        np.testing.assert_allclose(diagonal.values, sample22.values + second)
        diagonal.select('33')
        np.testing.assert_allclose(diagonal.values, sample33.values + second)

        in_plane_strain = StressField(sample11, sample22, None, YOUNG, POISSON, 'in-plane-strain')
        assert in_plane_strain
        # check coordinates
        np.testing.assert_equal(in_plane_strain.point_list.vx, X)
        np.testing.assert_equal(in_plane_strain.point_list.vy, Y)
        np.testing.assert_equal(in_plane_strain.point_list.vz, Z)
        # check values
        second = (sample11.values + sample22.values)
        in_plane_strain.select('11')
        np.testing.assert_allclose(in_plane_strain.values, sample11.values + second)
        in_plane_strain.select('22')
        np.testing.assert_allclose(in_plane_strain.values, sample22.values + second)
        in_plane_strain.select('33')
        np.testing.assert_allclose(in_plane_strain.values, second)

        # redefine values to simplify things
        POISSON = 1. / 2.  # makes nu / (1 - nu) == 1
        YOUNG = 1 + POISSON  # makes E / (1 + nu) == 1

        in_plane_stress = StressField(sample11, sample22, None, YOUNG, POISSON, 'in-plane-stress')
        assert in_plane_stress
        # check coordinates
        np.testing.assert_equal(in_plane_stress.point_list.vx, X)
        np.testing.assert_equal(in_plane_stress.point_list.vy, Y)
        np.testing.assert_equal(in_plane_stress.point_list.vz, Z)
        # check values
        second = (sample11.values + sample22.values)
        in_plane_stress.select('11')
        np.testing.assert_allclose(in_plane_stress.values, sample11.values + second)
        in_plane_stress.select('22')
        np.testing.assert_allclose(in_plane_stress.values, sample22.values + second)
        in_plane_stress.select('33')
        np.testing.assert_equal(in_plane_stress.values, 0.)
        np.testing.assert_equal(in_plane_stress.errors, 0.)


@pytest.fixture(scope='module')
def field_sample_collection():
    return {
        'sample1': SampleMock('strain',
                              [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090],  # values
                              [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                              [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
        # distance resolution is assumed to be 0.01
        # The first four points of sample2 still overlaps with the first four points of sample1
        # the last four points of sample1 are not in sample2, and viceversa
        'sample2': SampleMock('strain',
                              [1.071, 1.081, 1.091, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],  # values
                              [0.008, 0.008, 0.008, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # errors
                              [0.009, 1.009, 2.009, 3.009, 4.000, 5.000, 6.011, 7.011, 8.011, 9.011],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
        # the last two points of sample3 are redundant, as they are within resolution distance
        'sample3': SampleMock('strain',
                              [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090, 1.091],  # values
                              [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.008],  # errors
                              [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 8.991, 9.000],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
    }


def test_aggregate_scalar_field_samples(field_sample_collection):
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    aggregated_sample = aggregate_scalar_field_samples(sample1, sample2, sample3)
    assert len(aggregated_sample) == len(sample1) + len(sample2) + len(sample3)
    # check some key field values
    assert aggregated_sample.values[len(sample1)] == pytest.approx(sample2.values[0])
    assert aggregated_sample.values[len(sample1) + len(sample2)] == pytest.approx(sample3.values[0])
    assert aggregated_sample.values[-1] == pytest.approx(sample3.values[-1])
    # check some key point coordinate values
    assert aggregated_sample.x[len(sample1)] == pytest.approx(sample2.x[0])
    assert aggregated_sample.x[len(sample1) + len(sample2)] == pytest.approx(sample3.x[0])
    assert aggregated_sample.x[-1] == pytest.approx(sample3.x[-1])


def test_fuse_scalar_field_samples(field_sample_collection):
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    fused_sample = fuse_scalar_field_samples(sample1, sample2, sample3, resolution=0.01, criterion='min_error')
    assert len(fused_sample) == 14
    assert sorted(fused_sample.values) == pytest.approx([1., 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08,
                                                         1.091, 1.1, 1.13, 1.14, 1.15, 1.16])
    assert sorted(fused_sample.errors) == pytest.approx([0., 0., 0.001, 0.002, 0.004, 0.005, 0.006,
                                                         0.007, 0.008, 0.008, 0.03, 0.04, 0.05, 0.06])
    assert sorted(fused_sample.x) == pytest.approx([0.0, 1.0, 2.0, 3.009, 4.0, 5.0, 6.0, 6.011,
                                                    7.0, 7.011, 8.0, 8.011, 9.0, 9.011])


def test_stack_scalar_field_samples(field_sample_collection,
                                    approx_with_sorting, assert_almost_equal_with_sorting, allclose_with_sorting):
    r"""Stack three scalar fields"""
    # test stacking with the 'common' mode
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    sample1, sample2, sample3 = stack_scalar_field_samples(sample1, sample2, sample3, stack_mode='common')

    for sample in (sample1, sample2, sample3):
        assert len(sample) == 6
        assert_almost_equal_with_sorting(sample.x, [5.0, 4.0, 3.003, 2.003, 1.003, 0.003])

    # Assert evaluations for sample1
    sample1_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0]
    assert np.allclose(sample1.values, sample1_values)
    # Assert evaluations for sample2
    sample2_values = [1.12, 1.11, 1.1, 1.091, 1.081, 1.071]
    assert np.allclose(sample2.values, sample2_values)
    # Assert evaluations for sample3
    sample3_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0]
    assert np.allclose(sample3.values, sample3_values)

    # test stacking with the 'complete' mode
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    sample1, sample2, sample3 = stack_scalar_field_samples(sample1, sample2, sample3, stack_mode='complete')

    for sample in (sample1, sample2, sample3):
        assert len(sample) == 14
        approx_with_sorting(sample.x,
                            [5.0, 4.0, 3.003, 2.003, 1.003, 0.003, 9.0, 8.0, 6.0, 7.0, 9.011, 8.011, 6.011, 7.011])

    # Assert evaluations for sample1
    sample1_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0,
                      1.09, 1.08, 1.06, 1.07,
                      float('nan'), float('nan'), float('nan'), float('nan')]
    assert allclose_with_sorting(sample1.values, sample1_values, equal_nan=True)

    # Assert evaluations for sample2
    sample2_values = [1.12, 1.11, 1.1, 1.091, 1.081, 1.071,
                      float('nan'), float('nan'), float('nan'), float('nan'),
                      1.16, 1.15, 1.13, 1.14]
    assert allclose_with_sorting(sample2.values, sample2_values, equal_nan=True)

    # Assert evaluations for sample3
    sample3_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0,
                      1.091, 1.08, 1.06, 1.07,
                      float('nan'), float('nan'), float('nan'), float('nan')]
    assert allclose_with_sorting(sample3.values, sample3_values, equal_nan=True)


if __name__ == '__main__':
    pytest.main()
