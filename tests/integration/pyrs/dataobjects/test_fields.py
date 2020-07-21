import numpy as np
import pytest

from pyrs.dataobjects.constants import DEFAULT_POINT_RESOLUTION
from pyrs.dataobjects.fields import ScalarFieldSample, fuse_scalar_field_samples


@pytest.fixture(scope='module')
def fuse_data():
    r"""Data mimics runs 1347 and 1350. These runs have identical (vx, vy, and vz), specified below.
    The mimic data is for three runs, instead of two. All three runs have the same (vx, vy, and vz).
    Field values are created randomly, then for each sample point, the field with the smallest value
    is selected according to the 'min_error' criterion.
    """
    point_list_count = 87  # number of sample points
    vx = [-40.0000, -38.0833, -36.1667, -34.2500, -32.3333, -30.4167, -28.5000, -26.5833, -24.6667, -22.7500,
          -20.8333, -18.9167, -17.0000, -15.0000, -14.5000, -14.0000, -13.5000, -13.0000, -12.5000, -12.0000,
          -11.5000, -11.0000, -10.5000, -10.0000, -9.5000, -9.0000, -8.5000, -8.0000, -7.5000, -7.0000, -6.5000,
          -6.0000, -5.5000, -5.0000, -4.5000, -4.0000, -3.5000, -3.0000, -2.5000, -2.0000, -1.5000, -1.0000,
          -0.5000, 0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000, 5.0000,
          5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000, 9.0000, 9.5000, 10.0000, 10.5000, 11.0000,
          11.5000, 12.0000, 12.5000, 13.0000, 13.5000, 14.0000, 14.5000, 15.0000, 17.0000, 18.9167, 20.8333,
          22.7500, 24.6667, 26.5833, 28.5000, 30.4167, 32.3333, 34.2500, 36.1667, 38.0833, 40.0000]
    vy = vz = [0.0] * point_list_count
    # Create values and  for three scans, each scan having `point_list_count` sample points
    values = np.random.random(3 * point_list_count).reshape((point_list_count, 3))
    errors = 0.1 * np.random.random(3 * point_list_count).reshape((point_list_count, 3))
    i_min = np.argsort(errors, axis=1)[:, 0]  # For each point, pick the scan having the smallest error
    return {
        'name': 'lattice_constant',
        'logs_count': 3,
        'logs1': {'values': values[:, 0], 'errors': errors[:, 0], 'x': vx, 'y': vy, 'z': vz},  # mimic for first run
        'logs2': {'values': values[:, 1], 'errors': errors[:, 1], 'x': vx, 'y': vy, 'z': vz},  # mimic second run
        'logs3': {'values': values[:, 2], 'errors': errors[:, 2], 'x': vx, 'y': vy, 'z': vz},  # mimic third run
        'criterion': 'min_error',
        'resolution': 0.01,
        'fused': {
            'values': [values[i, i_min[i]] for i in range(point_list_count)],  # expected values after combining runs
            'errors': [errors[i, i_min[i]] for i in range(point_list_count)],
            'x': vx, 'y': vy, 'z': vz
        }
    }


def test_fuse_with(fuse_data):
    # Boiler-plate code to create a ScalarFieldSample object for every subrun
    scalar_field_samples = list()
    for i in range(1, 1 + fuse_data['logs_count']):
        logs = fuse_data[f'logs{i}']
        scalar_field_samples.append(ScalarFieldSample(fuse_data['name'], logs['values'], logs['errors'],
                                                      logs['x'], logs['y'], logs['z']))

    # Fuse the scalar field sample objects
    fused_field = fuse_scalar_field_samples(*scalar_field_samples,
                                            criterion=fuse_data['criterion'],
                                            resolution=fuse_data['resolution'])

    # Compare with the expected result. Different versions of scipy can return clusters
    # in different order. Thus, sort by increasing value of the field.
    permutation = np.argsort(fused_field.values)
    for item in ('values', 'errors', 'x', 'y', 'z'):
        item_sorted = np.array(getattr(fused_field, item))[permutation]
        list(item_sorted) == pytest.approx(fuse_data['fused'][item])


@pytest.fixture(scope='module')
def data_interpolate_surface_scans():
    r"""Three scans taken on the vz=0 plane"""

    def _lf(vx, vy):
        r"""Intensities are the sum of the absolute value of the coordinate"""
        assert len(vx) == len(vy)
        values = np.array(vx) + np.array(vy)
        errors = 0.1 * values
        return ScalarFieldSample('strain', values, errors, vx, vy, [0] * len(vx))

    def assert_checks(field):
        r""""""
        is_finite = np.isfinite(field.values)
        assert field.values[is_finite] == pytest.approx(np.sum(field.coordinates[is_finite], axis=1))

    return {
        'assert checks': assert_checks,
        'name': 'strain',
        'sample1': _lf([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6],
                       [0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16]),
        'sample2': _lf([10, 10, 10, 10, 10, 14, 14, 14, 14, 14, 18, 18, 18, 18, 18, 22, 22, 22, 22, 22,
                        26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 34, 34, 34, 34, 34],
                       [0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16,
                        0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16]),
        'sample3': _lf([38, 38, 38, 38, 38, 38, 38, 38, 40, 40, 40, 40, 40, 40, 40, 40, 42, 42, 42, 42, 42, 42, 42, 42,
                        44, 44, 44, 44, 44, 44, 44, 44, 46, 46, 46, 46, 46, 46, 46, 46, 48, 48, 48, 48, 48, 48, 48, 48,
                        50, 50, 50, 50, 50, 50, 50, 50, 52, 52, 52, 52, 52, 52, 52, 52],
                       [0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14,
                        0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14,
                        0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14]),
        'criterion': 'min_error',
        'resolution': DEFAULT_POINT_RESOLUTION,
        'fused': {
            'values': [],  # expected values after combining runs
            'errors': [],
            'x': [], 'y': [], 'z': []
        }
    }


def test_interpolate_surface_scans(data_interpolate_surface_scans, allclose_with_sorting):
    data = data_interpolate_surface_scans  # handy shortcut
    scalar_field_samples = [data['sample1'], data['sample2'], data['sample3']]
    # There are no sample points within resolution distance, thus fusing simply concatenate the three samples
    fused = fuse_scalar_field_samples(*scalar_field_samples, criterion=data['criterion'],
                                      resolution=data['resolution'])
    assert len(fused) == sum([len(sample) for sample in scalar_field_samples])
    assert allclose_with_sorting(fused.values, np.concatenate([sample.values for sample in scalar_field_samples]))

    interpolated = fused.interpolated_sample(keep_nan=True,
                                             criterion=data['criterion'], resolution=data['resolution'])
    data['assert checks'](interpolated)


@pytest.fixture(scope='module')
def data_interpolate_volume_scan():
    r"""
    Three surface scans on the (vx, vy) plane, taken at vz=0, vz=1, and vz=4. Their extents on
    the (vx, vy) plane are the same, but differ on the spacing.
    Scan at vz=0 is very detailed, mimicking and inspection of the surface. Scan at vz=1 is less
    detailed, and scan at vz=4 is even less detailed.
    Scans are elongated along the vx axis, to mimic inspecting a soldering
    - scan at vz=0:  0 <= vx <= 50, step = 1;  0 <= vy <= 20, step=1.  A total of 51 * 21 = 1071 points
    - scan at vz=1:  0 <= vz <= 50, step = 2,  0 <= vy <= 20, step=2.  A total of 25 * 11 = 275
    - scan at vz=4:  0 <= vz <= 50, step = 5,  0 <= vy <= 20, step=5.  A total of 11 * 5 = 55
    """

    def _field_generator(vx_extent, vy_extent, vz_extent):
        r"""
        Provide the start, end, and step along each axis.
        Intensities are the sum of the values of the coordinate components along each axis
        """
        epsilon = 1.e-9
        slices = [slice(e[0], e[1] + epsilon, e[2]) for e in (vx_extent, vy_extent, vz_extent)]
        coordinates = np.transpose(np.mgrid[slices], (1, 2, 3, 0)).reshape(-1, 3)
        values = np.sum(coordinates, axis=1)
        errors = 0.1 * values
        vx, vy, vz = coordinates.T
        return ScalarFieldSample('strain', values, errors, vx, vy, vz)

    def assert_checks(field):
        r"""Assert the intensity in the interpolated values is the sum of the value of the coordinate
        component along each axis."""
        is_finite = np.isfinite(field.values)
        assert field.values[is_finite] == pytest.approx(np.sum(field.coordinates[is_finite], axis=1))

    return {
        'assert checks': assert_checks,
        'name': 'strain',
        'sample1': _field_generator((0, 50, 1), (0, 20, 1), (0, 0, 1)),  # scan at vz=0
        'sample2': _field_generator((0, 50, 2), (0, 20, 2), (1, 1, 1)),  # scan at vz=1
        'sample3': _field_generator((0, 50, 5), (0, 20, 5), (3, 3, 1)),  # scan at vz=3
        'criterion': 'min_error',
        'resolution': DEFAULT_POINT_RESOLUTION,
        'fused': {
            'values': [],  # expected values after combining runs
            'errors': [],
            'x': [], 'y': [], 'z': []
        }
    }


# TODO implementation of this test is a duplication of test_interpolate_surface_scans, except of the data fixture.
def test_interpolate_volume_scan(data_interpolate_volume_scan, allclose_with_sorting):
    data = data_interpolate_volume_scan  # handy shortcut
    scalar_field_samples = [data['sample1'], data['sample2'], data['sample3']]
    # There are no sample points within resolution distance, thus fusing simply concatenate the three samples
    fused = fuse_scalar_field_samples(*scalar_field_samples, criterion=data['criterion'],
                                      resolution=data['resolution'])
    assert len(fused) == sum([len(sample) for sample in scalar_field_samples])
    assert allclose_with_sorting(fused.values, np.concatenate([sample.values for sample in scalar_field_samples]))

    interpolated = fused.interpolated_sample(keep_nan=True,
                                             criterion=data['criterion'], resolution=data['resolution'])
    data['assert checks'](interpolated)


if __name__ == '__main__':
    pytest.main()
