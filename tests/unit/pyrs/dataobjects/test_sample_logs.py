import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from pyrs.dataobjects.constants import HidraConstants, DEFAULT_POINT_RESOLUTION
from pyrs.dataobjects.sample_logs import DirectionExtents, PointList, aggregate_point_lists, SampleLogs


@pytest.fixture(scope='module')
def sample_logs_mock():
    logs = SampleLogs()
    # Simple subruns
    subruns_size = 14
    logs[HidraConstants.SUB_RUNS] = list(range(subruns_size))
    # Create sample coordinates
    xyz = [[-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
           [0.5, 0.495, 0.5, 0.509, 0.5, 0.5, 0.5, 0.595, 0.6, 0.601, 0.6, 0.6, 0.6, 0.6],
           [1.5, 1.5, 1.501, 1.499, 1.5, 1.498, 1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6, 1.7]
           ]
    for i, name in enumerate(HidraConstants.SAMPLE_COORDINATE_NAMES):
        logs[name] = xyz[i]
    # Extents
    extents = [[-0.6, 0.7, 0.1],  # min, max, delta
               [0.495, 0.601, (0.601 - 0.495) / 3],  # notice that 0.509 - 0.495 > DEFAULT_POINT_RESOLUTION
               [1.498, 1.7, (1.7 - 1.498) / 3]]
    return {'logs': logs, 'xyz': xyz, 'extents': extents, 'resolution': DEFAULT_POINT_RESOLUTION}


class TestSampleLogs:

    def test_setitem(self, sample_logs_mock):
        sample_logs, xyz = sample_logs_mock['logs'], sample_logs_mock['xyz']

        sample_logs['vx'] = xyz[0]
        assert sample_logs.units('vx') == ''
        sample_logs['vx', 'mm'] = xyz[0]
        assert sample_logs.units('vx') == 'mm'

    def test_reassign_subruns(self):
        sample = SampleLogs()
        sample.subruns = [1, 2, 3, 4]
        sample.subruns = [1, 2, 3, 4]  # setting same value is fine
        with pytest.raises(RuntimeError):
            sample.subruns = [1, 3, 4]
        with pytest.raises(RuntimeError):
            sample.subruns = [4, 3, 2, 1]

    def test_subrun_second(self):
        sample = SampleLogs()
        # do it wrong
        with pytest.raises(RuntimeError):
            sample['variable1'] = np.linspace(0., 100., 5)
        # do it right
        sample.subruns = [1, 2, 3, 4, 5]
        sample['variable1'] = np.linspace(0., 100., 5)

    def test_one(self):
        sample = SampleLogs()
        sample.subruns = 1
        assert len(sample) == 0

        sample['variable1'] = 27
        assert len(sample) == 1

        with pytest.raises(ValueError):
            sample['variable1'] = [27, 28]

        assert sorted(sample.plottable_logs()) == ['sub-runs', 'variable1']
        assert sample.constant_logs() == ['variable1']

    def test_multi(self):
        sample = SampleLogs()
        sample.subruns = [1, 2, 3, 4, 5]
        sample['constant1'] = np.zeros(5) + 42
        sample['variable1'] = np.linspace(0., 100., 5)
        sample['string1'] = np.array(['a'] * sample.subruns.size)  # will be constant as well

        # names of logs
        assert sorted(sample.plottable_logs()) == ['constant1', 'sub-runs', 'variable1']
        assert sorted(sample.constant_logs()) == ['constant1', 'string1']

        # slicing
        np.testing.assert_equal(sample['variable1'], [0., 25., 50., 75., 100.])
        np.testing.assert_equal(sample['variable1', 3], [50.])
        np.testing.assert_equal(sample['variable1', [1, 2, 3]], [0., 25., 50.])

        with pytest.raises(IndexError):
            np.testing.assert_equal(sample['variable1', [0]], [0., 50., 75., 100.])
        with pytest.raises(IndexError):
            np.testing.assert_equal(sample['variable1', [10]], [0., 50., 75., 100.])

    def test_get_pointlist(self):
        sample = SampleLogs()
        sample.subruns = np.arange(1, 6, dtype=int)

        with pytest.raises(ValueError):
            sample.get_pointlist()

        # check getting whole PointList
        sample['vx'] = np.arange(5, dtype=float)
        sample['vy'] = np.arange(5, dtype=float)
        sample['vz'] = np.arange(5, dtype=float)
        pointlist = sample.get_pointlist()

        assert pointlist
        assert len(pointlist) == 5
        np.testing.assert_equal(pointlist.vx, np.arange(5, dtype=float))
        np.testing.assert_equal(pointlist.vy, np.arange(5, dtype=float))
        np.testing.assert_equal(pointlist.vz, np.arange(5, dtype=float))

        # check getting partial PointList
        pointlist = sample.get_pointlist([1, 3, 4])

        assert pointlist
        assert len(pointlist) == 3
        np.testing.assert_equal(pointlist.vx, [0, 2, 3])
        np.testing.assert_equal(pointlist.vy, [0, 2, 3])
        np.testing.assert_equal(pointlist.vz, [0, 2, 3])

        # check unit conversion
        sample['vx', 'm'] = np.arange(5, dtype=float)
        sample['vy', 'm'] = np.arange(5, dtype=float)
        sample['vz', 'm'] = np.arange(5, dtype=float)
        pointlist = sample.get_pointlist()
        np.testing.assert_allclose(pointlist.vx, 1000 * np.arange(5, dtype=float), atol=0.1)


class TestDirectionExtents:

    def test_init(self):
        d = DirectionExtents(range(2))
        assert list(d) == pytest.approx([0, 1, 1])
        d = DirectionExtents(range(42))
        assert list(d) == pytest.approx([0, 41, 1])

        not_too_close = 1.01 * DEFAULT_POINT_RESOLUTION
        d = DirectionExtents(np.arange(0, 1.01, not_too_close), resolution=DEFAULT_POINT_RESOLUTION)
        assert d.numpoints == int(1 / DEFAULT_POINT_RESOLUTION)
        assert list(d) == [0, 1.01 - not_too_close, not_too_close]

        too_close_together = 0.1 * DEFAULT_POINT_RESOLUTION
        d = DirectionExtents(np.arange(0, 1, too_close_together))
        # we cannot resolve more coords than those limited by the precision
        assert d.numpoints == int(1 / DEFAULT_POINT_RESOLUTION)  # extra 1 due to function round()
        assert list(d) == [0, 1 - too_close_together, (1 - too_close_together) / (d.numpoints - 1)]

        # Corner cases
        d = DirectionExtents([0.001, 0.009], resolution=0.01)  # round() floors 0.004 and ceilings 0.006
        assert list(d) == pytest.approx([0.005, 0.005, 0.01])
        d = DirectionExtents([0.009, 0.011], resolution=0.01)
        assert list(d) == [0.009, 0.011, 0.002]

    def test_to_createmd(self):
        d = DirectionExtents(range(42))
        assert d.to_createmd() == '-0.500,41.500'
        d = DirectionExtents([0.001, 0.009], resolution=0.01)
        assert d.to_createmd() == '0.000,0.010'

    def test_to_binmd(self):
        d = DirectionExtents(range(42))
        assert d.to_binmd() == '-0.500,41.500,42'
        d = DirectionExtents([0.001, 0.009], resolution=0.01)
        assert d.to_binmd() == '0.000,0.010,1'


class TestPointList:

    def test_tolist(self, sample_logs_mock):
        for input_source in (sample_logs_mock['xyz'],
                             sample_logs_mock['logs'],
                             np.array(sample_logs_mock['xyz'])):
            list_of_list = PointList.tolist(input_source)
            assert isinstance(list_of_list, tuple)
            for i in range(3):
                assert isinstance(list_of_list[i], np.ndarray)
                np.testing.assert_equal(list_of_list[i], sample_logs_mock['xyz'][i])

    def test_init(self, sample_logs_mock):
        for input_source_type in ('logs', 'xyz'):
            input_source = sample_logs_mock[input_source_type]
            point_list = PointList(input_source)
            # dereference attributes 'vx', 'vy', 'vz'
            np.testing.assert_equal(point_list.vx, sample_logs_mock['xyz'][0])
            np.testing.assert_equal(point_list.vy, sample_logs_mock['xyz'][1])
            np.testing.assert_equal(point_list.vz, sample_logs_mock['xyz'][2])
            # dereference by item index
            xyz = np.array(sample_logs_mock['xyz'])
            assert point_list[5] == pytest.approx(list(xyz[:, 5]))

    def test_getattr(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        assert list(point_list.vx) == pytest.approx(sample_logs_mock['xyz'][0])
        assert list(point_list.vy) == pytest.approx(sample_logs_mock['xyz'][1])
        try:
            _ = point_list.dummy  # noqa F841
            assert False, 'Should not have been able to access attribute'
        except AttributeError:
            pass  # this is the correct behavior

    def test_is_contained_in(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        subset = np.array(sample_logs_mock['xyz'])[:, 0:5]  # pick only the first five coordinates
        other_list = PointList(subset)
        assert other_list.is_contained_in(point_list) is True
        assert point_list.is_contained_in(other_list) is False

    def test_is_equal_within_resolution(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        # perturb the positions of point_list within resolution
        original = np.array(sample_logs_mock['xyz'])
        perturbation = (sample_logs_mock['resolution'] / 3.0) * (np.random.random(original.shape) - 0.5)
        other_list = PointList(original + perturbation)
        assert point_list.is_equal_within_resolution(other_list, resolution=sample_logs_mock['resolution']) is True
        # perturb the very first coordinate too much
        perturbation[0][0] += 2 * sample_logs_mock['resolution']
        other_list = PointList(original + perturbation)
        assert point_list.is_equal_within_resolution(other_list, resolution=sample_logs_mock['resolution']) is False

    def test_aggregate(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs']).aggregate(PointList(sample_logs_mock['xyz']))
        np.testing.assert_equal(point_list.vx, sample_logs_mock['xyz'][0] + sample_logs_mock['xyz'][0])
        np.testing.assert_equal(point_list.vy, sample_logs_mock['xyz'][1] + sample_logs_mock['xyz'][1])
        np.testing.assert_equal(point_list.vz, sample_logs_mock['xyz'][2] + sample_logs_mock['xyz'][2])

    def test_cluster(self):
        xyz = [[0.0, 1.0, 2.0, 3.0, 1.009, 0.995, 2.0, 3.005, 4.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        clusters = PointList(xyz).cluster(resolution=DEFAULT_POINT_RESOLUTION)
        for cluster, comparison in zip(clusters, [[1, 4, 5], [2, 6], [3, 7], [0], [8]]):
            assert cluster == pytest.approx(comparison)

    def test_has_overlapping_points(self):
        xyz = [[0.0, 1.0, 2.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]]
        assert PointList(xyz).has_overlapping_points(resolution=DEFAULT_POINT_RESOLUTION) is False
        xyz = [[0.0, 1.0, 2.0, 2.009],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0]]
        assert PointList(xyz).has_overlapping_points(resolution=DEFAULT_POINT_RESOLUTION) is True

    def test_coordinates(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        np.testing.assert_allclose(point_list.coordinates, np.array(sample_logs_mock['xyz']).transpose())

    def test_sort(self):
        xyz = [[0, 0, 0, 0], [1, 2, 1, 2], [3, 3, 4, 4]]
        point_list = PointList(xyz)
        point_list.sort()
        assert_array_almost_equal(point_list.vy, [1, 1, 2, 2], decimal=1)
        assert_array_almost_equal(point_list.vz, [3, 4, 3, 4], decimal=1)

    def test_argsort(self):
        xyz = [[0, 0, 0, 0], [1, 2, 1, 2], [3, 3, 4, 4]]
        point_list = PointList(xyz)
        assert_array_equal(point_list.argsort(), [0, 2, 1, 3])

    def test_coordinates_along_direction(self):
        xyz = [[0, 1], [2, 3], [4, 5]]
        point_list = PointList(xyz)
        for r, index, label in zip(xyz, range(3), ('vx', 'vy', 'vz')):
            assert point_list.coordinates_along_direction(index) == pytest.approx(r)
            assert point_list.coordinates_along_direction(label) == pytest.approx(r)

    def test_coordinates_irreducible(self):
        xyz = [[0.0, 1.000, 1.001, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],  # x
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]  # z
        point_list = PointList(xyz)
        # Volume scan
        coordinates_irreducible = point_list.coordinates_irreducible(resolution=DEFAULT_POINT_RESOLUTION)
        assert coordinates_irreducible[0] == pytest.approx([0.0, 0.0, 0.0])  # check first point
        assert coordinates_irreducible[-1] == pytest.approx([4.0, 1.0, 1.0])  # check last point
        assert coordinates_irreducible.shape[-1] == 3  # xyz if a volume scan, thus three dimensions
        assert coordinates_irreducible == pytest.approx(np.array(xyz).transpose())
        # Surface scan
        xyz[2] = [0] * 11
        point_list = PointList(xyz)
        coordinates_irreducible = point_list.coordinates_irreducible(resolution=DEFAULT_POINT_RESOLUTION)
        assert coordinates_irreducible[0] == pytest.approx([0.0, 0.0])  # check first point
        assert coordinates_irreducible[-1] == pytest.approx([4.0, 1.0])  # check last point
        assert coordinates_irreducible.shape[-1] == 2  # xyz if a surface scan, thus two dimensions
        assert coordinates_irreducible == pytest.approx(np.array(xyz[0:2]).transpose())
        # Linear scan
        xyz[1] = [0] * 11
        point_list = PointList(xyz)
        coordinates_irreducible = point_list.coordinates_irreducible(resolution=DEFAULT_POINT_RESOLUTION)
        assert coordinates_irreducible[0] == pytest.approx([0.0])  # check first point
        assert coordinates_irreducible[-1] == pytest.approx([4.0])  # check last point
        assert coordinates_irreducible.shape[-1] == 1  # xyz if a linear scan, thus one dimensions
        assert coordinates_irreducible == pytest.approx(np.array(xyz[0]).reshape((11, 1)))

    def test_linear_scan_vector(self):
        # vx and vy are only one point, within resolution
        xyz = [[0, 0, 0.003, 0.004, 0], [0, 1, 2, 3, 4], [1, 1.001, 1.002, 1.003, 1.009]]
        point_list = PointList(xyz)
        assert point_list.linear_scan_vector(resolution=DEFAULT_POINT_RESOLUTION) == pytest.approx([0, 1, 0])
        xyz[0][0] = 0.011  # this point is beyond resolution with the rest of the other vx points
        point_list = PointList(xyz)
        assert point_list.linear_scan_vector(resolution=DEFAULT_POINT_RESOLUTION) is None

    def test_intersection(self):
        xyz1 = [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        xyz2 = [[1.009, 0.995, 2.0, 3.005, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        common = PointList(xyz1).intersection(PointList(xyz2), resolution=DEFAULT_POINT_RESOLUTION)
        assert common.vx == pytest.approx([1.0, 2.0, 3.0, 1.009, 0.995, 2.0, 3.005])

    def test_fuse_with(self):
        xyz1 = [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        xyz2 = [[1.009, 0.995, 2.0, 3.005, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        point_list = PointList(xyz1)
        common = point_list.fuse_with(PointList(xyz2), resolution=DEFAULT_POINT_RESOLUTION)
        assert common.vx == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_extents(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        for i, extent in enumerate(point_list.extents(resolution=DEFAULT_POINT_RESOLUTION)):
            assert list(extent) == pytest.approx(sample_logs_mock['extents'][i])

    def test_linspace(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        extents = sample_logs_mock['extents']
        vx, vy, vz = point_list.linspace(resolution=sample_logs_mock['resolution'])
        assert [vx[0], vx[-1], (vx[-1] - vx[0]) / (len(vx) - 1)] == pytest.approx(extents[0])
        assert len(vx) == 14
        assert [vy[0], vy[-1], (vy[-1] - vy[0]) / (len(vy) - 1)] == pytest.approx(extents[1])
        assert len(vy) == 4  # many values of point_list.vy are repeated
        assert [vz[0], vz[-1], (vz[-1] - vz[0]) / (len(vz) - 1)] == pytest.approx(extents[2])
        assert len(vz) == 4  # many values of point_list.vz are repeated

    def test_mgrid(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        grid_x, grid_y, grid_z = point_list.mgrid(resolution=sample_logs_mock['resolution'])
        assert grid_x[0][0] == pytest.approx([-0.6, -0.6, -0.6, -0.6])
        assert grid_x[13][-1] == pytest.approx([0.7, 0.7, 0.7, 0.7])
        assert grid_y[0][0] == pytest.approx([0.495, 0.495, 0.495, 0.495])
        assert grid_y[13][-1] == pytest.approx([0.601, 0.601, 0.601, 0.601])
        assert grid_z[0][0] == pytest.approx([1.498, 1.565, 1.633, 1.700], abs=0.001)
        assert grid_z[13][-1] == pytest.approx([1.498, 1.565, 1.633, 1.700], abs=0.001)
        # Test irreducibility option
        # volume scan: all three directions have more than unique point
        xyz = [[0.0, 1.000, 1.001, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],  # x
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]  # z
        point_list = PointList(xyz)
        grid = point_list.mgrid(resolution=sample_logs_mock['resolution'], irreducible=True)
        assert grid.shape == (3, 5, 2, 2)
        # Surface scan on the {vx, vy} plane
        xyz[2] = [0] * 11
        point_list = PointList(xyz)
        grid = point_list.mgrid(resolution=sample_logs_mock['resolution'], irreducible=True)
        assert grid.shape == (2, 5, 2)
        # Linear scan on the vx axis
        xyz[1] = [0] * 11
        point_list = PointList(xyz)
        grid = point_list.mgrid(resolution=sample_logs_mock['resolution'], irreducible=True)
        assert grid.shape == (1, 5)

    def test_grid_point_list(self):
        r"""
        The regular grid spanned by the three orthonormal vectors is the unit cube, with eight points
         The order of the coordinates must follow this nested loop structure:
         for vx in ...:
             for vy in ...:
                 for vz in ...:
        """
        # Passing the orthonormal vectors along each direction as three points
        point_list = PointList([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        other_list = point_list.grid_point_list(resolution=DEFAULT_POINT_RESOLUTION)
        cube_coordinates = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
                            [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]
        assert np.allclose(cube_coordinates, other_list.coordinates)

    def test_is_a_grid(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        assert point_list.is_a_grid(resolution=sample_logs_mock['resolution']) is False
        regular_point_list = point_list.grid_point_list(resolution=sample_logs_mock['resolution'])
        assert regular_point_list.is_a_grid(resolution=sample_logs_mock['resolution']) is True

    def test_sorted_indices(self):

        # The two point lists contain a different number of sample points
        point_list = PointList([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        point_list_other = PointList([[2.0, 0.005], [0.0, 0.0], [0.0, 0.0]])
        with pytest.raises(AssertionError) as exception_info:
            point_list.sorted_indices(point_list=point_list_other, resolution=DEFAULT_POINT_RESOLUTION)
        assert 'point lists do not contain same number of sample points' in str(exception_info.value)

        # Corner case: one of the point lists contains overlapping points
        point_list = PointList([[0.0, 0.009, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # three points, two overlap
        with pytest.raises(ValueError) as exception_info:
            point_list.sorted_indices(point_list=point_list, resolution=DEFAULT_POINT_RESOLUTION)
        assert 'point lists contains overlapping points' in str(exception_info.value)

        # The two point lists are not equal within resolution
        point_list = PointList([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        point_list_other = PointList([[0.0, 1.1, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError) as exception_info:
            point_list.sorted_indices(point_list=point_list_other, resolution=DEFAULT_POINT_RESOLUTION)
        assert 'point lists are not the same, within resolution' in str(exception_info.value)

        # Compare against itself
        point_list = PointList([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = point_list.sorted_indices(point_list=point_list, resolution=DEFAULT_POINT_RESOLUTION)
        np.testing.assert_equal(indices, np.array([0, 1, 2]))

        # General case
        point_list = PointList([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        point_list_other = PointList([[2.0, 0.005, 1.009], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = point_list.sorted_indices(point_list=point_list_other, resolution=DEFAULT_POINT_RESOLUTION)
        np.testing.assert_equal(indices, np.array([1, 2, 0]))

    def test_get_indices(self):
        x = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # x
        y = [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5]  # y
        z = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5]  # z
        point_list1 = PointList([x, y, z])

        # changes a single x-value in the last 4 points of point_list1
        x = [0.0, 0.5, 0.0, 0.5, 1.0, 1.5, 1.0, 1.5]  # x
        y = [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5]  # y
        z = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5]  # z
        point_list2 = PointList([x, y, z])

        # reverse order of point_list1
        x = [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]
        y = [1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0]
        z = [2.5, 2.5, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0]
        point_list3 = PointList([x, y, z])

        # last three sample points of point_list1
        x = [0.5, 0.0, 0.5]  # x
        y = [1.0, 1.5, 1.5]  # y
        z = [2.5, 2.5, 2.5]  # z
        point_list4 = PointList([x, y, z])

        # indices with self are running numbers
        for point_list in (point_list1, point_list2, point_list3, point_list4):
            np.testing.assert_equal(point_list.get_indices(point_list), np.arange(len(point_list)))

        # reversing values gets decreasing numbers
        for forward, reverse in zip((point_list1, point_list3), (point_list3, point_list1)):
            np.testing.assert_equal(forward.get_indices(reverse), [7, 6, 5, 4, 3, 2, 1, 0])

        # partial matching tests
        np.testing.assert_equal(point_list1.get_indices(point_list2), [0, 1, 2, 3, -1, -1, -1, -1])
        np.testing.assert_equal(point_list3.get_indices(point_list2), [-1, -1, -1, -1, 3, 2, 1, 0])

        np.testing.assert_equal(point_list4.get_indices(point_list1), [5, 6, 7])
        np.testing.assert_equal(point_list4.get_indices(point_list2), [-1, -1, -1])
        np.testing.assert_equal(point_list4.get_indices(point_list3), [2, 1, 0])


def test_aggregate_point_list(sample_logs_mock):
    point_list = aggregate_point_lists(*[PointList(sample_logs_mock['logs']) for _ in range(3)])  # three lists
    list_x, list_y, list_z = sample_logs_mock['xyz'][0], sample_logs_mock['xyz'][1], sample_logs_mock['xyz'][2]
    assert list(point_list.vx) == pytest.approx(list_x + list_x + list_x)
    assert list(point_list.vy) == pytest.approx(list_y + list_y + list_y)
    assert list(point_list.vz) == pytest.approx(list_z + list_z + list_z)


if __name__ == '__main__':
    pytest.main()
