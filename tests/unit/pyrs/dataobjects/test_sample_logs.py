import numpy as np
import pytest
from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import DirectionExtents, PointList, SampleLogs


class TestDirectionExtents:

    def test_init(self):
        d = DirectionExtents(range(2))
        assert list(d) == pytest.approx([0, 1, 1])
        d = DirectionExtents(range(42))
        assert list(d) == pytest.approx([0, 41, 1])

        precision = DirectionExtents.precision
        d = DirectionExtents(np.arange(0, 1, precision))
        assert d.numpoints == int(1 / precision)
        assert list(d) == [0, 1 - precision, precision]

        too_close_together = 0.1 * precision
        d = DirectionExtents(np.arange(0, 1, too_close_together))
        assert d.numpoints == int(1 / precision)  # we cannot resolve more coords than those limited by the precision
        assert list(d) == [0, 1 - too_close_together, (1 - too_close_together) / (d.numpoints - 1)]

    def test_to_createmd(self):
        d = DirectionExtents(range(42))
        assert d.to_createmd == '-0.5,41.5'

    def test_to_binmd(self):
        d = DirectionExtents(range(42))
        assert d.to_binmd('label') == 'label,-0.5,41.5,42'


@pytest.fixture(scope='module')
def sample_logs_mock():
    logs = SampleLogs()
    # Simple subruns
    subruns_size = 14
    logs[HidraConstants.SUB_RUNS] = list(range(subruns_size))
    # Create sample coordinates
    xyz = [[-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
           [0.5, 0.4995, 0.5, 0.5009, 0.5, 0.5, 0.5, 0.5995, 0.6, 0.6001, 0.6, 0.6, 0.6, 0.6],
           [1.5, 1.5, 1.5001, 1.4999, 1.5, 1.4998, 1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6, 1.7]
           ]
    for i, name in enumerate(HidraConstants.SAMPLE_COORDINATE_NAMES):
        logs[name] = xyz[i]
    # Extents
    extents = [[-0.6, 0.7, 0.1],
               [0.4995, 0.6001, (0.6001 - 0.4995) / 3],  # notice that 0.5009 - 0.4995 > PointList.precision
               [1.4998, 1.7, (1.7 - 1.4998) / 3]]
    return {'logs': logs, 'xyz': xyz, 'extents': extents}


class TestPointList:

    def test_init(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        # dereference attributes 'vx', 'vy', 'vz'
        assert [point_list.vx, point_list.vy, point_list.vz] == pytest.approx(sample_logs_mock['xyz'])
        # dereference by item index
        xyz = np.array(sample_logs_mock['xyz'])
        assert point_list[5] == pytest.approx(list(xyz[:, 5]))

    def test_getattr(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        assert point_list.vy == pytest.approx(sample_logs_mock['xyz'][1])
        assert point_list._points == pytest.approx(sample_logs_mock['xyz'])

    def test_extents(self, sample_logs_mock):
        point_list = PointList(sample_logs_mock['logs'])
        for i, extent in enumerate(point_list.extents):
            assert list(extent) == pytest.approx(sample_logs_mock['extents'][i])


if __name__ == '__main__':
    pytest.main()
