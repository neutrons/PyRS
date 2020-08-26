import numpy as np
import pytest
from pyrs.dataobjects import SampleLogs  # type: ignore


def test_reassign_subruns():
    sample = SampleLogs()
    sample.subruns = [1, 2, 3, 4]
    sample.subruns = [1, 2, 3, 4]  # setting same value is fine
    with pytest.raises(RuntimeError):
        sample.subruns = [1, 3, 4]
    with pytest.raises(RuntimeError):
        sample.subruns = [4, 3, 2, 1]


def test_subrun_second():
    sample = SampleLogs()
    # do it wrong
    with pytest.raises(RuntimeError):
        sample['variable1'] = np.linspace(0., 100., 5)
    # do it right
    sample.subruns = [1, 2, 3, 4, 5]
    sample['variable1'] = np.linspace(0., 100., 5)


def test_one():
    sample = SampleLogs()
    sample.subruns = 1
    assert len(sample) == 0

    sample['variable1'] = 27
    assert len(sample) == 1

    with pytest.raises(ValueError):
        sample['variable1'] = [27, 28]

    assert sorted(sample.plottable_logs()) == ['sub-runs', 'variable1']
    assert sample.constant_logs() == ['variable1']


def test_multi():
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


def test_get_pointlist():
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


if __name__ == '__main__':
    pytest.main([__file__])
