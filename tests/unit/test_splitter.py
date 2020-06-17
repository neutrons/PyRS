from mantid.kernel import Int32TimeSeriesProperty as IntLog
import numpy as np
from pyrs.core.nexus_conversion import Splitter
import pytest


def test_empty_constructor():
    # pass in empty dictionary
    try:
        _ = Splitter(dict())
        assert False, 'Should not have created a Splitter'
    except RuntimeError:
        pass

    try:
        logs = {'scan_index': IntLog('scan_index')}
        _ = Splitter(logs)
        assert False, 'Should not have created a Splitter'
    except RuntimeError:
        pass


def test_nominal():
    # single scan index
    scan_index = IntLog('scan_index')
    scan_index.addValue('1999-03-31T12:00Z', 0)
    scan_index.addValue('1999-03-31T12:01Z', 1)
    scan_index.addValue('1999-03-31T12:02Z', 0)

    splitter = Splitter({'scan_index': scan_index})

    assert splitter.size == 1
    np.testing.assert_equal(splitter.durations, [60])
    np.testing.assert_equal(splitter.subruns, [1])

    # add another scan index and create a new splitter
    scan_index.addValue('1999-03-31T12:03Z', 2)
    scan_index.addValue('1999-03-31T12:04Z', 0)

    splitter = Splitter({'scan_index': scan_index})

    assert splitter.size == 2
    np.testing.assert_equal(splitter.durations, [60, 60])
    np.testing.assert_equal(splitter.subruns, [1, 2])


def test_no_end():
    # when end isn't defined one day into the future is used
    ONE_DAY = 24*60*60

    # single scan index
    scan_index = IntLog('scan_index')
    scan_index.addValue('1999-03-31T12:00Z', 0)
    scan_index.addValue('1999-03-31T12:01Z', 1)

    splitter = Splitter({'scan_index': scan_index})

    assert splitter.size == 1
    np.testing.assert_equal(splitter.durations, [ONE_DAY])
    np.testing.assert_equal(splitter.subruns, [1])

    # add another scan index and create a new splitter
    scan_index.addValue('1999-03-31T12:02Z', 0)
    scan_index.addValue('1999-03-31T12:03Z', 2)

    splitter = Splitter({'scan_index': scan_index})

    assert splitter.size == 2
    np.testing.assert_equal(splitter.durations, [60, ONE_DAY])
    np.testing.assert_equal(splitter.subruns, [1, 2])


if __name__ == '__main__':
    pytest.main([__file__])
