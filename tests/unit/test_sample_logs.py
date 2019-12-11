import numpy as np
import pytest
from pyrs.dataobjects import SampleLogs
from pyrs.core.nexus_conversion import calculate_log_time_average


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

    # names of logs
    assert sorted(sample.plottable_logs()) == ['constant1', 'sub-runs', 'variable1']
    assert sample.constant_logs() == ['constant1']

    # slicing
    np.testing.assert_equal(sample['variable1'], [0., 25., 50., 75., 100.])
    np.testing.assert_equal(sample['variable1', 3], [50.])
    np.testing.assert_equal(sample['variable1', [1, 2, 3]], [0., 25., 50.])

    with pytest.raises(IndexError):
        np.testing.assert_equal(sample['variable1', [0]], [0., 50., 75., 100.])
    with pytest.raises(IndexError):
        np.testing.assert_equal(sample['variable1', [10]], [0., 50., 75., 100.])


def test_time_average():
    """Test method to do time average

    Example from 1086

    Returns
    -------

    """
    # sub run 1
    log1_times = np.array([1.5742809368379436e+18, 1.574280940122952e+18,
                           1.5742809402252196e+18, 1.5742809634411576e+18])
    log1_value = np.array([90.995, 81.9955, 81.99525, 84.])

    # splitter
    splitter1_times = np.array([1.5742809368379436e+18, 1.5742809601445215e+18])
    splitter1_value = np.array([1, 0])

    log1_average = calculate_log_time_average(log1_times, log1_value, splitter1_times, splitter1_value)
    assert abs(log1_average - 83.26374511281652) < 1E-10, '{} shall be close to {}' \
                                                          ''.format(log1_average, 83.26374511281652)

    # sub run 2
    log2_times = np.array([1.5742809634411576e+18, 1.5742809868603507e+18])
    log2_vlaue = np.array([84., 86.00025])

    splitter2_times = np.array([1.5742809368379436e+18, 1.574280963555972e+18, 1.574280983560895e+18])
    splitter2_value = np.array([0, 1, 0])
    log2_average = calculate_log_time_average(log2_times, log2_vlaue, splitter2_times, splitter2_value)
    assert abs(log2_average - 84. < 1E-10), '{} shall be close to 84'.format(log2_average)


if __name__ == '__main__':
    pytest.main([__file__])
