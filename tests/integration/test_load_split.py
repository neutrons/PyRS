from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from pyrs.projectfile import HidraProjectFile
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor
import os
import pytest

FILE_1017 = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.ORIG.nxs.h5'


def test_calibration_json():
    """Test reduce data with calibration file (.json)"""
    # Get simulated test data
    project_file_name = 'data/HB2B_000.h5'
    calib_file = 'data/HB2B_CAL_Si333.json'

    # Import file
    calib_obj = calibration_file_io.read_calibration_json_file(calib_file)
    shift, shift_error, wave_length, wl_error, status = calib_obj

    # Verify result
    assert shift
    assert shift_error
    assert abs(wave_length - 1.4499332864) < 1E-8
    assert wl_error
    assert status == 3

    # Import project file
    project_file = HidraProjectFile(project_file_name, 'r')

    # Reduce
    test_workspace = workspaces.HidraWorkspace('test calibration')
    test_workspace.load_hidra_project(project_file, load_raw_counts=True, load_reduced_diffraction=False)

    # Test set and get wave length
    test_workspace.set_wavelength(wave_length, True)
    wave_length_i = test_workspace.get_wavelength(True, True)
    assert wave_length_i == wave_length


@pytest.mark.skipif(not os.path.exists(FILE_1017), reason='File {} is not accessible'.format(FILE_1017))
def test_log_time_average():
    """Test the log time average calculation"""
    processor = NexusProcessor(FILE_1017)

    sub_run_times, sub_run_numbers = processor._splitter.times, processor._splitter.subruns

    # verify splitting information
    # 2theta is the motor that determines the first start time
    exp_times = np.array(['2019-11-10T16:31:02.645235328-0500', '2019-11-10T16:41:02.771317813-0500',   # scan_index=1
                          '2019-11-10T16:41:14.238680196-0500', '2019-11-10T17:11:14.249705287-0500',   # scan_index=2
                          '2019-11-10T17:11:33.208056929-0500', '2019-11-10T17:31:33.218615767-0500'],  # scan_index=3
                         dtype='datetime64[ns]')
    np.testing.assert_equal(sub_run_numbers, [1, 2, 3], err_msg='subrun numbers')
    np.testing.assert_equal(sub_run_times, exp_times, err_msg='subrun filtering')

    # previous calculations
    exp_durations = np.array([600., 1800., 1200.])
    np.testing.assert_almost_equal(processor._splitter.durations, exp_durations, decimal=0)

    # split the sample logs
    sample_logs = processor.split_sample_logs()

    # verify two of the properties
    np.testing.assert_allclose(sample_logs['2theta'], [69.99525,  80.,  97.50225])


if __name__ == '__main__':
    pytest.main([__file__])
