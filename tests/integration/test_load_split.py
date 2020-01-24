from pyrs.projectfile import HidraProjectFile
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor
from pyrs.core.powder_pattern import ReductionApp
import os
import pytest

DIAGNOSTIC_PLOTS = False
FILE_1017 = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5'


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
    assert wave_length
    assert wl_error
    assert status == 3

    # Import project file
    project_file = HidraProjectFile(project_file_name, 'r')

    # Reduce
    test_workspace = workspaces.HidraWorkspace('test calibration')
    test_workspace.load_hidra_project(project_file, load_raw_counts=True, load_reduced_diffraction=False)


@pytest.mark.skipif(not os.path.exists(FILE_1017), reason='File {} is not accessible'.format(FILE_1017))
def test_log_time_average():
    """Test the log time average calculation"""
    processor = NexusProcessor(FILE_1017)

    sub_run_times, sub_run_numbers = processor.get_sub_run_times_value()

    # verify splitting information
    # 2theta is the motor that determines the first start time
    exp_times = np.array(['2019-11-10T16:31:02.645235328-0500', '2019-11-10T16:41:02.771317813-0500',   # scan_index=1
                          '2019-11-10T16:41:14.238680196-0500', '2019-11-10T17:11:14.249705287-0500',   # scan_index=2
                          '2019-11-10T17:11:33.208056929-0500', '2019-11-10T17:31:33.218615767-0500'],  # scan_index=3
                         dtype='datetime64[ns]')
    np.testing.assert_equal(sub_run_numbers, [1, 2, 3], err_msg='subrun numbers')
    np.testing.assert_equal(sub_run_times, exp_times, err_msg='subrun filtering')

    # split the sample logs
    sample_logs = processor.split_sample_logs(sub_run_times, sub_run_numbers)

    # verify two of the properties
    np.testing.assert_allclose(sample_logs['2theta'], [69.99525,  80.,  97.50225])
    np.testing.assert_allclose(sample_logs['DOSC'], [-0.01139306,  0.00332028,  0.00635049], rtol=1.e-5)


@pytest.mark.skipif(not os.path.exists(FILE_1017), reason='File {} is not accessible'.format(FILE_1017))
@pytest.mark.parametrize('mask_file_name, filtered_counts, histogram_counts',
                         [('data/HB2B_Mask_12-18-19.xml', (540461, 1635432, 1193309), (510.8, 1555.7, 1136.3)),
                          (None, (548953, 1661711, 1212586), (518.7, 1580.5, 1154.4))],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_reduce_data(mask_file_name, filtered_counts, histogram_counts):
    """Verify NeXus converters including counts and sample log values"""
    SUBRUNS = (1, 2, 3)
    CENTERS = (69.99525,  80.,  97.50225)

    # reduce with PyRS/Python
    nexus_converter = NeXusConvertingApp(FILE_1017, mask_file_name=mask_file_name)
    hidra_ws = nexus_converter.convert()

    # verify subruns
    np.testing.assert_equal(hidra_ws.get_sub_runs(), SUBRUNS)

    for sub_run, total_counts in zip(hidra_ws.get_sub_runs(), filtered_counts):
        counts_array = hidra_ws.get_detector_counts(sub_run)
        np.testing.assert_equal(counts_array.shape, (1048576,))
        assert np.sum(counts_array) == total_counts, 'mismatch in subrun={} for filtered data'.format(sub_run)

    # Test reduction to diffraction pattern
    reducer = ReductionApp(False)
    reducer.load_hidra_workspace(hidra_ws)
    reducer.reduce_data(sub_runs=None,
                        instrument_file=None,
                        calibration_file=None,
                        mask=None)

    # plot the patterns
    if DIAGNOSTIC_PLOTS:
        from matplotlib import pyplot as plt
        for sub_run, angle in zip(SUBRUNS, CENTERS):
            x, y = reducer.get_diffraction_data(sub_run)
            plt.plot(x, y, label='SUBRUN {} at {:.1f} deg'.format(sub_run, angle))
        plt.legend()
        plt.show()

    # check ranges and total counts
    for sub_run, angle, total_counts in zip(SUBRUNS, CENTERS, histogram_counts):
        assert_label = 'mismatch in subrun={} for histogrammed data'.format(sub_run)
        x, y = reducer.get_diffraction_data(sub_run)
        assert x[0] < angle < x[-1], assert_label
        assert np.isnan(np.sum(y)), assert_label
        np.testing.assert_almost_equal(np.nansum(y), total_counts, decimal=1, err_msg=assert_label)

    # TODO add checks for against golden version


if __name__ == '__main__':
    pytest.main([__file__])
