from pyrs.projectfile import HidraProjectFile
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor
from pyrs.core.powder_pattern import ReductionApp
import os
import pytest


def test_calibration_json():
    """Test reduce data with calibration file (.json)

    Returns
    -------
    None
    """
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


def test_log_time_average():
    """Test the log time average calculation

    Returns
    -------

    """
    nexus_file_name = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5'
    if os.path.exists(nexus_file_name) is False:
        pytest.skip('File {} is not accessible'.format(nexus_file_name))

    processor = NexusProcessor(nexus_file_name)

    sub_run_times, sub_run_numbers = processor.get_sub_run_times_value()

    # sample log (TSP)
    tsp_sample_logs = processor.split_sample_logs(sub_run_times, sub_run_numbers)
    # sample log (PyRS)
    pyrs_sample_logs = processor.split_sample_logs_prototype(sub_run_times, sub_run_numbers)

    # compare some
    for log_name in ['2theta', 'DOSC']:
        np.testing.assert_allclose(tsp_sample_logs[log_name], pyrs_sample_logs[log_name])


@pytest.mark.parametrize('nexus_file_name, mask_file_name',
                         [('/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5', 'data/HB2B_Mask_12-18-19.xml'),
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5', None)],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_compare_nexus_reader(nexus_file_name, mask_file_name):
    """Verify NeXus converters including counts and sample log values

    Returns
    -------

    """
    from matplotlib import pyplot as plt

    # Test on a light weight NeXus
    if os.path.exists(nexus_file_name) is False:
        pytest.skip('Unable to access test file {}'.format(nexus_file_name))

    # reduce with Mantid
    nexus_mtd_converter = NeXusConvertingApp(nexus_file_name, mask_file_name=mask_file_name)
    hidra_mtd_ws = nexus_mtd_converter.convert(use_mantid=True)

    # reduce with PyRS/Python
    nexus_pyrs_converter = NeXusConvertingApp(nexus_file_name, mask_file_name=mask_file_name)
    hidra_pyrs_ws = nexus_pyrs_converter.convert(use_mantid=False)

    # compare sub runs
    sub_runs_mtd = hidra_mtd_ws.get_sub_runs()
    sub_run_pyrs = hidra_pyrs_ws.get_sub_runs()
    np.testing.assert_allclose(sub_runs_mtd, sub_run_pyrs)

    # compare counts
    for sub_run in sub_runs_mtd:
        mtd_counts = hidra_mtd_ws.get_detector_counts(sub_run)
        pyrs_counts = hidra_pyrs_ws.get_detector_counts(sub_run)

        diff_counts = mtd_counts - pyrs_counts
        print('Sub Run {}: Mantid counts = {}, PyRS counts = {}\n'
              'Number of pixels with different counts = {}.  Maximum difference = {}'
              ''.format(sub_run, np.sum(mtd_counts), np.sum(pyrs_counts),
                        len(np.where(diff_counts != 0)[0]), np.max(np.abs(diff_counts))))
        np.testing.assert_allclose(mtd_counts, pyrs_counts)

    # Test reduction to diffraction pattern
    # instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
    # hidra_mtd_ws.set_instrument_geometry(instrument)
    reducer_mtd = ReductionApp(False)
    reducer_mtd.load_hidra_workspace(hidra_mtd_ws)
    reducer_mtd.reduce_data(sub_runs=None,
                            instrument_file=None,
                            calibration_file=None,
                            mask=None)
    vec_mtd_x, vec_mtd_y = reducer_mtd.get_diffraction_data(1)

    # hidra_pyrs_ws.set_instrument_geometry(instrument)
    reducer_pyrs = ReductionApp(False)
    reducer_pyrs.load_hidra_workspace(hidra_pyrs_ws)
    reducer_pyrs.reduce_data(sub_runs=None,
                             instrument_file=None,
                             calibration_file=None,
                             mask=None)
    vec_rs_x, vec_rs_y = reducer_pyrs.get_diffraction_data(1)

    plt.plot(vec_mtd_x, vec_mtd_y)
    plt.plot(vec_rs_x, vec_rs_y, color='red')
    plt.show()

    for sub_run in [1, 2, 3]:
        print('Comparing pattern of sub run {}'.format(sub_run))
        vec_mtd_x, vec_mtd_y = reducer_mtd.get_diffraction_data(sub_run)
        vec_rs_x, vec_rs_y = reducer_pyrs.get_diffraction_data(sub_run)
        np.testing.assert_allclose(vec_mtd_y, vec_rs_y)

    # compare number of sample logs
    log_names_mantid = hidra_mtd_ws.get_sample_log_names()
    log_names_pyrs = hidra_pyrs_ws.get_sample_log_names()

    print('Diff set Mantid vs PyRS: {} ... {}'
          ''.format(set(log_names_mantid) - set(log_names_pyrs),
                    set(log_names_pyrs) - set(log_names_mantid)))
    if len(log_names_mantid) != len(log_names_pyrs):
        raise AssertionError('Sample logs entries are not same')

    not_compare_output = ''
    # compare sample log values
    for log_name in log_names_pyrs:
        if log_name == 'pulse_flags':
            continue

        mtd_log_values = hidra_mtd_ws.get_sample_log_values(log_name)
        pyrs_log_values = hidra_pyrs_ws.get_sample_log_values(log_name)

        if str(pyrs_log_values.dtype).count('float') == 0 and str(pyrs_log_values.dtype).count('int') == 0:
            # Not float or integer: cannot be compared
            not_compare_output += '{}: Mantid {} vs PyRS {}\n'.format(log_name, mtd_log_values, pyrs_log_values)
        else:
            # Int or float: comparable
            log_diff = np.sqrt(np.sum((mtd_log_values - pyrs_log_values) ** 2))
            print('{}: sum(diff) = {}, size = {}'.format(log_name, log_diff, len(mtd_log_values)))
            try:
                np.testing.assert_allclose(mtd_log_values, pyrs_log_values)
                print()
            except AssertionError as ass_err:
                print('........... Not matched!: \n{}'.format(ass_err))
                print('----------------------------------------------------\n')
    # END-FOR

    print(not_compare_output)


if __name__ == '__main__':
    pytest.main([__file__])
