from pyrs.projectfile import HidraProjectFile
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
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


@pytest.mark.parametrize('nexus_file_name, mask_file_name',
                         [('data/HB2B_938.nxs.h5', None),
                          ('data/HB2B_938_nxs.h5', 'data/xxx.xml')],
                         ids=('HB2B_938_NoMask', 'HB2B_938_Masked'))
def test_compare_nexus_reader(nexus_file_name, mask_file_name):
    """Verify NeXus converters including counts and sample log values

    Returns
    -------

    """
    # Test on a light weight NeXus
    if mask_file_name is not None:
        pytest.pytest.skip('Masking with H5PY/PYTHON NeXus conversion has not been implemented.')

    # reduce with Mantid
    nexus_converter = NeXusConvertingApp(nexus_file_name, mask_file_name=mask_file_name)
    hidra_mtd_ws = nexus_converter.convert(use_mantid=True)

    # reduce with PyRS/Python
    hidra_pyrs_ws = nexus_converter.convert(use_mantid=False)

    # compare sub runs
    sub_runs_mtd = hidra_mtd_ws.get_sub_runs()
    sub_run_pyrs = hidra_pyrs_ws.get_sub_runs()
    np.testing.assert_allclose(sub_runs_mtd, sub_run_pyrs)

    # compare counts
    for sub_run in sub_runs_mtd:
        mtd_counts = hidra_mtd_ws.get_detector_counts(sub_run)
        pyrs_counts = hidra_pyrs_ws.get_detector_counts(sub_run)
        np.testing.assert_allclose(mtd_counts, pyrs_counts)

    # compare number of sample logs
    log_names_mantid = hidra_mtd_ws.get_sample_log_names()
    log_names_pyrs = hidra_pyrs_ws.get_sample_log_names()
    np.testing.assert_allclose(log_names_mantid, log_names_pyrs)

    # compare sample log values
    for log_name in log_names_pyrs:
        mtd_log_values = hidra_mtd_ws.get_sample_log_values(log_name)
        pyrs_log_values = hidra_pyrs_ws.get_sample_log_values(log_name)
        np.testing.assert_allclose(mtd_log_values, pyrs_log_values)


if __name__ == '__main__':
    pytest.main([__file__])
