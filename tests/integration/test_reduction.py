from pyrs.utilities import rs_project_file
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
import pytest


def assert_delta(exp_value, test_value, delta_value, param_name):
    """Check whether two values are close enough.

    Exception: AssertionError

    Parameters
    ----------
    exp_value: float/ndarray
        expected value
    test_value: float/ndarray
        test value
    delta_value: float/ndarray
        allowed difference
    param_name: str
        parameter name

    Returns
    -------
    None
    """
    if np.abs(exp_value - test_value) < delta_value:
        raise AssertionError('Parameter {} value {} is different from expected value {} beyond allowed value {}'
                             ''.format(param_name, test_value, exp_value, delta_value))

    return


def test_calibration_json():
    """Test reduce data with calibration file (.json)

    Returns
    -------
    None
    """
    # Get simulated test data
    project_file_name = 'data/???.hdf'
    calib_file = 'data/???.json'

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
    project_file = rs_project_file.HydraProjectFile(project_file_name, rs_project_file.HydraProjectFileMode.READONLY)

    # Reduce
    test_workspace = workspaces.HidraWorkspace('test calibration')
    test_workspace.load_hidra_project(project_file, load_raw_counts=True, load_reduced_diffraction=False)
    # test_workspace.reduce()

    return


if __name__ == '__main__':
    pytest.main()
