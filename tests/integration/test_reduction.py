from pyrs.utilities import rs_project_file
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
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
    project_file = rs_project_file.HydraProjectFile(project_file_name, rs_project_file.HydraProjectFileMode.READONLY)

    # Reduce
    test_workspace = workspaces.HidraWorkspace('test calibration')
    test_workspace.load_hidra_project(project_file, load_raw_counts=True, load_reduced_diffraction=False)

    return


if __name__ == '__main__':
    pytest.main([__file__])
