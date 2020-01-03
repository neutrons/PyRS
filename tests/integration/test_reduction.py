from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.projectfile import HidraProjectFile
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import pytest
import time
import os


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


def test_performance_test():
    """Performance test for converting from NeXus to sub runs

    Returns
    -------

    """
    # Set up
    run_number = 1060
    # Convert the NeXus to project
    nexus_file = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_{}.nxs.h5'.format(run_number)
    if os.path.exists(nexus_file) is False:
        pytest.skip('{} is not available for test performance.'.format(nexus_file))

    project_file = 'HB2B_{}.h5'.format(run_number)

    # Convert
    time_0 = time.time()
    converter = NeXusConvertingApp(nexus_file)
    converter.convert()
    time_1 = time.time()
    converter.save(project_file, None)
    time_2 = time.time()

    print('Converting = {} seconds; Saving = {} seconds'.format(time_1 - time_0, time_2 - time_1))

    assert 'I am ' == '....'


if __name__ == '__main__':
    pytest.main([__file__])
