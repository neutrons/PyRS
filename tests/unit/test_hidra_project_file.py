"""
Test for reading and writing components to HiDRA project file
"""
from pyrs.utilities import rs_project_file
import os
import numpy as np
import datetime
from pyrs.core import peak_profile_utility
import pytest


def test_mask():
    """Test methods to read and write mask file

    Returns
    -------
    None
    """
    # Generate a HiDRA project file
    test_project_file = rs_project_file.HydraProjectFile('test_mask.hdf',
                                                         rs_project_file.HydraProjectFileMode.OVERWRITE)

    # Create a detector mask
    pixel_mask = np.zeros(shape=(1024**2,), dtype='int')
    pixel_mask += 1
    pixel_mask[123:345] = 0
    pixel_mask[21000:21019] = 0

    # Create a solid angle mask
    solid_mask = np.array([-20, -15, -10, -5, 5, 10, 15, 20])

    # Write detector mask
    test_project_file.add_mask_detector_array('test', pixel_mask)

    # Write solid angle mask
    test_project_file.add_mask_solid_angle('test', solid_mask)

    # Close file
    test_project_file.save_hydra_project(True)

    # Open file again
    verify_project_file = rs_project_file.HydraProjectFile('test_mask.hdf',
                                                           rs_project_file.HydraProjectFileMode.READONLY)

    # Read detector mask & compare
    verify_pixel_mask = verify_project_file.get_mask_detector_array('test')
    assert np.allclose(pixel_mask, verify_pixel_mask, 1.E-12)

    # Read solid angle mask & compare
    verify_solid_mask = verify_project_file.get_mask_solid_angle('test')
    assert np.allclose(solid_mask, verify_solid_mask, 1.E-2)

    # Clean
    os.remove('test_mask.hdf')

    return


def test_detector_efficiency():
    """
    Test methods to read and write detector efficiency

    Returns
    -------
    None
    """
    # Generate a HiDRA project file
    test_project_file = rs_project_file.HydraProjectFile('test_efficient.hdf',
                                                         rs_project_file.HydraProjectFileMode.OVERWRITE)

    # Create a detector efficiency array
    mock_test_run_number = 12345
    efficient_array = np.random.random_sample(1024**2)

    # Write to file
    test_project_file.set_efficiency_correction(mock_test_run_number, efficient_array)

    # Close file
    test_project_file.close()

    # Open file again
    verify_project_file = rs_project_file.HydraProjectFile('test_efficient.hdf',
                                                           rs_project_file.HydraProjectFileMode.READONLY)

    # Read detector efficiency & compare
    verify_eff_array = verify_project_file.get_efficiency_correction()

    # Check
    assert np.allclose(efficient_array, verify_eff_array, rtol=1E-12)

    # Clean
    os.remove('test_efficient.hdf')

    return


def next_test_monochromator_setup():
    """
    Test methods to read and write monochromator setup including

    Returns
    -------
    None
    """
    # Generate a HiDRA project file

    # Specify monochromator setting

    # Write to file

    # Specify calibrated wave length

    # Write to file

    # Close file

    # Open file again

    # Read monochromator setting & compare

    # Read calibrated wave length & compare

    # Clean
    os.remove('')

    return


# TODO - #89 - Peak fitting result R/W
def X_test_peak_fitting_result_io():
    """

    Returns
    -------

    """
    # Generate a unique test file
    now = datetime.datetime.now()
    test_file_name = 'test_peak_io_{}.hdf'.format(now.toordinal())

    # Generate a HiDRA project file
    test_project_file = rs_project_file.HydraProjectFile(test_file_name,
                                                         rs_project_file.HydraProjectFileMode.OVERWRITE)

    # Create a ND array for output parameters
    test_params_array = np.zeros((3, len(peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS)), float)
    for i in range(3):
        for j in range(len(peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS)):
            test_params_array[i, j] = 2**i + 0.1 * 3**j
    # END-FOR

    # Add test data to output
    test_project_file.set_peak_fit_result(peak_tag='test fake',
                                          peak_profile='PseudoVoigt',
                                          peak_param_names=peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS,
                                          sub_run_vec=np.array([1, 2, 3]),
                                          chi2_vec=np.array([0.323, 0.423, 0.523]),
                                          peak_params=test_params_array)

    test_project_file.save_hydra_project(False)

    # Check
    assert os.path.exists(test_file_name), 'Test project file for peak fitting result {} cannot be found.' \
                                           ''.format(test_file_name)
    print('[INFO] Peak parameter test project file: {}'.format(test_file_name))

    # Import
    verify_project_file = rs_project_file.HydraProjectFile(test_file_name,
                                                           rs_project_file.HydraProjectFileMode.READONLY)
    assert verify_project_file

    # TODO - NEXT Need to make the result to work
    # peaks = verify_project_file.get_peak_fit_result(peak_tag='test fake')

    # Then compare....

    os.remove(test_file_name)

    return


if __name__ == '__main__':
    pytest.main()
