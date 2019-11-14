"""
Test for reading and writing components to HiDRA project file
"""
from pyrs.utilities.rs_project_file import HidraConstants, HydraProjectFile, HidraProjectFileMode
import os
import numpy as np
import datetime
from pyrs.core import peak_profile_utility
import pytest


def assert_allclose_structured_numpy_arrays(expected, calculated):
    if expected.dtype.names != calculated.dtype.names:
        raise AssertionError('{} and {} do not match'.format(expected.dtype.names, calculated.dtype.names))

    for name in expected.dtype.names:
        if not np.allclose(expected[name], calculated[name], atol=1E-10):
            raise AssertionError('{}: Not same\nExpected: {}\nCalculated: {}'
                                 ''.format(name, expected[name], calculated[name]))

    return


def test_mask():
    """Test methods to read and write mask file

    Returns
    -------
    None
    """
    # Generate a HiDRA project file
    test_project_file = HydraProjectFile('test_mask.hdf', HidraProjectFileMode.OVERWRITE)

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
    verify_project_file = HydraProjectFile('test_mask.hdf', HidraProjectFileMode.READONLY)

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
    test_project_file = HydraProjectFile('test_efficient.hdf', HidraProjectFileMode.OVERWRITE)

    # Create a detector efficiency array
    mock_test_run_number = 12345
    efficient_array = np.random.random_sample(1024**2)

    # Write to file
    test_project_file.set_efficiency_correction(mock_test_run_number, efficient_array)

    # Close file
    test_project_file.close()

    # Open file again
    verify_project_file = HydraProjectFile('test_efficient.hdf', HidraProjectFileMode.READONLY)

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


def test_peak_fitting_result_io():
    """Test peak fitting result's writing and reading

    Returns
    -------

    """
    # Generate a unique test file
    now = datetime.datetime.now()
    test_file_name = 'test_peak_io_{}.hdf'.format(now.toordinal())

    # Generate a HiDRA project file
    test_project_file = HydraProjectFile(test_file_name, HidraProjectFileMode.OVERWRITE)

    # Create a ND array for output parameters
    data_type = list()
    for param_name in peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS:
        data_type.append((param_name, np.float32))
    test_error_array = np.zeros(3, dtype=data_type)
    data_type.append((HidraConstants.PEAK_FIT_CHI2, np.float32))
    test_params_array = np.zeros(3, dtype=data_type)

    for i in range(3):
        # sub run
        for j, par_name in enumerate(peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS):
            test_params_array[par_name][i] = 2**i + 0.1 * 3**j
            test_error_array[par_name][i] = np.sqrt(abs(test_params_array[par_name][i]))
    # END-FOR
    chi2_array = np.array([0.323, 0.423, 0.523])
    test_params_array[HidraConstants.PEAK_FIT_CHI2] = chi2_array

    # Add test data to output
    test_project_file.set_peak_fit_result(peak_tag='test fake',
                                          peak_profile='PseudoVoigt',
                                          background_type='Linear',
                                          sub_run_vec=np.array([1, 2, 3]),
                                          fit_cost_array=chi2_array,
                                          param_value_array=test_params_array,
                                          param_error_array=test_error_array)

    test_project_file.save_hydra_project(False)

    # Check
    assert os.path.exists(test_file_name), 'Test project file for peak fitting result {} cannot be found.' \
                                           ''.format(test_file_name)
    print('[INFO] Peak parameter test project file: {}'.format(test_file_name))

    # Import
    verify_project_file = HydraProjectFile(test_file_name, HidraProjectFileMode.READONLY)

    # get the tags
    peak_tags = verify_project_file.get_peak_tags()
    assert 'test fake' in peak_tags
    assert len(peak_tags) == 1

    # get the parameter of certain
    peak_info = verify_project_file.get_peak_parameters('test fake')

    # peak profile
    assert peak_info[0] == 'PseudoVoigt'
    assert peak_info[1] == 'Linear'

    # sub runs
    assert np.allclose(peak_info[2], np.array([1, 2, 3]))

    # parameter values
    # print('DEBUG:\n  Expected: {}\n  Found: {}'.format(test_params_array, peak_info[3]))
    assert_allclose_structured_numpy_arrays(test_params_array, peak_info[4])
    # np.testing.assert_allclose(peak_info[3], test_params_array, atol=1E-12)

    # parameter values
    # assert np.allclose(peak_info[4], test_error_array, 1E-12)
    assert_allclose_structured_numpy_arrays(test_error_array, peak_info[5])

    # Clean
    os.remove(test_file_name)

    return


if __name__ == '__main__':
    pytest.main()
