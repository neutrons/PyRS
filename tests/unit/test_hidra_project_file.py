"""
Test for reading and writing components to HiDRA project file
"""
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
import os
import numpy as np
import datetime
from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction
from pyrs.peaks import PeakCollection
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
    test_project_file = HidraProjectFile('test_mask.hdf', HidraProjectFileMode.OVERWRITE)

    # Create a detector mask
    pixel_mask = np.zeros(shape=(1024**2,), dtype='int')
    pixel_mask += 1
    pixel_mask[123:345] = 0
    pixel_mask[21000:21019] = 0

    # Create a solid angle mask
    solid_mask = np.array([-20, -15, -10, -5, 5, 10, 15, 20])

    # Write detector mask: default and etc
    test_project_file.write_mask_detector_array(None, pixel_mask)
    test_project_file.write_mask_detector_array('test', pixel_mask)

    # Write solid angle mask
    test_project_file.write_mask_solid_angle('test', solid_mask)

    # Close file
    test_project_file.save(True)

    # Open file again
    verify_project_file = HidraProjectFile('test_mask.hdf', HidraProjectFileMode.READONLY)

    # Read detector default mask
    default_pixel_mask = verify_project_file.read_default_masks()
    assert np.allclose(pixel_mask, default_pixel_mask, 1.E-12)

    # Read detector mask & compare
    verify_pixel_mask = verify_project_file.read_mask_detector_array('test')
    assert np.allclose(pixel_mask, verify_pixel_mask, 1.E-12)

    # Test to read all user detector mask
    user_mask_dict = dict()
    verify_project_file.read_user_masks(user_mask_dict)
    assert list(user_mask_dict.keys())[0] == 'test'

    # Read solid angle mask & compare
    verify_solid_mask = verify_project_file.read_mask_solid_angle('test')
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
    test_project_file = HidraProjectFile('test_efficient.hdf', HidraProjectFileMode.OVERWRITE)

    # Create a detector efficiency array
    mock_test_run_number = 12345
    efficient_array = np.random.random_sample(1024**2)

    # Write to file
    test_project_file.write_efficiency_correction(mock_test_run_number, efficient_array)

    # Close file
    test_project_file.close()

    # Open file again
    verify_project_file = HidraProjectFile('test_efficient.hdf', HidraProjectFileMode.READONLY)

    # Read detector efficiency & compare
    verify_eff_array = verify_project_file.read_efficiency_correction()

    # Check
    assert np.allclose(efficient_array, verify_eff_array, rtol=1E-12)

    # Clean
    os.remove('test_efficient.hdf')

    return


def test_wave_length_rw():
    """Test writing and reading for wave length

    Returns
    -------

    """
    # Set up for testing
    test_file_name = 'test_wave_length.h5'
    # Create a detector mask
    gold_wave_length = 1.23456

    # Generate a HiDRA project file
    test_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.OVERWRITE)
    test_project_file.save(True)
    test_project_file.close()

    # Open file
    verify_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.READONLY)

    # Read wave length (not exist)
    wave_length_test = verify_project_file.read_wavelengths()
    assert np.isnan(wave_length_test), 'No wave length read out'

    # Close
    verify_project_file.close()

    # Generate a HiDRA project file
    test_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.READWRITE)

    # Write wave length
    test_project_file.write_wavelength(gold_wave_length)

    # Save and close
    test_project_file.save(True)
    test_project_file.close()

    # Open file again to verify
    verify_project_file2 = HidraProjectFile(test_file_name, HidraProjectFileMode.READONLY)

    # Read wave length (not exist)
    wave_length_test = verify_project_file2.read_wavelengths()
    assert wave_length_test == gold_wave_length

    # Clean
    os.remove(test_file_name)


def test_peak_fitting_result_io():
    """Test peak fitting result's writing and reading

    Returns
    -------

    """
    # Generate a unique test file
    now = datetime.datetime.now()
    test_file_name = 'test_peak_io_{}.hdf'.format(now.toordinal())

    # Generate a HiDRA project file
    test_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.OVERWRITE)

    # Create a ND array for output parameters
    param_names = PeakShape.PSEUDOVOIGT.native_parameters + BackgroundFunction.LINEAR.native_parameters
    data_type = list()
    for param_name in param_names:
        data_type.append((param_name, np.float32))
    test_error_array = np.zeros(3, dtype=data_type)
    test_params_array = np.zeros(3, dtype=data_type)

    for i in range(3):
        # sub run
        for j, par_name in enumerate(param_names):
            test_params_array[par_name][i] = 2**i + 0.1 * 3**j
            test_error_array[par_name][i] = np.sqrt(abs(test_params_array[par_name][i]))
    # END-FOR
    chi2_array = np.array([0.323, 0.423, 0.523])

    # Add test data to output
    peaks = PeakCollection('test fake', PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR)
    peaks.set_peak_fitting_values(np.array([1, 2, 3]), test_params_array, test_error_array,
                                  chi2_array)

    test_project_file.write_peak_parameters(peaks)

    test_project_file.save(False)

    # Check
    assert os.path.exists(test_file_name), 'Test project file for peak fitting result {} cannot be found.' \
                                           ''.format(test_file_name)
    print('[INFO] Peak parameter test project file: {}'.format(test_file_name))

    # Import
    verify_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.READONLY)

    # get the tags
    peak_tags = verify_project_file.read_peak_tags()
    assert 'test fake' in peak_tags
    assert len(peak_tags) == 1

    # get the parameter of certain
    peak_info = verify_project_file.read_peak_parameters('test fake')

    # peak profile
    assert peak_info.peak_profile == str(PeakShape.PSEUDOVOIGT)
    assert peak_info.background_type == str(BackgroundFunction.LINEAR)

    # sub runs
    assert np.allclose(peak_info.sub_runs, np.array([1, 2, 3]))

    # parameter values
    # print('DEBUG:\n  Expected: {}\n  Found: {}'.format(test_params_array, peak_info[3]))
    peak_values, peak_errors = peak_info.get_native_params()
    assert_allclose_structured_numpy_arrays(test_params_array, peak_values)
    # np.testing.assert_allclose(peak_info[3], test_params_array, atol=1E-12)

    # parameter values
    # assert np.allclose(peak_info[4], test_error_array, 1E-12)
    assert_allclose_structured_numpy_arrays(test_error_array, peak_errors)

    # Clean
    os.remove(test_file_name)

    return


def test_strain_io():
    """Test PeakCollection writing and reading with *D reference*

    Returns
    -------

    """
    # Generate a unique test file
    now = datetime.datetime.now()
    test_file_name = 'test_strain_io_{}.h5'.format(now.toordinal())
    test_ref_d = 1.23454321
    test_ref_d2 = np.array([1.23, 1.24, 1.25])
    peak_tag = 'Fake Peak D'
    peak_tag_2 = 'Fake Peak D Diff'

    # Generate a HiDRA project file
    test_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.OVERWRITE)

    # Create a ND array for output parameters
    param_names = PeakShape.PSEUDOVOIGT.native_parameters + BackgroundFunction.LINEAR.native_parameters
    data_type = list()
    for param_name in param_names:
        data_type.append((param_name, np.float32))
    test_error_array = np.zeros(3, dtype=data_type)
    test_params_array = np.zeros(3, dtype=data_type)

    for i in range(3):
        # sub run
        for j, par_name in enumerate(param_names):
            test_params_array[par_name][i] = 2**i + 0.1 * 3**j
            test_error_array[par_name][i] = np.sqrt(abs(test_params_array[par_name][i]))
    # END-FOR
    chi2_array = np.array([0.323, 0.423, 0.523])

    # Add test data to output
    peaks = PeakCollection(peak_tag, PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR)
    peaks.set_peak_fitting_values(np.array([1, 2, 3]), test_params_array, test_error_array,
                                  chi2_array)
    peaks.set_d_reference(test_ref_d)

    # Add 2nd peak
    peaks2 = PeakCollection(peak_tag_2, PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR)
    peaks2.set_peak_fitting_values(np.array([1, 2, 3]), test_params_array, test_error_array,
                                   chi2_array)
    peaks2.set_d_reference(test_ref_d2)

    # Write
    test_project_file.write_peak_parameters(peaks)
    test_project_file.write_peak_parameters(peaks2)
    # Save
    test_project_file.save(verbose=False)

    # Verify
    assert os.path.exists(test_file_name), 'Test project file for peak fitting result {} cannot be found.' \
                                           ''.format(test_file_name)

    # import
    verify_project_file = HidraProjectFile(test_file_name, HidraProjectFileMode.READONLY)

    # check tags
    peak_tags = verify_project_file.read_peak_tags()
    assert peak_tag in peak_tags and peak_tag_2 in peak_tags
    assert len(peak_tags) == 2

    # Get d-reference of peak 1 to check
    peak_info = verify_project_file.read_peak_parameters(peak_tag)
    verify_d_ref = peak_info.get_d_reference()
    gold_ref_d = np.array([test_ref_d] * 3)
    np.testing.assert_allclose(verify_d_ref, gold_ref_d)

    # Get d-reference of peak 2 to check
    peak_info2 = verify_project_file.read_peak_parameters(peak_tag_2)
    verify_d_ref_2 = peak_info2.get_d_reference()
    np.testing.assert_allclose(verify_d_ref_2, test_ref_d2)

    # Clean
    os.remove(test_file_name)

    return


if __name__ == '__main__':
    pytest.main([__file__])
