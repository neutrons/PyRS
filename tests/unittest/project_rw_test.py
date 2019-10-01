"""
Test for reading and writing components to HiDRA project file
"""
from pyrs.utilities import rs_project_file
import os
import numpy as np


def test_mask(remove_test_file):
    """
    Test methods to read and write mask file
    Parameters
    ----------
    remove_test_file : bool
        Flag to remove temporary testing file

    Returns
    -------
    None
    """
    # Generate a HiDRA project file
    test_project_file = rs_project_file.HydraProjectFile('test.hdf',
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
    verify_project_file = rs_project_file.HydraProjectFile('test.hdf',
                                                           rs_project_file.HydraProjectFileMode.READONLY)

    # Read detector mask & compare
    verify_pixel_mask = verify_project_file.get_mask_detector_array('test')
    assert np.allclose(pixel_mask, verify_pixel_mask, 1.E-12)

    # Read solid angle mask & compare
    verify_solid_mask = verify_project_file.get_mask_solid_angle('test')
    assert np.allclose(solid_mask, verify_solid_mask, 1.E-2)

    # Clean
    if remove_test_file:
        os.remove('test.hdf')

    return


def test_detector_efficiency(remove_test_file):
    """
    Test methods to read and write detector efficiency
    Parameters
    ----------
    remove_test_file :  bool
        Flag to remove temporary testing file

    Returns
    -------
    None
    """
    # Generate a HiDRA project file
    test_project_file = rs_project_file.HydraProjectFile('efficient_testing.hdf',
                                                         rs_project_file.HydraProjectFileMode.OVERWRITE)

    # Create a detector efficiency array
    mock_test_run_number = 12345
    efficient_array = np.random.random_sample(1024**2)

    # Write to file
    test_project_file.set_efficiency_correction(mock_test_run_number, efficient_array)

    # Close file
    test_project_file.close()

    # Open file again
    verify_project_file = rs_project_file.HydraProjectFile('efficient_testing.hdf',
                                                           rs_project_file.HydraProjectFileMode.READONLY)

    # Read detector efficiency & compare
    verify_eff_array = verify_project_file.get_efficiency_correction()

    # Check
    assert np.allclose(efficient_array, verify_eff_array, rtol=1E-12)

    # Clean
    if remove_test_file:
        os.remove('efficient_testing.hdf')

    return


def test_monochromator_setup(remove_test_file):
    """
    Test methods to read and write monochromator setup including
    - calibrated wave length
    - monochromator setting
    Parameters
    ----------
    remove_test_file :  bool
        Flag to remove temporary testing file

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
    if remove_test_file:
        os.remove('')

    return


if __name__ == '__main__':
    test_mask(False)

    test_detector_efficiency(False)

    test_monochromator_setup(False)
>>>>>>> d774ee11c1baa96256262734cebfe30fb5d0b43b
