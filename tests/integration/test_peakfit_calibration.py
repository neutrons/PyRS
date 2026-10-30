# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
import pytest
import time
import os
import json
import numpy as np
from pyrs.calibration import mantid_peakfit_calibration


try:
    from scipy.optimize import least_squares
except ImportError as e:
    least_squares = str(e)  # import failed exception explains why

pytestmark = pytest.mark.integration


def are_equivalent_jsons(test_json_name, gold_json_name, atol):
    """Print out the difference of two JSON files

    Parameters
    ----------
    test_json_name
    gold_json_name
    atol : float
        absolute tolerance

    Returns
    -------
    boolean
        equivalent or not

    """
    # Load file 1
    with open(test_json_name, "r") as json1:
        test_json_dict = json.load(json1)

    # Load file 2
    with open(gold_json_name, "r") as json2:
        gold_json_dict = json.load(json2)

    # Compare
    diff = False
    for key in gold_json_dict.keys():
        # expect a file with same key
        if key not in test_json_dict:
            print(key)
            diff = True
            break

        # compare value
        gold_value = float(gold_json_dict[key])
        test_value = float(test_json_dict[key])
        print(abs(gold_value - test_value))
        if isinstance(gold_value, float):
            # float: check with tolerance
            diff = abs(gold_value - test_value) > atol
            print(abs(gold_value - test_value))
        else:
            # int or str: must be exactly same
            diff = gold_value != test_value
            print(gold_value, test_value)
        if diff:
            # quit loop if not same
            break
    # END-FOR

    return not diff


def print_out_json_diff(json_file1_name, json_file2_name):
    """Print out the difference of two JSON files

    Parameters
    ----------
    json_file1_name
    json_file2_name

    Returns
    -------

    """
    # Load file 1
    with open(json_file1_name, "r") as json1:
        json_dict1 = json.load(json1)

    # Load file 2
    with open(json_file2_name, "r") as json2:
        json_dict2 = json.load(json2)

    # Output difference
    if set(json_dict1.keys()) != set(json_dict2.keys()):
        # Compare keys
        print(
            "[JSON Keys are different]\n{}: {}\n{}: {}".format(
                json_file1_name, sorted(json_dict1.keys()), json_file2_name, sorted(json_dict2.keys())
            )
        )

    else:
        # Compare values
        keys = sorted(json_dict1.keys())
        print("[JSON Value are different]\nField\t{}\t{}".format(json_file1_name, json_file2_name))
        for k in keys:
            print("{}\t{}\t{}".format(k, json_dict1[k], json_dict2[k]))

    return


def test_wavelength():
    """Main test for the script

    Returns
    -------

    """
    # Define Fitting Routine
    nexus_file = "tests/data/calibration_tests/HB2B_3510.nxs.h5"

    goldfile = "tests/data/calibration_tests/HB2B_mantid_calib_lambda.json"

    t_start = time.time()

    # Initalize calibration
    calibrator = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file)

    # Setup test constraints
    calibrator._keep_subrun_list = [False] * calibrator.sy.size
    calibrator._keep_subrun_list[0] = True
    calibrator.set_max_nfev(3)

    # Calibrate
    calibrator.calibrate_wave_length()

    calibrator._calibstatus = 3

    calibrator.print_calibration()

    # write out
    if os.path.exists("HB2B_CAL_Test.json"):
        os.remove("HB2B_CAL_Test.json")
    file_name = os.path.join(os.getcwd(), "HB2B_CAL_Test.json")
    calibrator.write_calibration(file_name)

    t_stop = time.time()
    print("Total Time: {}".format(t_stop - t_start))

    # Compare output file with gold file for test
    if are_equivalent_jsons(goldfile, file_name, atol=5e-3):
        # Same: remove file generated in test
        os.remove(file_name)
    else:
        print_out_json_diff(goldfile, "HB2B_CAL_Test.json")
        assert False, "Test output {} is different from gold file {}".format(file_name, goldfile)

    return


def test_all_refinements():
    """Main test for the script

    Returns
    -------

    """
    # Define Fitting Routine
    nexus_file = "tests/data/calibration_tests/HB2B_3510.nxs.h5"

    goldfile = "tests/data/calibration_tests/HB2B_mantid_calib.json"

    t_start = time.time()

    # Initalize calibration
    calibrator = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file)

    # Setup test constraints
    calibrator._keep_subrun_list = [False] * calibrator.sy.size
    calibrator._keep_subrun_list[0] = True
    calibrator.set_max_nfev(1)

    # Calibrate
    calibrator.CalibrateRotation()
    calibrator.CalibrateGeometry()
    calibrator.CalibrateShift()
    calibrator.calibrate_shiftx()
    calibrator.calibrate_shifty()
    calibrator.calibrate_distance()
    calibrator.FullCalibration()
    calibrator.calibrate_wave_shift()

    calibrator._calibstatus = 3

    calibrator.print_calibration()

    # write out
    if os.path.exists("HB2B_CAL_Test.json"):
        os.remove("HB2B_CAL_Test.json")
    file_name = os.path.join(os.getcwd(), "HB2B_CAL_Test.json")
    calibrator.write_calibration(file_name)

    t_stop = time.time()
    print("Total Time: {}".format(t_stop - t_start))

    # Compare output file with gold file for test
    if are_equivalent_jsons(goldfile, file_name, atol=5e-3):
        # Same: remove file generated in test
        os.remove(file_name)
    else:
        print_out_json_diff(goldfile, "HB2B_CAL_Test.json")
        assert False, "Test output {} is different from gold file {}".format(file_name, goldfile)

    return


def test_load_print_calibration():
    """Main test for the script

    Returns
    -------

    """
    # Define Fitting Routine
    nexus_file = "tests/data/calibration_tests/HB2B_3510.nxs.h5"

    goldfile = "tests/data/calibration_tests/HB2B_mantid_calib_lambda.json"

    t_start = time.time()

    # Initalize calibration
    calibrator = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file)

    calibrator.get_archived_calibration(goldfile)

    # set calibration status and errors as they are not loaded
    calibrator._calibstatus = 3
    calibrator._caliberr[7] = 0.00770477092528105

    # write out
    if os.path.exists("HB2B_CAL_Test.json"):
        os.remove("HB2B_CAL_Test.json")
    file_name = os.path.join(os.getcwd(), "HB2B_CAL_Test.json")
    calibrator.write_calibration(file_name)

    t_stop = time.time()
    print("Total Time: {}".format(t_stop - t_start))

    # Compare output file with gold file for test
    if are_equivalent_jsons(goldfile, file_name, atol=5e-3):
        # Same: remove file generated in test
        os.remove(file_name)
    else:
        print_out_json_diff(goldfile, "HB2B_CAL_Test.json")
        assert False, "Test output {} is different from gold file {}".format(file_name, goldfile)

    return


if __name__ == "__main__":
    pytest.main([__file__])


def test_mask_file_is_applied():
    """Regression (#1024): a mask file raised ValueError because it never reached the NeXus conversion
    and its path was passed to reduce_data as a mask ID. It must become the default mask and be used."""
    # Arrange
    nexus_file = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
    mask_file = "tests/data/HB2B_Mask_12-18-19.xml"

    # Act
    masked = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file, mask_file=mask_file)
    unmasked = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file)

    # Assert - the file is the workspace's default mask ...
    default_mask = masked._hidra_ws.get_detector_mask(True)
    assert default_mask is not None and 0.5 < default_mask.mean() < 1.0
    assert unmasked._hidra_ws.get_detector_mask(True) is None

    # ... and the reduction uses it: masked and unmasked patterns of the same sub-run and wedge differ
    wedge = masked._hidra_ws.reduction_masks[0]
    _, masked_intensity, _ = masked.reducer.get_diffraction_data(1, mask_id=wedge)
    _, unmasked_intensity, _ = unmasked.reducer.get_diffraction_data(1, mask_id=wedge)
    assert masked_intensity.shape == unmasked_intensity.shape
    assert not np.allclose(masked_intensity, unmasked_intensity, equal_nan=True)

    # ... including in the calibration residual, which re-reduces the data
    residual = masked.get_alignment_residual(np.array(masked._calib))
    assert np.all(np.isfinite(residual))


def test_reference_lattice_is_the_2020_set():
    """#1022: FitCalibration's powders are the shared standards, whose lattice constants are the 2020 set
    (Ni 3.523799438, Fe 2.8663982, Mo 3.14719963 A), not the 2023 values fitted from HB2B data
    (see docs/ground_truths.md)"""
    # Arrange / Act
    calibrator = mantid_peakfit_calibration.FitCalibration(nexus_file="tests/data/calibration_tests/HB2B_3510.nxs.h5")

    # Assert - every HB2B_3510 sub-run is Ni or Fe, with the exact (float64) 2020 constant
    lattice_by_powder = {s.name: s.lattice_a for s in calibrator._powder_standards}
    assert lattice_by_powder == {"Ni": 3.523799438, "Fe": 2.8663982}
    assert all(calibrator._keep_subrun_list)
