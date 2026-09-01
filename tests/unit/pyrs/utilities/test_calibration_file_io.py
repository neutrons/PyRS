#!/usr/bin/python
# Unit test for classes and methods that will be used in instrument geometry calibration
import os
import pytest
from pyrs.utilities import calibration_file_io
import json
import datetime


@pytest.mark.parametrize("status", [-1, -4])
def test_check_calibration_status_negative_status_raises(status):
    """Test that a negative Status (unconverged or never-refined calibration) is rejected"""
    # Arrange - status already provided by parametrize

    # Act / Assert
    with pytest.raises(RuntimeError, match="never successfully refined"):
        calibration_file_io.check_calibration_status(status)


@pytest.mark.parametrize("status", [0, 1, 3])
def test_check_calibration_status_nonnegative_status_ok(status):
    """Test that a non-negative Status (a completed optimizer run) passes without error"""
    # Arrange - status already provided by parametrize

    # Act
    calibration_file_io.check_calibration_status(status)

    # Assert - no exception raised


@pytest.mark.integration
def test_calibration_json_io():
    """Test the calibration file (in Json format) I/O methods

    Returns
    -------
    None
    """
    # Set test data
    gold_calib_file = "tests/data/HB2B_CAL_Si333.json"
    assert os.path.exists(gold_calib_file), "Test calibration file {} does not exist".format(gold_calib_file)

    # Import for future test
    with open(gold_calib_file, "r") as f:
        calib_dict = json.load(f)

    # Load to Shift object
    test_shifts, test_shifts_error, wl, wl_error, status = calibration_file_io.read_calibration_json_file(
        gold_calib_file
    )

    # Export again
    # use ordinal current time to avoid pre-existed test file
    now = datetime.datetime.now()
    test_calib_file = "test_calib_{}.json".format(now.toordinal())

    calibration_file_io.write_calibration_to_json(
        test_shifts, test_shifts_error, wl, wl_error, status, test_calib_file
    )
    assert os.path.exists(test_calib_file), "blabla"

    # Check file existing or not

    # Read again and test
    with open(test_calib_file, "r") as t:
        test_calib_dict = json.load(t)

    print(calib_dict)
    print(test_calib_dict)
    # compare
    if calib_dict == test_calib_dict:
        os.remove(test_calib_file)
    else:
        raise AssertionError("blabla")

    return


if __name__ == "__main__":
    pytest.main()
