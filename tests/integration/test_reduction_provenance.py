"""Reduction records the calibration it applied in the project file it saves (#1053 part B)"""

import json

import pytest

from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import read_calibration_json_file, write_calibration_to_json

pytestmark = pytest.mark.integration

RUN = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
MASK = "tests/data/HB2B_Mask_12-18-19.xml"
REFERENCE = "tests/data/calibration_tests/HB2B_3510_ring_reference.json"  # written before #1053: no provenance


def _reduce(tmp_path, calibration_file=None, workspace_shift=None, workspace_wavelength=None, name="reduced.h5"):
    workspace = NeXusConvertingApp(RUN, MASK).convert()
    if workspace_shift is not None:
        workspace.set_detector_shift(workspace_shift)
    if workspace_wavelength is not None:
        workspace.set_wavelength(workspace_wavelength, True)
    reducer = ReductionApp()
    reducer.load_hidra_workspace(workspace)
    reducer.reduce_data(
        sub_runs=[1, 2], instrument_file=None, calibration_file=calibration_file, mask=None, num_bins=256
    )
    path = tmp_path / name
    reducer.save_diffraction_data(str(path))
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    record = project.read_reduction_provenance()
    project.close()
    # a regression in recording must fail a test, not only warn (#1053 part B review)
    assert "error" not in record and record.get("source") != "unknown", record
    return record


RUN_GEOMETRY = {"detector_size": [1024, 1024], "pixel_dimension": [0.3 / 1024, 0.3 / 1024], "arm_length": 0.985}


def _calibration(calibration_file):
    """The calibration JSON's dictionary of a calibration file, as the project file must record it: its
    values, and the geometry within the calibration (here the run's: the test files carry none)."""
    from pyrs.utilities.calibration_file_io import calibration_to_dict

    return dict(calibration_to_dict(*read_calibration_json_file(calibration_file)), geometry=RUN_GEOMETRY)


def test_calibration_with_provenance_is_recorded(tmp_path):
    # Arrange - the reference calibration, written with a provenance block
    shift, error, wavelength, wavelength_error, status = read_calibration_json_file(REFERENCE)
    provenance = {"schema": 1, "method": "ring", "created": "2026-11-03T11:40:37-05:00", "steps": []}
    calibration = tmp_path / "with_provenance.json"
    write_calibration_to_json(
        shift, error, wavelength, wavelength_error, status, str(calibration), provenance=provenance
    )

    # Act
    record = _reduce(tmp_path, str(calibration))

    # Assert
    assert record["source"] == "file" and record["calibration_file"] == str(calibration)
    assert record["calibration"] == _calibration(str(calibration)) and record["calibration"]["Status"] == status
    assert record["created"] == "2026-11-03T11:40:37-05:00"
    assert record["created_source"] == "calibration provenance"
    assert record["provenance"] == provenance


def test_old_calibration_uses_its_labelled_modification_time(tmp_path):
    # Act
    record = _reduce(tmp_path, REFERENCE)

    # Assert - no provenance in the file: the time is the file's modification time, and says so
    assert record["source"] == "file" and "provenance" not in record
    assert record["created_source"] == "file modification time"
    assert record["calibration"] == _calibration(REFERENCE)


def test_workspace_shift_is_recorded_without_invented_errors(tmp_path):
    # Arrange
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)

    # Act
    record = _reduce(tmp_path, workspace_shift=shift)

    # Assert - the nominal (mono) wavelength is not recorded as applied
    assert record["source"] == "workspace"
    shift = {
        k: record["calibration"][k] for k in ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0")
    }
    assert list(shift.values()) == [0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25]
    assert record["calibration"]["Lambda"] is None  # the nominal (mono) wavelength is not recorded as applied
    assert record["lambda_source"] == "workspace has no calibrated wavelength: not recorded"
    assert "Status" not in record["calibration"] and "provenance" not in record


def test_workspace_calibrated_wavelength_is_recorded(tmp_path):
    # Arrange
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)

    # Act
    record = _reduce(tmp_path, workspace_shift=shift, workspace_wavelength=1.5306)

    # Assert
    assert record["calibration"]["Lambda"] == 1.5306 and record["lambda_source"] == "workspace calibrated wavelength"
    assert record["calibration"]["geometry"] == RUN_GEOMETRY and record["geometry_source"] == "run"


def test_no_calibration_is_recorded_as_none(tmp_path):
    # Act
    record = _reduce(tmp_path)

    # Assert
    assert record == {"source": "none"}


def test_recorded_calibration_rebuilds_the_instrument(tmp_path):
    """The point of the instrument object: re-reduce with exactly the calibration that was applied"""
    # Arrange
    record = _reduce(tmp_path, REFERENCE)
    project = HidraProjectFile(str(tmp_path / "reduced.h5"), HidraProjectFileMode.READONLY)
    geometry = NeXusConvertingApp(RUN, MASK).convert().get_instrument_setup()

    # Act
    instrument = project.read_reduction_instrument(geometry)

    # Assert
    shift, error, wavelength, _, status = read_calibration_json_file(REFERENCE)
    assert instrument.geometry_shift.convert_to_dict() == shift.convert_to_dict()
    assert instrument.geometry_shift_error.convert_error_to_dict() == error.convert_error_to_dict()
    assert instrument.get_wavelength(None) == wavelength and instrument.calibration_status == status
    assert record["calibration"] == instrument.calibration_dict()
    json.dumps(record)
