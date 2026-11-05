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


def _values(calibration_file):
    shift, error, wavelength, wavelength_error, status = read_calibration_json_file(calibration_file)

    def vector(s, last):
        return [
            s.center_shift_x,
            s.center_shift_y,
            s.center_shift_z,
            s.rotation_x,
            s.rotation_y,
            s.rotation_z,
            s.two_theta_0,
            last,
        ]

    return vector(shift, wavelength), vector(error, wavelength_error), status


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
    values, errors, _ = _values(str(calibration))
    assert record["source"] == "file" and record["calibration_file"] == str(calibration)
    assert record["values"] == values and record["errors"] == errors and record["status"] == status
    assert record["created"] == "2026-11-03T11:40:37-05:00"
    assert record["created_source"] == "calibration provenance"
    assert record["provenance"] == provenance


def test_old_calibration_uses_its_labelled_modification_time(tmp_path):
    # Act
    record = _reduce(tmp_path, REFERENCE)

    # Assert - no provenance in the file: the time is the file's modification time, and says so
    assert record["source"] == "file" and "provenance" not in record
    assert record["created_source"] == "file modification time"
    assert record["values"] == _values(REFERENCE)[0]


def test_workspace_shift_is_recorded_without_invented_errors(tmp_path):
    # Arrange
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)

    # Act
    record = _reduce(tmp_path, workspace_shift=shift)

    # Assert - the nominal (mono) wavelength is not recorded as applied
    assert record["source"] == "workspace"
    assert record["values"][:7] == [0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25]
    assert record["values"][7] != record["values"][7]  # NaN
    assert record["lambda_source"] == "workspace has no calibrated wavelength: not recorded"
    assert not {"errors", "status", "provenance"} & set(record)


def test_workspace_calibrated_wavelength_is_recorded(tmp_path):
    # Arrange
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)

    # Act
    record = _reduce(tmp_path, workspace_shift=shift, workspace_wavelength=1.5306)

    # Assert
    assert record["values"][7] == 1.5306 and record["lambda_source"] == "workspace calibrated wavelength"


def test_no_calibration_is_recorded_as_none(tmp_path):
    # Act
    record = _reduce(tmp_path)

    # Assert
    assert record == {"source": "none"}
    json.dumps(record)
