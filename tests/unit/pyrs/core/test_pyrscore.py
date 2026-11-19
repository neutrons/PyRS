"""Unit tests for pyrs.core.pyrscore: how PyRsCore.reduce_diffraction_data takes a calibration (#1051)"""

import json

import pytest

from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.pyrscore import PyRsCore

CALIBRATION = {"Shift_x": 0.001, "Shift_y": -0.002, "Shift_z": 0.003, "Rot_x": 0.1, "Rot_y": -0.2, "Rot_z": 0.3}
CALIBRATION.update({"TTH_0": 0.25, "Lambda": 1.54})


@pytest.fixture
def core(monkeypatch):
    """A PyRsCore whose reduction service records the calibration it is asked to apply."""
    core = PyRsCore()
    applied = []
    monkeypatch.setattr(
        core.reduction_service,
        "reduce_diffraction_data",
        lambda session, calibration, *args: applied.append(calibration),
    )
    core.applied = applied
    return core


def _write_calibration(path, status):
    values = dict(CALIBRATION, Status=status)
    values.update({f"error_{key}": 0.0 for key in CALIBRATION})
    path.write_text(json.dumps(values))
    return str(path)


def test_calibration_file_is_read_with_its_two_theta_0(core, tmp_path):
    """Regression: the file branch built a 6-argument DENEXDetectorShift, and from_json would have read
    none of the standard keys, giving a zero calibration"""
    # Arrange
    file_name = _write_calibration(tmp_path / "calibration.json", status=3)

    # Act
    core.reduce_diffraction_data("session", 1000, "PyRS", geometry_calibration=file_name)

    # Assert
    shift = core.applied[0]
    assert isinstance(shift, DENEXDetectorShift)
    read = [shift.center_shift_x, shift.center_shift_y, shift.center_shift_z]
    read += [shift.rotation_x, shift.rotation_y, shift.rotation_z, shift.two_theta_0]
    assert read == [CALIBRATION[key] for key in ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0"]]


def test_uncalibrated_calibration_file_raises(core, tmp_path):
    # Arrange
    file_name = _write_calibration(tmp_path / "calibration.json", status=-1)

    # Act / Assert
    with pytest.raises(RuntimeError, match="never successfully refined"):
        core.reduce_diffraction_data("session", 1000, "PyRS", geometry_calibration=file_name)
    assert core.applied == []


def test_missing_error_is_applied_unchanged(core, tmp_path):
    """A calibration with an error missing is applied exactly as the complete one (#1098): errors do not move pixels"""
    # Arrange
    complete = _write_calibration(tmp_path / "complete.json", status=3)
    stored = json.loads(open(complete).read())
    del stored["error_Rot_y"]
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(stored))

    # Act
    core.reduce_diffraction_data("session", 1000, "PyRS", geometry_calibration=complete)
    core.reduce_diffraction_data("session", 1000, "PyRS", geometry_calibration=str(partial))

    # Assert
    assert core.applied[0].convert_to_dict() == core.applied[1].convert_to_dict()


def test_calibration_without_wavelength_is_refused(core, tmp_path):
    """Reduction refuses a JSON without Lambda rather than apply a wavelength of 0 (user decision, #1098)"""
    # Arrange
    stored = json.loads(open(_write_calibration(tmp_path / "complete.json", status=3)).read())
    del stored["Lambda"], stored["error_Lambda"]
    path = tmp_path / "no_lambda.json"
    path.write_text(json.dumps(stored))

    # Act / Assert
    with pytest.raises(ValueError, match="no wavelength"):
        core.reduce_diffraction_data("session", 1000, "PyRS", geometry_calibration=str(path))
    assert core.applied == []
