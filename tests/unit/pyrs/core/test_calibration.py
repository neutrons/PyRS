"""The applied-calibration record (#1141): its identity is a content hash, and it refuses what cannot be stated"""

import dataclasses
import hashlib
import math
import struct

import numpy as np
import pytest

from pyrs.core.calibration import (
    AppliedCalibration,
    CalibrationSource,
    DEFAULT_ERROR,
    UNKNOWN_ERROR,
    nominal_calibration,
    reduction_l2,
    run_monosetting,
    stated_l2,
)
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.monosetting import MonoSetting
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.calibration_file_io import calibration_to_dict
from tests.util.workspace_helpers import make_minimal_HidraWorkspace

SHIFT = DENEXDetectorShift(-0.0154027, -0.00255874, -0.0445212, 0.0642667, -0.254056, 0.16946, 0.85294)
ERROR = DENEXDetectorShift(9.889e-05, 0.001167, 0.00026, 0.188275, 0.322219, 0.010681, DEFAULT_ERROR)


def _refined(**changes) -> AppliedCalibration:
    calibration = AppliedCalibration.from_shift(
        SHIFT,
        ERROR,
        wavelength=1.531347247129065,
        wavelength_error=8.25e-05,
        l2=0.985,
        source=CalibrationSource.FILE,
        status=1,
    )
    return dataclasses.replace(calibration, **changes)


def test_hash_ignores_source_and_status():
    """Where a calibration came from is not what it is: the same values hash the same"""
    # Arrange
    from_file = _refined()

    # Act
    in_session = _refined(source=CalibrationSource.WORKSPACE, status=None)

    # Assert
    assert in_session.content_hash == from_file.content_hash
    assert in_session.same_calibration_as(from_file)


def test_hash_format_is_fixed():
    """The hash is a file contract (project files, NXstress): its order, byte order and packing never change"""
    # Arrange
    calibration = _refined()
    numbers = calibration.values + calibration.errors
    numbers += (calibration.wavelength, calibration.wavelength_error, calibration.l2)

    # Act
    expected = hashlib.sha256(struct.pack("<17d", *numbers)).hexdigest()

    # Assert
    assert calibration.content_hash == expected
    assert (
        nominal_calibration(MonoSetting.Si331, l2=0.985).content_hash
        == "e9d0bbb2bfdb9b21ced0231cf01ba209ad59b010d453f9e1b7c8d5f58ad1ef6f"
    )


@pytest.mark.parametrize("index", range(17))
def test_any_changed_number_changes_hash(index):
    """Each of the 17 hashed numbers (7 values, 7 errors, wavelength, its error, L2) is part of the identity"""
    # Arrange
    base = _refined()
    numbers = list(base.values + base.errors + (base.wavelength, base.wavelength_error, base.l2))
    numbers[index] = numbers[index] + 1e-3 if numbers[index] >= 0 else UNKNOWN_ERROR

    # Act
    changed = dataclasses.replace(
        base,
        values=tuple(numbers[:7]),
        errors=tuple(numbers[7:14]),
        wavelength=numbers[14],
        wavelength_error=numbers[15],
        l2=numbers[16],
    )

    # Assert
    assert changed.content_hash != base.content_hash
    assert not changed.same_calibration_as(base)


def test_hash_is_exact_to_the_last_bit():
    """Equality is bit-for-bit float64: the smallest change to a value is a different calibration"""
    # Arrange
    base = _refined()
    nudged = list(base.values)
    nudged[0] = math.nextafter(nudged[0], 1.0)

    # Act
    changed = dataclasses.replace(base, values=tuple(nudged))

    # Assert
    assert changed.content_hash != base.content_hash


def test_shift_round_trips():
    # Act
    calibration = _refined()

    # Assert
    assert calibration.shift.convert_to_dict() == SHIFT.convert_to_dict()
    assert calibration.shift_error.convert_to_dict() == ERROR.convert_to_dict()


def test_shift_is_a_copy():
    """A caller changing the shift it was given cannot change the record or its hash"""
    # Arrange
    calibration = _refined()
    before = calibration.content_hash

    # Act
    calibration.shift.center_shift_x = 0.04

    # Assert
    assert calibration.content_hash == before


@pytest.mark.parametrize("mono", list(MonoSetting))
def test_nominal_calibration_values(mono):
    # Act
    nominal = nominal_calibration(mono, l2=0.985)

    # Assert
    assert nominal.values == (0.0,) * 7
    assert nominal.errors == (DEFAULT_ERROR,) * 7
    assert (nominal.wavelength, nominal.wavelength_error) == (float(mono), DEFAULT_ERROR)
    assert (nominal.l2, nominal.source, nominal.status) == (0.985, CalibrationSource.NONE, None)


def test_nominal_differs_from_refined():
    # Act
    nominal = nominal_calibration(MonoSetting.Si331, l2=0.985)

    # Assert
    assert not nominal.same_calibration_as(_refined())


def test_nominal_differs_between_monochromators():
    """Different nominal wavelengths are different calibrations"""
    # Act
    si331, si422 = (nominal_calibration(mono, l2=0.985) for mono in (MonoSetting.Si331, MonoSetting.Si422))

    # Assert
    assert not si331.same_calibration_as(si422)


@pytest.mark.parametrize("source", list(CalibrationSource))
def test_stored_source_reads_back(source):
    # Act / Assert
    assert CalibrationSource.from_stored(source.value, where="run.h5") is source


@pytest.mark.parametrize("value", ["unknown", "File", ""])
def test_unrecognised_stored_source_raises(value):
    """A stored string that is not a member is rejected, naming the file and the value"""
    # Act / Assert
    with pytest.raises(ValueError, match=f"run.h5: calibration source {value!r}"):
        CalibrationSource.from_stored(value, where="run.h5")


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"values": (0.0,) * 6}, "values must hold 7"),
        ({"errors": (0.1,) * 8}, "errors must hold 7"),
        ({"values": (float("nan"),) + (0.0,) * 6}, "values must be finite"),
        ({"errors": (-3.0,) + (0.1,) * 6}, "calibration error must be"),
        ({"errors": (float("nan"),) + (0.1,) * 6}, "calibration error must be"),
        ({"wavelength": 0.0}, "wavelength must be"),
        ({"wavelength_error": -0.5}, "wavelength error must be"),
        ({"l2": float("inf")}, "L2 must be"),
        ({"source": CalibrationSource.NONE}, "must be the nominal one"),
    ],
    ids=[
        "six-values",
        "eight-errors",
        "nan-value",
        "bad-code",
        "nan-error",
        "zero-wavelength",
        "bad-wl-error",
        "infinite-l2",
        "none-not-nominal",
    ],
)
def test_invalid_record_raises(changes, match):
    # Act / Assert
    with pytest.raises(ValueError, match=match):
        _refined(**changes)


def test_source_must_be_member():
    # Act / Assert
    with pytest.raises(TypeError, match="CalibrationSource"):
        _refined(source="file")


def test_compare_with_non_record_raises():
    # Act / Assert
    with pytest.raises(TypeError, match="AppliedCalibration"):
        _refined().same_calibration_as(SHIFT)


def test_stored_record_round_trips():
    # Arrange
    calibration = _refined()

    # Act
    restored = AppliedCalibration.from_dict(calibration.to_dict(), where="run.h5")

    # Assert
    assert restored == calibration


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d.update(l2=0.9), "does not match its stored hash"),
        (lambda d: d.pop("content_hash"), "incomplete"),
        (lambda d: d.update(source="unknown"), "calibration source 'unknown'"),
        (lambda d: d.update(wavelength=-1.0), "invalid"),
    ],
    ids=["tampered", "no-hash", "unknown-source", "invalid-value"],
)
def test_corrupt_stored_record_raises(change, match):
    """A stored record that is not exactly what was written is corrupt, never read (#1141)"""
    # Arrange
    stored = _refined().to_dict()
    change(stored)

    # Act / Assert
    with pytest.raises(ValueError, match=f"run.h5: .*{match}"):
        AppliedCalibration.from_dict(stored, where="run.h5")


def test_reduction_l2_is_the_log():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("L2", np.array([1, 2]), np.array([0.95, 0.95]))

    # Act / Assert
    assert reduction_l2(workspace, [1, 2]) == 0.95


def test_reduction_l2_without_log_is_arm_length():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])

    # Act / Assert
    assert reduction_l2(workspace, [1, 2]) == workspace.get_instrument_setup().arm_length


def test_differing_l2_raises():
    """One reduction applies one calibration at one distance (#1141)"""
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("L2", np.array([1, 2]), np.array([0.95, 0.96]))

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"L2\) differs .*\[0.95, 0.96\]"):
        reduction_l2(workspace, [1, 2])


def test_monosetting_is_from_mrot():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([-170.0, -170.0]))

    # Act / Assert
    assert run_monosetting(workspace, [1, 2]) is MonoSetting.Si331


def test_differing_monochromators_raise():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([-170.0, -20.0]))

    # Act / Assert
    with pytest.raises(RuntimeError, match="different monochromators"):
        run_monosetting(workspace, [1, 2])


def test_missing_mrot_raises():
    # Arrange
    workspace = HidraWorkspace("no_mono")
    workspace.set_sub_runs(np.array([1, 2]))

    # Act / Assert
    with pytest.raises(RuntimeError, match="'no_mono' has no 'mrot' log"):
        run_monosetting(workspace, [1, 2])


def test_unknown_rotation_raises():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([100.0, 100.0]))

    # Act / Assert
    with pytest.raises(ValueError, match="monochromator rotation angle"):
        run_monosetting(workspace, [1, 2])


def _record(source, calibration, **more):
    return dict(source=source, calibration=calibration, **more)


def _loaded_workspace(**kwargs):
    """A workspace as a project file loads it: mrot 0 (Si511), instrument arm length 2.0, no L2 log."""
    return make_minimal_HidraWorkspace(sub_runs=[1, 2], **kwargs)


def test_older_none_record_raises():
    """A #1053 record of no calibration file is not rebuilt: the data must be reduced again (user decision)"""
    # Act / Assert
    with pytest.raises(ValueError, match="reduced without a calibration file before #1141"):
        AppliedCalibration.from_reduction_record(_record("none", None), workspace=_loaded_workspace(), where="run.h5")


def test_rebuilt_file_record_keeps_values():
    """A #1053 record of a calibration file carries everything: values, errors, wavelength and status"""
    # Arrange
    stored = calibration_to_dict(SHIFT, ERROR, 1.5306, 1e-5, 1)

    # Act
    rebuilt = AppliedCalibration.from_reduction_record(
        _record("file", stored), workspace=_loaded_workspace(), where="run.h5"
    )

    # Assert
    assert rebuilt.source is CalibrationSource.FILE and rebuilt.status == 1
    assert rebuilt.shift.convert_to_dict() == SHIFT.convert_to_dict()
    assert rebuilt.shift_error.convert_to_dict() == ERROR.convert_to_dict()
    assert (rebuilt.wavelength, rebuilt.wavelength_error, rebuilt.l2) == (1.5306, 1e-5, 2.0)


@pytest.mark.parametrize(
    "stored_lambda, expected",
    [(1.55, (1.55, UNKNOWN_ERROR)), (float("nan"), (float(MonoSetting.Si511), DEFAULT_ERROR))],
    ids=["lambda", "nan"],
)
def test_rebuilt_workspace_record_has_unknown_errors(stored_lambda, expected):
    """A #1053 workspace shift: its values, errors -2, and its wavelength if it stored one"""
    # Arrange
    stored = dict(SHIFT.convert_to_dict(), Lambda=stored_lambda)

    # Act
    rebuilt = AppliedCalibration.from_reduction_record(
        _record("workspace", stored), workspace=_loaded_workspace(), where="run.h5"
    )

    # Assert
    assert rebuilt.source is CalibrationSource.WORKSPACE and rebuilt.status is None
    assert rebuilt.shift.convert_to_dict() == SHIFT.convert_to_dict()
    assert rebuilt.errors == (UNKNOWN_ERROR,) * 7
    assert (rebuilt.wavelength, rebuilt.wavelength_error) == expected


@pytest.mark.parametrize(
    "record, match",
    [
        ({"source": "unknown", "error": "could not be recorded"}, "calibration source 'unknown'"),
        ({"calibration": None}, "calibration source None"),
        (_record("file", None), "holds no calibration"),
        (_record("file", {"Shift_x": 0.0}), "Lambda"),
    ],
    ids=["unknown", "no-source", "no-calibration", "no-wavelength"],
)
def test_unstatable_record_raises(record, match):
    """A record that cannot state its calibration is not rebuilt (#1141 G2)"""
    # Act / Assert
    with pytest.raises(ValueError, match=match):
        AppliedCalibration.from_reduction_record(record, workspace=_loaded_workspace(), where="run.h5")


@pytest.mark.parametrize(
    "record",
    [
        _record("file", {"geometry": {"arm_length": 0.95}}),
        _record("file", {"geometry": {"arm_length": 0.99}}, run_geometry={"arm_length": 0.95}),
    ],
    ids=["calibration-geometry", "run-geometry"],
)
def test_stated_l2_from_record_geometry(record):
    """No L2 log and no geometry in the file: the record's run geometry, else the calibration's"""
    # Arrange
    workspace = _loaded_workspace(with_instrument=False)

    # Act / Assert
    assert stated_l2(workspace, [1, 2], record=record, where="run.h5") == 0.95


def test_stated_l2_prefers_the_log():
    # Arrange
    workspace = _loaded_workspace(with_instrument=False)
    workspace.set_sample_log("L2", np.array([1, 2]), np.array([0.97, 0.97]))

    # Act / Assert
    assert (
        stated_l2(workspace, [1, 2], record=_record("file", {"geometry": {"arm_length": 0.95}}), where="run.h5")
        == 0.97
    )


def test_unstated_l2_raises():
    """Nothing in the file states the distance: never a default (user decision)"""
    # Arrange
    workspace = _loaded_workspace(with_instrument=False)

    # Act / Assert
    with pytest.raises(ValueError, match="run.h5: the distance its reduction used is not stated"):
        stated_l2(workspace, [1, 2], record=_record("file", {"Shift_x": 0.0}), where="run.h5")


def test_reduction_l2_without_log_or_geometry_raises():
    # Arrange
    workspace = HidraWorkspace("bare")
    workspace.set_sub_runs(np.array([1, 2]))

    # Act / Assert
    with pytest.raises(RuntimeError, match="no L2 log and no instrument geometry"):
        reduction_l2(workspace, [1, 2])
