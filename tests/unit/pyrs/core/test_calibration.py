"""The applied-calibration record (#1141): its identity is a content hash, and it refuses what cannot be stated"""

import dataclasses
import math

import pytest

from pyrs.core.calibration import (
    AppliedCalibration,
    CalibrationSource,
    DEFAULT_ERROR,
    UNKNOWN_ERROR,
    nominal_calibration,
)
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.monosetting import MonoSetting

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
