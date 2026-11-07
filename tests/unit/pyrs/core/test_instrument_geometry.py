"""Unit tests for pyrs.core.instrument_geometry.DENEXDetectorShift: values stored as float64 (#1052)"""

import numpy as np
import pytest

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift, HidraSetup

FIELDS = ["center_shift_x", "center_shift_y", "center_shift_z", "rotation_x", "rotation_y", "rotation_z"]
FIELDS += ["two_theta_0"]


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.float32(0.123456), float(np.float32(0.123456))),
        (np.float64(0.1), 0.1),
        (np.array([0.25]), 0.25),
        (2, 2.0),
        ("0.5", 0.5),
    ],
    ids=["float32", "float64", "one-element-array", "int", "string"],
)
@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")
def test_constructor_stores_python_floats(value, expected):
    # Act
    shift = DENEXDetectorShift(*[value] * 7)

    # Assert
    for field in FIELDS:
        stored = getattr(shift, field)
        assert type(stored) is float, field
        assert stored == expected, field


def test_error_above_360_degrees_constructs():
    """The class also holds errors, and a badly determined rotation's error may exceed 360 degrees"""
    # Act
    errors = DENEXDetectorShift(0.1, 0.1, 0.1, 400.0, 1e4, -2.0, 720.0)

    # Assert
    assert (errors.rotation_x, errors.rotation_y, errors.rotation_z, errors.two_theta_0) == (400.0, 1e4, -2.0, 720.0)


def test_constructor_non_number_raises():
    # Act / Assert
    with pytest.raises(TypeError, match="Rotation along X direction"):
        DENEXDetectorShift(0, 0, 0, "not a number", 0, 0, 0)


@pytest.mark.parametrize("field", ["rotation_x", "rotation_y", "rotation_z", "two_theta_0"])
def test_setter_still_rejects_angle_beyond_360(field):
    # Arrange
    shift = DENEXDetectorShift(0, 0, 0, 0, 0, 0, 0)

    # Act / Assert
    with pytest.raises(ValueError):
        setattr(shift, field, 400.0)


def test_set_from_dict_stores_python_floats():
    # Arrange
    shift = DENEXDetectorShift(0, 0, 0, 0, 0, 0, 0)

    # Act
    shift.set_from_dict({"shift x": np.float32(0.5), "rotation z": np.float32(1.5)})

    # Assert
    assert type(shift.center_shift_x) is float and shift.center_shift_x == 0.5
    assert type(shift.rotation_z) is float and shift.rotation_z == 1.5


def test_mask_record_defaults_to_not_recorded():
    # Arrange
    instrument = HidraSetup(DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False))

    # Act / Assert
    assert instrument.mask_record == {"source": "not recorded"}


def test_mask_record_without_source_key_raises():
    # Arrange
    instrument = HidraSetup(DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False))

    # Act / Assert
    with pytest.raises(ValueError, match="'source' key"):
        instrument.set_mask_record({"file": "HB2B_Mask.xml"})
