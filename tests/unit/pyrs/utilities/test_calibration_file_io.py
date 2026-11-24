#!/usr/bin/python
# Unit test for classes and methods that will be used in instrument geometry calibration
import os
import pytest
from pyrs.utilities import calibration_file_io
import json
import datetime
import numpy as np

from pyrs.core.detector_model import from_shift
from pyrs.core.instrument_geometry import DENEXDetectorShift


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


def _write_calibration(path, **two_theta_keys):
    """A calibration JSON with every key the writer produces, the 2theta_0 keys as given."""
    calibration = {
        "Shift_x": 0.001,
        "Shift_y": -0.002,
        "Shift_z": 0.003,
        "Rot_x": 0.1,
        "Rot_y": -0.2,
        "Rot_z": 0.3,
        "Lambda": 1.54,
        "error_Shift_x": 1e-4,
        "error_Shift_y": 2e-4,
        "error_Shift_z": 3e-4,
        "error_Rot_x": 0.01,
        "error_Rot_y": 0.02,
        "error_Rot_z": 0.03,
        "error_Lambda": 1e-4,
        "Status": 1.0,
    }
    calibration.update(two_theta_keys)
    path.write_text(json.dumps(calibration))
    return str(path)


def test_read_tth0_written_as_tth_0(tmp_path):
    """Regression (#1021): the writer stores 2theta_0 as TTH_0; the reader used to ignore it (0)"""
    # Arrange
    calibration_file = _write_calibration(tmp_path / "cal.json", TTH_0=0.25, error_TTH_0=0.01)

    # Act
    shift, shift_error, _, _, _ = calibration_file_io.read_calibration_json_file(calibration_file)

    # Assert
    assert shift.two_theta_0 == pytest.approx(0.25)
    assert shift_error.two_theta_0 == pytest.approx(0.01)
    assert shift.rotation_y == pytest.approx(-0.2)


def test_read_legacy_two_theta_0(tmp_path):
    # Arrange
    calibration_file = _write_calibration(tmp_path / "cal.json", two_theta_0=-0.15, error_two_theta_0=0.02)

    # Act
    shift, shift_error, _, _, _ = calibration_file_io.read_calibration_json_file(calibration_file)

    # Assert
    assert shift.two_theta_0 == pytest.approx(-0.15)
    assert shift_error.two_theta_0 == pytest.approx(0.02)


def test_conflicting_tth0_keys_raises(tmp_path):
    # Arrange
    calibration_file = _write_calibration(tmp_path / "cal.json", TTH_0=0.25, two_theta_0=0.0)

    # Act / Assert
    with pytest.raises(ValueError, match="TTH_0"):
        calibration_file_io.read_calibration_json_file(calibration_file)


def test_missing_tth0_reads_as_zero(tmp_path):
    # Arrange
    calibration_file = _write_calibration(tmp_path / "cal.json")

    # Act
    shift, shift_error, _, _, _ = calibration_file_io.read_calibration_json_file(calibration_file)

    # Assert
    assert shift.two_theta_0 == 0
    assert shift_error.two_theta_0 == -1


def test_tth0_without_its_error_still_loads(tmp_path):
    # Arrange - TTH_0 used to be ignored, so such a file loaded before #1021 and must still load
    calibration_file = _write_calibration(tmp_path / "cal.json", TTH_0=0.1)

    # Act
    shift, shift_error, _, _, _ = calibration_file_io.read_calibration_json_file(calibration_file)

    # Assert
    assert shift.two_theta_0 == pytest.approx(0.1)
    assert shift_error.two_theta_0 == -1


def test_round_trip_keeps_nonzero_tth0(tmp_path):
    """What write_calibration_to_json writes, read_calibration_json_file must read back"""
    # Arrange
    source = _write_calibration(tmp_path / "in.json", TTH_0=-0.4, error_TTH_0=0.05)
    shift, shift_error, wavelength, wavelength_error, status = calibration_file_io.read_calibration_json_file(source)
    target = str(tmp_path / "out.json")

    # Act
    calibration_file_io.write_calibration_to_json(shift, shift_error, wavelength, wavelength_error, status, target)
    reread, reread_error, _, _, _ = calibration_file_io.read_calibration_json_file(target)

    # Assert
    assert json.loads(open(target).read())["TTH_0"] == pytest.approx(-0.4)
    assert reread.two_theta_0 == pytest.approx(-0.4)
    assert reread_error.two_theta_0 == pytest.approx(0.05)


def _shift_and_errors():
    from pyrs.core.instrument_geometry import DENEXDetectorShift

    return DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25), DENEXDetectorShift(*[0.01] * 7)


def test_write_without_provenance_is_unchanged(tmp_path):
    """No provenance: the file is byte-for-byte what json.dump of the values gave before #1053"""
    # Arrange
    shift, error = _shift_and_errors()
    target = tmp_path / "cal.json"
    expected = shift.convert_to_dict()
    expected["Lambda"] = 1.54
    expected.update(error.convert_error_to_dict())
    expected["error_Lambda"] = 1e-4
    expected["Status"] = 1
    expected = {key: float(value) for key, value in expected.items()}

    # Act
    calibration_file_io.write_calibration_to_json(shift, error, 1.54, 1e-4, 1, str(target))

    # Assert
    assert target.read_text() == json.dumps(expected)


def test_provenance_round_trip(tmp_path):
    # Arrange
    shift, error = _shift_and_errors()
    target = str(tmp_path / "cal.json")
    provenance = {"schema": 1, "method": "ring", "steps": [{"step": 1, "routine": "CalibrateShift"}]}

    # Act
    calibration_file_io.write_calibration_to_json(shift, error, 1.54, 1e-4, 1, target, provenance=provenance)

    # Assert
    assert calibration_file_io.read_calibration_provenance(target) == provenance
    reread, _, wavelength, _, status = calibration_file_io.read_calibration_json_file(target)
    assert (reread.two_theta_0, wavelength, status) == (0.25, 1.54, 1)


def test_file_without_provenance_reads_none(tmp_path):
    # Arrange
    source = _write_calibration(tmp_path / "old.json", TTH_0=0.1, error_TTH_0=0.01)

    # Act / Assert
    assert calibration_file_io.read_calibration_provenance(source) is None


def test_unwritable_provenance_raises_and_writes_nothing(tmp_path):
    # Arrange
    shift, error = _shift_and_errors()
    target = tmp_path / "cal.json"

    # Act / Assert
    with pytest.raises(TypeError):
        calibration_file_io.write_calibration_to_json(
            shift, error, 1.54, 1e-4, 1, str(target), provenance={"x": object()}
        )
    assert not target.exists()


def test_provenance_not_a_dict_raises(tmp_path):
    # Arrange
    shift, error = _shift_and_errors()

    # Act / Assert
    with pytest.raises(TypeError, match="provenance must be a dict"):
        calibration_file_io.write_calibration_to_json(
            shift, error, 1.54, 1e-4, 1, str(tmp_path / "c.json"), provenance=[]
        )


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


def test_unknown_provenance_schema_warns(tmp_path):
    # Arrange
    shift, error = _shift_and_errors()
    target = str(tmp_path / "cal.json")
    calibration_file_io.write_calibration_to_json(shift, error, 1.54, 1e-4, 1, target, provenance={"schema": 99})

    # Act / Assert - returned as stored, with a warning
    with pytest.warns(UserWarning, match="provenance schema 99"):
        assert calibration_file_io.read_calibration_provenance(target) == {"schema": 99}


def test_geometry_round_trips_with_the_calibration(tmp_path):
    # Arrange
    from pyrs.core.instrument_geometry import DENEXDetectorGeometry

    shift, error = _shift_and_errors()
    geometry = DENEXDetectorGeometry(1024, 512, 3e-4, 6e-4, 0.985, False)
    target = str(tmp_path / "cal.json")

    # Act
    calibration_file_io.write_calibration_to_json(shift, error, 1.54, 1e-4, 1, target, geometry=geometry)
    read = calibration_file_io.read_calibration_geometry(target)

    # Assert - and the values still read as before
    assert (read.detector_size, read.pixel_dimension, read.arm_length) == ((1024, 512), (3e-4, 6e-4), 0.985)
    assert calibration_file_io.read_calibration_json_file(target)[2] == 1.54


def test_calibration_without_geometry_reads_none(tmp_path):
    # Arrange
    shift, error = _shift_and_errors()
    target = str(tmp_path / "cal.json")
    calibration_file_io.write_calibration_to_json(shift, error, 1.54, 1e-4, 1, target)

    # Act / Assert
    assert calibration_file_io.read_calibration_geometry(target) is None


# ---------------------------------------------------------------- starts and missing keys (#1098)

_SHIFT = DENEXDetectorShift(-0.0154027, -0.00255874, -0.0445212, 0.0642667, -0.254056, 0.16946, 0.85294)
_ERROR = DENEXDetectorShift(9.889e-05, 0.001167, 0.00026, 0.188275, 0.322219, 0.010681, 0.0042)
_WAVELENGTH, _WAVELENGTH_ERROR = 1.531347247129065, 8.25e-05
_PARAMETERS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")


def _json_without(tmp_path, *keys):
    """A complete calibration JSON with ``keys`` removed"""
    path = tmp_path / "cal.json"
    calibration_file_io.write_calibration_to_json(_SHIFT, _ERROR, _WAVELENGTH, _WAVELENGTH_ERROR, 1, str(path))
    stored = json.loads(path.read_text())
    for key in keys:
        del stored[key]
    path.write_text(json.dumps(stored))
    return str(path)


def _vector(shift, wavelength):
    return list(from_shift(shift)) + [wavelength]


def test_incomplete_json_is_refused_with_its_values(tmp_path):
    """A start without some parameters is refused by name (user decision, #1098); the values carry them as 0"""
    # Arrange
    path = _json_without(tmp_path, "Lambda", "error_Lambda")

    # Act
    with pytest.raises(calibration_file_io.IncompleteCalibrationError) as refused:
        calibration_file_io.read_calibration_values(path)

    # Assert
    assert refused.value.missing == ["Lambda"]
    assert "Lambda" in str(refused.value)
    assert refused.value.values["Lambda"] == 0.0
    assert refused.value.values["TTH_0"] == pytest.approx(0.85294)
    assert calibration_file_io.read_calibration_values(path, allow_missing=True) == refused.value.values


def test_legacy_two_theta_0_key_is_not_missing(tmp_path):
    """A start written with the legacy two_theta_0 key is complete"""
    # Arrange
    path = _json_without(tmp_path)
    stored = json.loads(open(path).read())
    stored["two_theta_0"] = stored.pop("TTH_0")
    stored["error_two_theta_0"] = stored.pop("error_TTH_0")
    with open(path, "w") as handle:
        json.dump(stored, handle)

    # Act
    values = calibration_file_io.read_calibration_values(path)

    # Assert
    assert values["TTH_0"] == pytest.approx(0.85294)


def test_several_missing_are_named_in_order(tmp_path):
    """The window shows this list: every missing parameter, in calibration order (#1098)"""
    # Arrange
    path = _json_without(tmp_path, "Lambda", "error_Lambda", "Rot_x", "error_Rot_x")

    # Act
    with pytest.raises(calibration_file_io.IncompleteCalibrationError) as refused:
        calibration_file_io.read_calibration_values(path)

    # Assert
    assert refused.value.missing == ["Rot_x", "Lambda"]
    assert "Rot_x, Lambda" in str(refused.value)
    assert "them as 0" in str(refused.value)


@pytest.mark.parametrize("parameter", _PARAMETERS)
def test_missing_error_reads_as_not_refined(tmp_path, parameter):
    """A value without its error is read with error -1 (user decision, #1098): only that error, every value"""
    # Arrange
    path = _json_without(tmp_path, f"error_{parameter}")
    expected = _vector(_ERROR, _WAVELENGTH_ERROR)
    expected[_PARAMETERS.index(parameter)] = -1.0

    # Act
    shift, shift_error, wavelength, wavelength_error, _ = calibration_file_io.read_calibration_file(path)

    # Assert
    np.testing.assert_allclose(_vector(shift_error, wavelength_error), expected)
    np.testing.assert_allclose(_vector(shift, wavelength), _vector(_SHIFT, _WAVELENGTH))


def test_json_without_wavelength_is_refused(tmp_path):
    """Reduction must not apply a wavelength of 0 (user decision, #1098): the reader and HidraSetup refuse the file;
    only a calibration start may read the missing Lambda as 0, after asking"""
    from pyrs.core.instrument_geometry import DENEXDetectorGeometry, HidraSetup

    # Arrange
    path = _json_without(tmp_path, "Lambda", "error_Lambda")

    # Act / Assert
    with pytest.raises(ValueError, match="no wavelength"):
        calibration_file_io.read_calibration_file(path)
    with pytest.raises(ValueError, match="no wavelength"):
        HidraSetup(DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)).load_calibration_file(path)
    with pytest.raises(calibration_file_io.IncompleteCalibrationError, match="Lambda"):
        calibration_file_io.read_calibration_values(path)


@pytest.mark.parametrize("wavelength", [0, -1.5, "nan", None], ids=["zero", "negative", "nan", "null"])
def test_unusable_wavelength_is_refused(tmp_path, wavelength):
    """A wavelength reduction cannot apply is refused like a missing one (user decision, #1098); a start may still
    read it, as the window asks about it"""
    # Arrange
    path = _json_without(tmp_path)
    stored = json.loads(open(path).read())
    stored["Lambda"] = float(wavelength) if wavelength == "nan" else wavelength
    with open(path, "w") as handle:
        handle.write(json.dumps(stored))

    # Act / Assert
    with pytest.raises(ValueError, match="unusable wavelength"):
        calibration_file_io.read_calibration_file(path)
    with pytest.raises(ValueError, match="unusable wavelength"):
        calibration_file_io.calibration_from_dict(stored, path)


def test_record_with_zero_wavelength_is_refused():
    """A project file reduced from a Lambda-less file before #1098 stores Lambda 0.0: reducing it again from that
    record is refused too"""
    # Arrange
    record = {name: 0.0 for name in _PARAMETERS}
    record.update({f"error_{name}": -1.0 for name in _PARAMETERS}, Status=1)

    # Act / Assert
    with pytest.raises(ValueError, match="unusable wavelength"):
        calibration_file_io.calibration_from_dict(record, "the project file's record")


def _write_run_file(path, mono_index=None, mrot=None):
    """A minimal NeXus run file holding only the logs the monochromator is read from."""
    import h5py

    with h5py.File(path, "w") as run:
        if mono_index is not None:
            run.create_dataset("entry/DASlogs/MonoSetting/value", data=np.array([mono_index, mono_index]))
        if mrot is not None:
            run.create_dataset("entry/DASlogs/mrot/value", data=np.array([mrot, mrot]))
    return str(path)


MONO_BY_INDEX = ["Si333", "Si511", "Si422", "Si331", "Si400", "Si311", "Si220"]
MONO_BY_MROT = [
    ("Si333", -39.5),
    ("Si511", -1.5),
    ("Si422", -19.2),
    ("Si331", -168.8),
    ("Si400", 15.0),
    ("Si311", -10.0),
    ("Si220", -180.0),
]


@pytest.mark.parametrize("index, mono", list(enumerate(MONO_BY_INDEX)))
def test_monochromator_of_run_from_monosetting(tmp_path, index, mono):
    """The MonoSetting log gives the monochromator, Si333 and Si511 included."""
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mono_index=index)

    assert calibration_file_io.monochromator_of_run(run_file).name == mono


@pytest.mark.parametrize("mono, mrot", MONO_BY_MROT)
def test_monochromator_of_run_from_mrot(tmp_path, mono, mrot):
    """Without a MonoSetting log, mrot gives the monochromator."""
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mrot=mrot)

    assert calibration_file_io.monochromator_of_run(run_file).name == mono


def test_monochromator_of_run_monosetting_wins_over_mrot(tmp_path):
    """A file with both logs is read through MonoSetting, as NeXusConvertingApp does."""
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mono_index=3, mrot=-19.2)

    assert calibration_file_io.monochromator_of_run(run_file).name == "Si331"


def test_monochromator_of_run_without_logs_raises(tmp_path):
    run_file = _write_run_file(tmp_path / "run.nxs.h5")

    with pytest.raises(ValueError, match="neither a MonoSetting nor an mrot"):
        calibration_file_io.monochromator_of_run(run_file)


def test_monochromator_of_run_unmatched_mrot_raises(tmp_path):
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mrot=100.0)

    with pytest.raises(ValueError, match="Unable to determine monosetting"):
        calibration_file_io.monochromator_of_run(run_file)


def test_monochromator_of_run_bad_index_raises(tmp_path):
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mono_index=9)

    with pytest.raises(ValueError, match="invalid MonoSetting"):
        calibration_file_io.monochromator_of_run(run_file)


def test_monochromator_of_run_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibration_file_io.monochromator_of_run(str(tmp_path / "absent.nxs.h5"))


def _shared_folder(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_text("{}")
    return str(tmp_path)


@pytest.mark.parametrize("index, mono", list(enumerate(MONO_BY_INDEX)))
def test_retrieve_calibration_picks_run_monochromator(tmp_path, index, mono):
    """Each monochromator, Si333 and Si511 included, gets its own file, not its neighbour's."""
    folder = _shared_folder(tmp_path, "HB2B_Latest.json", *(f"HB2B_Latest_{m}.json" for m in MONO_BY_INDEX))
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mono_index=index)

    result = calibration_file_io.retrieve_calibration(run_file, folder)

    assert result == os.path.join(folder, f"HB2B_Latest_{mono}.json")


def test_retrieve_calibration_from_mrot(tmp_path):
    folder = _shared_folder(tmp_path, "HB2B_Latest_Si331.json", "HB2B_Latest_Si422.json")
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mrot=-168.8)

    assert calibration_file_io.retrieve_calibration(run_file, folder).endswith("HB2B_Latest_Si331.json")


def test_retrieve_calibration_missing_falls_back_with_warning(tmp_path):
    """No per-monochromator file yet: HB2B_Latest.json, and the user is told."""
    folder = _shared_folder(tmp_path, "HB2B_Latest.json", "HB2B_Latest_Si422.json")
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mrot=-168.8)

    with pytest.warns(UserWarning, match="HB2B_Latest_Si331.json"):
        result = calibration_file_io.retrieve_calibration(run_file, folder)

    assert result == os.path.join(folder, "HB2B_Latest.json")


def test_retrieve_calibration_nothing_available_raises(tmp_path):
    folder = _shared_folder(tmp_path, "HB2B_Latest_Si422.json")
    run_file = _write_run_file(tmp_path / "run.nxs.h5", mrot=-168.8)

    with pytest.raises(FileNotFoundError, match="Si331"):
        calibration_file_io.retrieve_calibration(run_file, folder)


def test_retrieve_calibration_real_run_file():
    """A real HB2B run (mrot = -19.2, Si422) from the test data."""
    run_file = os.path.join(os.path.dirname(__file__), "../../../data/HB2B_1017.nxs.h5")

    assert calibration_file_io.monochromator_of_run(run_file).name == "Si422"
