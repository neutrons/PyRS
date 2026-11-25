"""A reduced project file records the calibration the reduction applied, read from its HidraSetup (#1053)"""

import json

import h5py
import pytest

from pyrs.core.calibration import UNKNOWN_ERROR, CalibrationSource, nominal_calibration
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift, HidraSetup
from pyrs.core.monosetting import MonoSetting
from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from tests.util.workspace_helpers import make_minimal_HidraWorkspace

GEOMETRY = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)
SHIFT = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)
ERROR = DENEXDetectorShift(1e-4, 2e-4, 3e-4, 0.01, 0.02, 0.03, -1.0)
PROVENANCE = {"schema": 1, "method": "ring", "created": "2026-11-03T11:40:37-05:00", "steps": [{"step": 1}]}


@pytest.fixture
def calibration_file(tmp_path):
    path = tmp_path / "HB2B_CAL.json"
    write_calibration_to_json(SHIFT, ERROR, 1.5306, 1e-5, 1, str(path), provenance=PROVENANCE, geometry=GEOMETRY)
    return path


@pytest.fixture
def calibrated(calibration_file):
    instrument = HidraSetup(GEOMETRY)
    instrument.load_calibration_file(str(calibration_file))
    return instrument


def write(path, mode=HidraProjectFileMode.OVERWRITE, **kwargs):
    project = HidraProjectFile(str(path), mode)
    project.write_reduction_provenance(**kwargs)
    project.save(verbose=False)


def read(path):
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    record = project.read_reduction_provenance()
    project.close()
    return record


def test_hidra_setup_holds_the_calibration_file(calibrated, calibration_file):
    # Assert - everything about the calibration lives on the instrument object
    assert calibrated.calibration_source == "file" and calibrated.calibration_status == 1
    assert calibrated.geometry_shift.two_theta_0 == 0.25 and calibrated.geometry_shift_error.center_shift_x == 1e-4
    assert calibrated.get_wavelength(None) == 1.5306 and calibrated.wavelength_error == 1e-5
    assert calibrated.calibration_provenance == PROVENANCE
    record = calibrated.calibration_record()
    assert record["calibration_file"] == str(calibration_file) and len(record["calibration_file_sha256"]) == 64
    assert record["created"] == PROVENANCE["created"] and record["created_source"] == "calibration provenance"
    assert record["calibration"]["TTH_0"] == 0.25 and record["calibration"]["Status"] == 1.0


def test_record_round_trips_to_an_equal_hidra_setup(tmp_path, calibrated):
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act - written from the instrument, rebuilt into an instrument
    write(path, instrument=calibrated)
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    rebuilt = project.read_reduction_instrument(GEOMETRY)

    # Assert
    assert rebuilt.calibration_record() == calibrated.calibration_record()


def test_calibration_is_stored_in_its_own_json_form(tmp_path, calibrated):
    """The calibration JSON's dictionary (as write_calibration_to_json), as a string dataset: no copy of
    the parameter names, and no ~64 kB attribute limit"""
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    write(path, instrument=calibrated)

    # Assert
    with h5py.File(path, "r") as f:
        group = f[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]
        assert isinstance(group["calibration_json"], h5py.Dataset) and "-2" in group["calibration_json"].attrs["codes"]
        assert not {"values", "errors", "status"} & set(group)
    assert read(path)["calibration"] == calibrated.calibration_dict()


def test_workspace_calibration_has_no_invented_errors(tmp_path):
    # Arrange - a workspace shift with no calibrated wavelength
    class Workspace:
        def get_detector_shift(self):
            return SHIFT

        def get_sub_runs(self):
            return [1, 2]

        def get_wavelength(self, calibrated, throw, sub_run=None):
            raise RuntimeError("There is no calibrated wave length")

    instrument = HidraSetup(GEOMETRY)
    instrument.use_workspace_calibration(Workspace())
    path = tmp_path / "reduced.h5"

    # Act
    write(path, instrument=instrument)
    record = read(path)

    # Assert - the shift that was applied; errors, status and the nominal wavelength are not made up
    assert record["source"] == "workspace" and record["calibration"]["Shift_x"] == 0.001
    assert record["calibration"]["Lambda"] is None and "error_Shift_x" not in record["calibration"]
    assert record["lambda_source"] == "workspace has no calibrated wavelength: not recorded"


def test_rereduction_replaces_the_earlier_record(tmp_path, calibrated):
    """#1053 part B review (B1): no attribute of the earlier reduction survives, e.g. its calibration file"""
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, vanadium_run="HB2B_931.h5", instrument=calibrated)

    # Act - reduced again into the same file, this time without a calibration
    write(path, HidraProjectFileMode.READWRITE, instrument=HidraSetup(GEOMETRY))

    # Assert - the instrument here has no mask record either
    assert read(path) == {"source": "none", "reduction": {"mask": {"source": "not recorded"}}}


def test_unbuildable_record_fails_the_save(tmp_path, monkeypatch, calibrated):
    """A record that cannot be built fails the save (#1141): "unknown" is never written"""

    # Arrange
    def broken():
        raise AttributeError("no shift")

    monkeypatch.setattr(calibrated, "calibration_record", broken)
    applied = nominal_calibration(MonoSetting.Si422, l2=0.985)

    # Act / Assert - the check a save runs before opening its file
    with pytest.raises(RuntimeError, match="could not be recorded, so nothing is saved: AttributeError: no shift"):
        HidraProjectFile.check_reduction_provenance(instrument=calibrated, applied=applied)


@pytest.mark.parametrize("failing_dataset", ["provenance_json", "applied_calibration_json"])
def test_failed_write_raises_and_removes_what_was_written(tmp_path, monkeypatch, calibrated, failing_dataset):
    """The file failing mid-record raises, and leaves nothing half-written: neither record survives alone (#1141)"""
    # Arrange - the file fails while one of the records is being written
    real = h5py.Group.create_dataset

    def failing(self, name, *args, **kwargs):
        if name == failing_dataset:
            raise OSError("disk full")
        return real(self, name, *args, **kwargs)

    path = tmp_path / "reduced.h5"
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    applied = calibrated.applied_calibration(l2=0.985, mono=MonoSetting.Si422)
    monkeypatch.setattr(h5py.Group, "create_dataset", failing)

    # Act
    with pytest.raises(RuntimeError, match="could not be written: OSError: disk full"):
        project.write_reduction_provenance(instrument=calibrated, applied=applied)
    monkeypatch.setattr(h5py.Group, "create_dataset", real)
    project.save(verbose=False)
    record = read(path)

    # Assert - neither "unknown" nor a partial record
    assert "source" not in record and not {"calibration", "provenance", "calibration_hash"} & set(record)


def test_missing_calibration_group_is_created(tmp_path, calibrated):
    """#1053 part B review (B2): an older file without the group must still be saved"""
    # Arrange
    path = tmp_path / "old.h5"
    write(path, vanadium_run="HB2B_931.h5")
    with h5py.File(path, "a") as f:
        del f[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]

    # Act
    write(path, HidraProjectFileMode.READWRITE, instrument=calibrated)

    # Assert
    assert read(path)["source"] == "file"


def test_provenance_failure_is_noted_not_raised(calibration_file, monkeypatch):
    # Arrange
    from pyrs.utilities import calibration_file_io

    def broken(path):
        raise OSError("calibration file vanished")

    monkeypatch.setattr(calibration_file_io, "read_calibration_provenance", broken)
    instrument = HidraSetup(GEOMETRY)

    # Act
    instrument.load_calibration_file(str(calibration_file))

    # Assert - the calibration is applied; the record says why its provenance is missing
    assert instrument.geometry_shift.center_shift_x == 0.001
    assert "calibration file vanished" in instrument.calibration_record()["error"]


def test_geometry_lives_within_the_calibration(tmp_path, calibrated):
    """#1053: the calibration carries the detector geometry it is valid for; a reduced file rebuilds its
    whole instrument from the calibration record alone"""
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, instrument=calibrated)

    # Act
    record = read(path)
    rebuilt = HidraProjectFile(str(path), HidraProjectFileMode.READONLY).read_reduction_instrument()

    # Assert
    assert record["calibration"]["geometry"] == {
        "detector_size": [1024, 1024],
        "pixel_dimension": [0.3 / 1024, 0.3 / 1024],
        "arm_length": 0.985,
    }
    assert record["geometry_source"] == "calibration file" and "geometry_note" not in record
    with h5py.File(path, "r") as f:  # not the separate group: only within the calibration
        assert (
            "detector size"
            not in f[HidraConstants.INSTRUMENT][HidraConstants.GEOMETRY_SETUP][HidraConstants.DETECTOR_PARAMS]
        )
    assert rebuilt.get_instrument_geometry(False).detector_size == (1024, 1024)
    assert rebuilt.calibration_record() == calibrated.calibration_record()


def test_calibration_for_another_geometry_is_flagged(tmp_path):
    # Arrange - a calibration made for a detector with a different L2
    other = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.95, False)
    path = tmp_path / "other.json"
    write_calibration_to_json(SHIFT, ERROR, 1.5306, 1e-5, 1, str(path), geometry=other)
    instrument = HidraSetup(GEOMETRY)

    # Act
    instrument.load_calibration_file(str(path))
    project_path = tmp_path / "reduced.h5"
    write(project_path, instrument=instrument)
    record = read(project_path)
    rebuilt = HidraProjectFile(str(project_path), HidraProjectFileMode.READONLY).read_reduction_instrument()

    # Assert - flagged; the file keeps the run's geometry, which the reduction used
    assert "differs from the run's" in record["geometry_note"]
    assert record["calibration"]["geometry"]["arm_length"] == 0.95 and record["run_geometry"]["arm_length"] == 0.985
    assert rebuilt.get_instrument_geometry(False).arm_length == 0.985


def test_older_calibration_takes_the_run_geometry_labelled(tmp_path):
    # Arrange - written before calibrations carried their geometry
    path = tmp_path / "old.json"
    write_calibration_to_json(SHIFT, ERROR, 1.5306, 1e-5, 1, str(path))
    instrument = HidraSetup(GEOMETRY)

    # Act
    instrument.load_calibration_file(str(path))

    # Assert
    record = instrument.calibration_record()
    assert record["geometry_source"] == "run (the calibration file has no geometry)"
    assert record["calibration"]["geometry"]["arm_length"] == 0.985 and "geometry_note" not in record


def test_rebuild_without_geometry_says_what_to_pass(tmp_path):
    # Arrange - a record with no geometry anywhere (written without an instrument)
    path = tmp_path / "reduced.h5"
    write(path, applied_calibration={"source": "file", "calibration": {"Shift_x": 0.0}})
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)

    # Act / Assert
    with pytest.raises(RuntimeError, match="pass the run's geometry"):
        project.read_reduction_instrument()


def test_file_without_a_record_reads_as_before(tmp_path):
    # Arrange - what a reduction before #1053 part B wrote (now with the file's format, #1098)
    path = tmp_path / "old.h5"
    write(path, calibration_file="calibration.json")

    # Act / Assert
    assert read(path) == {"calibration_file": "calibration.json", "calibration_format": "json"}


MASK = {"source": "file", "file": "/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml", "sha256": "f00d" * 16}
SETTINGS = {"num_bins": 720, "min_2theta": None, "eta_step": 3.0, "eta_min": -8.2, "eta_max": 8.2, "extra_mask": None}


@pytest.fixture
def masked(calibrated):
    calibrated.set_mask_record(MASK)
    return calibrated


def test_mask_and_settings_are_recorded_beside_calibration(tmp_path, masked):
    """#1068: a sibling REDUCTION group, read back under "reduction"; the calibration record is unchanged"""
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    write(path, instrument=masked, reduction_settings=SETTINGS)

    # Assert
    record = read(path)
    assert record["reduction"] == {"mask": MASK, "settings": SETTINGS}
    assert record["source"] == "file"
    with h5py.File(path, "r") as h5:
        assert sorted(h5[HidraConstants.INSTRUMENT][HidraConstants.REDUCTION]) == ["reduction_json"]
        assert "reduction_json" not in h5[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]


def test_rereduction_replaces_the_mask_record(tmp_path, masked):
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, instrument=masked, reduction_settings=SETTINGS)
    unmasked = HidraSetup(GEOMETRY)
    unmasked.set_mask_record({"source": "none"})

    # Act - reduced again without a mask and without settings
    write(path, HidraProjectFileMode.READWRITE, instrument=unmasked)

    # Assert - nothing of the earlier record survives
    assert read(path)["reduction"] == {"mask": {"source": "none"}}


def test_failed_reduction_write_is_recorded(tmp_path, monkeypatch, masked):
    # Arrange
    real = h5py.Group.create_dataset

    def failing(self, name, *args, **kwargs):
        if name == "reduction_json":
            raise OSError("disk full")
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", failing)
    path = tmp_path / "reduced.h5"

    # Act
    with pytest.warns(UserWarning, match="mask and binning could not be recorded"):
        write(path, instrument=masked, reduction_settings=SETTINGS)

    # Assert - the reason, and the calibration record is still complete
    record = read(path)
    assert record["reduction"] == {"error": "mask and binning could not be recorded: OSError: disk full"}
    assert record["calibration"]["TTH_0"] == 0.25


def test_recorded_mask_rebuilds_on_the_instrument(tmp_path, masked):
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, instrument=masked, reduction_settings=SETTINGS)

    # Act
    instrument = HidraProjectFile(str(path), HidraProjectFileMode.READONLY).read_reduction_instrument(GEOMETRY)

    # Assert
    assert instrument.mask_record == MASK


def test_file_without_reduction_group_has_no_reduction_entry(tmp_path, calibrated):
    """Files reduced before #1068 keep reading as before"""
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, applied_calibration=calibrated.calibration_record())

    # Act / Assert
    assert "reduction" not in read(path)


def test_record_names_the_calibration_format(tmp_path, calibrated):
    """The project file says whether the calibration file was JSON or NeXus (#1098)"""
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    write(path, instrument=calibrated)
    record = read(path)

    # Assert
    assert record["calibration_file"].endswith(".json") and record["calibration_format"] == "json"


@pytest.mark.parametrize("file_format", ["json", "nexus"])
def test_missing_error_is_recorded_as_not_refined(tmp_path, file_format):
    """The reduced file records -1 for an error the calibration file does not give, and every other value and error
    as given (#1098)"""
    import json

    from pyrs.utilities.calibration_nexus import write_calibration_nexus

    # Arrange - the calibration without Rot_y's error
    if file_format == "json":
        path = tmp_path / "HB2B_CAL.json"
        write_calibration_to_json(SHIFT, ERROR, 1.5306, 1e-5, 1, str(path), geometry=GEOMETRY)
        stored = json.loads(path.read_text())
        del stored["error_Rot_y"]
        path.write_text(json.dumps(stored))
    else:
        path = tmp_path / "HB2B_CAL.nxs"
        write_calibration_nexus(str(path), SHIFT, ERROR, 1.5306, 1e-5, 1, GEOMETRY)
        with h5py.File(path, "a") as nexus:
            del nexus["entry/instrument/detector/transformations/rotation_y_errors"]
    instrument = HidraSetup(GEOMETRY)
    instrument.load_calibration_file(str(path))
    reduced = tmp_path / "reduced.h5"

    # Act
    write(reduced, instrument=instrument)
    calibration = read(reduced)["calibration"]

    # Assert
    expected_errors = ERROR.convert_to_dict()
    expected_errors["Rot_y"] = -1.0
    assert {name: calibration[f"error_{name}"] for name in expected_errors} == pytest.approx(expected_errors)
    assert {name: calibration[name] for name in SHIFT.convert_to_dict()} == pytest.approx(SHIFT.convert_to_dict())
    assert calibration["error_Lambda"] == pytest.approx(1e-5) and calibration["Lambda"] == pytest.approx(1.5306)


def test_applied_calibration_from_a_file(calibrated):
    """The file's values, errors, wavelength and status, with the L2 reduction used (#1141)"""
    # Act
    applied = calibrated.applied_calibration(l2=0.985, mono=MonoSetting.Si422)

    # Assert
    assert applied.source is CalibrationSource.FILE and applied.status == 1
    assert applied.shift.convert_to_dict() == SHIFT.convert_to_dict()
    assert applied.shift_error.convert_to_dict() == ERROR.convert_to_dict()
    assert (applied.wavelength, applied.wavelength_error, applied.l2) == (1.5306, 1e-5, 0.985)


def test_applied_calibration_without_a_file_is_nominal():
    """No calibration file: the nominal calibration of the run's monochromator (#1141)"""
    # Act
    applied = HidraSetup(GEOMETRY).applied_calibration(l2=0.985, mono=MonoSetting.Si422)

    # Assert
    assert applied.source is CalibrationSource.NONE
    assert applied.same_calibration_as(nominal_calibration(MonoSetting.Si422, l2=0.985))


def test_applied_calibration_of_a_workspace_shift():
    """A workspace's own shift has unknown errors (-2) and no refinement status (#1141)"""
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2, 3])
    instrument = HidraSetup(GEOMETRY)
    instrument.use_workspace_calibration(workspace)

    # Act
    applied = instrument.applied_calibration(l2=0.985, mono=MonoSetting.Si422)

    # Assert
    assert applied.source is CalibrationSource.WORKSPACE and applied.status is None
    assert applied.shift.convert_to_dict() == workspace.get_detector_shift().convert_to_dict()
    assert applied.errors == (UNKNOWN_ERROR,) * 7
    assert (applied.wavelength, applied.wavelength_error) == (1.486, UNKNOWN_ERROR)


def test_recorded_source_is_the_plain_string(calibrated):
    """Project files keep the strings they always held: existing records read back unchanged (#1141)"""
    # Act
    source = calibrated.calibration_record()["source"]

    # Assert
    assert source == "file" and type(source) is str


@pytest.mark.parametrize("record", [{"source": "unknown"}, {"calibration": None}], ids=["unknown", "missing"])
def test_record_that_cannot_be_stated_raises(record):
    """A record whose source is "unknown" or missing is rejected, never read as no calibration (#1141)"""
    # Arrange
    instrument = HidraSetup(GEOMETRY)

    # Act / Assert
    with pytest.raises(ValueError, match="run.h5: calibration source"):
        instrument.set_calibration_record(record, where="run.h5")


def test_applied_record_round_trips(tmp_path, calibrated):
    """The record and its hash are stored, and read back checked against the hash (#1141)"""
    # Arrange
    path = tmp_path / "reduced.h5"
    applied = calibrated.applied_calibration(l2=0.985, mono=MonoSetting.Si422)

    # Act
    write(path, instrument=calibrated, applied=applied)
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    restored = project.read_applied_calibration()
    project.close()

    # Assert
    assert restored == applied
    with h5py.File(path, "r") as h5:
        stored_hash = h5[f"{HidraConstants.INSTRUMENT}/{HidraConstants.CALIBRATION}"].attrs["calibration_hash"]
    assert stored_hash == applied.content_hash


def test_file_without_applied_record_raises(tmp_path, calibrated):
    """A file reduced before #1141 has the descriptive record but not the applied one"""
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, instrument=calibrated)
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)

    # Act / Assert
    with pytest.raises(RuntimeError, match="no applied calibration record"):
        project.read_applied_calibration()
    project.close()


def test_tampered_applied_record_raises(tmp_path, calibrated):
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, instrument=calibrated, applied=calibrated.applied_calibration(l2=0.985, mono=MonoSetting.Si422))
    with h5py.File(path, "a") as h5:
        group = h5[f"{HidraConstants.INSTRUMENT}/{HidraConstants.CALIBRATION}"]
        stored = json.loads(group["applied_calibration_json"][()])
        stored["l2"] = 0.9
        del group["applied_calibration_json"]
        group.create_dataset("applied_calibration_json", data=json.dumps(stored))
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)

    # Act / Assert
    with pytest.raises(ValueError, match="does not match its stored hash"):
        project.read_applied_calibration()
    project.close()
