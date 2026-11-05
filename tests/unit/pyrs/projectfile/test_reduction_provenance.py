"""A reduced project file records the calibration the reduction applied, read from its HidraSetup (#1053)"""

import h5py
import pytest

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift, HidraSetup
from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import write_calibration_to_json

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

    # Assert
    assert read(path) == {"source": "none"}


def test_unbuildable_record_does_not_stop_the_save(tmp_path, monkeypatch, calibrated):
    # Arrange
    def broken():
        raise AttributeError("no shift")

    monkeypatch.setattr(calibrated, "calibration_record", broken)
    path = tmp_path / "reduced.h5"

    # Act
    with pytest.warns(UserWarning, match="could not be recorded"):
        write(path, instrument=calibrated)

    # Assert - saved; the record says it is unknown, and why
    assert read(path) == {
        "source": "unknown",
        "error": "applied calibration could not be recorded: AttributeError: no shift",
    }


def test_failed_write_removes_what_was_written(tmp_path, monkeypatch, calibrated):
    # Arrange - the file fails while the record is being written
    real = h5py.Group.create_dataset

    def failing(self, name, *args, **kwargs):
        if name == "provenance_json":
            raise OSError("disk full")
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", failing)
    path = tmp_path / "reduced.h5"

    # Act
    with pytest.warns(UserWarning, match="could not be written"):
        write(path, instrument=calibrated)
    monkeypatch.setattr(h5py.Group, "create_dataset", real)
    record = read(path)

    # Assert - nothing half-written beside "unknown"
    assert record["source"] == "unknown" and "disk full" in record["error"]
    assert not {"calibration", "provenance"} & set(record)


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
    # Arrange - what a reduction before #1053 part B wrote
    path = tmp_path / "old.h5"
    write(path, calibration_file="calibration.json")

    # Act / Assert
    assert read(path) == {"calibration_file": "calibration.json"}
