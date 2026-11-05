"""A reduced project file records the calibration the reduction applied (#1053 part B)"""

import h5py
import pytest

from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

VALUES = [0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25, 1.54]
ERRORS = [1e-4, 2e-4, 3e-4, 0.01, 0.02, 0.03, -1.0, 1e-5]
PROVENANCE = {"schema": 1, "method": "ring", "created": "2026-11-03T11:40:37-05:00", "steps": [{"step": 1}]}
FROM_FILE = {
    "source": "file",
    "values": VALUES,
    "errors": ERRORS,
    "status": 1,
    "created": PROVENANCE["created"],
    "created_source": "calibration provenance",
    "provenance": PROVENANCE,
}


def write(path, mode=HidraProjectFileMode.OVERWRITE, **kwargs):
    project = HidraProjectFile(str(path), mode)
    project.write_reduction_provenance(**kwargs)
    project.save(verbose=False)


def read(path):
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    record = project.read_reduction_provenance()
    project.close()
    return record


def test_file_calibration_round_trips(tmp_path):
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    write(path, calibration_file="calibration.json", applied_calibration=FROM_FILE)
    record = read(path)

    # Assert
    assert record["source"] == "file" and record["calibration_file"] == "calibration.json"
    assert record["values"] == VALUES and record["errors"] == ERRORS and record["status"] == 1
    assert record["created"] == PROVENANCE["created"]
    assert record["created_source"] == "calibration provenance"
    assert record["provenance"] == PROVENANCE


def test_provenance_is_a_string_dataset_with_names_on_values(tmp_path):
    """JSON text in a dataset, not an attribute: HDF5 attributes are limited to ~64 kB"""
    # Arrange
    path = tmp_path / "reduced.h5"
    large = dict(PROVENANCE, steps=[{"step": i, "message": "x" * 200} for i in range(500)])  # ~110 kB

    # Act
    write(path, applied_calibration=dict(FROM_FILE, provenance=large))

    # Assert
    with h5py.File(path, "r") as f:
        group = f[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]
        assert isinstance(group["provenance_json"], h5py.Dataset)
        assert list(group["values"].attrs["names"]) == [
            "Shift_x",
            "Shift_y",
            "Shift_z",
            "Rot_x",
            "Rot_y",
            "Rot_z",
            "TTH_0",
            "Lambda",
        ]
    assert read(path)["provenance"] == large


def test_workspace_calibration_has_no_invented_errors(tmp_path):
    # Arrange
    path = tmp_path / "reduced.h5"
    workspace = {
        "source": "workspace",
        "values": VALUES,
        "errors": None,
        "status": None,
        "created": None,
        "created_source": None,
        "provenance": None,
    }

    # Act
    write(path, applied_calibration=workspace)
    record = read(path)

    # Assert - the shift that was applied; what is not known is absent, not made up
    assert record["source"] == "workspace" and record["values"] == VALUES
    assert not {"errors", "status", "provenance", "created"} & set(record)


def test_rereduction_replaces_the_earlier_record(tmp_path):
    """#1053 part B review (B1): no attribute of the earlier reduction survives, e.g. its calibration file"""
    # Arrange
    path = tmp_path / "reduced.h5"
    write(path, calibration_file="calibration.json", vanadium_run="HB2B_931.h5", applied_calibration=FROM_FILE)

    # Act - reduced again into the same file, this time without a calibration
    write(path, HidraProjectFileMode.READWRITE, applied_calibration={"source": "none"})
    record = read(path)

    # Assert
    assert record == {"source": "none"}  # nothing of the earlier record survives, not even its file name


def test_unpreparable_record_writes_nothing_half(tmp_path):
    """#1053 part B review (B3): a record that cannot be prepared leaves only 'unknown' and why"""
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    with pytest.warns(UserWarning, match="could not be recorded"):
        write(path, applied_calibration=dict(FROM_FILE, provenance={"x": object()}))
    record = read(path)

    # Assert - no values without their provenance: the record says it is unknown, and why
    assert record["source"] == "unknown" and "TypeError" in record["error"]
    assert not {"values", "errors", "status", "provenance"} & set(record)


def test_failed_write_removes_what_was_written(tmp_path, monkeypatch):
    # Arrange - the file fails while the record is being written
    real = h5py.Group.create_dataset

    def failing(self, name, *args, **kwargs):
        if name == "status":
            raise OSError("disk full")
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", failing)
    path = tmp_path / "reduced.h5"

    # Act
    with pytest.warns(UserWarning, match="could not be written"):
        write(path, applied_calibration=FROM_FILE)
    monkeypatch.setattr(h5py.Group, "create_dataset", real)
    record = read(path)

    # Assert
    assert record["source"] == "unknown" and "disk full" in record["error"]
    assert not {"values", "errors", "status", "provenance"} & set(record)


def test_reduction_error_is_kept_beside_a_write_error(tmp_path):
    # Arrange - the reduction already could not record something
    path = tmp_path / "reduced.h5"
    applied = {
        "source": "unknown",
        "error": "applied calibration could not be recorded: OSError: gone",
        "provenance": {"x": object()},
    }

    # Act
    with pytest.warns(UserWarning):
        write(path, applied_calibration=applied)

    # Assert - both reasons survive
    error = read(path)["error"]
    assert "OSError: gone" in error and "TypeError" in error


def test_missing_calibration_group_is_created(tmp_path):
    """#1053 part B review (B2): an older file without the group must still be saved"""
    # Arrange
    path = tmp_path / "old.h5"
    write(path, calibration_file="calibration.json")
    with h5py.File(path, "a") as f:
        del f[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]

    # Act
    write(path, HidraProjectFileMode.READWRITE, applied_calibration=FROM_FILE)

    # Assert
    assert read(path)["source"] == "file"


def test_errors_say_what_minus_one_and_two_mean(tmp_path):
    # Arrange
    path = tmp_path / "reduced.h5"

    # Act
    write(path, applied_calibration=FROM_FILE)

    # Assert
    with h5py.File(path, "r") as f:
        codes = f[HidraConstants.INSTRUMENT][HidraConstants.CALIBRATION]["errors"].attrs["codes"]
    assert "-1" in codes and "-2" in codes


def test_provenance_nan_is_sanitised(tmp_path):
    # Arrange - an older or hand-edited block with a NaN
    path = tmp_path / "reduced.h5"

    # Act
    write(path, applied_calibration=dict(FROM_FILE, provenance={"schema": 1, "rmse": float("nan")}))

    # Assert
    assert read(path)["provenance"] == {"schema": 1, "rmse": None}


def test_file_without_a_record_reads_as_before(tmp_path):
    # Arrange - what a reduction before #1053 part B wrote
    path = tmp_path / "old.h5"
    write(path, calibration_file="calibration.json")

    # Act
    record = read(path)

    # Assert
    assert record == {"calibration_file": "calibration.json"}
