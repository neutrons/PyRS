"""Combine runs checks the calibrations of the project files it merges (#1063)"""

import shutil

import pytest

from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.projectfile.file_object import CalibrationMismatchError, VanadiumMismatchError

pytestmark = pytest.mark.integration

FILES = ["tests/data/HB2B_1327.h5", "tests/data/HB2B_1328.h5"]  # historic: reduced before #1053
CALIBRATION = {
    "Shift_x": 0.001,
    "Shift_y": -0.002,
    "Shift_z": 0.003,
    "Rot_x": 0.1,
    "Rot_y": -0.2,
    "Rot_z": 0.3,
    "TTH_0": 0.25,
    "Lambda": 1.5306,
    "Status": 1.0,
}
RECORD = {"source": "file", "calibration": CALIBRATION, "provenance": {"schema": 1}}


def _copies(tmp_path, records):
    """Copies of the test project files, each given the calibration record (None: left historic)."""
    paths = []
    for original, record in zip(FILES, records):
        path = tmp_path / original.split("/")[-1]
        shutil.copy(original, path)
        if record is not None:
            project = HidraProjectFile(str(path), HidraProjectFileMode.READWRITE)
            project.write_reduction_provenance(
                calibration_file="HB2B_CAL.json", vanadium_run=record.pop("vanadium", None), applied_calibration=record
            )
            project.save(verbose=False)
        paths.append(str(path))
    return paths


def _exported(model, tmp_path):
    out = tmp_path / "combined.h5"
    model.export_project_files(str(out))
    project = HidraProjectFile(str(out), HidraProjectFileMode.READONLY)
    record = project.read_reduction_provenance()
    project.close()
    return record


def test_same_calibration_is_combined_with_its_record(tmp_path):
    # Arrange
    paths = _copies(tmp_path, [dict(RECORD), dict(RECORD)])
    model = CombineRunsModel()

    # Act
    model.combine_project_files(paths)
    record = _exported(model, tmp_path)

    # Assert
    assert model.warnings == []
    assert record["source"] == "file" and record["calibration"] == CALIBRATION
    assert record["calibration_file"] == "HB2B_CAL.json"
    assert [source["file"] for source in record["sources"]] == paths


def test_different_calibrations_are_refused_before_loading(tmp_path):
    # Arrange
    paths = _copies(tmp_path, [dict(RECORD), dict(RECORD, calibration=dict(CALIBRATION, Lambda=1.5291))])
    model = CombineRunsModel()

    # Act / Assert - nothing combined
    with pytest.raises(CalibrationMismatchError, match="HB2B_1327.h5:.*\n.*HB2B_1328.h5:"):
        model.combine_project_files(paths)
    assert model._hidra_ws is None


def test_vanadium_difference_needs_approval(tmp_path):
    # Arrange
    paths = _copies(tmp_path, [dict(RECORD, vanadium="HB2B_931.h5"), dict(RECORD, vanadium="HB2B_932.h5")])
    model = CombineRunsModel()

    # Act / Assert
    with pytest.raises(VanadiumMismatchError):
        model.combine_project_files(paths)
    model.combine_project_files(paths, accept_vanadium_differences=True)
    record = _exported(model, tmp_path)
    assert "vanadium_run" not in record and "approved by the user" in record["warning"]


def test_historic_files_are_combined_and_flagged(tmp_path):
    # Arrange
    model = CombineRunsModel()

    # Act
    model.combine_project_files(FILES)
    record = _exported(model, tmp_path)

    # Assert
    assert len(model.warnings) == 1 and "2 of 2 files have no calibration record" in model.warnings[0]
    assert record["source"] == "not recorded" and len(record["sources"]) == 2
