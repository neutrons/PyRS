"""Saving a peak fit of several runs writes a new combined project file (#1080)

The run files are never modified, and, as for "Combine runs" (#1063), runs reduced with different
calibrations are refused.
"""

import hashlib
import os
import shutil

import pytest

from pyrs.core.pyrscore import PyRsCore
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.peaks.peak_fit_engine import FitResult
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration

FILES = ["tests/data/HB2B_1327.h5", "tests/data/HB2B_1328.h5"]
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
REDUCTION = {"mask": {"source": "none"}, "settings": {"num_bins": 720, "eta_step": None}}


def _record(calibration):
    return {"source": "file", "calibration": calibration, "provenance": {"schema": 1}, "reduction": REDUCTION}


def _run_copies(tmp_path, records):
    """Copies of the test project files, each given a calibration record."""
    paths = []
    for original, record in zip(FILES, records):
        path = tmp_path / os.path.basename(original)
        shutil.copy(original, path)
        project = HidraProjectFile(str(path), HidraProjectFileMode.READWRITE)
        project.write_reduction_provenance(calibration_file="HB2B_CAL.json", applied_calibration=record)
        project.save(verbose=False)
        paths.append(str(path))
    return paths


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _loaded_model(paths):
    """A model with the runs loaded and, as its fit, the peak stored in the first run."""
    model = PeakFittingModel(peak_fit_core=PyRsCore())
    model.load_hidra_project(paths)
    project = HidraProjectFile(paths[0], HidraProjectFileMode.READONLY)
    peaks = project.read_peak_parameters("peak0")
    project.close()
    model.fit_result = FitResult(peakcollections=(peaks,), fitted=None, difference=None)
    return model


def test_save_writes_combined_file_and_keeps_runs(tmp_path):
    """Save (which passes the loaded runs) writes <dir>/HB2B_1327HB2B_1328.h5 with data, record and fit."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(CALIBRATION), _record(CALIBRATION)])
    digests = [_digest(path) for path in paths]
    model = _loaded_model(paths)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(paths)

    # Assert
    assert failures == []
    combined = tmp_path / "HB2B_1327HB2B_1328.h5"
    project = HidraProjectFile(str(combined), HidraProjectFileMode.READONLY)
    assert list(project.read_sub_runs()) == list(model.hidra_workspace.get_sub_runs())
    assert project.read_peak_tags() == ["peak0"]
    record = project.read_reduction_provenance()
    project.close()
    assert record["calibration"] == CALIBRATION
    assert [source["file"] for source in record["sources"]] == paths
    assert [_digest(path) for path in paths] == digests


def test_save_as_onto_a_loaded_run_is_refused(tmp_path):
    """Save As onto one of the loaded runs reports an error and leaves the run untouched."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(CALIBRATION), _record(CALIBRATION)])
    digest = _digest(paths[0])
    model = _loaded_model(paths)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(paths[0])

    # Assert
    assert len(failures) == 1
    assert "one of the loaded runs" in failures[0][1]
    assert _digest(paths[0]) == digest


def test_save_of_runs_with_different_calibrations_is_refused(tmp_path):
    """Runs reduced with different calibrations are not combined into a file."""
    # Arrange
    other = dict(CALIBRATION, TTH_0=0.5)
    paths = _run_copies(tmp_path, [_record(CALIBRATION), _record(other)])
    model = _loaded_model(paths)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(str(tmp_path / "combined.h5"))

    # Assert
    assert len(failures) == 1
    assert not (tmp_path / "combined.h5").exists()
