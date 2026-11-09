"""Saving a peak fit writes the fit into a project file that can be read back (#1080)

With several runs loaded, a new combined project file is written: the run files are never
modified, and, as for "Combine runs" (#1063, #1068), runs reduced with different calibrations
are refused, while different vanadium runs, masks or binning need the user's approval.
"""

import hashlib
import os
import shutil

import pytest

from pyrs.core.pyrscore import PyRsCore
from pyrs.dataobjects.fields import StrainField
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.projectfile.file_object import ReductionMismatchError

pytestmark = pytest.mark.integration

FILES = ["tests/data/HB2B_1327.h5", "tests/data/HB2B_1328.h5"]
FIT_WINDOW = (88.6, 91.6)  # around the stored peak0 of both runs (~90.1 deg)
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


def _record(calibration=CALIBRATION, vanadium=None):
    record = {"source": "file", "calibration": calibration, "provenance": {"schema": 1}, "reduction": REDUCTION}
    return dict(record, vanadium=vanadium) if vanadium else record


def _run_copies(tmp_path, records):
    """Copies of the test project files, each given a calibration record (None: left historic)."""
    paths = []
    for original, record in zip(FILES, records):
        path = tmp_path / os.path.basename(original)
        shutil.copy(original, path)
        if record is not None:
            project = HidraProjectFile(str(path), HidraProjectFileMode.READWRITE)
            project.write_reduction_provenance(
                calibration_file="HB2B_CAL.json", vanadium_run=record.pop("vanadium", None), applied_calibration=record
            )
            project.save(verbose=False)
        paths.append(str(path))
    return paths


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _fitted_model(paths):
    """A model with the runs loaded and peak0 fitted over all of their sub-runs."""
    model = PeakFittingModel(peak_fit_core=PyRsCore())
    model.load_hidra_project(paths)
    model.fit_diff_peaks(["peak0"], [FIT_WINDOW[0]], [FIT_WINDOW[1]], "PseudoVoigt", "Linear")
    assert model.fit_result is not None
    return model


def _failures(model):
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))
    return failures


def _assert_readable_fit(path, model):
    """The saved peak reads back over the file's own sub-runs, and loads as a strain field."""
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        sub_runs = list(project.read_sub_runs())
        peaks = project.read_peak_parameters("peak0")
    assert sub_runs == list(model.hidra_workspace.get_sub_runs())
    assert list(peaks.sub_runs) == sub_runs
    assert len(StrainField(filename=str(path), peak_tag="peak0")) > 0


def test_save_writes_combined_file_and_keeps_runs(tmp_path):
    """Save (which passes the loaded runs) writes <dir>/HB2B_1327HB2B_1328.h5 with data, record and fit."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(), _record()])
    digests = [_digest(path) for path in paths]
    model = _fitted_model(paths)
    failures = _failures(model)

    # Act
    model.save_fit_result(paths)

    # Assert
    assert failures == [] and model.save_warnings == []
    combined = tmp_path / "HB2B_1327HB2B_1328.h5"
    _assert_readable_fit(combined, model)
    with HidraProjectFile(str(combined), HidraProjectFileMode.READONLY) as project:
        record = project.read_reduction_provenance()
    assert record["calibration"] == CALIBRATION
    assert [source["file"] for source in record["sources"]] == paths
    assert [_digest(path) for path in paths] == digests


def test_save_as_writes_the_chosen_file(tmp_path):
    """Save As writes the combined fit to the chosen name only."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(), _record()])
    model = _fitted_model(paths)
    failures = _failures(model)
    chosen = tmp_path / "chosen.h5"

    # Act
    model.save_fit_result(str(chosen))

    # Assert
    assert failures == []
    _assert_readable_fit(chosen, model)
    assert not (tmp_path / "HB2B_1327HB2B_1328.h5").exists()


@pytest.mark.parametrize("through_symlink", [False, True], ids=["same-path", "symlinked-path"])
def test_save_as_onto_a_loaded_run_is_refused(tmp_path, through_symlink):
    """Save As onto one of the loaded runs, however it is spelled, leaves the run untouched."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(), _record()])
    digest = _digest(paths[0])
    model = _fitted_model(paths)
    failures = _failures(model)
    target = paths[0]
    if through_symlink:
        link = tmp_path / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        target = str(link / os.path.basename(paths[0]))

    # Act
    model.save_fit_result(target)

    # Assert
    assert len(failures) == 1
    assert "one of the loaded runs" in failures[0][1]
    assert _digest(paths[0]) == digest


def test_save_into_read_only_directory_reports_failure(tmp_path):
    """A destination that cannot be written is reported, not raised into the Qt slot."""
    # Arrange
    if os.geteuid() == 0:
        pytest.skip("root can write to a read-only directory")
    paths = _run_copies(tmp_path, [_record(), _record()])
    model = _fitted_model(paths)
    failures = _failures(model)
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o555)

    # Act
    try:
        model.save_fit_result(str(read_only / "combined.h5"))
    finally:
        read_only.chmod(0o755)

    # Assert
    assert len(failures) == 1
    assert "not writable" in failures[0][1]


def test_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """If writing the peaks fails, the half-written combined file is removed."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(), _record()])
    model = _fitted_model(paths)
    failures = _failures(model)

    def failing_write(self, peaks):
        raise OSError("disk full")

    monkeypatch.setattr(HidraProjectFile, "write_peak_parameters", failing_write)
    chosen = tmp_path / "chosen.h5"

    # Act
    model.save_fit_result(str(chosen))

    # Assert
    assert len(failures) == 1 and "disk full" in failures[0][1]
    assert not chosen.exists()


def test_save_of_runs_with_different_calibrations_is_refused(tmp_path):
    """Runs reduced with different calibrations are not combined into a file."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(), _record(dict(CALIBRATION, TTH_0=0.5))])
    model = _fitted_model(paths)
    failures = _failures(model)

    # Act
    model.save_fit_result(str(tmp_path / "combined.h5"))

    # Assert
    assert len(failures) == 1
    assert "different calibrations" in failures[0][1]
    assert not (tmp_path / "combined.h5").exists()


def test_save_of_runs_with_different_vanadium_needs_approval(tmp_path):
    """Different vanadium runs raise for the window to ask; once approved, the save records it."""
    # Arrange
    paths = _run_copies(tmp_path, [_record(vanadium="HB2B_931.h5"), _record(vanadium="HB2B_932.h5")])
    model = _fitted_model(paths)
    chosen = tmp_path / "chosen.h5"

    # Act / Assert
    with pytest.raises(ReductionMismatchError):
        model.save_fit_result(str(chosen))
    assert not chosen.exists()
    model.save_fit_result(str(chosen), accept_reduction_differences=True)

    # Assert
    _assert_readable_fit(chosen, model)
    assert any("approved by the user" in warning for warning in model.save_warnings)


def test_save_of_historic_runs_warns(tmp_path):
    """Runs without a calibration record are saved, with a warning for the window to show."""
    # Arrange
    paths = _run_copies(tmp_path, [None, None])
    model = _fitted_model(paths)
    failures = _failures(model)

    # Act
    model.save_fit_result(paths)

    # Assert
    assert failures == []
    assert len(model.save_warnings) == 1 and "no calibration record" in model.save_warnings[0]
    _assert_readable_fit(tmp_path / "HB2B_1327HB2B_1328.h5", model)


def test_single_run_save_as_copies_and_keeps_source(tmp_path):
    """With one run loaded, Save As copies the run and writes the fit to the copy."""
    # Arrange
    paths = _run_copies(tmp_path, [_record()])
    digest = _digest(paths[0])
    model = _fitted_model(paths)
    copy_path = tmp_path / "single.h5"

    # Act
    model.save_fit_result(str(copy_path))

    # Assert
    _assert_readable_fit(copy_path, model)
    assert _digest(paths[0]) == digest
