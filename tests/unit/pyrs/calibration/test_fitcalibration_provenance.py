"""FitCalibration's provenance (#1053): one step per refinement routine, named after the routine run"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.calibration import mantid_peakfit_calibration as mpc
from pyrs.calibration.pixel_refinement import RANK_DEFICIENT


@pytest.fixture
def calibration(monkeypatch):
    """A FitCalibration without data; least_squares returns its start with a small residual."""

    def fake_least_squares(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=float)
        return SimpleNamespace(
            x=x0 + 0.001,
            jac=np.vstack([np.eye(x0.size)] * 4),
            fun=np.full(4 * x0.size, 0.01),
            status=2,
            nfev=7,
            message="`ftol` termination condition is satisfied.",
        )

    monkeypatch.setattr(mpc, "least_squares", fake_least_squares)
    fit = mpc.FitCalibration.__new__(mpc.FitCalibration)
    fit._ref_method = "trf"
    fit._max_nfev = None
    fit._calib = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.54])
    fit._caliberr = np.full(8, -1.0)
    fit._calibstatus = -1
    fit.steps = []
    fit._residual_labels = []
    fit._final_residual = None
    return fit


@pytest.mark.parametrize(
    "routine, free",
    [
        ("calibrate_wave_length", ["Lambda"]),
        ("calibrate_shiftx", ["Shift_x"]),
        ("calibrate_shifty", ["Shift_y"]),
        ("calibrate_distance", ["Shift_z"]),
        ("calibrate_tth0", ["TTH_0"]),
        ("calibrate_wave_shift", ["TTH_0", "Lambda"]),
        ("CalibrateShift", ["Shift_x", "Shift_y", "Shift_z"]),
        ("CalibrateRotation", ["Rot_x", "Rot_y", "Rot_z"]),
        ("CalibrateGeometry", ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0"]),
        ("FullCalibration", ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda"]),
    ],
)
def test_routine_records_one_step_under_its_name(calibration, routine, free):
    """Regression (#1053 prototype): wrapped routines were recorded as the helper calibrate_single"""
    # Act
    getattr(calibration, routine)()

    # Assert
    assert len(calibration.steps) == 1
    step = calibration.steps[0]
    assert (step["step"], step["routine"], step["free_parameters"]) == (1, routine, free)
    assert (step["status"], step["nfev"], step["n"]) == (2, 7, 4 * len(free))
    assert step["rmse_deg"] == pytest.approx(0.01)
    assert step["message"] == "`ftol` termination condition is satisfied."
    assert step["values"] == pytest.approx({name: 0.001 + (1.54 if name == "Lambda" else 0.0) for name in free})


def test_steps_accumulate_in_run_order(calibration):
    # Act
    calibration.calibrate_wave_length()
    calibration.CalibrateShift()
    calibration.CalibrateRotation()

    # Assert
    assert [(s["step"], s["routine"]) for s in calibration.steps] == [
        (1, "calibrate_wave_length"),
        (2, "CalibrateShift"),
        (3, "CalibrateRotation"),
    ]


def test_undetermined_parameter_has_its_reason(calibration, monkeypatch):
    # Arrange - a Jacobian whose second column is zero: Shift_y moves nothing
    def degenerate(fun, x0, **kwargs):
        x0 = np.asarray(x0, dtype=float)
        jac = np.zeros((6, 3))
        jac[:, 0], jac[:, 2] = 1.0, np.arange(6)
        return SimpleNamespace(x=x0, jac=jac, fun=np.full(6, 0.01), status=1, nfev=3, message="done")

    monkeypatch.setattr(mpc, "least_squares", degenerate)

    # Act
    calibration.CalibrateShift()

    # Assert - a zero column flags every parameter (the ring calibration's rule), each with the reason
    assert calibration.steps[0]["undetermined_reasons"] == {
        name: RANK_DEFICIENT for name in ["Shift_x", "Shift_y", "Shift_z"]
    }


def test_brute_force_step_has_its_reason(calibration):
    # Act
    calibration.FitDetector(
        lambda x, brute, i: float(np.sum((np.atleast_1d(x) - 0.2) ** 2)),
        [0.0],
        bounds=([1.0], [-1.0]),
        Brute=True,
        free=[0],
        routine="grid_search",
    )

    # Assert
    step = calibration.steps[0]
    assert step["routine"] == "grid_search" and step["errors"] == {"Shift_x": -2.0}
    assert step["undetermined_reasons"] == {"Shift_x": "brute-force grid search gives no covariance"}


def test_provenance_of_an_object_without_data_still_builds(calibration):
    """The provenance must never stop a calibration from being written"""
    # Arrange
    calibration._nexus_file, calibration.mask_file, calibration.vanadium = None, None, None
    calibration.bins, calibration.eta_slices = 512, 3
    calibration._start = {"source": "set_calibration_array", "values": calibration._calib.copy()}
    calibration.CalibrateShift()

    # Act
    block = calibration.provenance({"recipe": ["shifts"]})

    # Assert
    assert block["method"] == "peak-fit"
    assert block["inputs"]["recipe"] == ["shifts"] and block["inputs"]["monochromator"] is None
    assert [s["routine"] for s in block["steps"]] == ["CalibrateShift"]
    assert block["fit_quality"]["n"] == 12 and block["fit_quality"]["by_reflection_and_run"] == []
    assert block["start"]["source"] == "set_calibration_array"


def _with_inputs(fit):
    """The attributes provenance() reads that a real FitCalibration sets up from its run."""
    fit._nexus_file, fit.mask_file, fit.vanadium = None, None, None
    fit.bins, fit.eta_slices = 512, 3
    fit._start = {"source": "nominal geometry, monochromator wavelength", "values": fit._calib.copy(), "first_step": 0}
    return fit


def test_fit_quality_is_evaluated_at_the_solution(calibration, monkeypatch):
    """#1053 review (F4): the last peak fit of a refinement is a Jacobian probe; fit once more at out.x"""
    # Arrange
    _with_inputs(calibration)
    seen = []
    monkeypatch.setattr(calibration, "set_inst_shifts_wl", lambda params: seen.append(np.array(params)))

    def fit_peaks():
        calibration._residual_labels = [(1, "Ni", "FCC", 3, f"eta_{i}") for i in range(12)]
        return np.full(12, 0.02)

    monkeypatch.setattr(calibration, "fit_peaks", fit_peaks)

    def history_changed(x):
        raise AssertionError("get_alignment_residual appends to the GUI's residual history")

    monkeypatch.setattr(calibration, "get_alignment_residual", history_changed)

    # Act
    calibration.CalibrateShift()
    quality = calibration.provenance()["fit_quality"]

    # Assert - re-fitted at the solution (free Shift_x/y/z replaced by out.x), labelled by reflection
    np.testing.assert_array_equal(seen[-1], [0.001, 0.001, 0.001, 0, 0, 0, 0, 1.54])
    assert quality["residual_source"] == "residual at the solution"
    assert quality["rmse_deg"] == pytest.approx(0.02)
    assert {entry["reflection"] for entry in quality["by_reflection_and_run"]} == {"Ni 111"}


def test_failed_reevaluation_keeps_the_fit_residual(calibration):
    # Arrange - no run loaded, so fitting again at the solution fails
    _with_inputs(calibration)

    # Act
    calibration.CalibrateShift()
    quality = calibration.provenance()["fit_quality"]

    # Assert - the refinement completed; the quality says where its residual came from
    assert calibration.steps[0]["routine"] == "CalibrateShift"
    assert quality["n"] == 12 and quality["by_reflection_and_run"] == []
    assert quality["residual_source"].startswith("least_squares residual; re-evaluation failed")


def test_new_start_drops_earlier_steps(calibration):
    # Arrange
    _with_inputs(calibration)
    calibration.CalibrateShift()

    # Act
    calibration.set_calibration_array(calibration._calib)
    calibration.CalibrateRotation()
    block = calibration.provenance()

    # Assert
    assert [(s["step"], s["routine"]) for s in block["steps"]] == [(1, "CalibrateRotation")]
    assert block["start"]["source"] == "set_calibration_array"


def test_archived_start_records_the_file(calibration, tmp_path):
    # Arrange
    from pyrs.core.instrument_geometry import DENEXDetectorShift
    from pyrs.utilities.calibration_file_io import write_calibration_to_json

    _with_inputs(calibration)
    calibration.CalibrateShift()
    path = tmp_path / "archived.json"
    write_calibration_to_json(
        DENEXDetectorShift(*[0.0] * 7), DENEXDetectorShift(*[0.01] * 7), 1.53, 1e-4, 1, str(path)
    )

    # Act
    calibration.get_archived_calibration(str(path))
    block = calibration.provenance()

    # Assert - the start is the file, with its fingerprint; earlier steps and fit quality are dropped
    assert block["start"]["source"]["file"] == str(path) and block["start"]["source"]["sha256"]
    assert block["start"]["values"]["Lambda"] == 1.53
    assert block["steps"] == [] and block["fit_quality"] is None


def test_write_is_never_blocked_by_provenance(calibration, tmp_path, monkeypatch):
    # Arrange
    from pyrs.utilities.calibration_file_io import read_calibration_json_file, read_calibration_provenance

    calibration._calibstatus = 1

    def broken(inputs=None):
        raise AttributeError("monosetting")

    monkeypatch.setattr(calibration, "provenance", broken)
    path = tmp_path / "fit.json"

    # Act
    with pytest.warns(UserWarning, match="could not be built"):
        calibration.write_calibration(str(path))

    # Assert
    assert read_calibration_json_file(str(path))[2] == 1.54
    assert "AttributeError" in read_calibration_provenance(str(path))["error"]


def test_write_latest_gets_the_same_provenance(calibration, monkeypatch):
    # Arrange - the shared paths are hard-coded under /HFIR, so the writer is replaced
    _with_inputs(calibration)
    calibration._calibstatus = 1
    written = []
    monkeypatch.setattr(mpc.os, "access", lambda path, mode: True)
    monkeypatch.setattr(mpc, "write_calibration_to_json", lambda *a, **k: written.append((a[5], k["provenance"])))

    # Act
    calibration.write_calibration("user.json", write_latest=True)

    # Assert
    assert [name for name, _ in written] == ["user.json", "/HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json"]
    assert written[0][1] == written[1][1] and written[0][1]["method"] == "peak-fit"


def test_block_is_strict_json(calibration):
    # Arrange - numpy settings, as the GUI may pass them
    _with_inputs(calibration)
    calibration.bins, calibration._max_nfev = np.int64(512), np.int64(20)
    calibration.CalibrateShift()

    # Act
    from pyrs.calibration.provenance import safe_build

    block = safe_build(calibration.provenance)

    # Assert
    assert "error" not in block and block["inputs"]["reduction"]["two_theta_bins"] == 512
    json.dumps(block, allow_nan=False)
