"""Ring-based calibration in the calibration GUI (#1045): model and controller, no widgets

The GUI's ring path must give exactly what the pyrs-pixel-calibration driver gives for the same
settings, and Export writes the user's folder plus, with the shared flag, HB2B_Latest.json,
HB2B_Latest_<Si...>.json and <Si...>/HB2B_CAL_<time>.json.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.core import MonoSetting  # type: ignore
from pyrs.interface.detector_calibration import detector_calibration_model as model_module
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl
from pyrs.interface.detector_calibration.detector_calibration_model import (
    DEFAULT_STRIPS,
    METHOD_FITCALIBRATION,
    METHOD_RING,
    DetectorCalibrationModel,
)
from pyrs.utilities.calibration_file_io import read_calibration_json_file
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace

TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
MONO_WAVELENGTH = 1.540


@pytest.fixture(scope="module")
def workspace():
    ws = synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS)
    ws.set_wavelength(MONO_WAVELENGTH, False)
    return ws


def _model(workspace, method=METHOD_RING):
    """The GUI model over a loaded run (FitCalibration's object holds the converted run)"""
    gui_model = DetectorCalibrationModel(None)
    gui_model._calibration_obj = SimpleNamespace(
        _hidra_ws=workspace,
        monosetting=MonoSetting.Si422,
        calibration_array=np.r_[np.zeros(7), MONO_WAVELENGTH],
        calibration_error_array=np.full(8, -1.0),
        _calibstatus=1,
        set_keep_subrun_list=lambda keep: None,
    )
    gui_model._nexus_file = "synthetic"
    gui_model.set_method(method)
    return gui_model


@pytest.fixture(scope="module")
def refined(workspace):
    """The GUI's ring path with the driver's settings: default strips, recipe "iterate", trf"""
    gui_model = _model(workspace)
    gui_model.set_refinement_params("trf", 300)  # the GUI's default max nfev
    gui_model.extract_rings(DEFAULT_STRIPS)
    result = gui_model.calibrate_detector(["iterate", "", ""])
    return gui_model, result


def _driver():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "pyrs_pixel_calibration", Path(__file__).resolve().parents[5] / "scripts" / "pyrs_pixel_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ring_path_equals_the_driver(workspace, refined, tmp_path, monkeypatch):
    # Arrange - the real driver, with the NeXus conversion replaced by the synthetic run
    class _Converter:
        def __init__(self, nexus_file, mask_file=None):
            pass

        def convert(self):
            return workspace

    monkeypatch.setattr("pyrs.core.nexus_conversion.NeXusConvertingApp", _Converter)
    nexus = tmp_path / "HB2B_0001.nxs.h5"
    nexus.write_bytes(b"")
    driver_file = tmp_path / "driver.json"
    driver = _driver().run_calibration(str(nexus), str(driver_file), mask_file=None)

    # Act
    gui_model, (values, errors, r_sum, rmse) = refined
    (gui_file,) = gui_model.export_calibration(str(tmp_path), write_shared=False)

    # Assert - bit for bit, file and arrays
    np.testing.assert_array_equal(values[-1], driver.calibration_array)
    np.testing.assert_array_equal(errors[-1], driver.calibration_error_array)
    assert json.load(open(gui_file)) == json.load(open(driver_file))
    assert rmse == driver.stages[-1].rmse and errors[-1][6] == -1.0


def test_ring_recipe_steps_are_recorded(refined):
    gui_model, (values, _, _, _) = refined
    assert len(values) == 1  # empty routines are skipped
    x, y = gui_model.get_calibration_values(0, 7)
    np.testing.assert_array_equal(x, [0])
    np.testing.assert_array_equal(y, [values[0][7]])


def test_extract_rings_reports_points(workspace):
    # Act
    gui_model = _model(workspace)
    n_points, n_rings, rmse, r_sum = gui_model.extract_rings(3)

    # Assert
    assert n_points > 0 and n_rings > 0 and rmse > 0 and r_sum > 0
    assert f"{n_points} ring points on {n_rings} rings (3 strips per sub-run)" == gui_model.ring_summary


def test_excluded_sub_runs_not_extracted(workspace):
    # Arrange
    gui_model = _model(workspace)
    keep = np.ones(workspace.get_sub_runs().size, dtype=bool)
    keep[:2] = False
    gui_model.set_keep_subrun_list(keep)

    # Act
    gui_model.extract_rings(3)

    # Assert
    assert not {1, 2} & set(gui_model._ring_calibration.ring_points.sub_run.tolist())


@pytest.mark.parametrize("strips", [0, -3])
def test_extract_rings_bad_strips_raises(workspace, strips):
    with pytest.raises(ValueError, match="strips"):
        _model(workspace).extract_rings(strips)


def test_calibrate_before_extracting_raises(workspace):
    with pytest.raises(RuntimeError, match="extract the rings first"):
        _model(workspace).calibrate_detector(["iterate"])


def test_unknown_routine_raises(workspace):
    gui_model = _model(workspace)
    gui_model.extract_rings(3)
    with pytest.raises(ValueError, match="unknown recipe routines"):
        gui_model.calibrate_detector(["polish"])


def test_unknown_method_raises(workspace):
    with pytest.raises(ValueError, match="method must be one of"):
        _model(workspace).set_method("Best fit")


# ---------------------------------------------------------------- export


@pytest.fixture
def shared(tmp_path, monkeypatch):
    folder = tmp_path / "shared" / "CALIBRATION"
    folder.mkdir(parents=True)
    monkeypatch.setattr(model_module, "SHARED_CALIBRATION_DIR", str(folder))
    return folder


def _assert_calibration_file(path, values, errors):
    shift, shift_error, wavelength, wavelength_error, _ = read_calibration_json_file(str(path))
    written = [
        shift.center_shift_x,
        shift.center_shift_y,
        shift.center_shift_z,
        shift.rotation_x,
        shift.rotation_y,
        shift.rotation_z,
        shift.two_theta_0,
        wavelength,
    ]
    written_errors = [
        shift_error.center_shift_x,
        shift_error.center_shift_y,
        shift_error.center_shift_z,
        shift_error.rotation_x,
        shift_error.rotation_y,
        shift_error.rotation_z,
        shift_error.two_theta_0,
        wavelength_error,
    ]
    np.testing.assert_array_equal(written, values)
    np.testing.assert_array_equal(written_errors, errors)


def test_export_writes_folder_and_shared_files(refined, tmp_path, shared):
    # Arrange
    gui_model, _ = refined
    user = tmp_path / "mine"
    user.mkdir()

    # Act
    written = gui_model.export_calibration(str(user), write_shared=True)

    # Assert - four identical files, the ring-based result with its error codes
    names = sorted(os.path.relpath(path, tmp_path) for path in written)
    stamp = os.path.basename(written[0])[len("HB2B_CAL_") : -len(".json")]
    assert names == sorted(
        [
            f"mine/HB2B_CAL_{stamp}.json",
            "shared/CALIBRATION/HB2B_Latest.json",
            "shared/CALIBRATION/HB2B_Latest_Si422.json",
            f"shared/CALIBRATION/Si422/HB2B_CAL_{stamp}.json",
        ]
    )
    for path in written:
        _assert_calibration_file(
            path, gui_model.current_calibration_array, gui_model._ring_calibration.calibration_error_array
        )


def test_export_without_shared_flag_writes_only_folder(refined, tmp_path, shared):
    gui_model, _ = refined
    written = gui_model.export_calibration(str(tmp_path), write_shared=False)
    assert len(written) == 1 and os.path.dirname(written[0]) == str(tmp_path)
    assert not os.listdir(shared)


needs_permissions = pytest.mark.skipif(os.geteuid() == 0, reason="root can write read-only files")


@needs_permissions
def test_export_shared_not_writable_still_writes_folder(refined, tmp_path, shared):
    # Arrange
    gui_model, _ = refined
    user = tmp_path / "mine"
    user.mkdir()
    shared.chmod(0o500)

    # Act / Assert
    try:
        with pytest.raises(PermissionError, match="written to"):
            gui_model.export_calibration(str(user), write_shared=True)
        assert len(os.listdir(user)) == 1
    finally:
        shared.chmod(0o700)


def test_export_missing_folder_raises(refined, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        refined[0].export_calibration(str(tmp_path / "nowhere"), write_shared=False)


def test_export_ring_before_refining_raises(workspace, tmp_path):
    with pytest.raises(RuntimeError, match="nothing has been refined"):
        _model(workspace).export_calibration(str(tmp_path), write_shared=False)


def test_export_fitcalibration_result(workspace, tmp_path):
    # Arrange - FitCalibration's values, errors and status
    gui_model = _model(workspace, METHOD_FITCALIBRATION)
    gui_model._calibration_obj.calibration_array = TRUE_PARAMS.copy()
    gui_model._calibration_obj.calibration_error_array = np.full(8, 1e-4)

    # Act
    written = gui_model.export_calibration(str(tmp_path), write_shared=False)

    # Assert
    _assert_calibration_file(written[0], TRUE_PARAMS, np.full(8, 1e-4))


# ---------------------------------------------------------------- controller


def test_controller_fit_peaks_dispatches_by_method(workspace):
    # Arrange
    gui_model = _model(workspace)
    controller = DetectorCalibrationCrtl(gui_model)

    # Act
    rmse, r_sum = controller.fit_diffraction_peaks(np.ones(11, dtype=bool), strips=3)

    # Assert - ring extraction ran; RMS and signed sum of its residuals at the starting geometry
    calibration = gui_model._ring_calibration
    residual = calibration.residual(calibration.calibration_array)
    assert rmse == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert r_sum == pytest.approx(np.sum(residual))


# ---------------------------------------------------------------- picked ring points and rejected strips


@pytest.fixture(scope="module")
def strip0_masked():
    """A run whose detector rows 40-228 (strip 0 of 5) are masked: every sub-run rejects that strip"""
    ws = synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS)
    ws.set_wavelength(MONO_WAVELENGTH, False)
    mask = np.ones((1024, 1024))
    mask[:, 40:229] = 0
    ws.set_detector_mask(mask.ravel(), True)
    return ws


def test_ring_point_marks_are_the_extracted_points(refined):
    # Arrange
    gui_model, _ = refined
    points = gui_model._ring_calibration.ring_points

    # Act
    accepted, _ = gui_model.ring_point_marks(2)

    # Assert - exactly the sub-run's ring points, each described
    in_sub_run = points.sub_run == 2
    np.testing.assert_array_equal(accepted.index_0, points.index_0[in_sub_run])
    np.testing.assert_array_equal(accepted.index_1, points.index_1[in_sub_run])
    from pyrs.calibration.pixel_calibration_plots import hkl_label

    structure = {"Ni": "FCC", "Fe": "BCC", "Mo": "BCC"}
    expected = [
        f"{p} {hkl_label(int(n), structure[p])}, strip "
        for p, n in zip(points.powder[in_sub_run], points.hkl_squared[in_sub_run])
    ]
    assert all(text.startswith(label) and "2θ" in text for text, label in zip(accepted.text, expected))


def test_rejected_strips_marked_where_the_ring_was_sought(strip0_masked):
    # Arrange
    gui_model = _model(strip0_masked)
    gui_model.extract_rings(5)

    # Act
    _, rejected = gui_model.ring_point_marks(1)

    # Assert - strip 0 on its centre row, at the ring's column; described with the reason
    assert rejected.text and all("strip 1: rejected, band has no unmasked pixels" in t for t in rejected.text)
    np.testing.assert_array_equal(rejected.index_1, (40 + 229 - 1) / 2.0)
    assert np.all((rejected.index_0 > 0) & (rejected.index_0 < 1023))


def test_ring_point_marks_need_the_ring_method(workspace):
    gui_model = _model(workspace, METHOD_FITCALIBRATION)
    assert gui_model.ring_point_marks(1) is None and not gui_model.has_ring_points


def test_current_geometry_before_ring_extraction_is_the_start(workspace):
    # The default "Current calibration" geometry is usable before Fit Peaks: the window's start values
    np.testing.assert_array_equal(_model(workspace).current_calibration_array, np.r_[np.zeros(7), MONO_WAVELENGTH])


# ---------------------------------------------------------------- review fixes (PR #1046)


def _stand_in(workspace, status=1):
    return SimpleNamespace(
        _hidra_ws=workspace,
        monosetting=MonoSetting.Si422,
        calibration_array=TRUE_PARAMS.copy(),
        calibration_error_array=np.full(8, 1e-4),
        _calibstatus=status,
        set_keep_subrun_list=lambda keep: None,
    )


def test_new_run_clears_ring_state(workspace, monkeypatch, tmp_path):
    # Arrange - a ring-based result from run A, then load run B
    gui_model = _model(workspace)
    gui_model.extract_rings(3)
    gui_model.calibrate_detector(["iterate"])
    monkeypatch.setattr(model_module, "FitCalibration", lambda **kwargs: _stand_in(workspace))

    # Act
    gui_model._init_calibration("run_B.nxs.h5")

    # Assert - nothing of run A can be drawn, refined further or exported
    assert not gui_model.has_ring_points and gui_model.ring_point_marks(1) is None
    assert gui_model.ring_summary == "" and gui_model.calibration_warnings == []
    with pytest.raises(RuntimeError, match="nothing has been refined"):
        gui_model.export_calibration(str(tmp_path), write_shared=False)
    with pytest.raises(RuntimeError, match="extract the rings first"):
        gui_model.calibrate_detector(["iterate"])


@pytest.mark.parametrize("status, match", [(-1, "has not been refined"), (0, "before converging")])
def test_export_refuses_unrefined_or_unconverged(workspace, tmp_path, status, match):
    gui_model = _model(workspace, METHOD_FITCALIBRATION)
    gui_model._calibration_obj = _stand_in(workspace, status)
    with pytest.raises(RuntimeError, match=match):
        gui_model.export_calibration(str(tmp_path), write_shared=False)
    assert not os.listdir(tmp_path)


def test_unconverged_ring_result_warns_and_cannot_export(workspace, tmp_path):
    # Arrange - max nfev 1: every stage stops early
    gui_model = _model(workspace)
    gui_model.extract_rings(3)
    gui_model.set_refinement_params("trf", 1)

    # Act
    gui_model.calibrate_detector(["shifts"])

    # Assert
    assert any("before converging" in w for w in gui_model.calibration_warnings)
    with pytest.raises(RuntimeError, match="status 0"):
        gui_model.export_calibration(str(tmp_path), write_shared=False)


def test_partial_recipe_moves_only_its_parameters(workspace):
    # Arrange
    gui_model = _model(workspace)
    gui_model.extract_rings(3)
    start = gui_model.current_calibration_array

    # Act - only the wavelength is refined; the errors must not re-fit the geometry
    values, errors, _, _ = gui_model.calibrate_detector(["wavelength"])

    # Assert
    np.testing.assert_array_equal(values[-1][:7], start[:7])
    assert values[-1][7] != start[7]
    assert errors[-1][6] == -1.0 and np.all(errors[-1][[0, 1, 2, 3, 4, 5, 7]] > 0)


def test_failed_stage_restores_the_calibration(workspace):
    # Arrange - Shift_z 0.1 m: inside CalibrateShift's +/-0.15, outside CalibrateGeometry's +/-0.05
    gui_model = _model(workspace)
    start = np.r_[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, MONO_WAVELENGTH]
    gui_model.set_calibration_params(start)
    gui_model.extract_rings(3)
    n_stages, n_history = len(gui_model._ring_calibration.stages), len(gui_model._ring_history)

    # Act - the wavelength stage runs, then the geometry stage refuses its start
    with pytest.raises(ValueError, match="starting value outside bounds"):
        gui_model.calibrate_detector(["wavelength", "geometry"])

    # Assert - back to before the recipe
    np.testing.assert_array_equal(gui_model.current_calibration_array, start)
    assert len(gui_model._ring_calibration.stages) == n_stages and len(gui_model._ring_history) == n_history


def test_export_writes_archive_before_latest_atomically(refined, tmp_path, shared):
    # Arrange
    gui_model, _ = refined
    user = tmp_path / "mine"
    user.mkdir()

    # Act
    written = gui_model.export_calibration(str(user), write_shared=True)

    # Assert - user copy, archive, per-monochromator, and HB2B_Latest.json last; no temporary files left
    assert [os.path.relpath(p, tmp_path).split("/HB2B_CAL_")[0] for p in written] == [
        "mine",
        "shared/CALIBRATION/Si422",
        "shared/CALIBRATION/HB2B_Latest_Si422.json",
        "shared/CALIBRATION/HB2B_Latest.json",
    ]
    for folder in (user, shared, shared / "Si422"):
        assert not [name for name in os.listdir(folder) if name.startswith(".")]
    # rw-rw-r--: anyone who may write the folder may replace it, everyone may read it
    assert all(os.stat(p).st_mode & 0o777 == 0o664 for p in written)
    assert os.stat(shared / "Si422").st_mode & 0o7777 == 0o2775  # the archive folder it created


def test_export_replaces_a_read_only_latest(refined, tmp_path, shared):
    """Folder write access is enough to replace HB2B_Latest.json, whoever wrote it and whatever its mode"""
    # Arrange - someone else's read-only HB2B_Latest.json
    gui_model, _ = refined
    latest = shared / "HB2B_Latest.json"
    latest.write_text("{}")
    latest.chmod(0o444)

    # Act
    gui_model.export_calibration(str(tmp_path), write_shared=True)

    # Assert - replaced, and now rw-rw-r--
    assert json.loads(latest.read_text())["Lambda"] == gui_model.current_calibration_array[7]
    assert os.stat(latest).st_mode & 0o777 == 0o664


@needs_permissions
def test_export_read_only_folder_writes_no_shared_copy(refined, tmp_path, shared):
    # Arrange - a writable HB2B_Latest.json in a folder this user may not write: it cannot be replaced
    gui_model, _ = refined
    user = tmp_path / "mine"
    user.mkdir()
    latest = shared / "HB2B_Latest.json"
    latest.write_text("{}")
    latest.chmod(0o666)
    shared.chmod(0o555)

    # Act / Assert - refused before any shared file is written; the user copy is written
    try:
        with pytest.raises(PermissionError, match="HB2B_Latest.json"):
            gui_model.export_calibration(str(user), write_shared=True)
        assert len(os.listdir(user)) == 1
        assert sorted(os.listdir(shared)) == ["HB2B_Latest.json"] and latest.read_text() == "{}"
    finally:
        shared.chmod(0o755)


def _fit_stand_in(workspace, statuses, fail_on=None):
    """A FitCalibration stand-in whose recipe stages set the given statuses (or raise)."""
    fit = _stand_in(workspace, status=-1)
    fit.residual_sum, fit.residual_rmse = 0.01, 0.002

    def stage(name, status):
        def run():
            if name == fail_on:
                raise RuntimeError(f"{name} failed")
            fit._calibstatus = status

        return run

    for name, status in statuses.items():
        setattr(fit, name, stage(name, status))
    return fit


def test_export_refuses_an_unconverged_earlier_stage(workspace, tmp_path):
    # Arrange - FitCalibration: the wavelength stage stops at max nfev, the shift stage then converges
    gui_model = _model(workspace, METHOD_FITCALIBRATION)
    gui_model._calibration_obj = _fit_stand_in(workspace, {"calibrate_wave_length": 0, "CalibrateShift": 1})

    # Act
    gui_model.calibrate_detector(["wavelength", "shifts"])

    # Assert - the last status is 1, but a stage did not converge
    exportable, text = gui_model.export_description()
    assert not exportable and "status 0" in text
    with pytest.raises(RuntimeError, match="status 0"):
        gui_model.export_calibration(str(tmp_path), write_shared=False)


def test_ring_export_refuses_an_unconverged_earlier_stage(workspace, tmp_path):
    # Arrange - ring-based: one stage at max nfev 1, a later one converging
    gui_model = _model(workspace)
    gui_model.extract_rings(3)
    gui_model.set_refinement_params("trf", 1)
    gui_model.calibrate_detector(["shifts"])
    gui_model.set_refinement_params("trf", 300)

    # Act
    gui_model.calibrate_detector(["wavelength"])

    # Assert
    assert gui_model._ring_calibration.calibration_status > 0
    with pytest.raises(RuntimeError, match="status 0"):
        gui_model.export_calibration(str(tmp_path), write_shared=False)


def test_failed_fitcalibration_recipe_is_not_exportable(workspace, tmp_path):
    # Arrange
    gui_model = _model(workspace, METHOD_FITCALIBRATION)
    gui_model._calibration_obj = _fit_stand_in(
        workspace, {"CalibrateShift": 1, "CalibrateRotation": 1}, fail_on="CalibrateRotation"
    )

    # Act - the shift stage changed the calibration, then the rotation stage fails
    with pytest.raises(RuntimeError, match="CalibrateRotation failed"):
        gui_model.calibrate_detector(["shifts", "rotations"])

    # Assert
    with pytest.raises(RuntimeError, match="failed partway"):
        gui_model.export_calibration(str(tmp_path), write_shared=False)


def test_export_description_names_method_and_status(refined, workspace):
    exportable, text = refined[0].export_description()
    assert exportable and text.startswith("Ring-based calibration (status ") and "λ = " in text
    exportable, text = _model(workspace).export_description()
    assert not exportable and "cannot be exported" in text


def test_failed_recipe_keeps_its_warnings(workspace):
    # Arrange - the wavelength stage stops at max nfev (warns), then the geometry stage refuses its start
    gui_model = _model(workspace)
    gui_model.set_calibration_params(np.r_[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, MONO_WAVELENGTH])
    gui_model.extract_rings(3)
    gui_model.set_refinement_params("trf", 1)

    # Act
    with pytest.raises(ValueError, match="outside bounds"):
        gui_model.calibrate_detector(["wavelength", "geometry"])

    # Assert - this run's warning, not the previous run's list
    assert any("calibrate_wave_length" in w and "before converging" in w for w in gui_model.calibration_warnings)


@pytest.mark.parametrize("call", ["set_calibration_params", "set_refinement_params"])
def test_no_run_loaded_raises(call):
    args = (np.r_[np.zeros(7), MONO_WAVELENGTH],) if call == "set_calibration_params" else ("trf", 300)
    with pytest.raises(RuntimeError, match="load a calibration run first"):
        getattr(DetectorCalibrationModel(None), call)(*args)


def test_skipped_sub_runs_are_reported(workspace):
    # Arrange - sub-run 2 moved to an sy with no powder
    sy = np.array(workspace.get_sample_log_values("sy"), dtype=float)
    sub_runs = np.arange(1, sy.size + 1)
    workspace.set_sample_log("sy", sub_runs, np.r_[sy[0], 40.0, sy[2:]])
    gui_model = _model(workspace)

    # Act
    try:
        gui_model.extract_rings(3)
    finally:
        workspace.set_sample_log("sy", sub_runs, sy)

    # Assert
    assert any("sub-run 2" in w for w in gui_model.extraction_warnings)
    assert "1 warning(s)" in gui_model.ring_summary


def test_keep_list_length_must_match(workspace):
    gui_model = _model(workspace)
    gui_model.set_keep_subrun_list([True, False])
    with pytest.raises(ValueError, match="keep flags"):
        gui_model.extract_rings(3)
