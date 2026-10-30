"""Ring-based calibration in the calibration GUI (#1045): model and controller, no widgets

The GUI's ring path must give exactly what the pyrs-pixel-calibration driver gives for the same
settings, and Export writes the user's folder plus, with the shared flag, HB2B_Latest.json,
HB2B_Latest_<Si...>.json and <Si...>/HB2B_CAL_<time>.json.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
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


def test_ring_path_equals_the_driver(workspace, refined):
    # Arrange - the driver's steps (scripts/pyrs_pixel_calibration.py: run_calibration)
    points = extract_ring_points(workspace, MONO_WAVELENGTH, strips=5)
    driver = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=MONO_WAVELENGTH)
    driver.iterate(passes=3, polish=True)
    driver.estimate_errors()

    # Act
    gui_model, (values, errors, r_sum, rmse) = refined

    # Assert - bit for bit
    np.testing.assert_array_equal(values[-1], driver.calibration_array)
    np.testing.assert_array_equal(errors[-1], driver.calibration_error_array)
    np.testing.assert_array_equal(gui_model.current_calibration_array, driver.calibration_array)
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

    # Assert - ring extraction ran; its residual at the starting geometry is returned
    assert gui_model._ring_calibration is not None
    assert rmse == pytest.approx(np.sqrt(r_sum / len(gui_model._ring_calibration.ring_points)))


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
