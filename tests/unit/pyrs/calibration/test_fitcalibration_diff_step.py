"""FitCalibration's finite-difference steps (#1030): each stage passes per-parameter relative steps

The residual of FitCalibration has a noise floor of ~1e-3 deg (peaks are re-binned and re-fitted at every
evaluation), so scipy's default 3-point step (~6e-6 relative) gives noise-dominated derivatives. Every
stage passes ``DIFF_STEP`` for its free parameters: 0.5 mm shifts, 0.01 deg angles, 1e-4 relative wavelength.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.calibration import mantid_peakfit_calibration as mpc

SHIFT, ANGLE, WAVELENGTH = 5e-4, 1e-2, 1e-4


@pytest.fixture
def calibration(monkeypatch):
    """A FitCalibration without data; least_squares records its keyword arguments."""
    calls = []

    def recording_least_squares(fun, x0, **kwargs):
        calls.append(kwargs)
        x0 = np.asarray(x0, dtype=float)
        return SimpleNamespace(x=x0, jac=np.eye(x0.size), fun=np.zeros(x0.size), status=1, nfev=1, message="fake")

    monkeypatch.setattr(mpc, "least_squares", recording_least_squares)
    fit = mpc.FitCalibration.__new__(mpc.FitCalibration)
    fit._ref_method = "trf"
    fit._max_nfev = None
    fit.steps = []  # provenance (#1053), set by initialize_calib_arrays in a real FitCalibration
    fit._residual_labels = []
    fit._final_residual = None
    fit._calib = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.54])
    fit._caliberr = np.full(8, -1.0)
    fit._calibstatus = -1
    fit.calls = calls
    return fit


def test_diff_step_values():
    np.testing.assert_array_equal(mpc.DIFF_STEP, [SHIFT] * 3 + [ANGLE] * 4 + [WAVELENGTH])


@pytest.mark.parametrize(
    "hold, stage, expected",
    [
        (True, "calibrate_shiftx", [SHIFT]),
        (True, "calibrate_shifty", [SHIFT]),
        (True, "calibrate_distance", [SHIFT]),
        (True, "calibrate_wave_length", [WAVELENGTH]),
        # 2theta_0 held (the default, #1090): Shift_x, Rot_y and the wavelength in the wavelength stage
        (True, "calibrate_wave_shift", [SHIFT, ANGLE, WAVELENGTH]),
        (True, "CalibrateShift", [SHIFT] * 3),
        (True, "CalibrateRotation", [ANGLE] * 3),
        (True, "CalibrateGeometry", [SHIFT] * 3 + [ANGLE] * 3),
        (True, "FullCalibration", [SHIFT] * 3 + [ANGLE] * 3 + [WAVELENGTH]),
        # 2theta_0 freed (a joint fit)
        (False, "calibrate_tth0", [ANGLE]),
        (False, "calibrate_wave_shift", [ANGLE, WAVELENGTH]),
        (False, "CalibrateGeometry", [SHIFT] * 3 + [ANGLE] * 4),
        (False, "FullCalibration", [SHIFT] * 3 + [ANGLE] * 4 + [WAVELENGTH]),
    ],
)
def test_stage_passes_per_parameter_steps(calibration, hold, stage, expected):
    # Arrange
    calibration._hold_tth0 = hold

    # Act
    getattr(calibration, stage)()

    # Assert
    assert len(calibration.calls) == 1
    np.testing.assert_array_equal(calibration.calls[0]["diff_step"], expected)


def test_lm_method_passes_steps_too(calibration):
    # Arrange - "lm" ignores bounds and takes a separate least_squares call
    calibration._ref_method = "lm"

    # Act
    calibration.CalibrateRotation()

    # Assert
    np.testing.assert_array_equal(calibration.calls[0]["diff_step"], [ANGLE] * 3)
    assert "bounds" not in calibration.calls[0]


@pytest.mark.parametrize(
    "hold, stage, half_widths",
    [
        (True, "calibrate_wave_length", [0.025]),
        (True, "calibrate_shiftx", [0.05]),
        (True, "calibrate_shifty", [0.05]),
        (True, "calibrate_distance", [0.1]),
        (True, "calibrate_wave_shift", [0.05, 10.0, 0.025]),
        (True, "CalibrateShift", [0.05, 0.05, 0.15]),
        (True, "CalibrateRotation", [5.0, 10.0, 5.0]),
        (True, "CalibrateGeometry", [0.05, 0.05, 0.05, 5.0, 10.0, 5.0]),
        (True, "FullCalibration", [0.05, 0.05, 0.15, 5.0, 10.0, 5.0, 0.05]),
        (False, "calibrate_tth0", [5.0]),
        (False, "calibrate_wave_shift", [5.0, 0.025]),
        (False, "CalibrateGeometry", [0.05, 0.05, 0.05, 5.0, 10.0, 5.0, 5.0]),
        (False, "FullCalibration", [0.05, 0.05, 0.15, 5.0, 10.0, 5.0, 5.0, 0.05]),
    ],
)
def test_stage_passes_bounds(calibration, hold, stage, half_widths):
    """Each stage's bounds: +/- the half-width about 0, about the current wavelength for the wavelength (the last
    free parameter of the stages that refine it); Rot_y +/-10 deg (#1090)"""
    # Arrange
    calibration._hold_tth0 = hold
    refines_wavelength = stage in ("calibrate_wave_length", "calibrate_wave_shift", "FullCalibration")
    centre = np.zeros(len(half_widths))
    if refines_wavelength:
        centre[-1] = 1.54

    # Act
    getattr(calibration, stage)()

    # Assert
    lower, upper = calibration.calls[0]["bounds"]
    np.testing.assert_allclose(lower, centre - half_widths)
    np.testing.assert_allclose(upper, centre + half_widths)
