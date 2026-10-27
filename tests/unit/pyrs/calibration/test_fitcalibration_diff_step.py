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
        return SimpleNamespace(x=x0, jac=np.eye(x0.size), status=1)

    monkeypatch.setattr(mpc, "least_squares", recording_least_squares)
    fit = mpc.FitCalibration.__new__(mpc.FitCalibration)
    fit._ref_method = "trf"
    fit._max_nfev = None
    fit._calib = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.54])
    fit._caliberr = np.full(8, -1.0)
    fit._calibstatus = -1
    fit.calls = calls
    return fit


def test_diff_step_values():
    np.testing.assert_array_equal(mpc.DIFF_STEP, [SHIFT] * 3 + [ANGLE] * 4 + [WAVELENGTH])


@pytest.mark.parametrize(
    "stage, expected",
    [
        ("calibrate_shiftx", [SHIFT]),
        ("calibrate_shifty", [SHIFT]),
        ("calibrate_distance", [SHIFT]),
        ("calibrate_tth0", [ANGLE]),
        ("calibrate_wave_length", [WAVELENGTH]),
        ("calibrate_wave_shift", [ANGLE, WAVELENGTH]),
        ("CalibrateShift", [SHIFT] * 3),
        ("CalibrateRotation", [ANGLE] * 3),
        ("CalibrateGeometry", [SHIFT] * 3 + [ANGLE] * 4),
        ("FullCalibration", [SHIFT] * 3 + [ANGLE] * 4 + [WAVELENGTH]),
    ],
)
def test_stage_passes_per_parameter_steps(calibration, stage, expected):
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
