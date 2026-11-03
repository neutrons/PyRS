"""FitCalibration's parameter errors (#1049)

``FitDetector`` reports one-sigma errors ``sqrt(diag(inv(JᵀJ)) · Σr² / (n - p))``, scaled by the residual
scatter, and -2 (``UNDETERMINED``) for a parameter the data cannot determine; the same formula as the
ring calibration (``pixel_refinement._parameter_errors``). See docs/ground_truths.md.
"""

import numpy as np
import pytest

from pyrs.calibration import mantid_peakfit_calibration as mpc
from pyrs.calibration.pixel_refinement import UNDETERMINED


@pytest.fixture
def calibration():
    """A FitCalibration without data, refined with the real least_squares."""
    fit = mpc.FitCalibration.__new__(mpc.FitCalibration)
    fit._ref_method = "trf"
    fit._max_nfev = None
    fit.steps = []  # provenance (#1053), set by initialize_calib_arrays in a real FitCalibration
    fit._residual_labels = []
    fit._final_residual = None
    return fit


def noisy_values(n: int, sigma: float) -> np.ndarray:
    return np.random.default_rng(1049).normal(0.0, sigma, n)


def test_constant_fit_error_is_standard_error_of_mean(calibration):
    # Arrange - fitting a constant to n values with scatter s gives the textbook error s / sqrt(n)
    data = 3.0 + noisy_values(50, 0.1)

    def residual(x, brute, i_index):
        return x[0] - data

    # Act
    values, errors, status = calibration.FitDetector(residual, [0.0], bounds=([-10.0], [10.0]))

    # Assert
    assert status > 0
    np.testing.assert_allclose(values, [data.mean()], rtol=1e-6)
    np.testing.assert_allclose(errors, [data.std(ddof=1) / np.sqrt(data.size)], rtol=1e-6)


def test_line_fit_errors_match_covariance(calibration):
    # Arrange
    x_axis = np.linspace(-1.0, 1.0, 40)
    data = 0.5 + 2.0 * x_axis + noisy_values(x_axis.size, 0.03)
    design = np.column_stack([np.ones_like(x_axis), x_axis])

    def residual(x, brute, i_index):
        return design @ x - data

    coefficients, sum_squares = np.linalg.lstsq(design, data, rcond=None)[:2]
    expected = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * sum_squares[0] / (data.size - 2))

    # Act
    values, errors, _ = calibration.FitDetector(residual, [0.0, 0.0], bounds=([-10.0, -10.0], [10.0, 10.0]))

    # Assert
    np.testing.assert_allclose(values, coefficients, rtol=1e-6)
    np.testing.assert_allclose(errors, expected, rtol=1e-6)


def test_errors_scale_with_residual_scatter(calibration):
    # Arrange - the same Jacobian with 10x the scatter gives 10x the error
    noise = noisy_values(30, 1.0)

    def fit_with(sigma):
        return calibration.FitDetector(lambda x, b, i: x[0] - sigma * noise, [0.0], bounds=([-10.0], [10.0]))[1]

    # Act
    small, large = fit_with(0.01), fit_with(0.1)

    # Assert
    np.testing.assert_allclose(large, 10.0 * small, rtol=1e-6)


def test_parameter_that_moves_nothing_is_undetermined(calibration):
    # Arrange - the second parameter does not enter the residual; a zero Jacobian column flags every
    # parameter, as in the ring calibration
    data = 1.0 + noisy_values(20, 0.05)

    def residual(x, brute, i_index):
        return x[0] - data

    # Act
    _, errors, _ = calibration.FitDetector(residual, [0.0, 0.0], bounds=([-10.0, -10.0], [10.0, 10.0]))

    # Assert
    np.testing.assert_array_equal(errors, [UNDETERMINED, UNDETERMINED])


def test_degenerate_pair_is_undetermined(calibration):
    # Arrange - only x0 + x1 is constrained (like Shift_x/Rot_y/TTH_0 at one L2); x2 is independent
    x_axis = np.linspace(-1.0, 1.0, 30)
    data = 1.0 + 0.5 * x_axis + noisy_values(x_axis.size, 0.02)

    def residual(x, brute, i_index):
        return x[0] + x[1] + x[2] * x_axis - data

    # Act
    _, errors, _ = calibration.FitDetector(residual, [0.0, 0.0, 0.0], bounds=([-10.0] * 3, [10.0] * 3))

    # Assert
    np.testing.assert_array_equal(errors[:2], [UNDETERMINED, UNDETERMINED])
    assert 0.0 < errors[2] < 0.02


def test_lm_method_reports_same_errors(calibration):
    # Arrange
    data = 3.0 + noisy_values(50, 0.1)
    calibration._ref_method = "lm"

    # Act
    _, errors, _ = calibration.FitDetector(lambda x, b, i: x[0] - data, [0.0], bounds=([-10.0], [10.0]))

    # Assert
    np.testing.assert_allclose(errors, [data.std(ddof=1) / np.sqrt(data.size)], rtol=1e-6)


def test_brute_force_errors_are_undetermined(calibration):
    # Arrange - brute() gives no covariance; bounds are (upper, lower) on this path
    def cost(x, brute, i_index):
        return float(np.sum((np.atleast_1d(x) - [0.2, -0.3]) ** 2))

    # Act
    values, errors, _ = calibration.FitDetector(cost, [0.0, 0.0], bounds=([1.0, 1.0], [-1.0, -1.0]), Brute=True)

    # Assert
    np.testing.assert_allclose(values, [0.2, -0.3], atol=1e-3)
    np.testing.assert_array_equal(errors, [UNDETERMINED, UNDETERMINED])
