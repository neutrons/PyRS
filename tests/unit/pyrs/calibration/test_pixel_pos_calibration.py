"""Unit tests for pyrs.calibration.pixel_pos_calibration"""

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import (
    PARAMETER_NAMES,
    DetectorModel,
    PixelCalibration,
    RingPoints,
)
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_ring_points

NUM_PIXELS = 1024


@pytest.fixture(scope="module")
def setup():
    return DENEXDetectorGeometry(NUM_PIXELS, NUM_PIXELS, 0.3 / NUM_PIXELS, 0.3 / NUM_PIXELS, 0.985, False)


@pytest.fixture(scope="module")
def full_grid():
    return np.meshgrid(np.arange(NUM_PIXELS), np.arange(NUM_PIXELS), indexing="ij")


GEOMETRY_CASES = [
    pytest.param(0.0, None, None, id="nominal-arm0"),
    pytest.param(-85.0, 0.95, None, id="nominal-arm85"),
    pytest.param(-85.0, 0.95, DENEXDetectorShift(0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656), id="calib-arm85"),
    pytest.param(0.0, None, DENEXDetectorShift(0.001, 0.002, -0.01, 0.0, 0.2, 0.0, -0.1), id="calib-arm0"),
]


@pytest.mark.parametrize("arm_two_theta, l2, calibration", GEOMETRY_CASES)
def test_detector_model_matches_build_instrument(setup, full_grid, arm_two_theta, l2, calibration):
    """The calibration's forward model must reproduce reduction's pixel map exactly"""
    # Arrange
    instrument = ResidualStressInstrument(setup)
    instrument.build_instrument(arm_two_theta, l2, calibration)
    model = DetectorModel(setup)

    # Act
    two_theta = model.two_theta(*full_grid, arm_two_theta, l2, calibration)
    eta = model.eta(*full_grid, arm_two_theta, l2, calibration)

    # Assert
    np.testing.assert_array_equal(two_theta, instrument.get_pixels_2theta(2))
    np.testing.assert_array_equal(eta, instrument.get_eta_values(2))


def test_detector_model_fractional_pixel_between_neighbours(setup):
    # Arrange
    model = DetectorModel(setup)
    calibration = DENEXDetectorShift(0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656)

    # Act
    left, middle, right = model.two_theta(np.array([300.0, 300.5, 301.0]), 700.0, -85.0, None, calibration)

    # Assert
    assert min(left, right) < middle < max(left, right)
    assert middle == pytest.approx((left + right) / 2, abs=1e-4)


def test_detector_model_centre_pixel_at_arm_angle(setup):
    # Arrange
    model = DetectorModel(setup)
    centre = (NUM_PIXELS - 1) / 2

    # Act
    two_theta = model.two_theta(centre, centre, -90.0, None, None)

    # Assert
    assert two_theta == pytest.approx(90.0, abs=1e-5)


def test_detector_model_preserves_input_shape(setup):
    # Arrange
    model = DetectorModel(setup)
    index_0 = np.full((3, 4), 100.0)

    # Act
    two_theta = model.two_theta(index_0, np.arange(4.0), -85.0, None, None)

    # Assert
    assert two_theta.shape == (3, 4)


def test_detector_model_bad_setup_raises():
    with pytest.raises(TypeError, match="DENEXDetectorGeometry"):
        DetectorModel("not a geometry")


def test_detector_model_bad_calibration_raises(setup):
    with pytest.raises(TypeError, match="DENEXDetectorShift"):
        DetectorModel(setup).two_theta(0.0, 0.0, -85.0, None, [0.0] * 7)


# ---------------------------------------------------------------- refinement on synthetic rings

TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
NOMINAL_WAVELENGTH = 1.540  # Si422 mono setting the run starts from


@pytest.fixture(scope="module")
def hb2b():
    return default_hb2b_geometry()


@pytest.fixture(scope="module")
def true_rings(hb2b):
    return synthetic_ring_points(hb2b, TRUE_PARAMS)


def _central_derivative(calibration, unit, step):
    plus = calibration.residual(TRUE_PARAMS + step * unit)
    minus = calibration.residual(TRUE_PARAMS - step * unit)
    return (plus - minus) / (2 * step)


def _two_theta_map(model, params, arm):
    grid = np.meshgrid(np.arange(NUM_PIXELS), np.arange(NUM_PIXELS), indexing="ij")
    return model.two_theta(*grid, -arm, 0.985, DENEXDetectorShift(*params[:7]))


def test_residual_zero_at_true_geometry(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    residual = calibration.residual(TRUE_PARAMS)

    # Assert
    assert residual.shape == (len(true_rings),)
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_calibrate_wave_length_recovers_wavelength(hb2b):
    """Regression: the draft derived d from the nominal wavelength, leaving lambda unconstrained"""
    # Arrange - nominal geometry, wavelength 0.6% below the mono setting
    true_params = np.zeros(8)
    true_params[7] = 1.5308
    rings = synthetic_ring_points(hb2b, true_params)
    calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    stage = calibration.calibrate_wave_length()

    # Assert - converged to the true value, not pinned at the 1.515 lower bound
    assert stage.status > 0
    assert calibration.calibration_array[7] == pytest.approx(1.5308, abs=1e-7)


def test_full_calibration_recovers_identifiable_params(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    stage = calibration.FullCalibration()

    # Assert - Shift_x, Rot_y and TTH_0 are exactly degenerate as a set (see the refine() note), and
    # Shift_y is weak (a vertical shift moves 2theta only to second order), so the noise-free fit
    # stops on the optimizer's tolerance along it: tolerances follow each parameter's sensitivity
    fitted = calibration.calibration_array
    assert stage.rmse < 1e-5
    assert fitted[1] == pytest.approx(TRUE_PARAMS[1], abs=1e-4)  # Shift_y, m
    assert fitted[2] == pytest.approx(TRUE_PARAMS[2], abs=1e-4)  # Shift_z, m
    np.testing.assert_allclose(fitted[[3, 5]], TRUE_PARAMS[[3, 5]], atol=2e-3)  # Rot_x, Rot_z, deg
    assert fitted[7] == pytest.approx(TRUE_PARAMS[7], abs=1e-6)  # Lambda, A


def test_degenerate_errors_flagged_not_nan(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    calibration.FullCalibration()

    # Assert - every error is either a real sigma or the -2 "undetermined" flag
    errors = calibration.calibration_error_array
    assert np.all(np.isfinite(errors))
    assert np.all((errors >= 0) | (errors == -2.0))


@pytest.mark.parametrize("arm", [60.0, 90.0, 120.0])
def test_full_calibration_reproduces_two_theta_map(hb2b, true_rings, arm):
    """Whatever point of the degenerate family the fit lands on, reduction sees the same 2theta"""
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    model = DetectorModel(hb2b)

    # Act
    calibration.FullCalibration()

    # Assert - including arm angles outside the measured 76-103 deg range
    fitted_map = _two_theta_map(model, calibration.calibration_array, arm)
    np.testing.assert_allclose(fitted_map, _two_theta_map(model, TRUE_PARAMS, arm), atol=1e-5)


def test_refine_with_rot_y_fixed_recovers_all(hb2b, true_rings):
    """Fixing one member of the Shift_x/Rot_y/TTH_0 degeneracy makes every parameter identifiable"""
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    start = np.zeros(8)
    start[4] = TRUE_PARAMS[4]
    start[7] = NOMINAL_WAVELENGTH
    calibration.set_calibration_array(start)
    free = ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_z", "TTH_0", "Lambda"]

    # Act
    calibration.refine("no rot_y", free, [-0.05, -0.05, -0.15, -5, -5, -5, 1.49], [0.05, 0.05, 0.15, 5, 5, 5, 1.59])

    # Assert
    np.testing.assert_allclose(calibration.calibration_array, TRUE_PARAMS, atol=1e-5)


def test_stage_changes_only_free_parameters(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    before = calibration.calibration_array

    # Act
    calibration.CalibrateShift()

    # Assert
    after = calibration.calibration_array
    np.testing.assert_array_equal(after[3:], before[3:])
    assert np.all(after[:3] != before[:3])
    assert np.all(calibration.calibration_error_array[3:] == -1.0)
    assert np.all(calibration.calibration_error_array[:3] >= 0.0)


def test_stages_record_history(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    calibration.CalibrateShift()
    calibration.CalibrateRotation()
    calibration.calibrate_wave_shift()

    # Assert
    assert [stage.name for stage in calibration.stages] == [
        "CalibrateShift",
        "CalibrateRotation",
        "calibrate_wave_shift",
    ]
    assert calibration.stages[-1].free_parameters == ("TTH_0", "Lambda")
    assert calibration.residual_rmse == sorted(calibration.residual_rmse, reverse=True)
    assert len(calibration.residual_sum) == 3
    assert calibration.calibration_status == calibration.stages[-1].status


def test_errors_shrink_with_more_points(hb2b):
    # Arrange - same noise on the ring positions, 5 vs 40 points per ring
    errors = []
    for strips in (5, 40):
        rings = synthetic_ring_points(hb2b, TRUE_PARAMS, strips=strips, noise_deg=0.01)
        calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
        calibration.set_calibration_array(np.r_[TRUE_PARAMS[:6], 0.0, NOMINAL_WAVELENGTH])

        # Act
        calibration.calibrate_wave_shift()
        errors.append(calibration.calibration_error_array[6:])

    # Assert - about sqrt(8) smaller for 8x the points
    assert np.all(errors[1] < errors[0] / 2)


def test_errors_cover_true_values(hb2b):
    # Arrange
    rings = synthetic_ring_points(hb2b, TRUE_PARAMS, strips=20, noise_deg=0.01, rng=np.random.default_rng(1))
    calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.set_calibration_array(np.r_[TRUE_PARAMS[:6], 0.0, NOMINAL_WAVELENGTH])

    # Act
    calibration.calibrate_wave_shift()

    # Assert - within 4 sigma
    fitted, sigma = calibration.calibration_array[6:], calibration.calibration_error_array[6:]
    assert np.all(np.abs(fitted - TRUE_PARAMS[6:]) < 4 * sigma)


@pytest.mark.parametrize(
    "jac_step", [np.finfo(float).eps ** 0.5, np.finfo(float).eps ** (1 / 3)], ids=["2-point", "3-point"]
)
@pytest.mark.parametrize("parameter", ["Rot_x", "Rot_y", "Rot_z", "TTH_0"])
def test_scipy_default_step_gives_accurate_angle_derivative(hb2b, true_rings, parameter, jac_step):
    """Regression: with float32 rotation matrices these steps gave zero or noisy angle derivatives"""
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    unit = np.zeros(8)
    unit[PARAMETER_NAMES.index(parameter)] = 1.0

    # Act
    at_default = _central_derivative(calibration, unit, jac_step)
    reference = _central_derivative(calibration, unit, 1e-3)

    # Assert - per point, relative to the largest derivative
    assert np.abs(at_default - reference).max() < 1e-3 * np.abs(reference).max()


def test_init_nonpositive_diff_step_raises(hb2b, true_rings):
    with pytest.raises(ValueError, match="diff_step"):
        PixelCalibration(true_rings, hb2b, wavelength=1.54, diff_step=0.0)


def test_lm_method_ignores_bounds(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH, method="lm")
    calibration.set_calibration_array(np.r_[TRUE_PARAMS[:6], 0.0, NOMINAL_WAVELENGTH])

    # Act
    calibration.calibrate_wave_shift()

    # Assert
    np.testing.assert_allclose(calibration.calibration_array[6:], TRUE_PARAMS[6:], atol=1e-6)


def test_init_empty_ring_points_raises(hb2b, true_rings):
    with pytest.raises(ValueError, match="at least one ring point"):
        PixelCalibration(true_rings.select(np.zeros(len(true_rings), dtype=bool)), hb2b, wavelength=1.54)


@pytest.mark.parametrize("kwargs, match", [({"wavelength": 0.0}, "wavelength"), ({"method": "nelder"}, "method")])
def test_init_invalid_arguments_raise(hb2b, true_rings, kwargs, match):
    with pytest.raises(ValueError, match=match):
        PixelCalibration(true_rings, hb2b, **{"wavelength": 1.54, **kwargs})


def test_refine_start_outside_bounds_raises(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    calibration.set_calibration_array([0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 1.54])

    # Act / Assert - CalibrateGeometry bounds Shift_z to +/-0.05
    with pytest.raises(ValueError, match="Shift_z"):
        calibration.CalibrateGeometry()


def test_refine_too_few_points_raises(hb2b, true_rings):
    calibration = PixelCalibration(true_rings.select(np.arange(2)), hb2b, wavelength=1.54)
    with pytest.raises(ValueError, match="more ring points"):
        calibration.CalibrateShift()


def test_refine_unknown_parameter_raises(hb2b, true_rings):
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    with pytest.raises(ValueError, match="free parameters"):
        calibration.refine("bad", ["Shift_w"], [-1.0], [1.0])


def test_set_calibration_array_wrong_length_raises(hb2b, true_rings):
    with pytest.raises(ValueError, match="8 values"):
        PixelCalibration(true_rings, hb2b, wavelength=1.54).set_calibration_array([0.0] * 7)


def test_parameter_names_match_calibration_json_keys():
    assert PARAMETER_NAMES == ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")


# ---------------------------------------------------------------- RingPoints


def test_ring_points_save_load_round_trip(tmp_path, true_rings):
    # Arrange
    path = tmp_path / "rings.npz"

    # Act
    true_rings.save(path)
    loaded = RingPoints.load(path)

    # Assert
    for name in ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(true_rings, name))


def test_ring_points_load_wrong_file_raises(tmp_path):
    # Arrange
    path = tmp_path / "other.npz"
    np.savez(path, x=np.zeros(3))

    # Act / Assert
    with pytest.raises(ValueError, match="not a RingPoints file"):
        RingPoints.load(path)


def test_ring_points_unequal_lengths_raise():
    with pytest.raises(ValueError, match="equal length"):
        RingPoints([1, 1], [0.0], [0.0], [85.0], [0.985], [1.2], [8], ["Ni"])


def test_ring_points_nonpositive_d_spacing_raises():
    with pytest.raises(ValueError, match="d_spacing"):
        RingPoints([1], [0.0], [0.0], [85.0], [0.985], [0.0], [8], ["Ni"])


def test_ring_points_default_sigma_is_nan():
    points = RingPoints([1], [0.0], [0.0], [85.0], [0.985], [1.2], [8], ["Ni"])
    assert np.isnan(points.sigma).all()
