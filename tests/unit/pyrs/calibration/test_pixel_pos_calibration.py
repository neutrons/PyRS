"""Unit tests for pyrs.calibration.pixel_pos_calibration"""

import copy

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import (
    DEFAULT_RECIPE,
    PARAMETER_NAMES,
    DetectorModel,
    PixelCalibration,
    RingPoints,
)
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.utilities.calibration_file_io import check_calibration_status, read_calibration_json_file
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import concatenate_ring_points, synthetic_ring_points

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

    # Assert - the degenerate set is flagged -2 (not NaN, not a huge "sigma"); Rot_x and lambda are
    # orthogonal to it and get real sigmas. Parameters with a small share in the degenerate direction
    # (here Shift_y, Shift_z, Rot_z) are flagged too: their error is undefined while it is free
    errors = calibration.calibration_error_array
    assert np.all(errors[[0, 4, 6]] == -2.0)
    assert np.all(errors[[3, 7]] > 0)
    assert np.all((errors > 0) | (errors == -2.0))


def test_repeated_sub_run_numbers_use_each_points_position(hb2b):
    """Two runs both numbered from sub-run 1: each point keeps its own arm angle and L2"""
    # Arrange - concatenated without renumbering, so sub-run numbers collide across distances
    near = synthetic_ring_points(hb2b, TRUE_PARAMS, l2=0.985)
    far = synthetic_ring_points(hb2b, TRUE_PARAMS, l2=0.75)
    merged = RingPoints(
        **{name: np.concatenate([getattr(near, name), getattr(far, name)]) for name in RingPoints.__dataclass_fields__}
    )
    assert set(near.sub_run.tolist()) == set(far.sub_run.tolist())

    # Act
    residual = PixelCalibration(merged, hb2b, wavelength=NOMINAL_WAVELENGTH).residual(TRUE_PARAMS)

    # Assert
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_parameter_stopped_at_bound_flagged(hb2b):
    # Arrange - true Rot_z is 1.2 deg; bounds of +/-0.5 deg stop it
    calibration = PixelCalibration(synthetic_ring_points(hb2b, TRUE_PARAMS), hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.set_calibration_array(np.r_[TRUE_PARAMS[:5], 0.0, TRUE_PARAMS[6:]])

    # Act
    with pytest.warns(UserWarning, match=r"\['Rot_z'\] stopped at a bound"):
        stage = calibration.refine("rot_z", ["Rot_z"], [-0.5], [0.5])

    # Assert
    assert stage.values[0] == pytest.approx(0.5)
    assert stage.errors[0] == -2.0 and calibration.calibration_error_array[5] == -2.0


def test_parameter_inside_bounds_not_flagged(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.set_calibration_array(np.r_[TRUE_PARAMS[:5], 0.0, TRUE_PARAMS[6:]])

    # Act
    stage = calibration.refine("rot_z", ["Rot_z"], [-5.0], [5.0])

    # Assert
    assert stage.values[0] == pytest.approx(TRUE_PARAMS[5], abs=1e-6)
    assert stage.errors[0] > 0


def test_two_distances_recover_two_theta_0(hb2b):
    """A second sample-detector distance breaks the Shift_x/Rot_y/TTH_0 degeneracy (docs/ground_truths.md)"""
    # Arrange
    near = synthetic_ring_points(hb2b, TRUE_PARAMS, l2=0.985)
    far = synthetic_ring_points(hb2b, TRUE_PARAMS, l2=0.75)
    calibration = PixelCalibration(concatenate_ring_points(near, far), hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    stage = calibration.FullCalibration()

    # Assert - all three get real errors and their true values
    fitted, errors = calibration.calibration_array, calibration.calibration_error_array
    assert stage.rmse < 1e-5
    assert np.all(errors[[0, 4, 6]] > 0)
    assert fitted[0] == pytest.approx(TRUE_PARAMS[0], abs=1e-5)  # Shift_x, m
    np.testing.assert_allclose(fitted[[4, 6]], TRUE_PARAMS[[4, 6]], atol=1e-3)  # Rot_y, TTH_0, deg


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
    with pytest.raises(ValueError, match="unknown parameter 'Shift_w'"):
        calibration.refine("bad", ["Shift_w"], [-1.0], [1.0])


@pytest.mark.parametrize("parameter", [8, -1, 2.0])
def test_refine_bad_parameter_index_raises(hb2b, true_rings, parameter):
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    with pytest.raises(ValueError, match="indices 0-7"):
        calibration.refine("bad", [parameter], [-1.0], [1.0])


def test_refine_lower_not_below_upper_raises(hb2b, true_rings):
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    with pytest.raises(ValueError, match="lower bound"):
        calibration.refine("bad", ["Shift_x"], [0.0], [0.0])


@pytest.mark.parametrize(
    "stage, free, lower, upper",
    [
        ("calibrate_wave_shift", ("TTH_0", "Lambda"), [-5.0, 1.54 - 0.025], [5.0, 1.54 + 0.025]),
        ("CalibrateShift", ("Shift_x", "Shift_y", "Shift_z"), [-0.05, -0.05, -0.15], [0.05, 0.05, 0.15]),
        ("CalibrateGeometry", PARAMETER_NAMES[:7], [-0.05] * 3 + [-5.0] * 4, [0.05] * 3 + [5.0] * 4),
        (
            "FullCalibration",
            PARAMETER_NAMES,
            [-0.05, -0.05, -0.15] + [-5.0] * 4 + [1.49],
            [0.05, 0.05, 0.15] + [5.0] * 4 + [1.59],
        ),
    ],
)
def test_stage_bounds_follow_fitcalibration(hb2b, true_rings, monkeypatch, stage, free, lower, upper):
    # Arrange - capture what the stage hands to refine()
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    seen = {}
    monkeypatch.setattr(calibration, "refine", lambda name, f, lo, up, guess=None: seen.update(f=f, lo=lo, up=up))

    # Act
    getattr(calibration, stage)()

    # Assert
    assert tuple(PARAMETER_NAMES[i] for i in seen["f"]) == free
    np.testing.assert_allclose(seen["lo"], lower)
    np.testing.assert_allclose(seen["up"], upper)


def test_set_calibration_array_wrong_length_raises(hb2b, true_rings):
    with pytest.raises(ValueError, match="8 values"):
        PixelCalibration(true_rings, hb2b, wavelength=1.54).set_calibration_array([0.0] * 7)


def test_set_calibration_array_nonpositive_wavelength_raises(hb2b, true_rings):
    with pytest.raises(ValueError, match="wavelength must be positive"):
        PixelCalibration(true_rings, hb2b, wavelength=1.54).set_calibration_array([0.0] * 8)


def test_non_ring_points_raises(hb2b):
    with pytest.raises(TypeError, match="RingPoints"):
        PixelCalibration("rings.npz", hb2b, wavelength=1.54)


def test_refine_bounds_length_mismatch_raises(hb2b, true_rings):
    calibration = PixelCalibration(true_rings, hb2b, wavelength=1.54)
    with pytest.raises(ValueError, match="2 values"):
        calibration.refine("bad", ["Shift_x", "Shift_z"], [-1.0], [1.0, 1.0])


def test_with_two_theta_0_at_singularity_raises(setup):
    # Rot_y 89.5 deg, then a 1 deg swing: the compensated rotation passes Rot_y = 90 deg
    calibration = DENEXDetectorShift(0.0, 0.0, 0.0, 0.0, 89.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="singularity"):
        DetectorModel(setup).with_two_theta_0(calibration, -1.0, 0.985)


def test_ring_points_select_keeps_every_field(true_rings):
    # Act
    subset = true_rings.select(np.array([0, 2]))

    # Assert
    for name in RingPoints.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(subset, name), getattr(true_rings, name)[[0, 2]])


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


# ---------------------------------------------------------------- 2theta_0 re-expression


LARGE_PARAMS = np.array([-0.012, 0.006, 0.040, -1.0, 1.5, 2.0, -1.5, 1.5250])


@pytest.mark.parametrize("params", [TRUE_PARAMS, LARGE_PARAMS], ids=["gui-like", "large"])
@pytest.mark.parametrize("new_two_theta_0", [0.0, 1.2])
def test_with_two_theta_0_keeps_two_theta_map(hb2b, params, new_two_theta_0):
    """2theta_0 is a virtual swing: trading it for rotations + shifts leaves every pixel unchanged"""
    # Arrange
    model = DetectorModel(hb2b)
    calibration = DENEXDetectorShift(*params[:7])

    # Act
    swapped = model.with_two_theta_0(calibration, new_two_theta_0, 0.985)

    # Assert
    assert swapped.two_theta_0 == new_two_theta_0
    swapped_params = [
        swapped.center_shift_x,
        swapped.center_shift_y,
        swapped.center_shift_z,
        swapped.rotation_x,
        swapped.rotation_y,
        swapped.rotation_z,
        swapped.two_theta_0,
    ]
    for arm in (60.0, 90.0, 120.0):
        np.testing.assert_allclose(
            _two_theta_map(model, swapped_params, arm), _two_theta_map(model, params, arm), atol=1e-10
        )
        grid = np.meshgrid(np.arange(0, NUM_PIXELS, 16), np.arange(0, NUM_PIXELS, 16), indexing="ij")
        np.testing.assert_allclose(
            model.eta(*grid, -arm, 0.985, swapped), model.eta(*grid, -arm, 0.985, calibration), atol=1e-9
        )


def test_with_two_theta_0_round_trip(hb2b):
    # Arrange
    model = DetectorModel(hb2b)
    original = DENEXDetectorShift(*LARGE_PARAMS[:7])

    # Act
    back = model.with_two_theta_0(model.with_two_theta_0(original, 0.0, 0.985), LARGE_PARAMS[6], 0.985)

    # Assert
    for name in ("center_shift_x", "center_shift_y", "center_shift_z", "rotation_x", "rotation_y", "rotation_z"):
        assert getattr(back, name) == pytest.approx(getattr(original, name), abs=1e-10)


def test_with_two_theta_0_virtual_swing_size(hb2b):
    """From the nominal geometry, alpha of 2theta_0 is exactly its virtual swing: beam-centre move
    L*tan(alpha), distance change L*(1/cos(alpha) - 1) and a face rotation alpha"""
    # Arrange
    alpha, arm_l2 = 0.5, 0.985

    # Act
    swapped = DetectorModel(hb2b).with_two_theta_0(DENEXDetectorShift(0, 0, 0, 0, 0, 0, alpha), 0.0, arm_l2)

    # Assert
    alpha_rad = np.deg2rad(alpha)
    assert swapped.rotation_y == pytest.approx(alpha, abs=1e-12)
    assert swapped.center_shift_x == pytest.approx(arm_l2 * np.tan(alpha_rad), rel=1e-9)
    assert swapped.center_shift_z == pytest.approx(arm_l2 * (1 / np.cos(alpha_rad) - 1), rel=1e-6)
    assert swapped.rotation_x == pytest.approx(0.0, abs=1e-12)
    assert swapped.rotation_z == pytest.approx(0.0, abs=1e-12)


def test_with_two_theta_0_bad_calibration_raises(hb2b):
    with pytest.raises(TypeError, match="DENEXDetectorShift"):
        DetectorModel(hb2b).with_two_theta_0([0.0] * 7, 0.0, 0.985)


def test_set_two_theta_0_keeps_fit(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.set_calibration_array(TRUE_PARAMS)
    before = calibration.residual(TRUE_PARAMS)

    # Act
    stage = calibration.set_two_theta_0(0.0)

    # Assert
    after = calibration.calibration_array
    assert after[6] == 0.0
    assert after[7] == TRUE_PARAMS[7]
    np.testing.assert_allclose(calibration.residual(after), before, atol=1e-10)
    assert np.all(calibration.calibration_error_array[:7] == -1.0)
    assert stage.name == "set_two_theta_0" and calibration.stages[-1] is stage


# ---------------------------------------------------------------- iterate()


def test_iterate_fixed_passes_runs_recipe(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    n_pass = calibration.iterate(passes=2, polish=False)

    # Assert
    assert n_pass == 2
    assert [stage.name for stage in calibration.stages] == list(DEFAULT_RECIPE) * 2


def test_iterate_with_polish_reaches_true_map(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    model = DetectorModel(hb2b)

    # Act
    n_pass = calibration.iterate(passes=3)

    # Assert
    assert n_pass == 3
    assert calibration.stages[-1].name == "FullCalibration"
    assert calibration.calibration_array[7] == pytest.approx(TRUE_PARAMS[7], abs=1e-6)
    for arm in (76.0, 103.0):
        np.testing.assert_allclose(
            _two_theta_map(model, calibration.calibration_array, arm),
            _two_theta_map(model, TRUE_PARAMS, arm),
            atol=1e-5,
        )


def test_iterate_tolerance_stops_before_max(hb2b, true_rings):
    # Arrange
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)

    # Act
    n_pass = calibration.iterate(tolerance=1e-3, max_passes=50, polish=False)

    # Assert
    assert 1 < n_pass < 50
    assert calibration.stages[-1].rmse < 1e-3


def test_iterate_not_converged_warns(hb2b, true_rings):
    calibration = PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    with pytest.warns(UserWarning, match="not converged"):
        calibration.iterate(tolerance=1e-12, max_passes=2, polish=False)


@pytest.mark.parametrize(
    "kwargs", [{"recipe": ["CalibrateShift", "no_such_stage"]}, {"recipe": []}, {"passes": 0}, {"tolerance": 0.0}]
)
def test_iterate_invalid_arguments_raise(hb2b, true_rings, kwargs):
    with pytest.raises(ValueError):
        PixelCalibration(true_rings, hb2b, wavelength=NOMINAL_WAVELENGTH).iterate(**kwargs)


# ---------------------------------------------------------------- error estimate and calibration file


@pytest.fixture(scope="module")
def _refined_noisy_once(hb2b):
    rings = synthetic_ring_points(hb2b, TRUE_PARAMS, strips=20, noise_deg=0.01, rng=np.random.default_rng(5))
    calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.iterate(passes=3)
    return calibration


@pytest.fixture
def refined_noisy(_refined_noisy_once):
    """A fresh copy per test: estimate_errors() and friends change the calibration in place"""
    return copy.deepcopy(_refined_noisy_once)


@pytest.fixture
def refined_with_errors(refined_noisy):
    refined_noisy.estimate_errors()
    return refined_noisy


def test_estimate_errors_holds_two_theta_0(refined_noisy):
    # Arrange
    calibration = refined_noisy
    before = calibration.calibration_array
    assert np.all(calibration.calibration_error_array[[0, 4, 6]] == -2.0)  # the degenerate set after FullCalibration

    # Act
    stage = calibration.estimate_errors()

    # Assert - 2theta_0 marked "not refined", the other seven get real errors, values barely move
    errors = calibration.calibration_error_array
    assert stage.free_parameters == ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "Lambda")
    assert errors[6] == -1.0
    assert np.all(np.isfinite(errors[[0, 1, 2, 3, 4, 5, 7]])) and np.all(errors[[0, 1, 2, 3, 4, 5, 7]] > 0)
    assert np.all(
        np.abs(calibration.calibration_array - before)[[0, 1, 2, 3, 4, 5, 7]] < errors[[0, 1, 2, 3, 4, 5, 7]]
    )
    assert calibration.calibration_array[6] == before[6]


def test_estimate_errors_wavelength_error_is_realistic(refined_noisy):
    # Act
    refined_noisy.estimate_errors()

    # Assert - the true wavelength lies within 4 sigma of the fit
    fitted, sigma = refined_noisy.calibration_array[7], refined_noisy.calibration_error_array[7]
    assert abs(fitted - TRUE_PARAMS[7]) < 4 * sigma


def _shift_vector(shift):
    return np.array(
        [
            shift.center_shift_x,
            shift.center_shift_y,
            shift.center_shift_z,
            shift.rotation_x,
            shift.rotation_y,
            shift.rotation_z,
            shift.two_theta_0,
        ]
    )


def test_write_calibration_round_trip(tmp_path, refined_with_errors):
    # Arrange
    path = tmp_path / "pixel_calibration.json"

    # Act
    refined_with_errors.write_calibration(path)

    # Assert - reduction's own reader returns every value and error, in order
    shift, shift_error, wavelength, wavelength_error, status = read_calibration_json_file(str(path))
    check_calibration_status(status)
    values, errors = refined_with_errors.calibration_array, refined_with_errors.calibration_error_array
    np.testing.assert_array_equal(_shift_vector(shift), values[:7])
    np.testing.assert_array_equal(_shift_vector(shift_error), errors[:7])
    assert (wavelength, wavelength_error) == (values[7], errors[7])
    assert shift_error.two_theta_0 == -1.0


@pytest.mark.parametrize("arm", [76.0, 103.0])
def test_written_calibration_reproduces_fitted_map(tmp_path, hb2b, refined_with_errors, arm):
    """Reduction, applying the file as written (TTH_0 included), must see the 2theta that was fitted"""
    # Arrange
    path = tmp_path / "pixel_calibration.json"
    refined_with_errors.write_calibration(path)
    shift = read_calibration_json_file(str(path))[0]
    instrument = ResidualStressInstrument(hb2b)
    grid = np.meshgrid(np.arange(NUM_PIXELS), np.arange(NUM_PIXELS), indexing="ij")

    # Act
    instrument.build_instrument(-arm, 0.985, shift)

    # Assert
    fitted = DetectorModel(hb2b).two_theta(
        *grid, -arm, 0.985, DENEXDetectorShift(*refined_with_errors.calibration_array[:7])
    )
    np.testing.assert_array_equal(instrument.get_pixels_2theta(2), fitted)


def test_estimate_errors_hold_rot_y_gives_tth0_error(refined_noisy):
    # Arrange - Rot_y held at its true value makes 2theta_0 identifiable
    params = refined_noisy.calibration_array
    params[:7] = TRUE_PARAMS[:7]
    refined_noisy.set_calibration_array(params)

    # Act
    refined_noisy.estimate_errors(hold=("Rot_y",))

    # Assert
    values, errors = refined_noisy.calibration_array, refined_noisy.calibration_error_array
    assert errors[4] == -1.0 and errors[6] > 0
    assert abs(values[6] - TRUE_PARAMS[6]) < 4 * errors[6]


def test_estimate_errors_hold_by_index_same_as_name(refined_noisy):
    # Arrange
    by_name = copy.deepcopy(refined_noisy)

    # Act
    refined_noisy.estimate_errors(hold=(6,))
    by_name.estimate_errors(hold=("TTH_0",))

    # Assert
    np.testing.assert_array_equal(refined_noisy.calibration_error_array, by_name.calibration_error_array)


def test_estimate_errors_unaffected_by_degeneracy(refined_with_errors):
    """With 2theta_0 held the fit has full rank: errors are the plain inv(JᵀJ) formula, all positive"""
    errors = refined_with_errors.calibration_error_array
    assert np.all(errors[[0, 1, 2, 3, 4, 5, 7]] > 0)
    assert errors[1] < 0.01  # Shift_y, m: not inflated by the Shift_x/Rot_y/TTH_0 direction


def test_estimate_errors_unknown_name_raises(refined_noisy):
    with pytest.raises(ValueError):
        refined_noisy.estimate_errors(hold=("Rot_w",))


def test_write_calibration_unrefined_raises(tmp_path, hb2b, true_rings):
    with pytest.raises(RuntimeError, match="nothing has been refined"):
        PixelCalibration(true_rings, hb2b, wavelength=1.54).write_calibration(tmp_path / "x.json")
