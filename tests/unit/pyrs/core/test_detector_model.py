"""Unit tests for pyrs.core.detector_model: the detector forward model of reduction and the ring calibration

ResidualStressInstrument.build_instrument delegates to DetectorModel (#1011), so the pixel map is pinned
to reference values computed by the reduction code before that change; see docs/ground_truths.md.
"""

import numpy as np
import pytest

from pyrs.core.detector_model import DetectorModel, rotation_matrix, rotation_matrix_x, rotation_matrix_y
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from tests.util.instrument_helpers import default_hb2b_geometry

NUM_PIXELS = 1024
# a GUI-like calibration and a large misalignment (vectors: shifts, rotations, 2theta_0, wavelength)
TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
LARGE_PARAMS = np.array([-0.012, 0.006, 0.040, -1.0, 1.5, 2.0, -1.5, 1.5250])


@pytest.fixture(scope="module")
def setup():
    return DENEXDetectorGeometry(NUM_PIXELS, NUM_PIXELS, 0.3 / NUM_PIXELS, 0.3 / NUM_PIXELS, 0.985, False)


@pytest.fixture(scope="module")
def hb2b():
    return default_hb2b_geometry()


def _two_theta_map(model, params, arm):
    grid = np.meshgrid(np.arange(NUM_PIXELS), np.arange(NUM_PIXELS), indexing="ij")
    return model.two_theta(*grid, -arm, 0.985, DENEXDetectorShift(*params[:7]))


CALIBRATION_ARM85 = DENEXDetectorShift(0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656)
CALIBRATION_ARM0 = DENEXDetectorShift(0.001, 0.002, -0.01, 0.0, 0.2, 0.0, -0.1)
REFERENCE_PIXELS = [(0, 0), (0, 1023), (1023, 0), (1023, 1023), (511, 511), (200, 700)]
# 2theta and eta (degrees) of REFERENCE_PIXELS from ResidualStressInstrument.build_instrument as it was
# before #1011 (origin/next 7dde880b), when it had its own copy of the geometry
GEOMETRY_CASES = [
    pytest.param(
        0.0,
        None,
        None,
        [12.142212654967848] * 4 + [0.012050156728558248, 6.1806220502915075],
        [-135.0, 135.0, -45.0, 45.0, -135.0, 148.82033952701045],
        id="nominal-arm0",
    ),
    pytest.param(
        -85.0,
        0.95,
        None,
        [
            76.20579124627643,
            76.20579124627643,
            93.91667219492433,
            93.91667219492433,
            84.99116538981714,
            79.53054896737282,
        ],
        [
            -9.121496391514825,
            9.121496391514825,
            -8.87717003734764,
            8.87717003734764,
            -0.00886853652133368,
            3.3678465120396197,
        ],
        id="nominal-arm85",
    ),
    pytest.param(
        -85.0,
        0.95,
        CALIBRATION_ARM85,
        [
            75.2560645501971,
            75.63022630845553,
            92.7590457623418,
            93.11309640206474,
            84.09321549344203,
            78.77900793744644,
        ],
        [
            -9.02001226908476,
            9.025573764387502,
            -9.134905589008042,
            8.41851879218629,
            -0.1831636435654218,
            3.2684784035019447,
        ],
        id="calib-arm85",
    ),
    pytest.param(
        0.0,
        None,
        CALIBRATION_ARM0,
        [
            12.159062553565171,
            12.318217464831982,
            12.208811955703002,
            12.367079177539903,
            0.1137036024204848,
            6.27097693411383,
        ],
        [
            -135.25041236144665,
            134.48570727368795,
            -44.4813200292906,
            45.24602456865219,
            73.32464204776555,
            147.71073161192453,
        ],
        id="calib-arm0",
    ),
]
# loose enough for a last-bit difference in a future numpy, far tighter than any geometry change
REFERENCE_RTOL = 1e-12


@pytest.mark.parametrize("arm_two_theta, l2, calibration, two_theta_ref, eta_ref", GEOMETRY_CASES)
def test_detector_model_matches_reference(setup, arm_two_theta, l2, calibration, two_theta_ref, eta_ref):
    # Arrange
    index_0, index_1 = np.array(REFERENCE_PIXELS).T
    model = DetectorModel(setup)

    # Act
    two_theta = model.two_theta(index_0, index_1, arm_two_theta, l2, calibration)
    eta = model.eta(index_0, index_1, arm_two_theta, l2, calibration)

    # Assert
    np.testing.assert_allclose(two_theta, two_theta_ref, rtol=REFERENCE_RTOL)
    np.testing.assert_allclose(eta, eta_ref, rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("arm_two_theta, l2, calibration, two_theta_ref, eta_ref", GEOMETRY_CASES)
def test_build_instrument_matches_reference(setup, arm_two_theta, l2, calibration, two_theta_ref, eta_ref):
    """Reduction's pixel map, through its own 2D arrays, is unchanged by the delegation (#1011)"""
    # Arrange
    instrument = ResidualStressInstrument(setup)
    index_0, index_1 = np.array(REFERENCE_PIXELS).T

    # Act
    instrument.build_instrument(arm_two_theta, l2, calibration)

    # Assert
    np.testing.assert_allclose(instrument.get_pixels_2theta(2)[index_0, index_1], two_theta_ref, rtol=REFERENCE_RTOL)
    np.testing.assert_allclose(instrument.get_eta_values(2)[index_0, index_1], eta_ref, rtol=REFERENCE_RTOL)


def test_rotation_matrix_combined_rotation_is_orthogonal():
    """Regression: the three axis matrices were once combined element-wise (``*``) instead of by matrix
    product (``@``), which is not a rotation when two or more angles are non-zero. See docs/ground_truths.md.
    """
    # Act
    matrix = rotation_matrix(np.deg2rad(0.3), np.deg2rad(-0.5), np.deg2rad(1.2))

    # Assert - a rotation is orthogonal with determinant +1
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-6)


def test_rotation_matrix_single_axis_matches_axis_matrix():
    # Arrange
    rot_x_rad = np.deg2rad(2.0)

    # Act / Assert
    np.testing.assert_allclose(rotation_matrix(rot_x_rad, 0.0, 0.0), rotation_matrix_x(rot_x_rad), atol=1e-6)


def test_rotation_matrix_resolves_tiny_angle_change():
    """Regression: float32 matrices could not resolve angle changes below ~3e-6 deg, which made
    numerical derivatives of the calibration residual with respect to angles zero.
    See docs/ground_truths.md.
    """
    # Arrange
    arm_angle_rad = np.deg2rad(85.0)
    delta_rad = 1e-9

    # Act
    before = rotation_matrix_y(arm_angle_rad)
    after = rotation_matrix_y(arm_angle_rad + delta_rad)

    # Assert - the change is the analytic derivative times delta, not rounding noise
    assert before.dtype == np.float64
    derivative = np.array(
        [
            [-np.sin(arm_angle_rad), 0.0, np.cos(arm_angle_rad)],
            [0.0, 0.0, 0.0],
            [-np.cos(arm_angle_rad), 0.0, -np.sin(arm_angle_rad)],
        ]
    )
    np.testing.assert_allclose((after - before) / delta_rad, derivative, atol=1e-5)


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


def test_with_two_theta_0_at_singularity_raises(setup):
    # Rot_y 89.5 deg, then a 1 deg swing: the compensated rotation passes Rot_y = 90 deg
    calibration = DENEXDetectorShift(0.0, 0.0, 0.0, 0.0, 89.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="singularity"):
        DetectorModel(setup).with_two_theta_0(calibration, -1.0, 0.985)


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
