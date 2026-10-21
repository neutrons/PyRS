"""Unit tests for pyrs.calibration.pixel_pos_calibration"""

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument

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
