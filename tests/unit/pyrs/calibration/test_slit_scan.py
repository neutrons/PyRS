"""Slit-scan position-response analysis, tested on simulated scans (tests/util/slit_scan_helpers.py)"""

import numpy as np
import pytest

from pyrs.calibration.slit_scan import (
    SlitScanGeometry,
    active_edges,
    find_line_centres,
    fit_position_response,
    source_distance_from_two_gaps,
)
from tests.util.slit_scan_helpers import simulate_slit_scan

STAGE = np.linspace(-0.14, 0.14, 21)


def analyse(images, truth, gap=0.005, depth=None, degree=1, rows_per_band=64):
    geometry = SlitScanGeometry(0.985, gap, truth.mean_depth if depth is None else depth)
    centres = find_line_centres(images, rows_per_band=rows_per_band)
    return centres, fit_position_response(centres, STAGE, geometry, degree=degree)


def test_magnification():
    assert SlitScanGeometry(0.985, 0.0).magnification == 1.0
    assert SlitScanGeometry(0.985, 0.005, 0.005).magnification == pytest.approx(0.990 / 0.980)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(source_to_window=0.0, slit_gap=0.0),
        dict(source_to_window=1.0, slit_gap=1.0),
        dict(source_to_window=1.0, slit_gap=0.0, conversion_depth=-1e-3),
    ],
)
def test_invalid_geometry_raises(kwargs):
    with pytest.raises(ValueError):
        SlitScanGeometry(**kwargs)


def test_uniform_pitch_recovered_exactly_without_depth():
    # Arrange - a thin conversion layer leaves a symmetric line
    images, _, truth = simulate_slit_scan(STAGE, mean_depth=1e-7, gas_depth=4e-7, seed=None)

    # Act
    centres, response = analyse(images, truth)

    # Assert
    assert np.isfinite(centres.centre).all()
    assert response.pitch[0] == pytest.approx(0.314e-3, rel=1e-6)


def test_uniform_pitch_with_conversion_depth():
    # Arrange
    images, _, truth = simulate_slit_scan(STAGE, seed=None)

    # Act
    _, linear = analyse(images, truth, degree=1)
    _, cubic = analyse(images, truth, degree=3)

    # Assert - the exponential depth tail biases a Gaussian centre slightly; a cubic absorbs it
    assert linear.pitch[0] == pytest.approx(0.314e-3, rel=1e-4)
    assert cubic.pitch[0] == pytest.approx(0.314e-3, rel=1e-5)


def test_pitch_gradient_recovered():
    # Arrange - 1.5% change of the local pitch from the centre to the edge
    images, _, truth = simulate_slit_scan(STAGE, gradient=0.015, seed=None)

    # Act
    _, uniform = analyse(images, truth, degree=1)
    _, graded = analyse(images, truth, degree=2)

    # Assert
    assert uniform.chi2[0] / uniform.dof[0] > 100  # a uniform pitch cannot describe it
    assert graded.gradient[0] == pytest.approx(0.015, abs=2e-4)
    assert graded.local_pitch(1023, 0) > graded.local_pitch(0, 0)


def test_ignoring_the_magnification_biases_the_pitch():
    # Arrange
    images, _, truth = simulate_slit_scan(STAGE, seed=None)

    # Act - analysed as if the slit were on the conversion plane
    _, naive = analyse(images, truth, gap=0.0, depth=0.0)

    # Assert
    assert naive.pitch[0] == pytest.approx(0.314e-3 / truth.magnification_mean_depth, rel=1e-4)


def test_noisy_scan_errors_are_consistent():
    # Arrange
    images, efficiency, truth = simulate_slit_scan(STAGE, gradient=0.01, efficiency_spread=0.05, seed=3)

    # Act
    centres = find_line_centres(images, rows_per_band=32, efficiency=efficiency)
    response = fit_position_response(centres, STAGE, SlitScanGeometry(0.985, 0.005, truth.mean_depth), degree=2)

    # Assert
    pulls = (response.gradient - 0.01) / response.gradient_error
    assert np.all(np.abs(pulls) < 5)
    assert 0.3 < np.mean(response.chi2 / response.dof) < 3


def test_line_outside_active_area_is_not_found():
    # Arrange - the last position projects beyond the 300 mm active area
    stage = np.array([0.0, 0.10, 0.16])
    images, _, _ = simulate_slit_scan(stage, seed=None)

    # Act
    centres = find_line_centres(images, rows_per_band=64)

    # Assert
    assert np.isfinite(centres.centre[:2, 0]).all()
    assert np.isnan(centres.centre[2, 0])


def test_active_edges_of_flood():
    # Arrange - uniform flood on a 300 mm active area with 0.314 mm pixels
    centres = (np.arange(1024) - 511.5) * 0.314e-3
    flood = np.where(np.abs(centres) <= 0.150, 100.0, 0.0)[:, None] * np.ones((1, 64))

    # Act
    low, high = active_edges(flood)

    # Assert - 150 mm / 0.314 mm = 477.7 channels either side of 511.5
    assert low == pytest.approx(511.5 - 477.7, abs=1.0)
    assert high == pytest.approx(511.5 + 477.7, abs=1.0)


def test_source_distance_from_two_gaps():
    # Arrange - the same detector scanned with the slit 5 mm and 55 mm in front of the window
    near, _, truth = simulate_slit_scan(STAGE, slit_gap=0.005, seed=None)
    stage_far = STAGE * 0.9  # keep the more magnified line on the detector
    far, _, _ = simulate_slit_scan(stage_far, slit_gap=0.055, seed=None)
    unit = SlitScanGeometry(0.985, 0.0)  # M = 1: apparent pitch in stage metres per channel

    # Act
    p_near = fit_position_response(find_line_centres(near, rows_per_band=64), STAGE, unit, degree=3).pitch[0]
    p_far = fit_position_response(find_line_centres(far, rows_per_band=64), stage_far, unit, degree=3).pitch[0]

    # Assert
    assert source_distance_from_two_gaps(p_near, p_far, 0.005, 0.055) == pytest.approx(0.985, abs=2e-3)


def test_input_validation_raises():
    images, _, _ = simulate_slit_scan(STAGE[:3], n_rows=8, seed=None)
    centres = find_line_centres(images, rows_per_band=8)
    with pytest.raises(ValueError):
        find_line_centres(images[0])
    with pytest.raises(ValueError):
        fit_position_response(centres, STAGE, SlitScanGeometry(0.985, 0.0))
    with pytest.raises(ValueError):
        source_distance_from_two_gaps(1.0, 1.1, 0.005, 0.005)
