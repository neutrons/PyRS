"""Unit tests for ring extraction in pyrs.calibration.pixel_pos_calibration, on synthetic detector images"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyrs.calibration.pixel_pos_calibration import (
    DetectorModel,
    PixelCalibration,
    _reject_outliers,
    extract_ring_points,
)
from pyrs.calibration.powder_standards import HB2B_STANDARDS
from pyrs.core.instrument_geometry import DENEXDetectorShift
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace, synthetic_detector_image

NI, FE, _MO = HB2B_STANDARDS
TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
NOMINAL_WAVELENGTH = 1.540
# subset of the HB2B_4971 layout: one ring each, except arm 93 (Ni 311 and 222)
ARMS = (76.0, 78.0, 93.0, 103.0)
POWDERS = (NI, FE, NI, FE)
STRIPS = 8


@pytest.fixture(scope="module")
def hb2b():
    return default_hb2b_geometry()


@pytest.fixture(scope="module")
def workspace(hb2b):
    return synthetic_calibration_workspace(hb2b, TRUE_PARAMS, ARMS, POWDERS)


@pytest.fixture(scope="module")
def nominal_points(workspace):
    diagnostics = []
    points = extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, diagnostics=diagnostics)
    return points, diagnostics


def _true_columns(hb2b, points):
    """Exact column where each point's ring crosses its row, for the true geometry"""
    model = DetectorModel(hb2b)
    shift = DENEXDetectorShift(*TRUE_PARAMS[:7])
    columns = []
    for d, row, arm, l2 in zip(points.d_spacing, points.index_1, points.det_2theta, points.l2):
        bragg = 2 * np.degrees(np.arcsin(TRUE_PARAMS[7] / (2 * d)))
        columns.append(brentq(lambda c: float(model.two_theta(c, row, -arm, l2, shift)) - bragg, 0, 1023))
    return np.array(columns)


def test_points_lie_on_true_rings(hb2b, nominal_points):
    # Arrange
    points, _ = nominal_points

    # Act
    error = points.index_0 - _true_columns(hb2b, points)

    # Assert - sub-pixel and unbiased, although histogrammed with the nominal geometry
    assert np.sqrt(np.mean(error**2)) < 0.15
    assert abs(error.mean()) < 0.05


def test_every_ring_on_the_detector_is_fitted(nominal_points):
    # Arrange
    points, _ = nominal_points

    # Act
    rings = sorted(set(zip(points.sub_run.tolist(), points.hkl_squared.tolist())))

    # Assert - sub-run 3 (arm 93) sees both Ni 311 and Ni 222
    assert rings == [(1, 8), (2, 6), (3, 11), (3, 12), (4, 8)]
    assert len(points) == 5 * STRIPS
    np.testing.assert_allclose(points.d_spacing[points.hkl_squared == 11], NI.lattice_a / np.sqrt(11))


def test_points_independent_of_start_geometry(hb2b, workspace, nominal_points):
    """Histogramming and mapping back with the same geometry cancels the geometry error"""
    # Arrange
    nominal, _ = nominal_points

    # Act
    exact = extract_ring_points(
        workspace, TRUE_PARAMS[7], calibration=DENEXDetectorShift(*TRUE_PARAMS[:7]), strips=STRIPS
    )

    # Assert
    np.testing.assert_array_equal(exact.index_1, nominal.index_1)
    assert np.abs(exact.index_0 - nominal.index_0).max() < 0.05


def test_sigma_is_fit_center_uncertainty(nominal_points):
    points, diagnostics = nominal_points
    accepted = [d for d in diagnostics if d.accepted]
    np.testing.assert_allclose(points.sigma, [d.center_error for d in accepted])
    assert np.all((points.sigma > 0) & (points.sigma < 0.05))


def test_diagnostics_one_per_band_and_ring(nominal_points):
    # Arrange
    points, diagnostics = nominal_points

    # Assert
    assert len(diagnostics) == 5 * STRIPS
    assert all(d.accepted and d.reason == "" for d in diagnostics)
    first = diagnostics[0]
    assert first.two_theta.shape == first.intensity.shape
    assert first.center == pytest.approx(first.bragg_two_theta, abs=1.0)


def test_extract_then_iterate_recovers_two_theta_map(hb2b, nominal_points):
    """End to end: images -> ring points -> staged refinement from the nominal geometry"""
    # Arrange
    points, _ = nominal_points
    calibration = PixelCalibration(points, hb2b, wavelength=NOMINAL_WAVELENGTH)
    model = DetectorModel(hb2b)
    grid = np.meshgrid(np.arange(0, 1024, 16), np.arange(0, 1024, 16), indexing="ij")

    # Act
    calibration.iterate(passes=3)

    # Assert
    fitted = DENEXDetectorShift(*calibration.calibration_array[:7])
    truth = DENEXDetectorShift(*TRUE_PARAMS[:7])
    for arm in (76.0, 90.0, 103.0):
        np.testing.assert_allclose(
            model.two_theta(*grid, -arm, 0.985, fitted), model.two_theta(*grid, -arm, 0.985, truth), atol=5e-3
        )
    assert calibration.calibration_array[7] == pytest.approx(TRUE_PARAMS[7], abs=5e-4)


def test_masked_band_is_rejected(hb2b, workspace):
    # Arrange - mask the rows of the first band (rows 40 to 157 for 8 bands)
    mask = np.ones(1024 * 1024)
    mask.reshape(1024, 1024)[:, 40:158] = 0.0
    diagnostics = []

    # Act
    points = extract_ring_points(
        workspace, NOMINAL_WAVELENGTH, strips=STRIPS, sub_runs=[1], mask=mask, diagnostics=diagnostics
    )

    # Assert
    assert len(points) == STRIPS - 1
    assert points.index_1.min() > 158
    assert [d.reason for d in diagnostics if not d.accepted] == ["band has no unmasked pixels"]


def test_vanadium_normalisation_removes_efficiency_bias(hb2b):
    # Arrange - pixel efficiency falls 4x across the detector columns, so the sample image is distorted
    efficiency = np.repeat(np.linspace(1.0, 0.25, 1024), 1024)
    image = synthetic_detector_image(hb2b, TRUE_PARAMS, 93.0, NI, poisson=False) * efficiency
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, [93.0], [NI], poisson=False)
    workspace.set_raw_counts(1, image)

    # Act
    plain = extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS)
    normalised = extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, vanadium=1000.0 * efficiency)

    # Assert
    plain_error = np.abs(plain.index_0 - _true_columns(hb2b, plain)).max()
    normalised_error = np.abs(normalised.index_0 - _true_columns(hb2b, normalised)).max()
    assert normalised_error < 0.05
    assert normalised_error < plain_error


def test_smooth_keeps_rows_and_ring_shape(hb2b, workspace):
    # Act
    smoothed = extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, sub_runs=[1], smooth=True)

    # Assert - same points, columns on a quadratic through the ring, still on the true ring
    coefficients = np.polyfit(smoothed.index_1, smoothed.index_0, 2)
    np.testing.assert_allclose(np.polyval(coefficients, smoothed.index_1), smoothed.index_0, atol=1e-8)
    assert np.abs(smoothed.index_0 - _true_columns(hb2b, smoothed)).max() < 0.1


def test_unknown_powder_sub_run_skipped_with_warning(hb2b, workspace):
    # Arrange - a copy whose sub-run 2 sits at an sy with no standard
    odd = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, ARMS[:2], POWDERS[:2])
    odd.set_sample_log("sy", np.array([1, 2]), np.array([NI.sy, 40.0]))

    # Act / Assert
    with pytest.warns(UserWarning, match="sub-run 2"):
        points = extract_ring_points(odd, NOMINAL_WAVELENGTH, strips=STRIPS)
    assert set(points.sub_run.tolist()) == {1}


def test_weak_rings_rejected_then_raises(hb2b):
    # Arrange - rings barely above a high background
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, [76.0], [NI], peak_height=0.02, background=500.0)
    diagnostics = []

    # Act / Assert
    with pytest.raises(ValueError, match="no ring points"):
        extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, diagnostics=diagnostics)
    assert diagnostics and not any(d.accepted for d in diagnostics)


def test_reject_outliers_flags_far_point():
    # Arrange - a smooth ring with one point 5 px off
    rows = np.linspace(100, 900, 9)
    points = [{"sub_run": 1, "hkl_squared": 8, "index_1": r, "index_0": 300 + 1e-4 * (r - 500) ** 2} for r in rows]
    points[4]["index_0"] += 5.0

    # Act
    _reject_outliers(points, threshold=5.0, smooth=False)

    # Assert
    assert [i for i, p in enumerate(points) if p.get("reject")] == [4]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"wavelength": 0.0}, "wavelength"),
        ({"strips": 0}, "strips"),
        ({"bin_width": 0.0}, "bin_width"),
        ({"strips": 1000}, "do not fit"),
        ({"mask": np.ones(10)}, "one value per detector pixel"),
    ],
)
def test_invalid_arguments_raise(workspace, kwargs, match):
    with pytest.raises(ValueError, match=match):
        extract_ring_points(workspace, **{"wavelength": NOMINAL_WAVELENGTH, **kwargs})
