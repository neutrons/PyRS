"""Unit tests for ring extraction in pyrs.calibration.pixel_pos_calibration, on synthetic detector images"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyrs.calibration.pixel_pos_calibration import (
    DetectorModel,
    PixelCalibration,
    _fit_band_histograms,
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


@pytest.mark.parametrize("counts_per_pixel", [0.1, 1.0])
def test_center_error_matches_scatter(counts_per_pixel):
    """The fit's centre error must be the real scatter, which needs sigma (not variance) as DataE."""
    # Arrange - independent Poisson strips of one pseudo-Voigt peak, as a band's pixels see it
    rng = np.random.default_rng(1)
    true_centre, fwhm, n_sets, n_pixels = 80.0, 0.3, 40, 20000
    edges = np.arange(78.0, 82.0 + 1e-9, 0.03)
    pixel_sets = []
    for _ in range(n_sets):
        tth = rng.uniform(edges[0], edges[-1], n_pixels)
        offset = (tth - true_centre) / (fwhm / 2)
        shape = 0.5 / (1 + offset**2) + 0.5 * np.exp(-np.log(2) * offset**2)
        counts = rng.poisson(counts_per_pixel * (0.2 + shape)).astype(float)
        pixel_sets.append((tth, counts, np.ones(n_pixels)))

    # Act
    _, _, centres, errors, _, excluded = _fit_band_histograms(
        "test_center_error", pixel_sets, edges, "PseudoVoigt", "Linear", wavelength=1.54
    )

    # Assert
    good = ~excluded & np.isfinite(errors)
    assert good.sum() >= 0.9 * n_sets
    ratio = np.median(errors[good]) / np.std(centres[good], ddof=1)
    assert 0.5 < ratio < 2.0


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
        ({"side_margin": -1}, "side_margin"),
        ({"min_window_side": 0.0}, "min_window_side"),
        ({"min_window_side": 3.0}, "min_window_side"),
        ({"min_significance": -1.0}, "min_significance"),
    ],
)
def test_invalid_arguments_raise(workspace, kwargs, match):
    with pytest.raises(ValueError, match=match):
        extract_ring_points(workspace, **{"wavelength": NOMINAL_WAVELENGTH, **kwargs})


# ---------------------------------------------------------------- masks


def _first_band_mask():
    """Mask (1 = keep) hiding the first of 8 bands, rows 40-157"""
    mask = np.ones(1024 * 1024)
    mask.reshape(1024, 1024)[:, 40:158] = 0.0
    return mask


@pytest.fixture()
def masked_workspace(hb2b):
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, ARMS[:1], POWDERS[:1], poisson=False)
    workspace.set_detector_mask(_first_band_mask(), is_default=True)
    return workspace


def test_default_mask_applied(masked_workspace):
    # Act
    points = extract_ring_points(masked_workspace, NOMINAL_WAVELENGTH, strips=STRIPS)

    # Assert
    assert len(points) == STRIPS - 1
    assert points.index_1.min() > 158


def test_default_mask_ignored_when_disabled(masked_workspace):
    # Act
    points = extract_ring_points(masked_workspace, NOMINAL_WAVELENGTH, strips=STRIPS, apply_default_mask=False)

    # Assert
    assert len(points) == STRIPS


def test_mask_array_overrides_default_mask(masked_workspace):
    # Act - the explicit mask hides nothing, so it must replace (not combine with) the default mask
    points = extract_ring_points(masked_workspace, NOMINAL_WAVELENGTH, strips=STRIPS, mask=np.ones(1024 * 1024))

    # Assert
    assert len(points) == STRIPS


# ---------------------------------------------------------------- side-edge margin and peak significance


def _single_ring_extraction(hb2b, arm, **kwargs):
    """Extract Ni 220 from one synthetic sub-run at the given arm angle; returns (points or None, diagnostics)"""
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, [arm], [NI])
    diagnostics = []
    try:
        points = extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, diagnostics=diagnostics, **kwargs)
    except ValueError:
        points = None
    return points, diagnostics


def _reasons(diagnostics):
    return {d.reason for d in diagnostics if not d.accepted}


def test_ring_near_edge_with_clipped_window_is_accurate(hb2b):
    """Arm 84: Ni 220 sits ~1 deg from the side edge, so its fit window is clipped on one side"""
    # Act
    points, diagnostics = _single_ring_extraction(hb2b, 84.0, side_margin=40)

    # Assert - all bands used, columns within the margin and unbiased by the one-sided window
    assert points is not None and len(points) == STRIPS
    assert np.all(points.index_0 >= 40)
    assert np.abs(points.index_0 - _true_columns(hb2b, points)).max() < 0.15
    # the window really is clipped (the full one reaches 2 deg on both sides of the expected peak)
    assert any(
        min(d.bragg_two_theta - d.two_theta.min(), d.two_theta.max() - d.bragg_two_theta) < 1.5 for d in diagnostics
    )


def test_ring_predicted_inside_margin_not_fitted(hb2b):
    # Act - Ni 220 is predicted at columns ~80-86 for arm 83.5
    points, diagnostics = _single_ring_extraction(hb2b, 83.5, side_margin=120)

    # Assert
    assert points is None
    assert _reasons(diagnostics) == {"ring predicted within 120 px of the side edge"}
    assert all(d.two_theta.size == 0 for d in diagnostics)  # never histogrammed or fitted


def test_fitted_ring_inside_margin_rejected(hb2b):
    """The window is placed from one located 2theta, but the true ring is tilted (Rot_z): the fitted
    point of each band is checked against the margin again"""
    # Arrange - arm 84 puts Ni 220 near the side edge; take a margin just inside every predicted column
    from pyrs.calibration.pixel_pos_calibration import _column_at_two_theta

    _, unrestricted = _single_ring_extraction(hb2b, 84.0, side_margin=0)
    model = DetectorModel(hb2b)
    predicted = [_column_at_two_theta(model, d.start_center, d.row, -84.0, 0.985, None) for d in unrestricted]
    margin = int(np.floor(min(predicted)))
    inside = sum(d.index_0 < margin for d in unrestricted)
    assert 0 < inside < STRIPS  # the tilt really puts some fitted points inside the margin

    # Act
    points, diagnostics = _single_ring_extraction(hb2b, 84.0, side_margin=margin)

    # Assert - those bands pass the prediction and are dropped after the fit
    assert len(points) == STRIPS - inside
    assert np.all(points.index_0 >= margin)
    assert _reasons(diagnostics) == {f"fitted ring within {margin} px of the side edge"}


def test_ring_located_despite_wrong_start_wavelength(hb2b):
    """Two-pass extraction: a starting wavelength of 1.572 A (true 1.5308) puts the expected ring ~1.4 deg
    (4-5 peak widths) from where it appears; the all-bands fit finds it and the band windows are centred
    on it"""
    # Arrange
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, [76.0], [NI])
    diagnostics = []

    # Act
    points = extract_ring_points(workspace, 1.572, strips=STRIPS, diagnostics=diagnostics)

    # Assert
    centers = np.array([d.center for d in diagnostics])
    start = diagnostics[0].start_center
    assert np.all(np.abs(diagnostics[0].bragg_two_theta - centers) > 1.0)  # the expectation really is far off
    # with the uncalibrated geometry the ring's apparent 2theta varies by band; the start is their mean
    assert start == pytest.approx(centers.mean(), abs=0.05)
    assert np.all(np.abs(centers - start) < 0.3)  # every band's fit starts within a peak width
    assert len(points) == STRIPS
    assert np.abs(points.index_0 - _true_columns(hb2b, points)).max() < 0.15


def test_window_too_clipped_rejected(hb2b):
    points, diagnostics = _single_ring_extraction(hb2b, 84.0, side_margin=40, min_window_side=1.5)
    assert points is None
    assert _reasons(diagnostics) == {"fit window clipped by the detector edge"}


def test_significance_check_rejects_invisible_ring(hb2b):
    """A fit to pure noise can report a small centre error (docs/ground_truths.md); the significance
    check, not the centre-error limit, is what rejects such fits"""
    # Arrange - an invisible ring on a high background
    workspace = synthetic_calibration_workspace(hb2b, TRUE_PARAMS, [76.0], [NI], peak_height=0.02, background=500.0)
    strict, lenient = [], []

    # Act
    with pytest.raises(ValueError, match="no ring points"):
        extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, diagnostics=strict)
    try:
        extract_ring_points(workspace, NOMINAL_WAVELENGTH, strips=STRIPS, min_significance=0.0, diagnostics=lenient)
    except ValueError:
        pass

    # Assert - with the check, the fits are refused as not significant (or as failed, where the fit has no
    # valid intensity uncertainty); without it, none is refused for significance
    strict_reasons = [d.reason.split(" (")[0] for d in strict]
    assert set(strict_reasons) <= {"peak not significant", "fit failed"}
    assert strict_reasons.count("peak not significant") > len(strict) / 2
    assert not any(r.startswith("peak not significant") for r in _reasons(lenient))
