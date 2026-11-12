"""Unit tests for the zero-count pixel statistics helpers (synthetic data only)."""

import numpy as np
import pytest

from scripts.pixelation_study.zero_counts import (
    angle_bin_index,
    dead_pixel_mask,
    dominant_period,
    expected_counts,
    smooth_along_axis,
    to_detector_image,
    zero_fraction_profile,
)


def test_to_detector_image_indexes_x_then_y():
    """Pixel ID = x * N + y lands at image[x, y]."""
    # Arrange
    ids = np.arange(16)

    # Act
    image = to_detector_image(ids, num_pixel_1d=4)

    # Assert
    assert image[2, 3] == 2 * 4 + 3
    assert image.shape == (4, 4)


def test_to_detector_image_wrong_size_raises():
    with pytest.raises(ValueError, match="Expected 16"):
        to_detector_image(np.zeros(15), num_pixel_1d=4)


def test_dead_pixel_mask_uses_reduction_threshold():
    """Below 0.9 counts is dead, matching histogram_by_numpy."""
    # Arrange
    vanadium = np.array([0.0, 0.89, 0.9, 5.0])

    # Act
    dead = dead_pixel_mask(vanadium)

    # Assert
    assert dead.tolist() == [True, True, False, False]


def test_dead_pixel_mask_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        dead_pixel_mask(np.array([1.0, -1.0]))


def test_angle_bin_index_separates_bins():
    # Arrange
    tth = np.array([[0.0, 0.04], [0.06, 0.06]])
    eta = np.array([[0.0, 0.0], [0.0, 3.0]])

    # Act
    bins = angle_bin_index(tth, eta, two_theta_step=0.05, eta_step=2.0)

    # Assert
    assert bins[0, 0] == bins[0, 1]
    assert len({bins[0, 0], bins[1, 0], bins[1, 1]}) == 3


def test_angle_bin_index_nonpositive_step_raises():
    with pytest.raises(ValueError, match="positive"):
        angle_bin_index(np.zeros(3), np.zeros(3), 0.0, 1.0)


def test_expected_counts_scales_by_efficiency():
    """Within one bin λ_i = (Σ N / Σ v) · v_i; invalid pixels are NaN."""
    # Arrange
    counts = np.array([2.0, 4.0, 100.0])
    efficiency = np.array([1.0, 2.0, 1.0])
    bins = np.zeros(3, dtype=np.int64)
    valid = np.array([True, True, False])

    # Act
    lam = expected_counts(counts, efficiency, bins, valid)

    # Assert
    np.testing.assert_allclose(lam[:2], [2.0, 4.0])
    assert np.isnan(lam[2])


def test_expected_counts_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share one shape"):
        expected_counts(np.zeros(3), np.zeros(2), np.zeros(3, dtype=np.int64), np.ones(3, dtype=bool))


def test_profile_matches_simulated_poisson():
    """Poisson counts with known λ give observed ≈ expected and z-scores of unit scatter."""
    # Arrange
    rng = np.random.default_rng(1)
    lam = np.tile(np.linspace(0.1, 2.0, 200)[:, None], (1, 400))
    counts = rng.poisson(lam).astype(float)
    valid = np.ones_like(lam, dtype=bool)

    # Act
    profile = zero_fraction_profile(counts, lam, valid, axis=1)

    # Assert
    assert abs(np.mean(profile.observed - profile.expected)) < 0.005
    assert np.max(np.abs(profile.z_score)) < 5
    assert 0.8 < np.sqrt(np.mean(profile.z_score**2)) < 1.2
    np.testing.assert_allclose(profile.count_ratio, 1.0, atol=0.25)


def test_profile_flags_excess_zeros():
    """Lines with extra dead-but-unmasked pixels show a large positive z-score."""
    # Arrange
    rng = np.random.default_rng(2)
    lam = np.full((50, 500), 2.0)
    counts = rng.poisson(lam).astype(float)
    counts[10, :100] = 0.0

    # Act
    profile = zero_fraction_profile(counts, lam, np.ones_like(lam, dtype=bool), axis=1)

    # Assert
    assert profile.z_score[10] > 10
    assert np.max(np.abs(np.delete(profile.z_score, 10))) < 5


def test_profile_bad_axis_raises():
    image = np.zeros((2, 2))
    with pytest.raises(ValueError, match="axis"):
        zero_fraction_profile(image, image, np.ones((2, 2), dtype=bool), axis=2)


def test_smooth_along_axis_ignores_dead_pixels():
    # Arrange
    values = np.full((5, 3), 4.0)
    values[2, 1] = 1000.0
    live = np.ones((5, 3), dtype=bool)
    live[2, 1] = False

    # Act
    smoothed = smooth_along_axis(values, live, width=3, axis=0)

    # Assert
    assert np.isnan(smoothed[2, 1])
    np.testing.assert_allclose(smoothed[live], 4.0)


def test_smooth_along_axis_zero_width_raises():
    with pytest.raises(ValueError, match="width"):
        smooth_along_axis(np.zeros((2, 2)), np.ones((2, 2), dtype=bool), 0, axis=0)


def test_smoothed_vanadium_removes_noise_bias():
    """Noisy per-pixel efficiency over-predicts zeros; smoothing it along x removes most of the bias."""
    # Arrange
    rng = np.random.default_rng(3)
    true_eff = np.tile(1.0 + 0.5 * np.sin(np.arange(300) * 2 * np.pi / 6.6), (300, 1))
    vanadium = rng.poisson(30.0 * true_eff).astype(float)
    counts = rng.poisson(3.0 * true_eff).astype(float)
    live = np.ones_like(counts, dtype=bool)
    bins = np.zeros(counts.shape, dtype=np.int64)

    def obs_over_exp(efficiency: np.ndarray) -> float:
        lam = expected_counts(counts, efficiency, bins, live)
        return float((counts == 0).sum() / np.exp(-lam).sum())

    # Act
    raw = obs_over_exp(vanadium)
    smoothed = obs_over_exp(smooth_along_axis(vanadium, live, 15, axis=0))

    # Assert
    assert raw < 0.95
    assert abs(smoothed - 1.0) < 0.03


def test_dominant_period_finds_wire_pitch():
    # Arrange
    signal = np.sin(np.arange(965) * 2 * np.pi / 6.6)

    # Act
    period, share = dominant_period(signal)

    # Assert
    assert period == pytest.approx(6.6, rel=0.02)
    assert share > 0.5


def test_dominant_period_too_short_raises():
    with pytest.raises(ValueError, match="8 finite"):
        dominant_period(np.ones(5))
