"""Unit tests for the 2theta histogram of PyHB2BReduction: Poisson bin variance (#1109).

A bin's variance is the Poisson variance of its total count, sum(N_i). Zero-count pixels add nothing; the
former rule (sigma = 1 per empty pixel) added their number. See docs/ground_truths.md, "Poisson bin variance
in reduction".
"""

import numpy as np
import pytest

from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction

EDGES = np.array([0.0, 1.0, 2.0, 3.0])


def histogram(tth, counts, vanadium=None):
    """Point-data histogram on EDGES."""
    return PyHB2BReduction.histogram_by_numpy(
        np.asarray(tth, dtype=float),
        np.asarray(counts, dtype=float),
        EDGES,
        True,
        None if vanadium is None else np.asarray(vanadium, dtype=float),
    )


def test_bin_variance_is_sum_of_counts():
    """A bin holding counts 0, 0, 3 has sigma^2 = 3, not 3 + 2 empty pixels."""
    # Arrange: bin 0 has pixels with 0, 0, 3 counts; bin 1 has 4 and 1; bin 2 has 2
    tth = [0.2, 0.5, 0.8, 1.3, 1.7, 2.5]
    counts = [0, 0, 3, 4, 1, 2]

    # Act
    _, intensity, sigma = histogram(tth, counts)

    # Assert
    np.testing.assert_allclose(intensity, [3.0, 5.0, 2.0])
    np.testing.assert_allclose(sigma, np.sqrt([3.0, 5.0, 2.0]))


def test_empty_bin_keeps_unit_sigma():
    """A bin whose pixels all have zero counts gets intensity 0 and sigma 1."""
    _, intensity, sigma = histogram([0.5, 0.6, 1.5], [0, 0, 7])

    assert intensity[0] == 0.0 and sigma[0] == 1.0
    assert intensity[1] == 7.0 and sigma[1] == pytest.approx(np.sqrt(7.0))


def test_bin_with_no_pixels_keeps_unit_sigma():
    _, intensity, sigma = histogram([0.5], [4])

    assert intensity[2] == 0.0 and sigma[2] == 1.0


def test_zero_count_pixels_do_not_change_sigma():
    """Adding zero-count pixels to a bin leaves its sigma unchanged."""
    _, _, sigma_few = histogram([0.5, 0.6], [2, 3])
    _, _, sigma_many = histogram([0.5, 0.6] + [0.7] * 50, [2, 3] + [0] * 50)

    assert sigma_many[0] == pytest.approx(sigma_few[0])


def test_vanadium_normalised_sigma_uses_poisson_data_variance():
    """With vanadium, I = sum(N) * V_max / sum(V) and sigma propagates sum(N), sum(V) and V_max."""
    # Arrange: bin 0: counts 0, 0, 3 with vanadium 10, 10, 10; bin 1: counts 4, 1 with vanadium 20, 20;
    # bin 2: counts 2 with vanadium 5
    tth = [0.2, 0.5, 0.8, 1.3, 1.7, 2.5]
    counts = [0, 0, 3, 4, 1, 2]
    vanadium = [10, 10, 10, 20, 20, 5]
    sum_n = np.array([3.0, 5.0, 2.0])
    sum_v = np.array([30.0, 40.0, 5.0])
    v_max = 40.0

    # Act
    _, intensity, sigma = histogram(tth, counts, vanadium)

    # Assert
    expected = sum_n * v_max / sum_v
    np.testing.assert_allclose(intensity, expected)
    relative = np.sqrt(1.0 / sum_n + 1.0 / sum_v + 1.0 / v_max)
    np.testing.assert_allclose(sigma, expected * relative)


def test_vanadium_dead_pixels_are_excluded():
    """Pixels with vanadium < 0.9 add neither counts nor variance (unchanged behaviour)."""
    _, intensity, sigma = histogram([0.5, 0.6], [4, 9], vanadium=[10, 0.5])

    assert intensity[0] == pytest.approx(4.0 * 10 / 10)
    assert sigma[0] == pytest.approx(4.0 * np.sqrt(1 / 4 + 1 / 10 + 1 / 10))
