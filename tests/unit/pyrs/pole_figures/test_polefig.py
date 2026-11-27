"""Measured pole figures: ignored points, corrected rates and the per-point checks."""

from dataclasses import replace

import numpy as np
import pytest

from pyrs.pole_figures.core.polefig import PolePeak
from tests.util.pole_figures_helpers import synthetic_pole_figure


def test_nan_rate_is_ignored() -> None:
    # Arrange
    base = synthetic_pole_figure()
    rate = np.where(np.arange(len(base.rate)) == 0, np.nan, 1.0)

    # Act
    r = replace(base, rate=rate)

    # Assert
    assert r.ignored[0]  # a NaN rate is never a live point


# ---------------------------------------------------------------- corrected rates
@pytest.fixture
def corrected_peak() -> PolePeak:
    """Three points with corrections 1, 2.5 and inf (the gauge missed the sample)."""
    base = synthetic_pole_figure()
    keep = slice(0, 3)
    fields = ("chi", "phi", "count_s", "h")
    gauge = replace(base.gauge.take(keep), correction=np.array([1.0, 2.5, np.inf]))
    return replace(base, **{f: getattr(base, f)[keep] for f in fields}, rate=np.array([10.0, 20.0, 30.0]),
                   sigma=np.array([1.0, 2.0, 3.0]), gauge=gauge)  # fmt: skip


def test_corrected_is_rate_times_correction(corrected_peak: PolePeak) -> None:
    # Arrange
    pk = corrected_peak

    # Act
    corrected = pk.corrected

    # Assert
    np.testing.assert_array_equal(corrected, [10.0, 50.0, np.nan])


def test_corrected_sigma_is_sigma_times_correction(corrected_peak: PolePeak) -> None:
    # Arrange
    pk = corrected_peak

    # Act
    sigma = pk.corrected_sigma

    # Assert
    np.testing.assert_array_equal(sigma, [1.0, 5.0, np.nan])


def test_pole_peak_wrong_pole_shape_raises() -> None:
    # Arrange
    base = synthetic_pole_figure()

    message = rf"h must hold one pole vector per point, shape \({base.n_points}, 3\), got \(2, 3\)"

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        replace(base, h=base.h[:2])


def test_pole_peak_wrong_gauge_length_raises() -> None:
    # Arrange
    base = synthetic_pole_figure()

    # Act / Assert
    with pytest.raises(ValueError, match=rf"gauge must hold one result per point \({base.n_points}\), got 2"):
        replace(base, gauge=base.gauge.take(slice(0, 2)))
