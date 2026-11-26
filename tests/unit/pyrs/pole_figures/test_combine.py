"""Combining runs that measured the same pole figure into one."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.combine import combine, combine_by_hkl
from pyrs.pole_figures.core.geometry import Mount
from pyrs.pole_figures.core.polefig import PolePeak
from tests.util.pole_figures_helpers import synthetic_pole_figure


def split(r: PolePeak, cut: int) -> tuple[PolePeak, PolePeak]:
    """Two runs that each hold part of one pole figure."""
    fields = ("chi", "phi", "count_s", "rate", "sigma", "h")
    a = replace(r, run="2001", gauge=r.gauge.take(np.arange(cut)), **{f: getattr(r, f)[:cut] for f in fields})
    b = replace(
        r, run="2002", gauge=r.gauge.take(np.arange(cut, r.n_points)), **{f: getattr(r, f)[cut:] for f in fields}
    )
    return a, b


@pytest.fixture
def whole_and_combined() -> tuple[PolePeak, PolePeak]:
    whole = synthetic_pole_figure()
    a, b = split(whole, 20)
    return whole, combine([a, b], reference=a.run)


def test_combine_split_scan_names_the_runs(whole_and_combined: tuple[PolePeak, PolePeak]) -> None:
    # Arrange
    whole, c = whole_and_combined

    # Act
    names = (c.run, c.source_runs, c.n_points)

    # Assert
    assert names == ("2001+2002", ("2001", "2002"), whole.n_points)


@pytest.mark.parametrize("field", ["rate", "correction", "ignored"])
def test_combine_split_scan_is_the_whole_scan(whole_and_combined: tuple[PolePeak, PolePeak], field: str) -> None:
    # Arrange
    whole, c = whole_and_combined

    # Act
    got, want = getattr(c, field), getattr(whole, field)

    # Assert
    np.testing.assert_array_equal(got, want)


def test_combine_split_scan_keeps_poles(whole_and_combined: tuple[PolePeak, PolePeak]) -> None:
    # Arrange
    whole, c = whole_and_combined

    # Act
    poles = c.poles()

    # Assert
    np.testing.assert_allclose(poles, whole.poles(), atol=1e-15)


def test_combine_without_scale_notes_it(whole_and_combined: tuple[PolePeak, PolePeak]) -> None:
    # Arrange
    _, c = whole_and_combined

    # Act
    notes = c.notes

    # Assert
    assert "no scaling" in notes


@pytest.fixture
def median_scaled() -> tuple[PolePeak, PolePeak]:
    a, b = split(synthetic_pole_figure(), 20)
    b = replace(b, rate=b.rate * 2.0, sigma=b.sigma * 2.0)  # a detector region twice as efficient
    return a, combine([a, b], scale="median", reference=a.run)


def test_combine_median_scale_matches_medians(median_scaled: tuple[PolePeak, PolePeak]) -> None:
    # Arrange
    a, c = median_scaled
    kept = ~c.ignored

    # Act
    ra, rb = c.rate[: a.n_points][kept[: a.n_points]], c.rate[a.n_points :][kept[a.n_points :]]

    # Assert
    assert np.median(rb * c.correction[a.n_points :][kept[a.n_points :]]) == pytest.approx(
        np.median(ra * c.correction[: a.n_points][kept[: a.n_points]]), rel=1e-9
    )


def test_combine_median_scale_removes_injected_factor() -> None:
    """Doubling one part's rates before a median-scaled combine gives the same result as not doubling them."""
    # Arrange
    a, b = split(synthetic_pole_figure(), 20)
    doubled = replace(b, rate=b.rate * 2.0, sigma=b.sigma * 2.0)
    plain = combine([a, b], scale="median", reference=a.run)

    # Act
    c = combine([a, doubled], scale="median", reference=a.run)

    # Assert
    np.testing.assert_allclose(c.rate, plain.rate, rtol=1e-12)
    np.testing.assert_allclose(c.sigma, plain.sigma, rtol=1e-12)


def test_combine_median_scale_notes_it(median_scaled: tuple[PolePeak, PolePeak]) -> None:
    # Arrange
    _, c = median_scaled

    # Act
    notes = c.notes

    # Assert
    assert "scaled by median" in notes


def test_combine_by_hkl_groups_same_reflection() -> None:
    # Arrange
    p = synthetic_pole_figure(run="1", hkl="200")
    q = synthetic_pole_figure(run="2", hkl="200", seed=1)
    s = synthetic_pole_figure(run="3", hkl="111")

    # Act
    out = combine_by_hkl([p, s, q])

    # Assert
    assert [r.label for r in out] == ["1+2 {200}", "3 {111}"]


def test_combine_different_reflections_raises() -> None:
    # Arrange
    p, s = synthetic_pole_figure(run="1", hkl="200"), synthetic_pole_figure(run="3", hkl="111")

    # Act / Assert
    with pytest.raises(ValueError, match="different reflections"):
        combine([p, s], reference=p.run)


def test_combine_single_run_raises() -> None:
    # Arrange
    p = synthetic_pole_figure(run="1", hkl="200")

    # Act / Assert
    with pytest.raises(ValueError, match="at least two"):
        combine([p], reference=p.run)


def test_combine_unknown_reference_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match="reference run 9999"):
        combine([a, b], scale="median", reference="9999")


def test_combine_different_cutoff_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match="different cutoff"):
        combine([a, b.with_cutoff(2.0)], reference=a.run)


def test_combine_median_without_positive_rates_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)
    dead = replace(b, rate=np.full_like(b.rate, -1.0))

    # Act / Assert
    with pytest.raises(ValueError, match="run 1002 has no kept points with a positive"):
        combine([a, dead], scale="median", reference=a.run)


def test_combine_unknown_scale_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match="scale must be one of none, median, got 'mean'"):
        combine([a, b], scale="mean", reference=a.run)  # type: ignore[arg-type]


def remounted(r: PolePeak, *, omega: float | None = None, **conventions: Any) -> PolePeak:
    """The pole figure with its mount changed: another omega (None keeps it) or other mount conventions."""
    conv = replace(r.mount.conv, **conventions)
    return replace(r, mount=Mount(conv, omega=r.omega if omega is None else omega, tth=r.mount.tth))


def test_combine_different_chi_offset_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match=r"chi_offset_deg \(run 1001: 0.0, run 1002: 2.5\)"):
        combine([a, remounted(b, chi_offset_deg=2.5)], reference=a.run)


def test_combine_different_conventions_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match="different mount conventions: phi_sense"):
        combine([a, remounted(b, phi_sense="counterclockwise_from_above")], reference=a.run)


def test_combine_different_omega_raises() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act / Assert
    with pytest.raises(ValueError, match="different omega .*run 1001: 28.5, run 1002: 30"):
        combine([a, remounted(b, omega=30.0)], reference=a.run)


def test_combine_omega_within_tolerance_kept() -> None:
    # Arrange
    a, b = synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "200", seed=1)

    # Act
    c = combine([a, remounted(b, omega=a.omega + 0.005)], reference=a.run)

    # Assert
    assert (c.omega, c.chi_offset_deg) == (a.omega, a.chi_offset_deg)
