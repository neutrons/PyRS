"""
Pole-figure plot helpers: cells, hatching, the rate color scale and interpolated panels.

Covers the vectorized cells, the default cell width, negative rates and repeated poles.
"""

from dataclasses import replace
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.grid import Grid
from pyrs.pole_figures.core.polefig import PolePeak
from pyrs.pole_figures.core.projection import ViewFrom
from pyrs.pole_figures.plots import polefig_cells
from pyrs.pole_figures.plots.polefig_cells import (
    DEFAULT_PHI_STEP_DEG,
    cell_outline,
    cells,
    draw_cells,
    pf_frame,
)
from pyrs.pole_figures.plots.polefig_interp import _pixel_grid, draw_interp
from pyrs.pole_figures.plots.polefigures import pole_figures_figure, rate_norm
from pyrs.pole_figures.plots.style import IGNORED, SEQUENTIAL
from tests.util.pole_figures_helpers import synthetic_pole_figure

FloatArray = npt.NDArray[np.float64]


def _mount() -> Mount:
    return Mount(MountConventions(incident_from="+vy"), omega=28.5, tth=57.0)


def _loop_cells(m: Mount, chi: FloatArray, phi: FloatArray) -> list[FloatArray]:
    """The per-point loop that ``cells`` replaced (kept here as the reference)."""
    uc = np.unique(np.round(chi, 1))
    dchi = float(np.min(np.diff(uc))) if len(uc) > 1 else 10.0
    out = []
    for c, p in zip(chi, phi, strict=True):
        same = np.abs(np.round(chi, 1) - round(c, 1)) < 0.05
        ps = np.sort(np.mod(phi[same], 360))
        dphi = float(np.median(np.diff(ps))) if same.sum() > 2 else 5.0
        if c > 90 - dchi / 2 - 1e-6 and same.sum() == 1:
            out.append(cell_outline(m, chi_lo=90 - dchi / 2, chi_hi=90, phi_lo=0, phi_hi=360, n=24))
        elif abs(c) < 0.05:
            out.append(cell_outline(m, chi_lo=0, chi_hi=dchi / 2, phi_lo=p - dphi / 2, phi_hi=p + dphi / 2))
        else:
            out.append(cell_outline(m, chi_lo=max(c - dchi / 2, 0), chi_hi=min(c + dchi / 2, 90), phi_lo=p - dphi / 2,
                                    phi_hi=p + dphi / 2))  # fmt: skip
    return out


def _ax(view: ViewFrom = "+vz") -> Axes:
    ax = Figure().add_subplot()
    pf_frame(ax, view=view)
    return ax


def test_cells_grid_equals_loop_reference() -> None:
    # Arrange: a measured-style grid (chi 0 ring, a cap point at chi 90) and a ring of two points
    chi, phi = Grid(10, 15).points()
    chi, phi = np.r_[chi, 45.0, 45.0], np.r_[phi, 3.0, 7.0]
    m = _mount()

    # Act
    new = cells(m, chi, phi)

    # Assert
    old = _loop_cells(m, chi, phi)
    assert len(new) == len(old)
    assert all(np.array_equal(a, b) for a, b in zip(new, old, strict=True))


def test_cells_same_input_returns_read_only_outlines() -> None:
    # Arrange
    chi, phi = Grid(30, 30).points()
    m = _mount()

    # Act
    first, second = cells(m, chi, phi), cells(m, chi.copy(), phi.copy())

    # Assert: equal outlines, shared read-only through the cache
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not any(o.flags.writeable for o in first + second)


def test_cells_repeated_phi_get_default_width() -> None:
    # Arrange: a spatial map, every point at chi 0, phi 0
    chi, phi = np.zeros(5), np.zeros(5)

    # Act
    out = cells(_mount(), chi, phi)

    # Assert: the cell spans DEFAULT_PHI_STEP_DEG in phi, as an outline built with that width
    ref = cell_outline(
        _mount(), chi_lo=0, chi_hi=5.0, phi_lo=-DEFAULT_PHI_STEP_DEG / 2, phi_hi=DEFAULT_PHI_STEP_DEG / 2
    )
    assert np.allclose(out[0], ref) and np.ptp(out[0], axis=0).max() > 0.01


def _draw_ignored(ax: Axes, *, ignored_colored: bool) -> None:
    """Draw a 15 deg grid with every third cell ignored."""
    chi, phi = Grid(15, 15).points()
    ignored = np.arange(len(chi)) % 3 == 0
    draw_cells(ax, cells(_mount(), chi, phi), values=np.linspace(0, 100, len(chi)), ignored=ignored,
               norm=Normalize(0, 100), cmap=SEQUENTIAL, view="+vz", ignored_colored=ignored_colored)  # fmt: skip


def _hatch_rgba(coll: PolyCollection) -> tuple[float, ...]:
    """Hatch color of a collection (matplotlib >= 3.11 has get_hatchcolor, older keeps _hatch_color)."""
    getter = getattr(coll, "get_hatchcolor", None)
    rgba = getter()[0] if getter is not None else coll._hatch_color  # type: ignore[attr-defined,unused-ignore]
    return tuple(float(x) for x in rgba)


@pytest.mark.parametrize("ignored_colored", [False, True])
def test_draw_cells_ignored_leaves_rcparams_unchanged(ignored_colored: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: record rcParams whenever draw_cells makes a collection
    before = dict(matplotlib.rcParams)
    seen: list[dict[str, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> PolyCollection:
        seen.append(dict(matplotlib.rcParams))
        return PolyCollection(*args, **kwargs)

    monkeypatch.setattr(polefig_cells, "PolyCollection", spy)

    # Act
    _draw_ignored(_ax(), ignored_colored=ignored_colored)

    # Assert
    assert seen and all(s == before for s in seen)
    assert dict(matplotlib.rcParams) == before


@pytest.mark.parametrize(("ignored_colored", "color"), [(False, "0.55"), (True, "white")])
def test_draw_cells_ignored_hatch_has_intended_color(ignored_colored: bool, color: str) -> None:
    # Arrange
    ax = _ax()

    # Act
    _draw_ignored(ax, ignored_colored=ignored_colored)

    # Assert
    hatched = [c for c in ax.collections if isinstance(c, PolyCollection) and c.get_hatch()]
    assert len(hatched) == 1
    assert _hatch_rgba(hatched[0]) == pytest.approx(to_rgba(color))


def test_pole_figures_figure_negative_rates_use_data_range() -> None:
    # Arrange: net rates all below zero (background above the peak)
    r = synthetic_pole_figure(rate=-10.0)

    # Act
    fig = pole_figures_figure([r])

    # Assert: the shared color bar spans the negative data, not 0 to 1
    cbar = fig.axes[2]
    lo, hi = cbar.get_ylim()
    assert hi < 0 and lo < hi
    assert "data range" in cbar.get_ylabel()


def test_pole_figures_figure_empty_raises() -> None:
    # Arrange
    results: list[PolePeak] = []

    # Act / Assert
    with pytest.raises(ValueError, match="no pole figures"):
        pole_figures_figure(results)


def test_draw_interp_repeated_poles_same_image() -> None:
    # Arrange: the same pole figure, once with every pole given three times
    r = synthetic_pole_figure(chi_step=15.0, phi_step=15.0)
    poles, vals, ig = r.poles(), r.corrected, r.ignored
    norm = Normalize(0, 20)
    ax1, ax2 = _ax(), _ax()

    # Act
    draw_interp(ax1, poles, values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, view="+vz", n=101)
    draw_interp(ax2, np.vstack([poles] * 3), values=np.r_[vals, vals, vals], ignored=np.r_[ig, ig, ig], norm=norm,
                cmap=SEQUENTIAL, view="+vz", n=101)  # fmt: skip

    # Assert
    a, b = ax1.images[0].get_array(), ax2.images[0].get_array()
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_draw_interp_repeated_poles_are_averaged() -> None:
    # Arrange: one pole figure, then the same with a second copy of each pole at a different value
    r = synthetic_pole_figure(chi_step=15.0, phi_step=15.0)
    poles, ig = r.poles(), np.zeros(r.n_points, bool)
    norm = Normalize(0, 20)
    ax1, ax2 = _ax(), _ax()

    # Act: values 4 and 6 at each pole average to 5
    draw_interp(ax1, poles, values=np.full(r.n_points, 5.0), ignored=ig, norm=norm, cmap=SEQUENTIAL, view="+vz",
                n=61)  # fmt: skip
    n = r.n_points
    draw_interp(ax2, np.vstack([poles, poles]), values=np.r_[np.full(n, 4.0), np.full(n, 6.0)], ignored=np.r_[ig, ig],
                norm=norm, cmap=SEQUENTIAL, view="+vz", n=61)  # fmt: skip

    # Assert
    a, b = ax1.images[0].get_array(), ax2.images[0].get_array()
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


@pytest.mark.parametrize("view", ["+vz", "-vz", "+vx"])
def test_draw_interp_grid_reused_across_panels(view: ViewFrom) -> None:
    # Arrange
    r = synthetic_pole_figure()
    norm = Normalize(0, 20)
    ax1, ax2 = _ax(), _ax()
    n = {"+vz": 81, "-vz": 83, "+vx": 85}[view]  # a size no other test uses

    # Act: two panels of one figure
    for ax in (ax1, ax2):
        draw_interp(ax, r.poles(), values=r.rate, ignored=r.ignored, norm=norm, cmap=SEQUENTIAL, view=view, n=n)

    # Assert: both panels drew the same image, and the shared pixel grid is read-only
    a, b = ax1.images[0].get_array(), ax2.images[0].get_array()
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not any(arr.flags.writeable for arr in _pixel_grid(n, view))


# ---------------------------------------------------------------- hatching
def _mid_grid() -> tuple[FloatArray, FloatArray]:
    """
    The chi rings 15 to 60 in 15 deg steps, phi every 30 deg.

    Seen from +vx (down the phi axis) the chi rings are circles about the center and no cell reaches the rim, so
    every cell is one polygon.
    """
    chi, phi = Grid(15, 30).points()
    keep = (chi > 10) & (chi < 65)
    return chi[keep], phi[keep]


def _hatched(ax: Axes) -> list[PolyCollection]:
    return [c for c in ax.collections if isinstance(c, PolyCollection) and c.get_hatch()]


def _kept(ax: Axes) -> list[PolyCollection]:
    return [c for c in ax.collections if isinstance(c, PolyCollection) and not c.get_hatch()]


@pytest.mark.parametrize("ignored_colored", [False, True])
def test_draw_cells_hatches_each_ignored_cell(ignored_colored: bool) -> None:
    # Arrange: every fourth cell ignored
    chi, phi = _mid_grid()
    values = np.linspace(10, 90, len(chi))
    ignored = np.arange(len(chi)) % 4 == 0
    norm = Normalize(0, 100)
    ax = _ax("+vx")

    # Act
    draw_cells(ax, cells(_mount(), chi, phi), values=values, ignored=ignored, norm=norm, cmap=SEQUENTIAL, view="+vx",
               ignored_colored=ignored_colored)  # fmt: skip

    # Assert
    (hatched,) = _hatched(ax)
    assert hatched.get_hatch() == "////"
    assert len(hatched.get_paths()) == ignored.sum()
    want = SEQUENTIAL(norm(values[ignored])) if ignored_colored else [to_rgba(IGNORED)] * int(ignored.sum())
    np.testing.assert_allclose(np.array(hatched.get_facecolor(), dtype=float), want)
    (kept,) = _kept(ax)
    assert len(kept.get_paths()) == (~ignored).sum()


def test_draw_cells_nothing_ignored_no_hatch() -> None:
    # Arrange
    chi, phi = _mid_grid()
    ax = _ax("+vx")

    # Act
    draw_cells(ax, cells(_mount(), chi, phi), values=np.full(len(chi), 50.0), ignored=np.zeros(len(chi), bool),
               norm=Normalize(0, 100), cmap=SEQUENTIAL, view="+vx")  # fmt: skip

    # Assert
    assert _hatched(ax) == []
    assert len(_kept(ax)[0].get_paths()) == len(chi)


def test_draw_cells_nan_not_ignored_not_drawn() -> None:
    # Arrange: two cells without a value that are not ignored
    chi, phi = _mid_grid()
    values = np.full(len(chi), 50.0)
    values[[1, 5]] = np.nan
    ax = _ax("+vx")

    # Act
    draw_cells(ax, cells(_mount(), chi, phi), values=values, ignored=np.zeros(len(chi), bool), norm=Normalize(0, 100),
               cmap=SEQUENTIAL, view="+vx")  # fmt: skip

    # Assert
    assert _hatched(ax) == []
    assert len(_kept(ax)[0].get_paths()) == len(chi) - 2


def _mid_pole_figure() -> PolePeak:
    """The synthetic pole figure on ``_mid_grid``, with the corrections of the chi 45 ring above the cutoff."""
    r = synthetic_pole_figure(chi_step=15.0, phi_step=30.0)
    k = np.flatnonzero((r.chi > 10) & (r.chi < 65))
    g = r.gauge.take(k)
    corr = np.where(np.isclose(r.chi[k], 45.0), 4.0, 1.5)
    return replace(r, chi=r.chi[k], phi=r.phi[k], count_s=r.count_s[k], rate=r.rate[k], sigma=r.sigma[k], h=r.h[k],
                   gauge=replace(g, correction=corr))  # fmt: skip


def test_pole_figures_figure_hatches_only_corrected_panels() -> None:
    # Arrange
    r = _mid_pole_figure()

    # Act
    fig = pole_figures_figure([r], view="+vx")

    # Assert: uncorrected panel has no hatch; corrected and correction panels hatch each ignored cell
    uncorrected, corrected, correction = fig.axes[0], fig.axes[1], fig.axes[3]
    assert r.ignored.sum() == 12
    assert _hatched(uncorrected) == []
    assert [len(c.get_paths()) for c in _hatched(corrected)] == [int(r.ignored.sum())]
    assert [len(c.get_paths()) for c in _hatched(correction)] == [int(r.ignored.sum())]


# ---------------------------------------------------------------- color scale of the rates
@pytest.mark.parametrize(
    ("rates", "limits", "note"),
    [(np.linspace(1.0, 100.0, 100), (0.0, float(np.percentile(np.linspace(1.0, 100.0, 100), 99))), ""),
     (np.full(5, -3.0), (-4.0, -2.0), "data range"),
     (np.array([]), (0.0, 1.0), "no finite rates")],
)  # fmt: skip
def test_rate_norm_limits_and_note(rates: FloatArray, limits: tuple[float, float], note: str) -> None:
    # Arrange
    finite = rates[np.isfinite(rates)]

    # Act
    norm, got_note = rate_norm(finite)

    # Assert
    assert (norm.vmin, norm.vmax) == pytest.approx(limits)
    assert note in got_note and (note != "" or got_note == "")


# ---------------------------------------------------------------- interpolated pole figures
def test_pole_figures_figure_interpolated_images() -> None:
    # Arrange: corrections above the cutoff at high chi, so some points are ignored
    r = synthetic_pole_figure(chi_step=15.0, phi_step=15.0)
    assert r.ignored.any()

    # Act
    fig = pole_figures_figure([r], interpolated=True)

    # Assert: one image per pole-figure panel, the hatch contour on the corrected and correction panels
    panels = (fig.axes[0], fig.axes[1], fig.axes[3])
    assert [len(ax.images) for ax in panels] == [1, 1, 1]
    assert [any(isinstance(c, ContourSet) for c in ax.collections) for ax in panels] == [False, True, True]


def test_pole_figures_figure_interpolated_correction_finite() -> None:
    # Arrange: a 10 x 10 deg grid, so few pixels are more than 7 deg from a measured pole
    r = synthetic_pole_figure(chi_step=10.0, phi_step=10.0)

    # Act
    fig = pole_figures_figure([r], interpolated=True)

    # Assert: the correction panel (log interpolation) has finite colors inside the circle
    rgba = np.asarray(fig.axes[3].images[0].get_array())
    n = rgba.shape[0]
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= 1.0
    assert np.isfinite(rgba[inside]).all()
    assert (rgba[inside][:, 3] > 0).mean() > 0.9


def test_draw_interp_log_matches_geometric_mean() -> None:
    # Arrange: values 1 and 100 on alternate poles; log interpolation gives 10 halfway, linear 50.5
    r = synthetic_pole_figure(chi_step=15.0, phi_step=15.0)
    vals = np.where(np.arange(r.n_points) % 2 == 0, 1.0, 100.0)
    ig = np.zeros(r.n_points, bool)
    norm = Normalize(0, 100)
    ax_log, ax_lin = _ax(), _ax()

    # Act
    draw_interp(ax_log, r.poles(), values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, view="+vz", log=True, n=61)
    draw_interp(ax_lin, r.poles(), values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, view="+vz", n=61)

    # Assert: the log field is lighter on average (lower values are lighter in this colormap)
    a = np.asarray(ax_log.images[0].get_array())[..., :3].sum(axis=-1)
    b = np.asarray(ax_lin.images[0].get_array())[..., :3].sum(axis=-1)
    assert a.mean() > b.mean()
