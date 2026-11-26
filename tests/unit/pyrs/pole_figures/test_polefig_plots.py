"""Pole-figure plot helpers: vectorized cells, default cell width, negative rates, repeated poles."""

from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
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
from pyrs.pole_figures.plots.polefigures import pole_figures_figure
from pyrs.pole_figures.plots.style import SEQUENTIAL
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


def _ax() -> Axes:
    ax = Figure().add_subplot()
    pf_frame(ax, view="+vz")
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


def test_cells_same_input_returns_cached_outlines() -> None:
    # Arrange
    chi, phi = Grid(30, 30).points()
    m = _mount()

    # Act
    first, second = cells(m, chi, phi), cells(m, chi.copy(), phi.copy())

    # Assert
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert not first[0].flags.writeable


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
               norm=Normalize(0, 100), cmap=SEQUENTIAL, ignored_colored=ignored_colored)  # fmt: skip


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
    draw_interp(ax1, poles, values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, n=101)
    draw_interp(ax2, np.vstack([poles] * 3), values=np.r_[vals, vals, vals], ignored=np.r_[ig, ig, ig], norm=norm,
                cmap=SEQUENTIAL, n=101)  # fmt: skip

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
    draw_interp(ax1, poles, values=np.full(r.n_points, 5.0), ignored=ig, norm=norm, cmap=SEQUENTIAL, n=61)
    n = r.n_points
    draw_interp(ax2, np.vstack([poles, poles]), values=np.r_[np.full(n, 4.0), np.full(n, 6.0)], ignored=np.r_[ig, ig],
                norm=norm, cmap=SEQUENTIAL, n=61)  # fmt: skip

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
    hits = _pixel_grid.cache_info().hits
    for ax in (ax1, ax2):
        draw_interp(ax, r.poles(), values=r.rate, ignored=r.ignored, norm=norm, cmap=SEQUENTIAL, view=view, n=n)

    # Assert: the second panel took the grid from the cache and drew the same image
    assert _pixel_grid.cache_info().hits == hits + 1
    a, b = ax1.images[0].get_array(), ax2.images[0].get_array()
    assert np.array_equal(np.asarray(a), np.asarray(b))
