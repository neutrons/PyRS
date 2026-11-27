"""
Predicted beam path and correction pole figures.

- ``draw_predicted`` / ``predicted_figure``: all planes, path (top) and correction (bottom), as the reference
  script's predicted_correction_pole_figures.png.
- ``draw_plane`` / ``plane_figure``: one plane, path and correction side by side, larger.

Each ``draw_*`` function draws into a given figure (cleared first, for a GUI canvas). Each ``*_figure`` function
makes a new figure of the designed size and draws into it. The correction against phi is in ``correction_profile``,
and the predicted CSV is written by ``io.predicted_csv``.

Typical workflow:

1. Predict the correction with ``core.prediction.predict`` (no measured data needed).
2. Draw the summary with ``predicted_figure``, or one plane with ``plane_figure``.
3. Change ``cutoff`` to see which cells would be ignored. This needs no new prediction.

Example:
    >>> fig = predicted_figure(pred, cutoff=3.0, view="+vz", vz_name="BD")
"""

from collections.abc import Sequence

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.prediction import PlanePrediction, Prediction
from ..core.projection import ViewFrom
from ..core.types import FloatArray
from .annotate import add_colorbar, setup_line, view_caption
from .polefig_cells import cells, draw_cells, pf_frame
from .style import CORR_CMAP, CORR_TICKS, PATH_CMAP, PATH_TICKS, clip_correction, correction_norm, norm_limits

#: size (in) of the plane figure
PLANE_FIGSIZE = (13.0, 6.6)


def predicted_figsize(n_planes: int) -> tuple[float, float]:
    """
    Size (in) of the summary figure of ``n_planes`` planes.

    Args:
        n_planes: Number of planes.

    Returns:
        Width and height.
    """
    return 4.2 * n_planes + 1.0, 9.6


def _norms(planes: Sequence[PlanePrediction], cmax: float) -> tuple[mcolors.LogNorm, mcolors.LogNorm]:
    """Shared path and correction color scales."""
    Ls = np.concatenate([q.gauge.mean_path[np.isfinite(q.gauge.mean_path)] for q in planes])
    if Ls.size == 0:
        Ls = np.array([1.0, 10.0])
    lo = max(1.0, float(np.floor(Ls.min())))
    lnorm = mcolors.LogNorm(lo, max(lo * 1.5, float(np.ceil(np.percentile(Ls, 99.5)))))
    return lnorm, correction_norm(cmax)


def _draw_path(ax: Axes, q: PlanePrediction, *, cl: list[FloatArray], lnorm: mcolors.LogNorm, view: ViewFrom,
               vz_name: str) -> None:  # fmt: skip
    lvmin, lvmax = norm_limits(lnorm)
    L = q.gauge.mean_path
    pf_frame(ax, view=view, vz_name=vz_name)
    draw_cells(ax, cl, values=np.clip(np.nan_to_num(L, nan=lvmax), lvmin, lvmax), ignored=np.zeros(len(L), bool),
               norm=lnorm, cmap=PATH_CMAP, view=view)  # fmt: skip
    rng = f"{np.nanmin(L):.1f} to {np.nanmax(L):.0f} mm" if np.isfinite(L).any() else "gauge misses the sample"
    ax.set_title(f"{{{q.plane.hkl}}}: 2theta {q.plane.two_theta:.2f}, omega {q.plane.omega:.2f}\n"
                 f"beam path in the sample: {rng}", fontsize=9, loc="left")  # fmt: skip


def _draw_corr(ax: Axes, pred: Prediction, *, q: PlanePrediction, cl: list[FloatArray], cnorm: mcolors.LogNorm,
               cmax: float, view: ViewFrom, vz_name: str) -> None:  # fmt: skip
    ign = q.correction > cmax
    pf_frame(ax, view=view, vz_name=vz_name)
    draw_cells(ax, cl, values=clip_correction(q.correction, cnorm), ignored=ign, norm=cnorm, cmap=CORR_CMAP,
               view=view, ignored_colored=True)  # fmt: skip
    frac = pred.grid.sphere_fraction(q.chi, ign)
    ax.set_title(f"{{{q.plane.hkl}}} correction F(0,0) / F\nhatched (> {cmax:g}x, ignored): {100 * frac:.0f}% of the "
                 "sphere", fontsize=9, loc="left")  # fmt: skip


def _notes(fig: Figure, pred: Prediction, *, view: ViewFrom, vz_name: str) -> None:
    g = pred.grid
    last = float(np.arange(0, 90, g.chi_step).max())
    off = pred.planes[0].mount.conv.chi_offset_deg if pred.planes else 0.0
    fig.text(0.01, 0.015,
             f"Equal-area projection {view_caption(view, vz_name=vz_name)}. Grid: logged chi {0 - off:g} to "
             f"{last - off:g} in {g.chi_step:g} deg steps plus chi {90 - off:g}, " + (
                 f"phi every {g.phi_step:g} deg at chi 0 and coarser toward chi 90 (equal spacing on the sphere). "
                 if g.equal_spacing else f"phi every {g.phi_step:g} deg. ") +
             f"{pred.n_points:,} Monte Carlo points per orientation.\nPath = mean of (in + out) over the part of the "
             "gauge inside the sample. Correction = F(0,0) / F, with F the gauge volume in the sample weighted by "
             "exp(-mu path).\nHatched cells (correction above the cut) are ignored in the measured pole figures. "
             "Sphere fractions are solid-angle weighted.",
             fontsize=7.5, va="bottom", ha="left", color="0.25")  # fmt: skip


def draw_predicted(
    fig: Figure,
    pred: Prediction,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> None:
    """
    All planes. Top row: mean beam path in the sample. Bottom row: correction F(0,0) / F, ignored cells hatched.

    Args:
        fig: Figure to draw into (cleared first). ``predicted_figsize`` gives its designed size.
        pred: Result of ``core.prediction.predict``.
        cutoff: Hatch cells above this correction (``pred.cutoff`` is the one the prediction was made with).
            Changing it does not need a new prediction.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
    """
    cmax = float(cutoff)
    res = pred.planes
    n = len(res)
    fig.clear()
    gs = fig.add_gridspec(2, n + 1, width_ratios=(1,) * n + (0.06,), wspace=0.12, hspace=0.32,
                          left=0.02, right=0.93, top=0.86, bottom=0.12)  # fmt: skip
    lnorm, cnorm = _norms(res, cmax)
    for k, q in enumerate(res):
        cl = cells(q.mount, q.chi, q.phi)
        _draw_path(fig.add_subplot(gs[0, k]), q, cl=cl, lnorm=lnorm, view=view, vz_name=vz_name)
        _draw_corr(fig.add_subplot(gs[1, k]), pred, q=q, cl=cl, cnorm=cnorm, cmax=cmax, view=view, vz_name=vz_name)
    add_colorbar(fig, fig.add_subplot(gs[0, n]), norm=lnorm, cmap=PATH_CMAP, ticks=PATH_TICKS,
                 label="Mean beam path in the sample, in + out (mm, log scale)")  # fmt: skip
    add_colorbar(fig, fig.add_subplot(gs[1, n]), norm=cnorm, cmap=CORR_CMAP, ticks=CORR_TICKS,
                 label="Predicted correction F(0,0) / F (log scale)", mark=cmax)  # fmt: skip
    fig.suptitle("Predicted beam path and correction per plane (sample model only), " + setup_line(pred),
                 fontsize=9.5, x=0.01, ha="left", y=0.97)  # fmt: skip
    _notes(fig, pred, view=view, vz_name=vz_name)


def predicted_figure(
    pred: Prediction,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    A new summary figure (``predicted_figsize``): ``draw_predicted`` into a new ``Figure``.

    Args:
        pred: Result of ``core.prediction.predict``.
        cutoff: Hatch cells above this correction.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    fig = Figure(figsize=predicted_figsize(len(pred.planes)))
    draw_predicted(fig, pred, cutoff=cutoff, view=view, vz_name=vz_name)
    return fig


def draw_plane(
    fig: Figure,
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> None:
    """
    One plane: path and correction pole figures side by side (color scales shared with the summary).

    Args:
        fig: Figure to draw into (cleared first). Its designed size is ``PLANE_FIGSIZE``.
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Hatch cells above this correction (``pred.cutoff`` is the one the prediction was made with).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
    """
    cmax = float(cutoff)
    q = pred.planes[index]
    fig.clear()
    gs = fig.add_gridspec(1, 7, width_ratios=(1, 0.14, 0.045, 0.3, 1, 0.14, 0.045), wspace=0.04,
                          left=0.02, right=0.94, top=0.84, bottom=0.17)  # fmt: skip
    lnorm, cnorm = _norms(pred.planes, cmax)
    cl = cells(q.mount, q.chi, q.phi)
    _draw_path(fig.add_subplot(gs[0, 0]), q, cl=cl, lnorm=lnorm, view=view, vz_name=vz_name)
    add_colorbar(fig, fig.add_subplot(gs[0, 2]), norm=lnorm, cmap=PATH_CMAP, ticks=PATH_TICKS,
                 label="Mean beam path, in + out (mm, log)")  # fmt: skip
    _draw_corr(fig.add_subplot(gs[0, 4]), pred, q=q, cl=cl, cnorm=cnorm, cmax=cmax, view=view, vz_name=vz_name)
    add_colorbar(fig, fig.add_subplot(gs[0, 6]), norm=cnorm, cmap=CORR_CMAP, ticks=CORR_TICKS,
                 label="Correction F(0,0) / F (log)", mark=cmax)  # fmt: skip
    fig.suptitle(f"Predicted {{{q.plane.hkl}}} beam path and correction (sample model only)\n" + setup_line(pred),
                 fontsize=9.5, x=0.01, ha="left", y=0.98)  # fmt: skip
    _notes(fig, pred, view=view, vz_name=vz_name)


def plane_figure(
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    A new plane figure (``PLANE_FIGSIZE``): ``draw_plane`` into a new ``Figure``.

    Args:
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Hatch cells above this correction.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    fig = Figure(figsize=PLANE_FIGSIZE)
    draw_plane(fig, pred, index, cutoff=cutoff, view=view, vz_name=vz_name)
    return fig
