"""
Predicted beam path and correction figures.

- ``predicted_figure``: all planes, path (top) and correction (bottom), as the reference script's
  predicted_correction_pole_figures.png.
- ``plane_figure``: one plane, path and correction side by side, larger.
- ``correction_profile_figure``: one plane, correction against phi per chi ring (the reference script's geometry-check
  style), split into its gauge-volume and attenuation parts.
- ``write_predicted_csv``: the reference script's predicted_correction.csv.

Typical workflow:

1. Predict the correction with ``core.prediction.predict`` (no measured data needed).
2. Draw the summary with ``predicted_figure``, or one plane with ``plane_figure`` and ``correction_profile_figure``.
3. Change ``cutoff`` to see which cells would be ignored. This needs no new prediction.

Example:
    >>> fig = predicted_figure(pred, cutoff=3.0, view="+vz", vz_name="BD")
    >>> write_predicted_csv("predicted_correction.csv", pred, cutoff=3.0)
"""

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ..core.prediction import PlanePrediction, Prediction
from ..core.projection import ViewFrom
from .polefig_cells import cells, draw_cells, pf_frame

PATH_CMAP = mcolors.ListedColormap(colormaps["YlGnBu"](np.linspace(0.08, 1.0, 256)))
CORR_CMAP = mcolors.ListedColormap(colormaps["magma_r"](np.linspace(0.0, 0.88, 256)))
RING_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0", "#c9a227", "#d14d72", "#4a4a4a", "#17becf", "#7f7f7f", "k"]
PATH_TICKS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100)
CORR_TICKS = (1, 1.5, 2, 3, 5, 10, 20, 30)

#: Parts of the correction that ``correction_profile_figure`` can show.
PARTS = ("total", "volume", "attenuation")


def add_colorbar(
    fig: Figure,
    cax: Axes,
    *,
    norm: mcolors.Normalize,
    cmap: mcolors.Colormap,
    ticks: tuple[float, ...],
    label: str,
    mark: float = math.inf,
) -> None:
    """
    Color bar with the given ticks (only those inside the color range), and an optional cutoff mark.

    Args:
        fig: Figure that holds the color bar.
        cax: Axes to draw the color bar in.
        norm: Color scale.
        cmap: Colormap.
        ticks: Candidate tick values.
        label: Color bar label.
        mark: Value marked by a dashed line labeled "ignore > mark". Infinity draws no mark (default: inf).
    """
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    vmin, vmax = float(norm.vmin or 0), float(norm.vmax or 1)
    shown = [t for t in ticks if vmin <= t <= vmax]
    cb.set_ticks(shown)
    cb.set_ticklabels([f"{t:g}" for t in shown])
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=8)
    if math.isfinite(mark):
        cb.ax.axhline(mark, color="white", lw=2.4)
        cb.ax.axhline(mark, color="black", lw=0.9, ls=(0, (3, 2)))
        cb.ax.text(-0.25, mark, f"ignore > {mark:g}", transform=cb.ax.get_yaxis_transform(), fontsize=7,
                   va="center", ha="right", clip_on=False,
                   bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})  # fmt: skip


def _norms(planes: Sequence[PlanePrediction], cmax: float) -> tuple[mcolors.LogNorm, mcolors.LogNorm]:
    """Shared path and correction color scales."""
    Ls = np.concatenate([q.gauge.mean_path[np.isfinite(q.gauge.mean_path)] for q in planes])
    if Ls.size == 0:
        Ls = np.array([1.0, 10.0])
    lo = max(1.0, float(np.floor(Ls.min())))
    lnorm = mcolors.LogNorm(lo, max(lo * 1.5, float(np.ceil(np.percentile(Ls, 99.5)))))
    return lnorm, mcolors.LogNorm(1, max(cmax * 10, 30))


def sphere_fraction(pred: Prediction, ignored: npt.NDArray[np.bool_]) -> float:
    """
    Solid-angle-weighted fraction of the prediction grid that is ignored.

    Args:
        pred: Prediction. The weights come from its grid and the chi of its first plane.
        ignored: True for each ignored grid point, in the order of the plane's points.

    Returns:
        The ignored fraction, from 0 to 1.
    """
    w = pred.grid.weights(pred.planes[0].chi)
    return float((w * ignored).sum() / w.sum())


def _draw_path(ax: Axes, q: PlanePrediction, *, cl: list, lnorm: mcolors.LogNorm, view: ViewFrom,
               vz_name: str) -> None:  # fmt: skip
    lvmin, lvmax = float(lnorm.vmin or 1), float(lnorm.vmax or 10)
    L = q.gauge.mean_path
    pf_frame(ax, view=view, vz_name=vz_name)
    draw_cells(ax, cl, values=np.clip(np.nan_to_num(L, nan=lvmax), lvmin, lvmax), ignored=np.zeros(len(L), bool),
               norm=lnorm, cmap=PATH_CMAP, view=view)  # fmt: skip
    rng = f"{np.nanmin(L):.1f} to {np.nanmax(L):.0f} mm" if np.isfinite(L).any() else "gauge misses the sample"
    ax.set_title(f"{{{q.plane.hkl}}}: 2theta {q.plane.two_theta:.2f}, omega {q.plane.omega:.2f}\n"
                 f"beam path in the sample: {rng}", fontsize=9, loc="left")  # fmt: skip


def _draw_corr(ax: Axes, pred: Prediction, *, q: PlanePrediction, cl: list, cnorm: mcolors.LogNorm, cmax: float,
               view: ViewFrom, vz_name: str) -> None:  # fmt: skip
    top = float(cnorm.vmax or 30)
    ign = q.correction > cmax
    pf_frame(ax, view=view, vz_name=vz_name)
    corr = np.clip(np.where(np.isfinite(q.correction), q.correction, top), 1, top)
    draw_cells(ax, cl, values=corr, ignored=ign, norm=cnorm, cmap=CORR_CMAP, view=view, ignored_colored=True)
    frac = sphere_fraction(pred, ign)
    ax.set_title(f"{{{q.plane.hkl}}} correction F(0,0) / F\nhatched (> {cmax:g}x, ignored): {100 * frac:.0f}% of the "
                 "sphere", fontsize=9, loc="left")  # fmt: skip


def setup_line(pred: Prediction) -> str:
    """
    One-line summary of the setup for figure titles: gauge center (mm), slits (mm) and mu (1/mm).

    Args:
        pred: Prediction.

    Returns:
        The summary line.
    """
    c, b = pred.center, pred.beams
    return (f"gauge at vx {c[0]:g}, vy {c[1]:g}, vz {c[2]:g} mm, {b.incident_width:g} x {b.incident_height:g} mm "
            f"incident, {b.diffracted_width:g} mm diffracted slit, mu {pred.mu:.4g} /mm")  # fmt: skip


def _notes(fig: Figure, pred: Prediction, *, view: ViewFrom, vz_name: str) -> None:
    g = pred.grid
    last = float(np.arange(0, 90, g.chi_step).max())
    view_txt = {"+vz": f"seen from +vz (down {vz_name}), +vx right, +vy up",
                "-vz": "seen from -vz, vx reversed (as the 2D maps)",
                "+vx": "seen from +vx (down the phi axis), +vy right, +vz up"}[view]  # fmt: skip
    off = pred.planes[0].mount.conv.chi_offset_deg if pred.planes else 0.0
    fig.text(0.01, 0.015,
             f"Equal-area projection {view_txt}. Grid: logged chi {0 - off:g} to {last - off:g} in "
             f"{g.chi_step:g} deg steps plus chi {90 - off:g}, " + (
                 f"phi every {g.phi_step:g} deg at chi 0 and coarser toward chi 90 (equal spacing on the sphere). "
                 if g.equal_spacing else f"phi every {g.phi_step:g} deg. ") +
             f"{pred.n_points:,} Monte Carlo points per orientation.\nPath = mean of (in + out) over the part of the "
             "gauge inside the sample. Correction = F(0,0) / F, with F the gauge volume in the sample weighted by "
             "exp(-mu path).\nHatched cells (correction above the cut) are ignored in the measured pole figures. "
             "Sphere fractions are solid-angle weighted.",
             fontsize=7.5, va="bottom", ha="left", color="0.25")  # fmt: skip


def predicted_figure(
    pred: Prediction,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    All planes. Top row: mean beam path in the sample. Bottom row: correction F(0,0) / F, ignored cells hatched.

    Args:
        pred: Result of ``core.prediction.predict``.
        cutoff: Hatch cells above this correction (``pred.cutoff`` is the one the prediction was made with).
            Changing it does not need a new prediction.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    cmax = float(cutoff)
    res = pred.planes
    n = len(res)
    fig = Figure(figsize=(4.2 * n + 1.0, 9.6))
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
    return fig


def plane_figure(
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    One plane: path and correction pole figures side by side (color scales shared with the summary).

    Args:
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Hatch cells above this correction (``pred.cutoff`` is the one the prediction was made with).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    cmax = float(cutoff)
    q = pred.planes[index]
    fig = Figure(figsize=(13.0, 6.6))
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
    return fig


def correction_parts(q: PlanePrediction) -> dict[str, npt.NDArray[np.float64]]:
    """
    The correction and its two factors (correction = volume x attenuation, exactly).

    - volume: V(0,0) / V, the sample volume in the gauge relative to chi 0, phi 0
    - attenuation: A(0,0) / A, with A = F / V the mean of exp(-mu path) over the sample part of the gauge

    Args:
        q: Predicted plane. The reference is its grid point at chi 0, phi 0.

    Returns:
        Arrays "total", "volume" and "attenuation", one value per grid point. The volume part is inf and the
        attenuation part NaN where the gauge misses the sample (V = 0).

    Raises:
        ValueError: If the plane has no grid point at chi 0, phi 0.
    """
    V, F = q.gauge.volume_in_sample, q.gauge.F
    at_zero = np.flatnonzero((np.abs(q.chi) < 1e-9) & (np.abs(q.phi) < 1e-9))
    if at_zero.size == 0:
        raise ValueError(f"plane {q.plane.hkl} has no grid point at chi 0, phi 0 ({len(q.chi)} points)")
    i0 = int(at_zero[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(V > 0, F / np.where(V > 0, V, 1), np.nan)
        vol = np.where(V > 0, V[i0] / np.where(V > 0, V, 1), np.inf)
        att = A[i0] / A
    return {"total": q.correction, "volume": vol, "attenuation": att}


def correction_profile_figure(
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    parts: Sequence[str] = PARTS,
) -> Figure:
    """
    Correction against phi for each chi ring (the reference script's geometry-check style), with the chosen parts.

    Total is drawn as solid lines with the predicted points, the gauge-volume part dashed and the attenuation part
    dotted. The lower panel is the mean beam path in the sample (mm).

    Args:
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Correction drawn as the ignore line (``pred.cutoff`` is the one the prediction was made with).
        parts: Parts to draw, from ``PARTS`` (default: all three). The ring labels show the logged chi (the cradle
            chi minus the mount's ``chi_offset_deg``).

    Returns:
        The figure.

    Raises:
        ValueError: If a part is not one of ``PARTS``, or the plane has no grid point at chi 0, phi 0.
    """
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown correction parts {unknown}, expected some of {PARTS}")
    cmax = float(cutoff)
    q = pred.planes[index]
    chi_offset_deg = q.mount.conv.chi_offset_deg
    vals = correction_parts(q)
    fig = Figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=(1.6, 1), hspace=0.22, left=0.07, right=0.87, top=0.88, bottom=0.14)
    ax, axL = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    styles = {"total": ("o-", 1.3), "volume": ("--", 1.1), "attenuation": (":", 1.4)}
    rings = np.unique(np.round(q.chi, 3))
    shown: list[float] = []
    for k, c in enumerate(rings):
        col = RING_COLORS[k % len(RING_COLORS)]
        sel = np.abs(q.chi - c) < 1e-6
        o = np.argsort(q.phi[sel])
        p = q.phi[sel][o]
        lab = f"chi {c - chi_offset_deg:g}"
        for part in PARTS:
            if part not in parts:
                continue
            fmt, lw = styles[part]
            y = np.where(np.isfinite(vals[part][sel][o]), vals[part][sel][o], np.nan)
            if sel.sum() == 1:
                fmt = {"o-": "o", "--": "s", ":": "^"}[fmt]
            ax.plot(p, y, fmt, ms=2.8, lw=lw, color=col, label=lab if part == parts[0] else None)
            shown.extend(y[np.isfinite(y)])
        axL.plot(p, q.gauge.mean_path[sel][o], "o-" if sel.sum() > 1 else "o", ms=2.5, lw=1, color=col, label=lab)
    ax.set_yscale("log")
    if shown:
        lo, hi = min(min(shown), 1.0), max(max(shown), cmax * 1.2)
        ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.axhline(cmax, color="#a32e2e", lw=0.8, ls="--")
    ax.text(2, cmax * 1.05, f"ignore above {cmax:g}", fontsize=7, color="#a32e2e")
    ax.axhline(1, color="0.5", lw=0.6)
    names = {"total": "total (solid, points)", "volume": "gauge volume in sample (dashed)",
             "attenuation": "attenuation (dotted)"}  # fmt: skip
    ax.set_title(f"{{{q.plane.hkl}}} predicted correction vs phi per chi ring: "
                 + ", ".join(names[p] for p in PARTS if p in parts), fontsize=9, loc="left")  # fmt: skip
    ax.set_ylabel("Correction factor (log)", fontsize=8)
    axL.set_ylabel("Mean beam path in + out (mm)", fontsize=8)
    axL.set_xlabel("phi (deg)", fontsize=8)
    axL.set_title("Mean beam path in the sample", fontsize=9, loc="left")
    for a in (ax, axL):
        a.set_xticks(range(0, 361, 30))
        a.set_xlim(-5, 360)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
    ax.legend(
        fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="chi ring", title_fontsize=8
    )
    fig.suptitle(f"Predicted {{{q.plane.hkl}}} correction at 2theta {q.plane.two_theta:.2f}, omega "
                 f"{q.plane.omega:.2f} (sample model only)\n" + setup_line(pred), fontsize=9.5, x=0.01, ha="left",
                 y=0.98)  # fmt: skip
    fig.text(0.01, 0.015,
             "Correction = F(0,0) / F = [V(0,0) / V] x [A(0,0) / A]. V: sample volume in the gauge. A = F / V: mean of "
             "exp(-mu path) over the sample part of the gauge.\nThe volume part is the geometry alone (the gauge "
             "leaving the sample). The attenuation part is the beam path. Values are the predicted grid points.",
             fontsize=7.5, va="bottom", ha="left", color="0.25")  # fmt: skip
    return fig


def write_predicted_csv(path: str | Path, pred: Prediction, *, cutoff: float) -> None:
    """
    Write the reference script's predicted_correction.csv: one row per plane and grid point.

    chi is the logged chi (the cradle chi minus the mount's ``chi_offset_deg``). Paths are in mm, volumes in mm^3.
    ``ignored`` is 1 where the correction is above the cutoff.

    Args:
        path: Output CSV file.
        pred: Result of ``core.prediction.predict``.
        cutoff: Correction above which a point is marked ignored (``pred.cutoff`` is the one the prediction was
            made with).
    """
    cmax = float(cutoff)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["hkl", "two_theta", "omega", "chi", "phi", "pole_vx", "pole_vy", "pole_vz", "mean_path_mm",
                     "sample_volume_in_gauge_mm3", "correction", "ignored"])  # fmt: skip
        for q in pred.planes:
            g, chi_offset_deg = q.gauge, q.mount.conv.chi_offset_deg
            for c, f, h, L, V, cc in zip(q.chi, q.phi, q.poles(), g.mean_path, g.volume_in_sample, q.correction,
                                         strict=True):  # fmt: skip
                wr.writerow([q.plane.hkl, round(q.plane.two_theta, 3), round(q.plane.omega, 3), c - chi_offset_deg,
                             f, *np.round(h, 5), round(L, 3) if np.isfinite(L) else "", round(V, 3),
                             round(cc, 4) if np.isfinite(cc) else "", int(cc > cmax)])  # fmt: skip
