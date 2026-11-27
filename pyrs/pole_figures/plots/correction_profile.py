"""
Predicted correction against phi for each chi ring (the reference script's geometry-check style).

The correction is split into its gauge-volume and attenuation factors (``core.correction.correction_parts``).
``draw_correction_profile`` draws into a given figure, ``correction_profile_figure`` makes a new one.

Example:
    >>> fig = correction_profile_figure(pred, 0, cutoff=3.0, parts=("total", "volume"))
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

from ..core.correction import PARTS, CorrectionPart, correction_parts
from ..core.prediction import Prediction
from .annotate import setup_line
from .style import RING_COLORS, WARNING_TEXT

#: size (in) of the correction profile figure
PROFILE_FIGSIZE = (12.0, 8.6)
#: line style and width of each part, and the marker used when a ring has one point
PART_STYLES: dict[CorrectionPart, tuple[str, float, str]] = {
    "total": ("o-", 1.3, "o"),
    "volume": ("--", 1.1, "s"),
    "attenuation": (":", 1.4, "^"),
}
PART_NAMES: dict[CorrectionPart, str] = {
    "total": "total (solid, points)",
    "volume": "gauge volume in sample (dashed)",
    "attenuation": "attenuation (dotted)",
}


def draw_correction_profile(
    fig: Figure,
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    parts: Sequence[CorrectionPart] = PARTS,
) -> None:
    """
    Correction against phi for each chi ring, with the chosen parts.

    Total is drawn as solid lines with the predicted points, the gauge-volume part dashed and the attenuation part
    dotted. The lower panel is the mean beam path in the sample (mm). The ring labels show the logged chi (the
    cradle chi minus the mount's ``chi_offset_deg``).

    Args:
        fig: Figure to draw into (cleared first). Its designed size is ``PROFILE_FIGSIZE``.
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Correction drawn as the ignore line (``pred.cutoff`` is the one the prediction was made with).
        parts: Parts to draw, from ``core.correction.PARTS`` (default: all three).

    Raises:
        ValueError: If a part is not one of ``PARTS``, or the plane has no grid point at chi 0, phi 0.
    """
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown correction parts {unknown}, expected some of {PARTS}")
    cmax = float(cutoff)
    q = pred.planes[index]
    chi_offset_deg = q.mount.conv.chi_offset_deg
    vals = correction_parts(q.gauge, chi=q.chi, phi=q.phi)
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=(1.6, 1), hspace=0.22, left=0.07, right=0.87, top=0.88, bottom=0.14)
    ax, axL = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
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
            fmt, lw, single = PART_STYLES[part]
            y = np.where(np.isfinite(vals[part][sel][o]), vals[part][sel][o], np.nan)
            ax.plot(p, y, single if sel.sum() == 1 else fmt, ms=2.8, lw=lw, color=col,
                    label=lab if part == parts[0] else None)  # fmt: skip
            shown.extend(y[np.isfinite(y)])
        axL.plot(p, q.gauge.mean_path[sel][o], "o-" if sel.sum() > 1 else "o", ms=2.5, lw=1, color=col, label=lab)
    ax.set_yscale("log")
    if shown:
        lo, hi = min(min(shown), 1.0), max(max(shown), cmax * 1.2)
        ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.axhline(cmax, color=WARNING_TEXT, lw=0.8, ls="--")
    ax.text(2, cmax * 1.05, f"ignore above {cmax:g}", fontsize=7, color=WARNING_TEXT)
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_title(f"{{{q.plane.hkl}}} predicted correction vs phi per chi ring: "
                 + ", ".join(PART_NAMES[p] for p in PARTS if p in parts), fontsize=9, loc="left")  # fmt: skip
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


def correction_profile_figure(
    pred: Prediction,
    index: int,
    *,
    cutoff: float,
    parts: Sequence[CorrectionPart] = PARTS,
) -> Figure:
    """
    A new correction profile figure (``PROFILE_FIGSIZE``): ``draw_correction_profile`` into a new ``Figure``.

    Args:
        pred: Result of ``core.prediction.predict``.
        index: Index of the plane in ``pred.planes``.
        cutoff: Correction drawn as the ignore line.
        parts: Parts to draw, from ``core.correction.PARTS`` (default: all three).

    Returns:
        The figure.

    Raises:
        ValueError: If a part is not one of ``PARTS``, or the plane has no grid point at chi 0, phi 0.
    """
    fig = Figure(figsize=PROFILE_FIGSIZE)
    draw_correction_profile(fig, pred, index, cutoff=cutoff, parts=parts)
    return fig
