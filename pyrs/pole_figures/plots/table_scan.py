"""
Table-scan preview: the points kept for one plane, colored by count time. Left-out points are gray and hatched.

Typical workflow:

1. Predict the correction with ``core.prediction.predict``.
2. Plan the scan of a plane with ``core.table_scan`` (a ``PlaneScan``).
3. Draw it with ``table_scan_figure`` before writing the workbook (``io.table_scan``).

Example:
    >>> fig = table_scan_figure(pred, 0, scan=scan, view="+vz", vz_name="BD")
"""

import math

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from ..core.prediction import PlanePrediction, Prediction
from ..core.projection import ViewFrom
from ..core.table_scan import PlaneScan, phi_travel
from ..io.table_scan import format_duration
from .polefig_cells import cells, draw_cells, pf_frame
from .predicted import add_colorbar, setup_line

TIME_CMAP = mcolors.ListedColormap(colormaps["Blues"](np.linspace(0.18, 1.0, 256)))


def table_scan_figure(
    pred: Prediction,
    index: int,
    *,
    scan: PlaneScan,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    Pole figure of the planned table scan for one plane.

    Each predicted grid point is one cell, colored by its count time (s). Points whose predicted correction is above
    ``scan.cutoff`` (or not finite) are left out and drawn gray and hatched.

    Args:
        pred: Prediction the scan was planned from.
        index: Index of the plane in ``pred.planes``.
        scan: Table scan of that plane. Its points carry the logged chi, matched to the cradle-chi grid through
            ``scan.chi_offset_deg``.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    q = pred.planes[index]
    keep = np.isfinite(q.correction) & (q.correction <= scan.cutoff)
    by_index = {(round(p.chi + scan.chi_offset_deg, 6), round(p.phi, 6)): p.count_s for p in scan.points}
    times = np.array([by_index.get((round(c, 6), round(f, 6)), np.nan) for c, f in zip(q.chi, q.phi, strict=True)],
                     float)  # fmt: skip
    fig = Figure(figsize=(8.6, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 0.12, 0.045), wspace=0.04, left=0.03, right=0.9, top=0.84,
                          bottom=0.16)  # fmt: skip
    ax = fig.add_subplot(gs[0, 0])
    pf_frame(ax, view=view, vz_name=vz_name)
    lo = float(np.nanmin(times)) if keep.any() else 1.0
    hi = float(np.nanmax(times)) if keep.any() else 2.0
    norm = mcolors.Normalize(lo, max(hi, lo + 1))
    draw_cells(ax, cells(q.mount, q.chi, q.phi), values=times, ignored=~keep, norm=norm, cmap=TIME_CMAP, view=view)
    ticks = tuple(float(t) for t in np.unique(np.round(np.linspace(lo, max(hi, lo + 1), 6))))
    add_colorbar(
        fig, fig.add_subplot(gs[0, 2]), norm=norm, cmap=TIME_CMAP, ticks=ticks, label="Count time per point (s)"
    )
    ax.set_title(f"{{{q.plane.hkl}}} table scan: {len(scan.points)} of {scan.n_grid} points kept, "
                 f"{scan.n_skipped} left out (hatched)\ncount time {format_duration(scan.total_s)} "
                 f"(without motor moves), {_rule(scan)}\n"
                 f"order: {'phi alternates up and down by chi ring' if scan.serpentine else 'phi up on every ring'}, "
                 f"total phi travel {phi_travel(scan):,.0f} deg", fontsize=9,
                 loc="left")  # fmt: skip
    fig.suptitle(f"Planned {{{q.plane.hkl}}} pole figure at 2theta {q.plane.two_theta:.2f}, omega "
                 f"{q.plane.omega:.2f}\n" + setup_line(pred), fontsize=9.5, x=0.01, ha="left", y=0.98)  # fmt: skip
    fig.text(0.01, 0.015,
             f"Points with a predicted correction above {scan.cutoff:g} are left out of the table scan. "
             f"Count time: {_rule(scan)}.\nEqual-area projection, one cell per chi, phi point. "
             + _free_phi_note(q),
             fontsize=7.5, va="bottom", ha="left", color="0.25")  # fmt: skip
    return fig


def _rule(scan: PlaneScan) -> str:
    rule = f"{scan.base_s:g} s x correction per point" if scan.scaled else f"{scan.base_s:g} s per point"
    if math.isfinite(scan.max_s):
        rule += (
            f", at most {scan.max_s:g} s ({scan.n_capped} capped)" if scan.scaled else f", at most {scan.max_s:g} s"
        )
    return rule


def _free_phi_note(q: PlanePrediction) -> str:
    notes = []
    for i, (trial, paths) in sorted(q.phi_sweeps.items()):
        worst = f"{np.nanmax(paths):.1f} mm at worst, {len(trial)} phi tried"
        notes.append(f"chi {q.chi[i]:g}: phi axis along Q, phi {q.phi[i]:g} has the shortest path "
                     f"({np.nanmin(paths):.1f} mm, {worst}).")  # fmt: skip
    return " ".join(notes)
