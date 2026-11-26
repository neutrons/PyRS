"""
Measured pole figures: per run and peak, uncorrected, corrected and the correction.

Ported from the reference script's ``pole_figures``.

Per-point cells (one cell per chi, phi measurement) or interpolated (linear on the projection, the reference
script's pole_figures_interpolated.png). Ignored points (correction above the cutoff) are hatched.
``draw_pole_figures`` draws into a given figure, ``pole_figures_figure`` makes a new one.

Example:
    >>> fig = pole_figures_figure(results, interpolated=False, view="+vz", vz_name="BD")
    >>> save_figure(fig, "pole_figures.png", dpi=150)
"""

from collections.abc import Sequence

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ..core.peaks import Method
from ..core.polefig import PolePeak
from ..core.projection import ViewFrom
from .annotate import add_colorbar, view_caption
from .polefig_cells import cells, draw_cells, pf_frame
from .polefig_interp import draw_interp
from .style import CORR_CMAP, CORR_TICKS, SEQUENTIAL, clip_correction, correction_norm

#: the intensity method of a pole figure, as the figure notes describe it (every ``core.peaks.Method``)
METHOD_NOTES: dict[Method, str] = {
    "window": "net counts in a fixed window of +- 1.5 FWHM above a straight background",
    "fit": "area of a pseudo-Voigt plus linear background fit per point",
}
ROW_HEIGHT_IN = 5.4  # height of one pole-figure row (in)
MARGIN_IN = 1.2  # room for the notes under the rows (in)


def pole_figures_figsize(n_rows: int) -> tuple[float, float]:
    """
    Size (in) of the figure of ``n_rows`` pole figures.

    Args:
        n_rows: Number of pole figures (rows).

    Returns:
        Width and height.
    """
    return 17.0, ROW_HEIGHT_IN * n_rows + MARGIN_IN


def rate_norm(rates: npt.NDArray[np.float64]) -> tuple[mcolors.Normalize, str]:
    """
    Color scale of the net rates: 0 to their 99th percentile, or the data range when that percentile is not positive.

    Args:
        rates: Finite net rates (counts/s) of a row's uncorrected and corrected panels.

    Returns:
        The scale, and a note for the color-bar label ("" for the usual 0 to 99th percentile scale).

    Example:
        >>> "data range" in rate_norm(np.array([-3.0, -1.0]))[1]
        True
    """
    if not rates.size:
        return mcolors.Normalize(0, 1.0), "\nno finite rates"
    vmax = float(np.percentile(rates, 99))
    if vmax > 0:
        return mcolors.Normalize(0, vmax), ""
    lo, hi = float(np.percentile(rates, 1)), vmax
    if hi - lo <= 0:
        lo, hi = lo - 1.0, hi + 1.0
    return mcolors.Normalize(lo, hi), "\nnet rates not positive: data range"


def draw_pole_figures(
    fig: Figure,
    results: Sequence[PolePeak],
    *,
    interpolated: bool = False,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> None:
    """
    One row per pole figure: uncorrected, corrected (ignored hatched), and the correction factor.

    The uncorrected and corrected panels of a row share one color scale, 0 to the 99th percentile of the rates
    (counts/s), or their data range when that percentile is not positive (``rate_norm``). The correction panel uses a
    log scale from 1 to max(10 x cutoff, 30) (``style.correction_norm``). The intensity method and the receiving slit
    in the note are those of the first pole figure.

    Args:
        fig: Figure to draw into (cleared first). ``pole_figures_figsize`` gives its designed size.
        results: Pole figures, one row each. Must not be empty.
        interpolated: Draw linear interpolation on the projection instead of one cell per point (default: False).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Raises:
        ValueError: If ``results`` is empty or the view is unknown.
    """
    n = len(results)
    if n == 0:
        raise ValueError("no pole figures to draw (results is empty)")
    caption = view_caption(view, vz_name=vz_name)
    height = pole_figures_figsize(n)[1]
    fig.clear()
    gs = fig.add_gridspec(n, 6, width_ratios=(1, 1, 0.045, 0.32, 1, 0.045), wspace=0.16, hspace=0.3,
                          left=0.02, right=0.95, top=1 - 0.5 / height, bottom=1.0 / height)  # fmt: skip
    for i, r in enumerate(results):
        poles = r.poles()
        outl = cells(r.mount, r.chi, r.phi)
        cv = r.corrected
        good = ~r.ignored
        both = np.r_[r.rate, cv[good]]
        both = both[np.isfinite(both)]
        norm, scale_note = rate_norm(both)
        tag = f"Run {r.run}{' (live)' if r.live else ''}, {{{r.hkl}}}"
        rings = ", ".join(f"{c - r.chi_offset_deg:.0f}" for c in np.unique(np.round(r.chi, 0)))
        panels = ((r.rate, np.zeros(r.n_points, bool), "uncorrected", f"{r.n_points} points, chi rings {rings}"),
                  (cv, r.ignored, "corrected (path, gauge volume)",
                   f"hatched = ignored ({int(r.ignored.sum())} points, correction > {r.cutoff:g})"))  # fmt: skip
        for j, (vals, ig, lab, sub) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            pf_frame(ax, view=view, vz_name=vz_name)
            if interpolated:
                draw_interp(ax, poles, values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, view=view)
            else:
                draw_cells(ax, outl, values=vals, ignored=ig, norm=norm, cmap=SEQUENTIAL, view=view)
            ax.set_title(f"{tag}: {lab}\n{sub}", fontsize=9, loc="left")
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENTIAL), cax=fig.add_subplot(gs[i, 2]))
        cb.set_label(f"{{{r.hkl}}} net counts/s (corrected: chi 0, phi 0 scale){scale_note}", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[i, 4])
        pf_frame(ax, view=view, vz_name=vz_name)
        cn = correction_norm(r.cutoff)
        cvals = clip_correction(r.correction, cn)
        if interpolated:
            draw_interp(ax, poles, values=cvals, ignored=r.ignored, norm=cn, cmap=CORR_CMAP, view=view,
                        ignored_colored=True, log=True)  # fmt: skip
        else:
            draw_cells(ax, outl, values=cvals, ignored=r.ignored, norm=cn, cmap=CORR_CMAP, view=view,
                       ignored_colored=True)  # fmt: skip
        ax.set_title(f"Correction factor F(0,0) / F\n(attenuation and sample volume in the gauge), mu {r.mu:.4g} /mm",
                     fontsize=9, loc="left")  # fmt: skip
        add_colorbar(fig, fig.add_subplot(gs[i, 5]), norm=cn, cmap=CORR_CMAP, ticks=CORR_TICKS,
                     label="correction (log scale)", mark=r.cutoff)  # fmt: skip
    how = ("Linear interpolation on the projection between measured poles, each used with its antipode. Blank more "
           "than 7 deg from any measurement. Hatched where the nearest pole is ignored." if interpolated else
           "Each measurement is one cell. The chi 0 ring is a great circle: phi and phi + 180 are the same pole and "
           "fill the two halves of its band.")  # fmt: skip
    b = results[0].beams
    fig.text(0.01, 0.2 / height,
             f"Equal-area projection {caption}. {how}\nIntensity = {METHOD_NOTES[results[0].method]}, per second. "
             f"Correction = Monte Carlo gauge (incident slits from each run's logs, {b.diffracted_width:g} mm "
             "receiving slit) inside the sample with Beer-Lambert paths.",
             fontsize=7.5, va="bottom", ha="left", color="0.25")  # fmt: skip


def pole_figures_figure(
    results: Sequence[PolePeak],
    *,
    interpolated: bool = False,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    A new pole-figure figure (``pole_figures_figsize``): ``draw_pole_figures`` into a new ``Figure``.

    Args:
        results: Pole figures, one row each. Must not be empty.
        interpolated: Draw linear interpolation on the projection instead of one cell per point (default: False).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.

    Raises:
        ValueError: If ``results`` is empty or the view is unknown.
    """
    if not results:
        raise ValueError("no pole figures to draw (results is empty)")
    fig = Figure(figsize=pole_figures_figsize(len(results)))
    draw_pole_figures(fig, results, interpolated=interpolated, view=view, vz_name=vz_name)
    return fig
