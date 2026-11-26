"""
Captions and color bars shared by the figures: the color bar with a cutoff mark, the setup line, the view caption.

Example:
    >>> view_caption("+vz", vz_name="BD")
    'seen from +vz (down BD), +vx right, +vy up'
    >>> vz_label("BD")
    '+vz (BD)'
"""

import math

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from ..core.prediction import Prediction
from ..core.projection import ViewFrom, check_view
from .style import norm_limits


def add_colorbar(
    fig: Figure,
    cax: Axes,
    *,
    norm: Normalize,
    cmap: Colormap,
    ticks: tuple[float, ...],
    label: str,
    mark: float = math.inf,
) -> None:
    """
    Color bar with the given ticks (only those inside the color range), and an optional cutoff mark.

    Args:
        fig: Figure that holds the color bar.
        cax: Axes to draw the color bar in.
        norm: Color scale. Its limits must be set.
        cmap: Colormap.
        ticks: Candidate tick values.
        label: Color bar label.
        mark: Value marked by a dashed line labeled "ignore > mark". Infinity draws no mark (default: inf).

    Raises:
        ValueError: If the color scale has no limits.
    """
    vmin, vmax = norm_limits(norm)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
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


def vz_label(vz_name: str) -> str:
    """
    Label of the +vz axis: "+vz", or "+vz (name)" when +vz has a name such as "BD".

    Args:
        vz_name: Name of +vz ("vz" or "+vz" for none).

    Returns:
        The label.
    """
    return "+vz" if vz_name in ("vz", "+vz") else f"+vz ({vz_name})"


def view_caption(view: ViewFrom, *, vz_name: str) -> str:
    """
    Where a pole figure is seen from, for the notes under a figure.

    Args:
        view: Pole-figure view, "+vz", "-vz" or "+vx".
        vz_name: Name of +vz, for example "BD" (shown for the +vz view).

    Returns:
        The caption, for example "seen from +vz (down BD), +vx right, +vy up".

    Raises:
        ValueError: If the view is not one of "+vz", "-vz" and "+vx".
    """
    check_view(view)
    captions: dict[ViewFrom, str] = {
        "+vz": f"seen from +vz (down {vz_name}), +vx right, +vy up",
        "-vz": "seen from -vz, vx reversed (as the 2D maps)",
        "+vx": "seen from +vx (down the phi axis), +vy right, +vz up",
    }
    return captions[view]
