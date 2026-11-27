"""
Shared plot colors, colormaps, color-bar ticks and the correction color scale, as the reference script uses them.

``AXIS_COLORS`` and ``TEXT_COLORS`` mark the sample axes vx, vy and vz (arrows and labels). The other names color
the gauge, the incident and diffracted beams, Q, the sample, the phi dial, ignored cells and cell edges, and the
text drawn next to them. ``SEQUENTIAL`` is the colormap for measured intensities, ``PATH_CMAP``, ``CORR_CMAP`` and
``TIME_CMAP`` those of the beam path, the correction and the count time. Every correction panel uses
``correction_norm`` and ``clip_correction``, so the same correction has the same color in every figure.

Example:
    >>> from pyrs.pole_figures.plots.style import AXIS_COLORS, correction_norm
    >>> AXIS_COLORS["vx"]
    '#2a78d6'
    >>> correction_norm(3.0).vmax
    30.0
"""

import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize

from ..core.types import FloatArray

AXIS_COLORS = {"vx": "#2a78d6", "vy": "#eb6834", "vz": "#1baf7a"}
TEXT_COLORS = {"vx": "#1c5cab", "vy": "#b0461a", "vz": "#0f7a55"}
GAUGE = "#c0392b"
GAUGE_FILL = (0.75, 0.16, 0.13, 0.6)  # the part of the gauge inside the sample, shaded in the beam-path GIF
INCIDENT = "#4d4d4d"
DIFFRACTED = "#a0522d"
Q_COLOR = "#7b2fa8"
SAMPLE = "#c9a227"
GIF_SAMPLE = "#9db7d5"  # the sample in the beam-path GIF (light, so the beams and the gauge show through)
DIAL = "#e6e3dc"
IGNORED = "#c9c6bf"
SURFACE = "#fcfcfb"
CURRENT_POLE = "#00a5a5"  # the pole of the current GIF frame
CHI_AXIS = "#888888"
RING_ZERO = "#3987e5"  # the chi 0 ring of the pole-figure key
RING_OTHER = "#9dbbe3"  # the other chi rings of the pole-figure key

# text next to the beams, Q and the axes, a shade darker than the lines
INCIDENT_TEXT = "#333333"
DIFFRACTED_TEXT = "#8a4424"
Q_TEXT = "#6a2a91"
LABEL_TEXT = "#555555"
WARNING_TEXT = "#a32e2e"  # ignored points and the cutoff line

#: the reference script's colormap for measured intensities
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
PATH_CMAP = ListedColormap(colormaps["YlGnBu"](np.linspace(0.08, 1.0, 256)))
CORR_CMAP = ListedColormap(colormaps["magma_r"](np.linspace(0.0, 0.88, 256)))
TIME_CMAP = ListedColormap(colormaps["Blues"](np.linspace(0.18, 1.0, 256)))
#: line colors of the chi rings in the correction profile, in ring order (repeated after the last)
RING_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0", "#c9a227", "#d14d72", "#4a4a4a", "#17becf", "#7f7f7f", "k"]
PATH_TICKS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100)
CORR_TICKS = (1, 1.5, 2, 3, 5, 10, 20, 30)

#: cradle (e_x, e_y, e_z) -> plot (X, Y, Z) = (e_y, e_z, e_x), so the phi axis (e_x) is drawn up in the 3D views
CRADLE_TO_PLOT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
#: light direction (plot frame) of the shaded sample in the 3D views
LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])

CORR_MIN_TOP = 30.0  # the correction color scale reaches at least this
CORR_TOP_PER_CUTOFF = 10.0  # and at least this many times the cutoff


def norm_limits(norm: Normalize) -> tuple[float, float]:
    """
    The limits of a color scale.

    Args:
        norm: The color scale.

    Returns:
        vmin and vmax.

    Raises:
        ValueError: If a limit is not set (a scale that has not seen data yet).
    """
    if norm.vmin is None or norm.vmax is None:
        raise ValueError(f"the color scale needs limits, got vmin {norm.vmin}, vmax {norm.vmax}")
    return float(norm.vmin), float(norm.vmax)


def correction_norm(cutoff: float) -> LogNorm:
    """
    Log color scale of a correction panel: 1 to max(10 x cutoff, 30).

    Args:
        cutoff: Correction above which a point is ignored.

    Returns:
        The color scale.

    Example:
        >>> correction_norm(5.0).vmax
        50.0
    """
    return LogNorm(1.0, max(float(cutoff) * CORR_TOP_PER_CUTOFF, CORR_MIN_TOP))


def clip_correction(values: npt.ArrayLike, norm: LogNorm) -> FloatArray:
    """
    Corrections ready to color: non-finite values (the gauge misses the sample) at the top, all within the scale.

    Args:
        values: Corrections.
        norm: The color scale (``correction_norm``). Its limits must be set.

    Returns:
        The values clipped to the scale, with inf and NaN at its top.

    Raises:
        ValueError: If the scale has no limits.
    """
    bottom, top = norm_limits(norm)
    v = np.asarray(values, float)
    return np.asarray(np.clip(np.where(np.isfinite(v), v, top), bottom, top), float)
