"""
Pole-figure cells: one polygon per chi, phi measurement on the equal-area projection.

Ported from the reference script's cells and draw_cells.

Typical workflow:

1. ``cells`` builds the outline of each measurement from the mount and the cradle chi, phi.
2. ``pf_frame`` draws the unit circle and the axis labels.
3. ``draw_cells`` fills the outlines with the values.

Example:
    >>> outlines = cells(r.mount, r.chi, r.phi)
    >>> pf_frame(ax, view="+vz", vz_name="BD")
    >>> draw_cells(ax, outlines, values=r.rate, ignored=r.ignored, norm=Normalize(0, 100), cmap=SEQUENTIAL,
    ...            view="+vz")
"""

import threading
from collections import OrderedDict
from collections.abc import Hashable, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Circle

from ..core.geometry import Mount
from ..core.projection import ViewFrom, project, to_view
from ..core.types import FloatArray
from .annotate import vz_label
from .style import AXIS_COLORS, IGNORED, SURFACE

_HATCHCOLOR_KW = hasattr(PolyCollection, "set_hatchcolor")  # matplotlib >= 3.11


DEFAULT_PHI_STEP_DEG = 5.0  # cell width in phi when the ring's phi step is 0 or cannot be found
DEFAULT_CHI_STEP_DEG = 10.0  # cell height in chi for a single chi ring
EDGE_POINTS = 8  # points along each edge of a cell outline
CACHED_POLE_FIGURES = 32  # cell outlines kept for this many pole figures

_cache: OrderedDict[Hashable, list[FloatArray]] = OrderedDict()
_cache_lock = threading.Lock()  # figures are built on a worker thread


def cell_outline(m: Mount, *, chi_lo: float, chi_hi: float, phi_lo: float, phi_hi: float, n: int = 8) -> FloatArray:
    """
    Outline of a chi, phi cell as pole vectors in the sample frame.

    Args:
        m: Mount that turns cradle chi, phi into poles.
        chi_lo: Lower cradle chi of the cell (deg).
        chi_hi: Upper cradle chi of the cell (deg).
        phi_lo: Lower phi of the cell (deg).
        phi_hi: Upper phi of the cell (deg).
        n: Points along each edge (default: 8).

    Returns:
        Unit pole vectors around the cell, shape (4 n, 3).
    """
    cs = np.linspace(chi_lo, chi_hi, n)
    ps = np.linspace(phi_lo, phi_hi, n)
    chi = np.r_[np.full(n, chi_lo), cs, np.full(n, chi_hi), cs[::-1]]
    phi = np.r_[ps, np.full(n, phi_hi), ps[::-1], np.full(n, phi_lo)]
    return m.poles(chi, phi)


def _phi_steps(ring: npt.NDArray[np.intp], counts: npt.NDArray[np.intp], phi: FloatArray) -> FloatArray:
    """Phi step of each chi ring: the median spacing of its sorted phi (mod 360), or DEFAULT_PHI_STEP_DEG."""
    steps = np.full(len(counts), DEFAULT_PHI_STEP_DEG)
    for k in np.flatnonzero(counts > 2):
        ps = np.sort(np.mod(phi[ring == k], 360))
        step = float(np.median(np.diff(ps)))
        if np.isfinite(step) and step > 0:  # repeated phi (a step of 0) would give invisible cells
            steps[k] = step
    return steps


def cells(m: Mount, chi: FloatArray, phi: FloatArray) -> list[FloatArray]:
    """
    Cell outline for each measurement.

    The chi step is the smallest spacing of the chi rings (DEFAULT_CHI_STEP_DEG, 10 deg, for a single ring). The phi
    step is the median spacing on the point's ring, or DEFAULT_PHI_STEP_DEG (5 deg) for two points or fewer or when
    that spacing is 0 (repeated phi). A single point near chi 90 gets the whole cap. chi 0 cells take only the half
    band on their side, since phi and phi + 180 are the same pole there. The outlines of a pole figure are cached.

    Args:
        m: Mount that turns cradle chi, phi into poles.
        chi: Cradle chi of each measurement (deg).
        phi: Phi of each measurement (deg).

    Returns:
        One outline per measurement (``cell_outline``), in the order of ``chi``. The arrays are read-only.
    """
    chi, phi = np.asarray(chi, float), np.asarray(phi, float)
    key = (m.s_chi, m.s_phi, np.asarray(m.q, float).tobytes(), np.asarray(m.M0, float).tobytes(), chi.tobytes(),
           phi.tobytes())  # fmt: skip
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return list(_cache[key])
    out = _cells(m, chi, phi)
    for o in out:
        o.flags.writeable = False  # shared between figures through the cache
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > CACHED_POLE_FIGURES:
            _cache.popitem(last=False)
    return list(out)


def _cells(m: Mount, chi: FloatArray, phi: FloatArray) -> list[FloatArray]:
    """``cells`` without the cache: every outline of the regular cells from one vectorized ``Mount.poles`` call."""
    n = len(chi)
    if n == 0:
        return []
    rc = np.round(chi, 1)
    uc, ring, counts = np.unique(rc, return_inverse=True, return_counts=True)
    ring = ring.ravel()
    dchi = float(np.min(np.diff(uc))) if len(uc) > 1 else DEFAULT_CHI_STEP_DEG
    dphi = _phi_steps(ring, counts, phi)[ring]
    cap = (chi > 90 - dchi / 2 - 1e-6) & (counts[ring] == 1)
    zero = ~cap & (np.abs(chi) < 0.05)
    reg = ~cap
    chi_lo = np.where(zero, 0.0, np.maximum(chi - dchi / 2, 0))[reg]
    chi_hi = np.where(zero, dchi / 2, np.minimum(chi + dchi / 2, 90))[reg]
    phi_lo, phi_hi = (phi - dphi / 2)[reg], (phi + dphi / 2)[reg]
    e = EDGE_POINTS
    cs = np.linspace(chi_lo, chi_hi, e, axis=1)
    ps = np.linspace(phi_lo, phi_hi, e, axis=1)
    full = np.ones(e)
    ch = np.hstack([chi_lo[:, None] * full, cs, chi_hi[:, None] * full, cs[:, ::-1]])
    ph = np.hstack([ps, phi_hi[:, None] * full, ps[:, ::-1], phi_lo[:, None] * full])
    poles = m.poles(ch, ph).reshape(len(chi_lo), 4 * e, 3)
    out: list[FloatArray] = [np.empty((0, 3))] * n
    for i, o in zip(np.flatnonzero(reg), poles, strict=True):
        out[i] = np.array(o)
    if cap.any():
        top = cell_outline(m, chi_lo=90 - dchi / 2, chi_hi=90, phi_lo=0, phi_hi=360, n=24)
        for i in np.flatnonzero(cap):
            out[i] = top.copy()
    return out


PF_LIMIT = 1.18  # a pole-figure axes spans -PF_LIMIT to PF_LIMIT (room for the axis labels outside the circle)


def pf_frame(ax: Axes, *, view: ViewFrom, vz_name: str = "vz", labels: bool = True) -> None:
    """
    Unit circle and axis labels for one pole figure.

    Args:
        ax: Axes to set up (equal aspect, no axis lines, limits -PF_LIMIT to PF_LIMIT).
        view: Pole-figure view, "+vz", "-vz" or "+vx".
        vz_name: Label of +vz, for example "BD" (default: "vz").
        labels: Draw the axis labels (default: True).
    """
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.9, ec="0.25", zorder=5))
    ax.set_xlim(-PF_LIMIT, PF_LIMIT)
    ax.set_ylim(-PF_LIMIT, PF_LIMIT)
    ax.set_aspect("equal")
    ax.axis("off")
    if not labels:
        return
    sgn = -1.0 if view == "-vz" else 1.0
    vzlab = vz_label(vz_name)
    if view == "+vx":
        labs = [(vzlab, (0.0, 1.09), AXIS_COLORS["vz"]), ("+vy", (1.03, 0.0), AXIS_COLORS["vy"]),
                ("vx\n(phi axis)", (0.0, 0.0), AXIS_COLORS["vx"])]  # fmt: skip
    else:
        labs = [("+vy", (0.0, 1.09), AXIS_COLORS["vy"]), ("+vx\n(phi axis)", (sgn * 1.03, 0.0), AXIS_COLORS["vx"]),
                (vz_name, (0.0, 0.0), AXIS_COLORS["vz"])]  # fmt: skip
    for txt, xy, col in labs:
        ha = "right" if xy[0] < -0.5 else "left" if xy[0] > 0.5 else "center"
        box = {"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.75} if xy == (0.0, 0.0) else None
        ax.text(xy[0], xy[1], txt, fontsize=7, color=col, ha=ha, va="center", zorder=7, bbox=box)


def draw_cells(
    ax: Axes,
    outlines: Sequence[FloatArray],
    *,
    values: FloatArray,
    ignored: npt.NDArray[np.bool_],
    norm: Normalize,
    cmap: Colormap,
    view: ViewFrom,
    ignored_colored: bool = False,
) -> None:
    """
    One polygon per measurement, with its antipode on the visible side.

    Ignored cells are gray and hatched, or keep their color under a hatch when ``ignored_colored`` is True. Cells
    that are not ignored and have a non-finite value are not drawn. The polygons go into three PolyCollections (kept,
    ignored colored, ignored gray): one artist each instead of one patch per cell, which builds and draws many times
    faster and looks the same.

    Args:
        ax: Axes to draw into (a ``pf_frame`` axes).
        outlines: Cell outlines as sample-frame pole vectors (``cells``).
        values: Value of each cell.
        ignored: True for ignored cells.
        norm: Color scale.
        cmap: Colormap.
        view: Pole-figure view, "+vz", "-vz" or "+vx".
        ignored_colored: Ignored cells keep their color under a white hatch instead of turning gray (default: False).
    """
    sgn_view = -1.0 if view == "-vz" else 1.0
    groups: dict[str, tuple[list[FloatArray], list[float]]] = {"keep": ([], []), "ign_col": ([], []), "ign": ([], [])}
    for o, v, ig in zip(outlines, values, ignored, strict=True):
        ctr = to_view(o.mean(axis=0), view=view)
        sgn = -1.0 if sgn_view * ctr[2] < -1e-9 else 1.0
        for s in (sgn, -sgn):
            if view == "+vx" and sgn_view * s * ctr[2] < -0.5:
                continue  # far-side copy: an exact antipode of the view axis would project to the center
            xy = project(s * o, view=view, fold=False)
            if np.all(np.hypot(*xy.T) > 1.0):
                continue
            if ig and ignored_colored and np.isfinite(v):
                key = "ign_col"
            elif ig:
                key = "ign"
            elif np.isfinite(v):
                key = "keep"
            else:
                continue
            groups[key][0].append(xy)
            groups[key][1].append(float(v))
    styles = {
        "keep": {"edgecolors": SURFACE, "linewidths": 0.25},
        "ign_col": {"edgecolors": "white", "linewidths": 0.25, "hatch": "////", "hatchcolor": "white"},
        "ign": {"edgecolors": "0.55", "linewidths": 0.2, "hatch": "////", "hatchcolor": "0.55"},
    }
    for key, (verts, vals) in groups.items():
        if not verts:
            continue
        fc = [IGNORED] * len(verts) if key == "ign" else cmap(norm(np.asarray(vals)))
        style = dict(styles[key])
        if not _HATCHCOLOR_KW:
            # matplotlib < 3.11 has no hatchcolor keyword. The hatch takes the edge color (the same as the wanted
            # hatch color) when the collection is made. rcParams are not touched: figures are built on a worker thread.
            style.pop("hatchcolor", None)
        coll = PolyCollection(verts, facecolors=fc, zorder=3, **style)  # type: ignore[arg-type]
        coll.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
        ax.add_collection(coll, autolim=False)
