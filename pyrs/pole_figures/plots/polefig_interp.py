"""
Interpolated pole figure (the reference script's ``draw_interp``): linear interpolation on the equal-area projection.

Each pole is used with its antipode (a pole figure is centrosymmetric), projected without folding, so the field is
continuous across the rim. Pixels whose nearest measured pole is ignored are gray and hatched (or keep their color
under the hatch with ``ignored_colored``). Pixels more than ``max_gap_deg`` from any measured pole stay blank, so
nothing is extrapolated.

Repeated poles (the same pole vector within POLE_TOL, for example points measured twice) are merged into one with
the mean value before the interpolation, so the triangulation does not see coincident points. The pixel grid of a
size and view is computed once and reused for every panel.

Use it in place of ``polefig_cells.draw_cells`` on an axes set up by ``polefig_cells.pf_frame``.

Example:
    >>> pf_frame(ax, view="+vz")
    >>> draw_interp(ax, r.poles(), values=r.corrected, ignored=r.ignored, norm=Normalize(0, 100), cmap=SEQUENTIAL)
"""

from functools import lru_cache

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.spatial import QhullError, cKDTree

from ..core.projection import ViewFrom, from_view, project
from ..core.types import BoolArray, FloatArray
from .style import IGNORED

POLE_TOL = 1e-6  # pole vectors closer than this (per component) are the same pole


def merge_repeated(poles: FloatArray, values: FloatArray, *, tol: float = POLE_TOL) -> tuple[FloatArray, FloatArray]:
    """
    Merge repeated poles into one with the mean of their values.

    Args:
        poles: Pole vectors, shape (n, 3).
        values: Value at each pole, shape (n,).
        tol: Poles whose components agree after rounding to this step are the same pole (default: POLE_TOL).

    Returns:
        The distinct poles (the first of each group, in order of first appearance) and their mean values. A group
        of equal values keeps that value exactly.

    Example:
        >>> merge_repeated(np.array([[0.0, 0, 1], [0, 0, 1]]), np.array([4.0, 6.0]))[1]
        array([5.])
    """
    key = np.round(poles / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    mean = np.bincount(inv, weights=values) / np.bincount(inv)
    lo, hi = np.full(len(first), np.inf), np.full(len(first), -np.inf)
    np.minimum.at(lo, inv, values)
    np.maximum.at(hi, inv, values)
    mean = np.where(lo == hi, lo, mean)  # no rounding change when the repeats agree
    order = np.argsort(first)
    return poles[first[order]], mean[order]


@lru_cache(maxsize=8)
def _pixel_grid(n: int, view: ViewFrom) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray]:
    """
    Pixel grid of an n x n image on the projection, computed once per size and view.

    Returns:
        X and Y (n, n), the pixels inside the unit circle (n, n), and their unit vectors in the sample frame (m, 3).
        The arrays are read-only (shared between panels and figures).
    """
    sgn = -1.0 if view == "-vz" else 1.0
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    inside = R <= 1.0
    a = 2 * np.arcsin(np.clip(R / np.sqrt(2), 0, 1))
    Rs = np.where(R > 0, R, 1)
    U = np.dstack([sgn * X / Rs * np.sin(a), Y / Rs * np.sin(a), sgn * np.cos(a)])[inside]
    U = from_view(U, view=view)  # display frame back to the sample frame of the poles
    for arr in (X, Y, inside, U):
        arr.flags.writeable = False
    return X, Y, inside, U


def draw_interp(
    ax: Axes,
    poles: FloatArray,
    *,
    values: FloatArray,
    ignored: npt.NDArray[np.bool_],
    norm: Normalize,
    cmap: Colormap,
    view: ViewFrom = "+vz",
    ignored_colored: bool = False,
    log: bool = False,
    n: int = 401,
    max_gap_deg: float = 7.0,
) -> None:
    """
    Draw the interpolated field of ``values`` measured at ``poles``, clipped to the unit circle.

    Nothing is drawn when fewer than three distinct poles are usable. Repeated poles are merged (``merge_repeated``).
    Linear interpolation falls back to the nearest pole when the poles cannot be triangulated.

    Args:
        ax: Axes to draw into (a ``pf_frame`` axes).
        poles: Unit pole vectors in the sample frame (vx, vy, vz), shape (n, 3).
        values: Value at each pole, shape (n,). Non-finite values are not used.
        ignored: True for ignored poles, shape (n,).
        norm: Color scale.
        cmap: Colormap.
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        ignored_colored: If True, ignored poles are interpolated and keep their color under a white hatch. If False,
            pixels nearest an ignored pole are gray with a gray hatch (default: False).
        log: Interpolate log(values), using only positive values (default: False).
        n: Pixels across the image (default: 401).
        max_gap_deg: Pixels farther than this angle (deg) from every measured pole stay blank (default: 7.0).
    """
    X, Y, inside, U = _pixel_grid(n, view)
    H = np.vstack([poles, -poles])
    v = np.r_[values, values].astype(float)
    ig = np.r_[ignored, ignored]
    use = np.isfinite(v) & ((v > 0) if log else True) & (~ig | ignored_colored)
    pix = np.c_[X[inside], Y[inside]]
    if use.sum() < 3:
        return
    hu, vv = merge_repeated(H[use], np.log(v[use]) if log else v[use])
    if len(hu) < 3:
        return
    xy = project(hu, view=view, fold=False)
    try:
        f = griddata(xy, vv, pix, method="linear")
    except (QhullError, ValueError):  # poles that cannot be triangulated (collinear): nearest pole instead
        f = griddata(xy, vv, pix, method="nearest")
    if log:
        f = np.exp(f)
    dist, k = cKDTree(H).query(U)
    far = 2 * np.degrees(np.arcsin(np.clip(dist / 2, 0, 1))) > max_gap_deg
    near_ig = ig[k]
    vmin = float(norm.vmin if norm.vmin is not None else 0.0)
    c = cmap(norm(np.where(np.isfinite(f), f, vmin)))
    c[~np.isfinite(f) | far, 3] = 0
    if not ignored_colored:
        c[near_ig & ~far] = mcolors.to_rgba(IGNORED)
    rgba = np.zeros((n, n, 4))
    rgba[inside] = c
    clip = Circle((0, 0), 1, transform=ax.transData)
    e = 1 + 1 / (n - 1)
    im = ax.imshow(rgba, extent=(-e, e, -e, e), origin="lower", interpolation="nearest", zorder=3)
    im.set_clip_path(clip)
    if near_ig.any():
        M = np.zeros((n, n))
        M[inside] = (near_ig & ~far).astype(float)
        cs = ax.contourf(X, Y, M, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=4)
        cs.set_edgecolor("white" if ignored_colored else "0.55")
        cs.set_linewidth(0)
        cs.set_clip_path(clip)
