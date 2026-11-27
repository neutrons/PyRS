"""
MTEX-style ODF plots drawn with matplotlib from :class:`core.odf.OdfResult` (the reference script's ODF plots).

Peterson et al. 2021 (ODF section, Fig. 5; recalculated pole figures, Fig. 6): filled contours on MTEX's WhiteJet
colormap from 0 to the color range, and the "white" versions with everything below 1 mrd (random) white. Plotkowski
et al. 2021 Fig. 5C/E: eight log-spaced bands with edges max^(k/6), k = -1 to 5. Pole figures are the upper hemisphere
in an equal-area projection seen from +vz with +vx to the right (or from +vx with +vy to the right), as the
reference script.

Typical workflow:

1. Fit the ODF with ``core.odf.compute`` (an ``OdfResult``).
2. Draw each plot with the ``*_figure`` functions. ``io.odf.write_plots`` does this for every plot.

Example:
    >>> fig = recalculated_figure(res, ["111", "200", "220"], n_levels=20, vz_label="BD", white_below_mrd=None)
    >>> fig = section_figure(res, phi2_deg=45.0, n_levels=20, white_below_mrd=1.0)
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull

from ..core.odf import OdfResult, PFGrid, equal_area
from ..core.types import FloatArray

AXIS_COLORS = {"vx": "#2a78d6", "vy": "#eb6834", "vz": "#1baf7a"}  # as the reference script
FONT = 11
#: Plotkowski 2021 Fig. 5 legend colors, low to high
PLOTKOWSKI_COLORS = np.array([[57, 83, 167], [13, 146, 109], [112, 192, 69], [186, 215, 52], [255, 222, 4],
                              [251, 148, 29], [241, 78, 35], [240, 31, 36]]) / 255.0  # fmt: skip


def white_jet(n: int = 256) -> FloatArray:
    """
    MTEX's WhiteJetColorMap: white at 0, then the jet colors.

    Args:
        n: Number of colors (default: 256).

    Returns:
        RGB colors from 0 to 1, low to high, shape (n, 3).
    """
    # deferred: pymtex is an optional dependency (installed separately, not in every environment), and its first
    # import costs seconds (numba compiles its kernels)
    import mtex

    return np.asarray(mtex.WhiteJetColorMap(n), float)


def white_below(top: float, *, below: float, n: int = 256) -> ListedColormap:
    """
    WhiteJet from ``below`` to ``top``, white under it (the script's white-background map, range 0 to top).

    Args:
        top: Top of the color range (mrd).
        below: Values under this (mrd) are white.
        n: Number of colors (default: 256).

    Returns:
        The colormap, for a 0 to ``top`` color scale.
    """
    n_white = max(1, round(n * below / top))
    wj = white_jet(round((n - n_white) / 0.95))
    wj = wj[len(wj) - (n - n_white) :]  # skip the first near-white entries so 1-1.1 mrd shows
    return ListedColormap(np.vstack([np.ones((n_white, 3)), wj]))


def _pf_xy(g: PFGrid) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The grid as 2D arrays (polar rows, azimuth columns) of x, y (equal-area, 1 at the rim) and f."""
    n_az = len(np.unique(g.azimuth))
    shape = (-1, n_az)
    r = equal_area(g.polar)
    x = (r * np.cos(np.radians(g.azimuth))).reshape(shape)
    y = (r * np.sin(np.radians(g.azimuth))).reshape(shape)
    return x, y, g.f.reshape(shape)


def _pf_frame(ax: Axes, labels: tuple[str, str, str], colors: tuple[str, str, str]) -> None:
    """Circle, the in-plane axis labels (right, up) outside it, and '(.) out' in the top-left corner."""
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=0.8, color="k"))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    out, right, up = labels
    c_out, c_right, c_up = colors
    ax.text(1.03, 0.0, right, color=c_right, ha="left", va="center", fontsize=FONT)
    ax.text(0.0, 1.03, up, color=c_up, ha="center", va="bottom", fontsize=FONT)
    ax.text(-1.05, 1.03, f"⊙ {out}", color=c_out, ha="left", va="bottom", fontsize=FONT)


def _view_labels(res: OdfResult, vz_label: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    out = res.view_axes[0]
    if out[0] > 0.5:  # seen down +vx: +vy right, +vz up
        return ("+vx", "+vy", vz_label), (AXIS_COLORS["vx"], AXIS_COLORS["vy"], AXIS_COLORS["vz"])
    return (vz_label, "+vx", "+vy"), (AXIS_COLORS["vz"], AXIS_COLORS["vx"], AXIS_COLORS["vy"])


def _contour_pf(ax: Axes, g: PFGrid, *, levels: Sequence[float], cmap: Colormap, norm: Normalize) -> None:
    x, y, f = _pf_xy(g)
    ax.contourf(x, y, np.clip(f, levels[0], None), levels=list(levels), cmap=cmap, norm=norm, extend="max")


def _colorbar(fig: Figure, axes: Sequence[Axes], *, cmap: Colormap, norm: Normalize, top: float) -> None:
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), shrink=0.8, pad=0.03)
    cb.ax.set_title("mrd", fontsize=FONT)
    cb.ax.set_ylim(0, top)


def recalculated_figure(res: OdfResult, hkls: Sequence[str], *, n_levels: int, vz_label: str,
                        white_below_mrd: float | None) -> Figure:  # fmt: skip
    """
    Recalculated pole figures (Peterson 2021 Fig. 6), one panel per hkl, on a shared 0 to ``res.cmax_pf`` scale.

    Args:
        res: ODF result.
        hkls: Reflections to draw. Each must be in ``res.pf_plot``.
        n_levels: Number of contour levels.
        vz_label: Label of +vz, for example "BD".
        white_below_mrd: Draw values below this (mrd) white. None uses the plain WhiteJet map.

    Returns:
        The figure.
    """
    top = res.cmax_pf
    if white_below_mrd is None:
        cmap: Colormap = ListedColormap(white_jet())
        levels: list[float] = np.linspace(0, top, n_levels + 1).tolist()
    else:
        cmap = white_below(top, below=white_below_mrd)
        levels = [0.0, *np.linspace(white_below_mrd, top, n_levels + 1).tolist()]
    norm = Normalize(0, top)
    labels, colors = _view_labels(res, vz_label)
    fig = Figure(figsize=(3.3 * len(hkls) + 1.2, 3.8))
    axes = fig.subplots(1, len(hkls), squeeze=False)[0]
    for ax, h in zip(axes, hkls, strict=True):
        _contour_pf(ax, res.pf_plot[h], levels=levels, cmap=cmap, norm=norm)
        _pf_frame(ax, labels, colors)
        ax.set_title(f"{{{h}}}", fontsize=FONT, pad=18)
    _colorbar(fig, axes, cmap=cmap, norm=norm, top=top)
    return fig


def measured_vs_recalculated_figure(res: OdfResult, *, n_levels: int, vz_label: str) -> Figure:
    """
    Measured points (scaled to the ODF) above the recalculated pole figure of each measured hkl, same color range.

    Measured poles on the lower hemisphere are shown by their antipodes. The measured values are in mrd when the
    scaling to the ODF worked, else in counts/s (the panel title says so).

    Args:
        res: ODF result.
        n_levels: Number of contour levels.
        vz_label: Label of +vz, for example "BD".

    Returns:
        The figure.
    """
    top = res.cmax_pf
    cmap = ListedColormap(white_jet())
    norm = Normalize(0, top)
    levels = list(np.linspace(0, top, n_levels + 1))
    labels, colors = _view_labels(res, vz_label)
    out, east, north = res.view_axes
    n = len(res.measured)
    fig = Figure(figsize=(3.3 * n + 1.2, 7.2))
    axes = fig.subplots(2, n, squeeze=False)
    for k, m in enumerate(res.measured):
        ax = axes[0, k]
        c, e, nn = m.pole @ out, m.pole @ east, m.pole @ north
        flip = c < 0  # antipodal: shown on the upper hemisphere
        c, e, nn = np.where(flip, -c, c), np.where(flip, -e, e), np.where(flip, -nn, nn)
        po = np.degrees(np.arccos(np.clip(c, -1, 1)))
        az = np.arctan2(nn, e)
        r = equal_area(po)
        ax.scatter(r * np.cos(az), r * np.sin(az), c=res.measured_scaled[k], cmap=cmap, norm=norm, s=9,
                   edgecolors="none")  # fmt: skip
        _pf_frame(ax, labels, colors)
        unit = "" if res.measured_in_mrd else " (counts/s)"
        ax.set_title(f"measured {m.label}{unit}", fontsize=FONT, pad=18)
        ax2 = axes[1, k]
        _contour_pf(ax2, res.pf_plot[m.hkl], levels=levels, cmap=cmap, norm=norm)
        _pf_frame(ax2, labels, colors)
        ax2.set_title(f"recalculated, RP error {res.rp_error[k]:.2f}", fontsize=FONT, pad=18)
    _colorbar(fig, list(axes.ravel()), cmap=cmap, norm=norm, top=top)
    return fig


def plotkowski_figure(res: OdfResult, hkls: Sequence[str], *, vz_label: str) -> Figure:
    """
    Pole figures on the eight-band log scale of Plotkowski et al. 2021 Fig. 5C/E, with the legend panel.

    The band edges are max^(k/6), k = -1 to 5, with max ``res.plotkowski_max`` (``res.cmax_pf`` when None). The
    figure is a 2 x 2 grid: the pole figures fill the first panels and the legend the last.

    Args:
        res: ODF result.
        hkls: Reflections to draw, normally "100", "110" and "111". Each must be in ``res.pf_plot``.
        vz_label: Label of +vz, for example "BD".

    Returns:
        The figure.
    """
    k_max = res.plotkowski_max if res.plotkowski_max is not None else res.cmax_pf
    edges = k_max ** (np.arange(-1, 6) / 6.0)  # band edges, as in the paper's legend
    bounds = [0.0, *edges, max(k_max, edges[-1]) * 10]
    cmap = ListedColormap(PLOTKOWSKI_COLORS)
    norm = BoundaryNorm(bounds, cmap.N)
    labels, colors = _view_labels(res, vz_label)
    fig = Figure(figsize=(8.0, 8.0))
    axes = fig.subplots(2, 2).ravel()
    for ax, h in zip(axes, hkls, strict=False):
        x, y, f = _pf_xy(res.pf_plot[h])
        ax.contourf(x, y, np.clip(f, 1e-3, None), levels=bounds, cmap=cmap, norm=norm)
        ax.plot([-1, 1], [0, 0], color="0.2", lw=0.6)
        ax.plot([0, 0], [-1, 1], color="0.2", lw=0.6)
        _pf_frame(ax, labels, colors)
        ax.set_title(f"{{{h}}}", fontsize=FONT, pad=18)
    lg = axes[-1]  # legend panel, as in the paper
    for ax in axes[len(hkls) : -1]:
        ax.axis("off")
    lg.cla()
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    texts = [f"max = {k_max:.3f}", *[f"{v:.3f}" for v in edges[::-1]]]
    for k in range(8):
        y0 = 0.9 - (k + 1) * 0.1
        lg.add_patch(Rectangle((0.25, y0), 0.12, 0.1, facecolor=PLOTKOWSKI_COLORS[7 - k], edgecolor="k"))
        lg.text(0.40, y0 + 0.1, texts[k], va="center", fontsize=FONT)
    lg.text(0.25, 0.97, "mrd", fontsize=FONT)
    return fig


def section_figure(res: OdfResult, *, phi2_deg: float, n_levels: int, white_below_mrd: float | None,
                   phi1_ticks: float = 45.0, Phi_ticks: float = 30.0) -> Figure:  # fmt: skip
    """
    ODF section at fixed phi2 (Bunge), Peterson 2021 Fig. 5: phi1 across (0 to 360 deg), Phi down (0 to 90 deg).

    Args:
        res: ODF result. The section is ``res.section_plot``.
        phi2_deg: phi2 of the section (deg), used for the title.
        n_levels: Number of contour levels.
        white_below_mrd: Draw values below this (mrd) white. None uses the plain WhiteJet map.
        phi1_ticks: Tick spacing of phi1 (deg) (default: 45.0).
        Phi_ticks: Tick spacing of Phi (deg) (default: 30.0).

    Returns:
        The figure.
    """
    top = res.cmax_odf
    if white_below_mrd is None:
        cmap: Colormap = ListedColormap(white_jet())
        levels: list[float] = np.linspace(0, top, n_levels + 1).tolist()
    else:
        cmap = white_below(top, below=white_below_mrd)
        levels = [0.0, *np.linspace(white_below_mrd, top, n_levels + 1).tolist()]
    norm = Normalize(0, top)
    p1, PP, f = res.section_plot
    fig = Figure(figsize=(9.0, 3.4))
    ax = fig.add_subplot()
    ax.contourf(p1, PP, np.clip(f, 0, None), levels=levels, cmap=cmap, norm=norm, extend="max")
    ax.set_xlim(0, 360)
    ax.set_ylim(90, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 360.1, phi1_ticks))
    ax.set_yticks(np.arange(0, 90.1, Phi_ticks))
    ax.set_xlabel("φ₁ (deg)", fontsize=FONT)
    ax.set_ylabel("Φ (deg)", fontsize=FONT)
    ax.set_title(f"φ₂ = {phi2_deg:g}°", fontsize=FONT)
    _colorbar(fig, [ax], cmap=cmap, norm=norm, top=top)
    return fig


def ipf_figure(res: OdfResult, names: dict[str, str], *, n_levels: int, cubic: bool) -> Figure:
    """
    Inverse pole figures, one per sample direction: crystal directions in the fundamental sector, equal-area.

    Args:
        res: ODF result.
        names: Title of each direction ("vx", "vy", "vz"). Directions not in it are titled by their key.
        n_levels: Number of contour levels.
        cubic: Label the [001], [101] and [111] corners of the cubic sector.

    Returns:
        The figure.
    """
    top = res.cmax_ipf
    cmap = ListedColormap(white_jet())
    norm = Normalize(0, top)
    levels = list(np.linspace(0, top, n_levels + 1))
    dirs = list(res.ipf)
    fig = Figure(figsize=(3.6 * len(dirs) + 1.2, 3.2))
    axes = fig.subplots(1, len(dirs), squeeze=False, gridspec_kw={"wspace": 0.35})[0]
    for ax, d in zip(axes, dirs, strict=True):
        xyz, f = res.ipf[d]
        po = np.degrees(np.arccos(np.clip(xyz[:, 2], -1, 1)))
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        r = equal_area(po)
        x, y = r * np.cos(az), r * np.sin(az)
        tri = Triangulation(x, y)
        ax.tricontourf(tri, np.clip(f, 0, None), levels=levels, cmap=cmap, norm=norm, extend="max")
        _sector_outline(ax, x, y)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(names.get(d, d), fontsize=FONT, pad=16 if cubic else 6)
        if cubic:
            corners = (((0, 0, 1), "[001]", "top"), ((1, 0, 1), "[101]", "top"), ((1, 1, 1), "[111]", "bottom"))
            for v, lab, va in corners:
                u = np.asarray(v, float) / np.linalg.norm(v)
                rr = equal_area(np.degrees(np.arccos(u[2])))
                a = np.arctan2(u[1], u[0])
                dy = -0.015 if va == "top" else 0.015
                ax.text(rr * np.cos(a), rr * np.sin(a) + dy, lab, ha="center", va=va, fontsize=FONT - 1)
    _colorbar(fig, axes, cmap=cmap, norm=norm, top=top)
    return fig


def _sector_outline(ax: Axes, x: FloatArray, y: FloatArray) -> None:
    if len(x) < 3:
        return
    hull = ConvexHull(np.column_stack([x, y]))
    v = np.append(hull.vertices, hull.vertices[0])
    ax.plot(x[v], y[v], color="k", lw=0.8)
