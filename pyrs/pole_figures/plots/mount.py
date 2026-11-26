"""
Mount check: the sample in its frame, as mounted on the cradle at any chi/phi, and the pole-figure key.

Generalizes the reference script's ``sketch_stl`` (which draws chi 0, phi 0 only) so the user can step through chi
and phi and check the mount conventions against the instrument cameras.

Typical workflow:

1. Load the sample with ``core.stl_mesh`` and build a ``core.geometry.Mount`` for one omega and 2theta.
2. Call ``mount_figure`` for each logged chi, phi to check, or ``draw_mount`` to redraw into the same figure.

Example:
    >>> fig = mount_figure(mesh, center=(0.0, 0.0, 5.0), mount=mount, chi_logged=0.0, phi=90.0, view="+vz",
    ...                    vz_name="BD")
"""

from collections.abc import Callable
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..core.geometry import Mount
from ..core.projection import ViewFrom, project
from ..core.stl_mesh import SampleMesh
from ..core.types import FloatArray
from ..io.matrix_text import format_matrix
from .style import AXIS_COLORS, DIAL, DIFFRACTED, GAUGE, INCIDENT, Q_COLOR, SAMPLE, TEXT_COLORS

Transform = Callable[[FloatArray], FloatArray]

# cradle (e_x, e_y, e_z) -> plot (X, Y, Z) = (e_y, e_z, e_x): the phi axis (e_x) is drawn up
_MP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
_LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])
#: size (in) of a new mount figure
FIGSIZE = (16.0, 6.2)


def _mesh_poly(ax: Axes3D, mesh: SampleMesh, *, T: Transform, z: float, alpha: float = 0.6) -> None:
    P = T(mesh.triangles().reshape(-1, 3)).reshape(-1, 3, 3)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.55 + 0.45 * np.abs(n @ _LIGHT)
    base = np.array(mcolors.to_rgb(SAMPLE))
    pc = Poly3DCollection(
        list(P), facecolors=np.clip(base[None] * shade[:, None], 0, 1), edgecolors="none", alpha=alpha
    )
    pc.set_zorder(z)
    ax.add_collection3d(pc)


def _arrow(ax: Axes3D, start: npt.ArrayLike, *, vec: npt.ArrayLike, col: str, lw: float = 1.6, z: float = 20,
           ratio: float = 0.12) -> None:  # fmt: skip
    s, v = np.asarray(start, float), np.asarray(vec, float)
    q = ax.quiver(s[0], s[1], s[2], v[0], v[1], v[2], color=col, lw=lw, arrow_length_ratio=ratio)
    q.set_zorder(z)
    q.set_path_effects([pe.withStroke(linewidth=lw + 1.2, foreground="white")])


def _line(ax: Axes3D, P: FloatArray, *, col: str, lw: float, z: float, ls: str = "-") -> None:
    (ln,) = ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=lw, ls=ls, zorder=z)
    ln.set_path_effects([pe.withStroke(linewidth=lw + 2, foreground="white")])


def _text(ax: Axes3D, xyz: npt.ArrayLike, *, s: str, col: str, fs: float = 7.5) -> None:
    p = np.asarray(xyz, float)
    ax.text(p[0], p[1], p[2], s, color=col, fontsize=fs, zorder=60, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})  # fmt: skip


def _cube_limits(ax: Axes3D, center: FloatArray, half: float) -> None:
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def path_at_gauge(
    mesh: SampleMesh, center: FloatArray, *, mount: Mount, chi: float, phi: float
) -> tuple[float, float]:
    """
    Beam path in and out from the gauge center at cradle chi, phi, along the beam centerlines.

    Args:
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm).
        mount: Beams and conventions for one omega and 2theta.
        chi: Cradle chi (deg).
        phi: Phi (deg).

    Returns:
        Path in and path out (mm). Both are NaN when the center is outside the sample.
    """
    R = mount.R(chi, phi)
    ki, kf = R.T @ mount.ki, R.T @ mount.kf
    wind, L_in = mesh.trace(center[None], -ki)
    if wind[0] < 0.5:
        return float("nan"), float("nan")
    return float(L_in[0]), float(mesh.trace(center[None], kf)[1][0])


def mount_figure(
    mesh: SampleMesh,
    *,
    center: npt.ArrayLike,
    mount: Mount,
    chi_logged: float,
    phi: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> Figure:
    """
    A new mount figure (``FIGSIZE``): ``draw_mount`` into a new ``Figure``.

    Args:
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        mount: Beams and conventions for one omega and 2theta.
        chi_logged: Chi as logged on the goniometer (deg). The cradle chi adds the mount's ``chi_offset_deg``.
        phi: Phi (deg).
        view: Pole-figure view for panel (c), "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").

    Returns:
        The figure.
    """
    fig = Figure(figsize=FIGSIZE)
    draw_mount(fig, mesh, center=center, mount=mount, chi_logged=chi_logged, phi=phi, view=view, vz_name=vz_name)
    return fig


def draw_mount(
    fig: Figure,
    mesh: SampleMesh,
    *,
    center: npt.ArrayLike,
    mount: Mount,
    chi_logged: float,
    phi: float,
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
) -> None:
    """
    Three panels: (a) on the stage at chi 0, phi 0, (b) as mounted at chi, phi with the beams, (c) pole key.

    Panels (a) and (b) are drawn in the cradle frame with the phi axis up.

    Args:
        fig: Figure to draw into (cleared first).
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        mount: Beams and conventions for one omega and 2theta.
        chi_logged: Chi as logged on the goniometer (deg). The cradle chi adds the mount's ``chi_offset_deg``.
        phi: Phi (deg).
        view: Pole-figure view for panel (c), "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
    """
    c = np.asarray(center, float)
    conv = mount.conv
    ax0 = conv.axes_at_zero()
    chi = float(conv.cradle_chi(chi_logged))
    fig.clear()
    gs = fig.add_gridspec(
        1, 3, width_ratios=(1, 1.25, 0.95), wspace=0.02, left=0.01, right=0.99, top=0.86, bottom=0.03
    )
    span = float(np.max(mesh.size))
    t_ring = np.linspace(0, 2 * np.pi, 80)
    vzl = "+vz" if vz_name in ("vz", "+vz") else f"+vz ({vz_name})"

    # (a) on the stage at chi 0, phi 0: the sample turned by the sample-to-instrument matrix M0, gauge at the rotation
    # point, plot (X, Y, Z) = (e_y, e_z, e_x) so the phi axis is up, the phi dial under the sample
    ax = cast(Axes3D, fig.add_subplot(gs[0, 0], projection="3d"))
    ax.computed_zorder = False
    M0 = conv.M0

    def Ta(P: FloatArray) -> FloatArray:  # noqa: N802
        return np.asarray((np.asarray(P) - c) @ M0.T @ _MP)

    P0 = (mesh.triangles().reshape(-1, 3) - c) @ M0.T  # instrument frame at chi 0, phi 0
    reach0 = max(float(np.max(np.linalg.norm(P0, axis=1))), 5.0)
    ex, ey0 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    z0 = float(P0[:, 0].min()) - 0.03 * reach0
    ring = z0 * ex + 0.8 * reach0 * (np.cos(t_ring)[:, None] * ey0 + np.sin(t_ring)[:, None] * np.array([0, 0, 1.0]))
    dial = Poly3DCollection([np.asarray(ring @ _MP)], fc=DIAL, ec="0.55", lw=0.6, alpha=0.8)
    dial.set_zorder(1)
    ax.add_collection3d(dial)
    _mesh_poly(ax, mesh, T=Ta, z=3, alpha=0.5)
    ax.scatter(0, 0, 0, color=GAUGE, s=55, depthshade=False, zorder=40, edgecolors="white", linewidths=1.2)
    La = 1.25 * reach0
    _arrow(ax, np.asarray(z0 * ex @ _MP), vec=np.asarray((La - z0) * ex @ _MP), col=AXIS_COLORS["vx"], lw=1.4, z=31,
           ratio=0.06)  # fmt: skip
    _text(ax, np.asarray(1.05 * La * ex @ _MP), s="phi axis", col="#555555")
    _arrow(ax, [0, 0, 0], vec=np.asarray(0.95 * La * ey0 @ _MP), col="#888888", lw=1.2, z=31, ratio=0.08)
    _text(ax, np.asarray(1.02 * La * ey0 @ _MP), s="chi axis", col="#555555")
    _arrow(ax, [0, 0, 0], vec=np.asarray(0.95 * La * np.array([0, 0, 1.0]) @ _MP), col=Q_COLOR, lw=1.2, z=31,
           ratio=0.08)  # fmt: skip
    _text(ax, np.asarray(1.02 * La * np.array([0, 0, 1.0]) @ _MP), s="Q (omega = theta)", col="#6a2a91")
    # the sample's own axes, from a corner of its bounding box, turned as mounted
    o = Ta(mesh.lo[None])[0]
    L = 0.4 * span
    for k, (lab, key) in enumerate((("+vx", "vx"), ("+vy", "vy"), (vzl, "vz"))):
        v = np.asarray((M0[:, k] * L) @ _MP)
        _arrow(ax, o, vec=v, col=AXIS_COLORS[key], lw=2.0, ratio=0.25, z=32)
        _text(ax, o + 1.18 * v, s=lab, col=TEXT_COLORS[key])
    ax.view_init(elev=22, azim=-55)
    _cube_limits(ax, np.zeros(3), 0.75 * La)
    sz = mesh.size
    ax.set_title(
        f"(a) On the stage at chi 0, phi 0: {mesh.name}\n"
        f"{ax0.up} up (phi axis), {ax0.across} across, {ax0.q} along Q\n"
        f"matrix (rows phi, chi, Q): {format_matrix(M0)}\n"
        f"{sz[0]:.4g} x {sz[1]:.4g} x {sz[2]:.4g} mm (vx, vy, vz), {mesh.volume:.0f} mm3, "
        f"{'closed' if mesh.closed else 'NOT closed'}\ngauge (red) at {c[0]:g}, {c[1]:g}, {c[2]:g}",
        fontsize=8.5,
        loc="left",
    )

    # (b) as mounted at cradle chi, phi: gauge at the origin, plot (X, Y, Z) = (e_y, e_z, e_x)
    ax = cast(Axes3D, fig.add_subplot(gs[0, 1], projection="3d"))
    ax.computed_zorder = False
    R = mount.R(chi, phi)

    def Tb(P: FloatArray) -> FloatArray:  # noqa: N802
        return np.asarray((np.asarray(P) - c) @ R.T @ _MP)

    def Lv(v: npt.ArrayLike) -> FloatArray:  # noqa: N802
        return np.asarray(np.asarray(v, float) @ _MP)

    Pc = (mesh.triangles().reshape(-1, 3) - c) @ R.T  # cradle frame
    reach = float(np.max(np.linalg.norm(Pc, axis=1)))
    Lb = max(1.4 * reach, 10.0)
    a_phi = mount.phi_axis(chi)
    ey = np.array([0.0, 1.0, 0.0])
    zb = float((Pc @ a_phi).min()) - 0.03 * Lb
    t = np.linspace(0, 2 * np.pi, 80)
    w = np.cross(a_phi, ey)
    dial = zb * a_phi + 0.75 * Lb * (np.cos(t)[:, None] * ey + np.sin(t)[:, None] * w)
    pc = Poly3DCollection([Lv(dial)], fc=DIAL, ec="0.55", lw=0.6, alpha=0.8)
    pc.set_zorder(1)
    ax.add_collection3d(pc)
    _mesh_poly(ax, mesh, T=Tb, z=3, alpha=0.6)

    ki, kf, q = Lv(mount.ki), Lv(mount.kf), Lv(mount.q)
    _arrow(ax, -1.15 * Lb * ki, vec=1.05 * Lb * ki, col=INCIDENT, lw=2.6, z=30, ratio=0.06)
    _text(ax, -1.2 * Lb * ki, s="incident", col="#333333")
    _arrow(ax, 0.05 * Lb * kf, vec=1.05 * Lb * kf, col=DIFFRACTED, lw=2.6, z=30, ratio=0.07)
    _text(ax, 1.18 * Lb * kf, s="to detector", col="#8a4424")
    _arrow(ax, [0, 0, 0], vec=0.45 * Lb * q, col=Q_COLOR, lw=2.4, z=31, ratio=0.2)
    _text(ax, 0.55 * Lb * q, s="Q", col="#6a2a91")
    _arrow(ax, Lv(zb * a_phi), vec=Lv((0.9 * Lb - zb) * a_phi), col=AXIS_COLORS["vx"], lw=1.8, z=31, ratio=0.06)
    _text(ax, Lv(0.95 * Lb * a_phi), s="phi axis", col=TEXT_COLORS["vx"])
    _arrow(ax, [0, 0, 0], vec=0.8 * Lb * Lv(ey), col=AXIS_COLORS["vy"], lw=1.8, z=31, ratio=0.08)
    _text(ax, 0.88 * Lb * Lv(ey), s="chi axis", col=TEXT_COLORS["vy"])

    ki_s, kf_s = R.T @ mount.ki, R.T @ mount.kf
    # (sample-frame direction for the tracer, the same direction in the cradle frame for drawing)
    for d_s, d_c, col in ((-ki_s, -mount.ki, INCIDENT), (kf_s, mount.kf, DIFFRACTED)):
        for t0, t1 in mesh.ray_segments(c, d_s):
            _line(ax, Lv(np.array([t0 * d_c, t1 * d_c])), col=col, lw=3.6, z=35)
    ax.scatter(0, 0, 0, color=GAUGE, s=45, depthshade=False, zorder=40, edgecolors="white", linewidths=1.2)
    ax.view_init(elev=30, azim=117)
    _cube_limits(ax, np.zeros(3), 0.8 * Lb)
    L_in, L_out = path_at_gauge(mesh, c, mount=mount, chi=chi, phi=phi)
    path = (
        "gauge center is OUTSIDE the sample"
        if np.isnan(L_in)
        else f"path from the gauge center: in {L_in:.2f} + out {L_out:.2f} = {L_in + L_out:.2f} mm"
    )
    sense = "clockwise" if conv.s_phi < 0 else "counterclockwise"
    chi_txt = f"chi {chi_logged:g} (logged)" + (f", cradle {chi:g}" if conv.chi_offset_deg else "")
    ax.set_title(
        f"(b) As mounted: {chi_txt}, phi {phi:g}, omega {mount.omega:g}, 2theta {mount.tth:g}\n"
        f"incident from {conv.incident_from}, +phi {sense} from above\n" + path,
        fontsize=8.5,
        loc="left",
    )

    # (c) pole-figure key with the current pole
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.add_patch(Circle((0, 0), 1, fill=False, lw=0.9, ec="0.25", zorder=5))
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    phis = np.linspace(0, 360, 361)
    for cc in range(0, 90, 10):
        h = np.array([mount.pole(cc, p) for p in phis])
        for s in (1, -1):
            xy = project(s * h, view=view, fold=False)
            xy[np.hypot(*xy.T) > 1.0001] = np.nan
            ax2.plot(*xy.T, color="#3987e5" if cc == 0 else "#9dbbe3", lw=1.3 if cc == 0 else 0.6,
                     ls="-" if cc == 0 else "--")  # fmt: skip
    hp = mount.pole(chi, phi)
    xy = project(hp, view=view)[0]
    ax2.plot(*xy, "o", ms=9, color=Q_COLOR, mec="white", mew=1.5, zorder=8)
    if view == "+vx":
        keys = (("+vz", [0, 0, 1], "vz"), ("+vy", [0, 1, 0], "vy"), ("vx", [1, 0, 0], "vx"))
    else:
        keys = (("+vy", [0, 1, 0], "vy"), ("+vx", [1, 0, 0], "vx"), (vz_name, [0, 0, 1], "vz"))
    for txt, axis, col in keys:
        p = project(np.array(axis, float), view=view)[0]
        p = p * 1.1 if np.hypot(*p) > 0.5 else p
        ax2.text(float(p[0]), float(p[1]), txt, fontsize=8, color=AXIS_COLORS[col], ha="center", va="center",
                 bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.85})  # fmt: skip
    ax2.set_title(
        f"(c) Pole figure key, seen from {view}: solid chi {conv.logged_chi(0.0):g} ring,\n"
        f"dashed every 10 deg of chi. Current pole (purple):\n"
        f"h = ({hp[0]:+.3f}, {hp[1]:+.3f}, {hp[2]:+.3f}) in (vx, vy, vz)",
        fontsize=8.5,
        loc="left",
    )
