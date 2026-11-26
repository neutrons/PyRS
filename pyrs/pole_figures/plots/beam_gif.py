"""
Beam-path animation: one frame per chi, phi orientation (the reference script's ``beam_path_gif``).

The orientations are a predicted plane's grid, or a measured pole figure's sub-runs in the order they were taken.

Left: the STL turned to (chi, phi) in the cradle frame (phi axis up), with the incident (blue) and diffracted (orange)
beam centerlines through the gauge. Thick = inside the sample (every crossing), dashed = air. The shaded red region is
the part of the gauge inside the sample (convex hull of the Monte Carlo points inside, or the points themselves when
that part is not convex). Right: where the pole falls on the pole figure, cells colored by the correction.

Typical workflow:

1. Predict a plane with ``core.prediction.predict``, or load a measured pole figure (``core.polefig.PolePeak``).
2. Call ``beam_path_gif``. Long GIFs are drawn in worker processes (``gif_worker``), each with its own
   ``FrameRenderer``.

Example:
    >>> n = beam_path_gif("111.gif", mesh, center=center, q=pred.planes[0], beams=pred.beams, cutoff=3.0,
    ...                   settings=GifSettings(fps=8.0, every=2))
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial import ConvexHull, QhullError

from ..core.correction import Beams
from ..core.gauge import gauge_points
from ..core.geometry import Mount
from ..core.parallel import Workers, n_workers, run_chunks
from ..core.polefig import PolePeak
from ..core.projection import ViewFrom, project
from ..core.protocols import PoleFigureLike
from ..core.stl_mesh import SampleMesh
from ..core.types import Progress, no_progress
from . import gif_worker
from .polefig_cells import cells, draw_cells, pf_frame
from .predicted import CORR_CMAP
from .style import AXIS_COLORS, TEXT_COLORS

_MP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # plot (X, Y, Z) = (e_y, e_z, e_x): phi axis up
_PROBE = np.array([0.31, 0.53, 0.79]) / np.linalg.norm([0.31, 0.53, 0.79])
_LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])
_BASE = np.array(mcolors.to_rgb("#9db7d5"))
MIN_FRAMES_PARALLEL = 24  # shorter GIFs: starting processes costs more than it saves
INCIDENT_C, DIFFRACTED_C, Q_C = "#3b82f6", "#f97316", "#7b2fa8"


@dataclass(frozen=True)
class GifSettings:
    """
    Beam-path GIF options (the reference script's ``beam_path_gif`` settings).

    Attributes:
        fps: Frames per second.
        every: Use every n-th orientation.
        dpi: Resolution of each frame (the figure is 12.5 x 5.8 in).
        workers: Processes that draw the frames in parallel, a positive whole number or "auto". "auto" follows
            ``core.parallel.n_workers``: 1 for short GIFs, else up to its maximum, at low priority.

    Example:
        >>> GifSettings(fps=10.0, every=2).dpi
        80.0
    """

    fps: float = 8.0
    every: int = 1
    dpi: float = 80.0
    workers: Workers = "auto"


def frame_indices(n: int, every: int) -> list[int]:
    """
    Orientation index of each frame: every ``every``-th of ``n`` orientations, starting at 0.

    Args:
        n: Number of orientations.
        every: Step between frames. Values below 1 count as 1.

    Returns:
        The orientation indices.

    Example:
        >>> frame_indices(7, 3)
        [0, 3, 6]
    """
    return list(range(0, n, max(1, int(every))))


@dataclass(frozen=True)
class _Track:
    """The orientations to animate, from a prediction or a measured pole figure."""

    hkl: str
    mount: Mount
    chi: np.ndarray
    phi: np.ndarray
    correction: np.ndarray
    mean_path: np.ndarray
    poles: np.ndarray
    measured: bool
    runs: str  # the measured runs ("" for a prediction)
    count_s: np.ndarray  # count time of each measured point (NaN for a prediction)


def _track(q: PoleFigureLike) -> _Track:
    if isinstance(q, PolePeak):  # a measured pole figure is animated in sub-run order, with its runs and count times
        runs = ", ".join(q.source_runs) if q.source_runs else q.run
        return _Track(q.hkl, q.mount, q.chi, q.phi, q.correction, q.gauge.mean_path, q.poles(), True, runs, q.count_s)
    return _Track(q.hkl, q.mount, q.chi, q.phi, q.correction, q.gauge.mean_path, q.poles(), False, "",
                  np.full(len(q.chi), np.nan))  # fmt: skip


def beam_path_gif(
    path: str | Path,
    mesh: SampleMesh,
    *,
    center: np.ndarray,
    q: PoleFigureLike,
    beams: Beams,
    cutoff: float,
    settings: GifSettings = GifSettings(),
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
    progress: Progress = no_progress,
) -> int:
    """
    Write the animation for one predicted plane or measured pole figure.

    A measured pole figure (PolePeak) is shown point by point in the order of its sub-runs, with its own beams and
    cutoff passed in by the caller. Frames are drawn in parallel processes for longer GIFs (``settings.workers``).
    The titles show the logged chi (the cradle chi minus the mount's ``chi_offset_deg``).

    Args:
        path: Output GIF file.
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm).
        q: Predicted plane or measured pole figure.
        beams: Slit widths and height (mm) of the gauge drawn in red.
        cutoff: Correction above which a point is ignored (hatched, title marked).
        settings: GIF options (default: ``GifSettings()``).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
        progress: Called as ``progress(fraction, message)`` as frames finish. It may raise to stop the job (the
            processes are then stopped) (default: no reporting).

    Returns:
        The number of frames written.

    Raises:
        ValueError: If ``q`` has no orientations.
    """
    renderer_kw: dict[str, Any] = {"center": center, "q": q, "beams": beams, "cutoff": cutoff, "settings": settings,
                                   "view": view, "vz_name": vz_name}  # fmt: skip
    order = frame_indices(len(q.chi), settings.every)
    n = len(order)
    if n == 0:
        raise ValueError(f"{{{q.hkl}}}: no chi, phi orientations to animate")
    hkl = q.hkl
    jobs = list(enumerate(order))
    n_workers = gif_workers(n, settings.workers)
    frames: list[Image.Image | None] = [None] * n
    if n_workers <= 1:
        r = FrameRenderer(mesh, **renderer_kw)
        for k, i in jobs:
            frames[k] = r.frame(k, i, order)
            progress((k + 1) / n * 0.97, f"{{{hkl}}} GIF: frame {k + 1} of {n}")
    else:
        # frames are independent: render chunks in worker processes (each builds the figure once)
        size = max(2, min(25, n // (4 * n_workers) or 2))
        chunks = [jobs[j : j + size] for j in range(0, n, size)]
        done = 0

        def on_done(_chunk: Any, result: list[tuple[int, Image.Image]]) -> None:
            nonlocal done
            for k, img in result:
                frames[k] = img
            done += len(result)
            progress(done / n * 0.97, f"{{{hkl}}} GIF: {done} of {n} frames ({n_workers} processes)")

        run_chunks(chunks, init=gif_worker.init, task=gif_worker.render,
                   payload=pickle.dumps((mesh, renderer_kw, order)),
                   workers=n_workers, on_done=on_done)  # fmt: skip
    progress(0.98, "Writing the GIF...")
    _save_gif(Path(path), [f for f in frames if f is not None], settings.fps)
    return n


def gif_workers(n_frames: int, requested: Workers = "auto") -> int:
    """
    Worker processes for a GIF of ``n_frames`` frames (``core.parallel.n_workers``).

    Below ``MIN_FRAMES_PARALLEL`` frames "auto" gives 1, since starting processes costs more than it saves.

    Args:
        n_frames: Number of frames.
        requested: A positive whole number, or "auto" (default: "auto").

    Returns:
        The number of processes.
    """
    return n_workers(float(n_frames), requested=requested, min_work=MIN_FRAMES_PARALLEL)


class FrameRenderer:
    """
    The figure of one GIF, set up once, then redrawn for each frame.

    The parts that never change (pole-figure cells, color bar, title) are drawn once and kept as a background. Each
    frame redraws only the 3D view and the pole markers. All frames share the palette of orientation 0, so a frame
    looks the same whichever process draws it.

    Args:
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm).
        q: Predicted plane or measured pole figure.
        beams: Slit widths and height (mm) of the gauge drawn in red.
        cutoff: Correction above which a point is ignored.
        settings: GIF options (only ``dpi`` is used here).
        view: Pole-figure view, "+vz", "-vz" or "+vx".
        vz_name: Label of +vz.

    The titles show the logged chi (the cradle chi minus the mount's ``chi_offset_deg``).

    Example:
        >>> r = FrameRenderer(mesh, center=center, q=q, beams=beams, cutoff=3.0, settings=GifSettings(), view="+vz",
        ...                   vz_name="vz")
        >>> img = r.frame(0, 0, [0])
    """

    def __init__(self, mesh: SampleMesh, *, center: np.ndarray, q: PoleFigureLike, beams: Beams, cutoff: float,
                 settings: GifSettings, view: ViewFrom, vz_name: str) -> None:  # fmt: skip
        t = _track(q)
        m = t.mount
        chi_offset_deg = m.conv.chi_offset_deg
        center = np.asarray(center, float)
        tris = mesh.triangles()
        nrm = mesh.n / np.linalg.norm(mesh.n, axis=1)[:, None]
        lim = 1.15 * float(np.linalg.norm(tris.reshape(-1, 3) - center, axis=1).max())
        ki, kf, qv = m.ki @ _MP, m.kf @ _MP, m.q @ _MP
        gp, gvol = gauge_points(m, incident_width=beams.incident_width, incident_height=beams.incident_height,
                                diffracted_width=beams.diffracted_width, n_points=6000, seed=5)  # fmt: skip
        ignored = ~(t.correction <= cutoff)  # inf and NaN (gauge misses the sample) are ignored

        fig = Figure(figsize=(12.5, 5.8))
        canvas = FigureCanvasAgg(fig)
        ax3 = cast(Axes3D, fig.add_axes((0.0, 0.0, 0.58, 0.86), projection="3d"))
        axp = fig.add_axes((0.58, 0.1, 0.32, 0.72))
        cax = fig.add_axes((0.915, 0.2, 0.012, 0.52))
        pf_frame(axp, view=view, vz_name=vz_name)
        cn = mcolors.LogNorm(1, max(cutoff * 10, 30))
        top = float(cn.vmax or 30)
        corr = np.clip(np.where(np.isfinite(t.correction), t.correction, top), 1, top)
        draw_cells(axp, cells(m, t.chi, t.phi), values=corr, ignored=ignored, norm=cn, cmap=CORR_CMAP, view=view,
                   ignored_colored=True)  # fmt: skip
        cb = fig.colorbar(ScalarMappable(norm=cn, cmap=CORR_CMAP), cax=cax)
        ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 30) if t <= top]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.minorticks_off()
        cb.ax.tick_params(labelsize=7)
        cb.ax.axhline(cutoff, color="black", lw=0.9, ls=(0, (3, 2)))
        cb.set_label(f"correction (log), hatched > {cutoff:g} ignored", fontsize=8)
        (visited,) = axp.plot([], [], ".", ms=2.5, color="0.15", zorder=8)
        (mark,) = axp.plot([], [], "o", ms=11, mfc="none", mec="#00a5a5", mew=2.2, zorder=9)
        mark.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
        axp.set_title("Pole in this frame (circle), earlier frames (dots)", fontsize=9, loc="left")
        xy_all = project(t.poles, view=view)
        what = (f"the measured chi, phi of run {t.runs} in the order taken" if t.measured
                else "the predicted chi, phi grid")  # fmt: skip
        fig.suptitle(f"{mesh.name} turned through {what} (cradle frame, phi axis up, beams fixed). "
                     f"Gauge at vx {center[0]:g}, vy {center[1]:g}, vz {center[2]:g}. Slits {beams.incident_width:g} x "
                     f"{beams.incident_height:g} in, {beams.diffracted_width:g} out (mm)", fontsize=8.5, x=0.01,
                     ha="left", y=0.985, wrap=True)  # fmt: skip
        fig.set_dpi(settings.dpi)
        # the parts that never change (pole-figure cells, color bar, title) are drawn once and kept as a background;
        # each frame pastes it back and draws only the 3D view and the pole markers
        ax3.set_visible(False)
        canvas.draw()
        background = canvas.copy_from_bbox(fig.bbox)
        ax3.set_visible(True)
        palette: Image.Image | None = None  # one 256-color palette for every frame (from the first frame drawn)

        state = dict(locals())  # the set-up objects, used by frame()
        state.pop("self")
        self.__dict__.update(state)
        self.frame(0, 0, [0])  # sets the shared palette from the first orientation (the same in every process)

    def frame(self, k: int, i: int, order: list[int]) -> Image.Image:
        """
        Draw one frame.

        Args:
            k: Frame number (0-based).
            i: Orientation index of the frame.
            order: Orientation index of every frame. Frames up to k are marked as visited.

        Returns:
            The frame as a palette (GIF) image.
        """
        g = self.__dict__
        t, m, mesh, center, tris, nrm, lim = g["t"], g["m"], g["mesh"], g["center"], g["tris"], g["nrm"], g["lim"]
        ki, kf, qv, gp, gvol, ignored = g["ki"], g["kf"], g["qv"], g["gp"], g["gvol"], g["ignored"]
        ax3, canvas, visited, mark, xy_all = g["ax3"], g["canvas"], g["visited"], g["mark"], g["xy_all"]
        chi_offset_deg = g["chi_offset_deg"]
        chi, phi = float(t.chi[i]), float(t.phi[i])
        R = m.R(chi, phi)
        ax3.cla()
        ax3.computed_zorder = False  # beams drawn over the translucent sample
        T = ((tris - center) @ R.T) @ _MP
        shade = 0.5 + 0.5 * np.clip(((nrm @ R.T) @ _MP) @ _LIGHT, 0, 1)
        fc = np.c_[_BASE[None, :] * shade[:, None], np.full(len(T), 0.5)]
        ax3.add_collection3d(Poly3DCollection(list(T), facecolors=fc, edgecolors="none", zorder=1))
        seg_in = mesh.ray_segments(center, R.T @ (-m.ki))
        seg_out = mesh.ray_segments(center, R.T @ m.kf)
        for segs, kv, col, sgn in ((seg_in, ki, INCIDENT_C, -1), (seg_out, kf, DIFFRACTED_C, 1)):
            ax3.plot(*np.array([[0, 0, 0], sgn * lim * kv]).T, color=col, lw=1.2, ls=(0, (3, 2)), zorder=5)
            for a0, b0 in segs:
                ax3.plot(*np.array([sgn * a0 * kv, sgn * b0 * kv]).T, color=col, lw=4.5, solid_capstyle="butt",
                         zorder=6)  # fmt: skip
        ax3.quiver(*(-lim * ki), *(0.22 * lim * ki), color=INCIDENT_C, lw=1.8, arrow_length_ratio=0.35)
        ax3.quiver(*(0.75 * lim * kf), *(0.22 * lim * kf), color=DIFFRACTED_C, lw=1.8, arrow_length_ratio=0.35)
        ax3.text(*(-lim * ki), " incident", color="#1d4ed8", fontsize=8)
        ax3.text(*(0.97 * lim * kf), " to detector", color="#c2410c", fontsize=8)
        ax3.quiver(0, 0, 0, *(0.3 * lim * qv), color=Q_C, lw=1.8, arrow_length_ratio=0.25)
        ax3.text(*(0.32 * lim * qv), "Q", color="#6a2a91", fontsize=8)
        inside = mesh.trace(gp @ R + center, _PROBE)[0] > 0.5
        v_in = gvol * float(inside.mean())
        G = gp[inside] @ _MP
        if len(G) >= 10:
            try:
                hull = ConvexHull(G)
                convex = hull.volume <= 1.15 * v_in
            except QhullError:
                convex = False
            if convex:  # the part in the sample is convex: shade its hull
                ax3.add_collection3d(Poly3DCollection(list(G[hull.simplices]), facecolors=(0.75, 0.16, 0.13, 0.6),
                                                      edgecolors="none", zorder=9))  # fmt: skip
            else:  # not convex (hole, edge, concave part): show the points
                ax3.scatter(*G.T, s=1, color="#c0392b", alpha=0.35, depthshade=False, zorder=9)
        o = ((np.zeros(3) - center) @ R.T) @ _MP  # STL origin (vx 0, vy 0, vz 0)
        for j, key in enumerate(("vx", "vy", "vz")):
            v = ((np.eye(3)[j] * 0.22 * lim) @ R.T) @ _MP
            ax3.quiver(*o, *v, color=AXIS_COLORS[key], lw=1.8, arrow_length_ratio=0.2)
            ax3.text(*(o + 1.12 * v), "+" + key, color=TEXT_COLORS[key], fontsize=8)
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-lim, lim)
        ax3.set_box_aspect((1, 1, 1), zoom=1.45)
        ax3.set_proj_type("ortho")
        ax3.view_init(elev=30, azim=117)
        ax3.set_axis_off()
        L_in, L_out = sum(b - a for a, b in seg_in), sum(b - a for a, b in seg_out)
        ign = bool(ignored[i])
        head = f"{{{t.hkl}}}: chi {chi - chi_offset_deg:g}, phi {phi:g}   (frame {k + 1} of {len(order)})"
        if t.measured:
            head = (
                f"{{{t.hkl}}} run {t.runs}, point {i + 1} of {len(t.chi)}: chi {chi - chi_offset_deg:.4g}, "
                f"phi {phi:.4g}, {t.count_s[i]:.0f} s   (frame {k + 1} of {len(order)})"
            )
        ax3.set_title(f"{head}\n"
                      f"centerline in sample: incident {L_in:.1f} mm ({len(seg_in)} seg.), diffracted {L_out:.1f} mm "
                      f"({len(seg_out)} seg.)\ngauge in sample (red) {v_in:.1f} of {gvol:.1f} mm3, mean path "
                      f"{t.mean_path[i]:.1f} mm, correction {t.correction[i]:.2f}{'  IGNORED' if ign else ''}",
                      fontsize=9, loc="left", color="#a32e2e" if ign else "black")  # fmt: skip
        upto = order[: k + 1]
        visited.set_data(xy_all[upto, 0], xy_all[upto, 1])
        mark.set_data([xy_all[i, 0]], [xy_all[i, 1]])
        canvas.restore_region(g["background"])
        fig = g["fig"]
        fig.draw_artist(ax3)
        g["axp"].draw_artist(visited)
        g["axp"].draw_artist(mark)
        w, h = canvas.get_width_height()
        img = Image.frombuffer("RGBA", (w, h), bytes(canvas.buffer_rgba()), "raw", "RGBA", 0, 1).convert("RGB")
        if g["palette"] is None:  # first frame: its adaptive palette is used for all frames (no color flicker)
            g["palette"] = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        return img.quantize(palette=g["palette"], dither=Image.Dither.NONE)


def _save_gif(path: Path, frames: list[Image.Image], fps: float) -> None:
    kw: dict[str, Any] = {"save_all": True, "append_images": frames[1:], "duration": int(round(1000 / fps)),
                          "loop": 0, "optimize": True}  # fmt: skip
    for attempt in range(20):
        try:
            frames[0].save(path, **kw)
            return
        except OSError:
            if attempt == 19:
                raise
            time.sleep(1.5)
