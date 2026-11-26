"""
One frame of the beam-path animation: the sample turned to a chi, phi orientation, the beams and the pole.

Left: the STL turned to (chi, phi) in the cradle frame (phi axis up), with the incident and diffracted beam
centerlines through the gauge (``style.INCIDENT`` and ``style.DIFFRACTED``). Thick = inside the sample (every
crossing), dashed = air. The shaded red region is the part of the gauge inside the sample (convex hull of the Monte
Carlo points inside, or the points themselves when that part is not convex). Right: where the pole falls on the pole
figure, cells colored by the correction.

``Track`` holds the orientations of a predicted plane or of a measured pole figure. ``FrameRenderer`` builds the
figure once and draws each frame. ``beam_gif`` writes the GIF, and ``gif_worker`` draws frames in other processes.

Example:
    >>> r = FrameRenderer(mesh, center=center, track=Track.from_plane(q), beams=beams, cutoff=3.0, dpi=80.0,
    ...                   view="+vz", vz_name="vz")
    >>> img = r.frame(0, 0, [0])
"""

from dataclasses import dataclass
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial import ConvexHull, QhullError

from ..core.correction import Beams
from ..core.gauge import gauge_points
from ..core.geometry import Mount
from ..core.polefig import PolePeak
from ..core.prediction import PlanePrediction
from ..core.projection import ViewFrom, project
from ..core.stl_mesh import SampleMesh
from ..core.types import BoolArray, FloatArray
from .annotate import add_colorbar
from .polefig_cells import cells, draw_cells, pf_frame
from .style import (
    AXIS_COLORS,
    CORR_CMAP,
    CORR_TICKS,
    CRADLE_TO_PLOT,
    CURRENT_POLE,
    DIFFRACTED,
    DIFFRACTED_TEXT,
    GAUGE,
    GAUGE_FILL,
    GIF_SAMPLE,
    INCIDENT,
    INCIDENT_TEXT,
    LIGHT,
    Q_COLOR,
    Q_TEXT,
    TEXT_COLORS,
    WARNING_TEXT,
    clip_correction,
    correction_norm,
)

FIGSIZE = (12.5, 5.8)  # size (in) of a GIF frame
GAUGE_POINTS = 6000  # Monte Carlo points that draw the part of the gauge inside the sample
GAUGE_SEED = 5  # their random seed (the same points in every process)
MIN_HULL_POINTS = 10  # fewer points inside the sample are not shaded
CONVEX_TOLERANCE = 1.15  # hull volume up to this times the inside volume counts as convex (Monte Carlo noise)
VIEW_MARGIN = 1.15  # the 3D view spans this times the farthest mesh vertex from the gauge center
#: direction of the inside test of the gauge points (any direction not along a mesh face)
PROBE = np.array([0.31, 0.53, 0.79]) / np.linalg.norm([0.31, 0.53, 0.79])
_BASE = np.array(mcolors.to_rgb(GIF_SAMPLE))


@dataclass(frozen=True, kw_only=True)
class Track:
    """
    The orientations to animate, from a predicted plane or a measured pole figure.

    Attributes:
        hkl: Plane label.
        mount: Beams and rotation conventions.
        chi: Cradle chi of each orientation (deg).
        phi: Phi of each orientation (deg).
        correction: Correction of each orientation.
        mean_path: Mean beam path of each orientation (mm).
        poles: Pole vector of each orientation in the sample frame, shape (n, 3).
        measured: True for a measured pole figure (animated in sub-run order, with run and count time).
        runs: The measured runs, for example "1234" or "1234, 1235" ("" for a prediction).
        count_s: Count time of each measured point (s), NaN for a prediction.
    """

    hkl: str
    mount: Mount
    chi: FloatArray
    phi: FloatArray
    correction: FloatArray
    mean_path: FloatArray
    poles: FloatArray
    measured: bool
    runs: str
    count_s: FloatArray

    @classmethod
    def from_plane(cls, q: PlanePrediction) -> "Track":
        """
        The grid of a predicted plane.

        Args:
            q: Predicted plane.

        Returns:
            The track, in grid order.
        """
        return cls(hkl=q.hkl, mount=q.mount, chi=q.chi, phi=q.phi, correction=q.correction,
                   mean_path=q.gauge.mean_path, poles=q.poles(), measured=False, runs="",
                   count_s=np.full(len(q.chi), np.nan))  # fmt: skip

    @classmethod
    def from_peak(cls, peak: PolePeak) -> "Track":
        """
        The sub-runs of a measured pole figure, in the order they were taken.

        Args:
            peak: Measured pole figure.

        Returns:
            The track, with its runs and count times.
        """
        runs = ", ".join(peak.source_runs) if peak.source_runs else peak.run
        return cls(hkl=peak.hkl, mount=peak.mount, chi=peak.chi, phi=peak.phi, correction=peak.correction,
                   mean_path=peak.gauge.mean_path, poles=peak.poles(), measured=True, runs=runs,
                   count_s=peak.count_s)  # fmt: skip


class FrameRenderer:
    """
    The figure of one GIF, set up once, then redrawn for each frame.

    The parts that never change (pole-figure cells, color bar, title) are drawn once and kept as a background. Each
    frame redraws only the 3D view and the pole markers. All frames share the palette of orientation 0, so a frame
    looks the same whichever process draws it. The titles show the logged chi (the cradle chi minus the mount's
    ``chi_offset_deg``).

    Args:
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        track: The orientations to animate.
        beams: Slit widths and height (mm) of the gauge drawn in red.
        cutoff: Correction above which a point is ignored.
        dpi: Resolution of each frame.
        view: Pole-figure view, "+vz", "-vz" or "+vx".
        vz_name: Label of +vz.
    """

    def __init__(self, mesh: SampleMesh, *, center: npt.ArrayLike, track: Track, beams: Beams, cutoff: float,
                 dpi: float, view: ViewFrom, vz_name: str) -> None:  # fmt: skip
        m = track.mount
        self._mesh = mesh
        self._track = track
        self._center: FloatArray = np.asarray(center, float)
        self._tris: FloatArray = mesh.triangles()
        self._normals: FloatArray = mesh.n / np.linalg.norm(mesh.n, axis=1)[:, None]
        self._lim = VIEW_MARGIN * float(np.linalg.norm(self._tris.reshape(-1, 3) - self._center, axis=1).max())
        self._ki: FloatArray = m.ki @ CRADLE_TO_PLOT
        self._kf: FloatArray = m.kf @ CRADLE_TO_PLOT
        self._q: FloatArray = m.q @ CRADLE_TO_PLOT
        self._gauge_points, self._gauge_volume = gauge_points(
            m, incident_width=beams.incident_width, incident_height=beams.incident_height,
            diffracted_width=beams.diffracted_width, n_points=GAUGE_POINTS, seed=GAUGE_SEED)  # fmt: skip
        self._ignored: BoolArray = ~(track.correction <= cutoff)  # inf and NaN (gauge misses the sample) are ignored
        self._xy_all: FloatArray = project(track.poles, view=view)
        self._palette: Image.Image | None = None  # one 256-color palette for every frame (from the first frame drawn)

        fig = Figure(figsize=FIGSIZE)
        self._fig = fig
        self._canvas = FigureCanvasAgg(fig)
        self._ax3 = cast(Axes3D, fig.add_axes((0.0, 0.0, 0.58, 0.86), projection="3d"))
        self._axp = fig.add_axes((0.58, 0.1, 0.32, 0.72))
        cax = fig.add_axes((0.915, 0.2, 0.012, 0.52))
        pf_frame(self._axp, view=view, vz_name=vz_name)
        cn = correction_norm(cutoff)
        draw_cells(self._axp, cells(m, track.chi, track.phi), values=clip_correction(track.correction, cn),
                   ignored=self._ignored, norm=cn, cmap=CORR_CMAP, view=view, ignored_colored=True)  # fmt: skip
        add_colorbar(fig, cax, norm=cn, cmap=CORR_CMAP, ticks=CORR_TICKS, mark=cutoff,
                     label=f"correction (log), hatched > {cutoff:g} ignored")  # fmt: skip
        (self._visited,) = self._axp.plot([], [], ".", ms=2.5, color="0.15", zorder=8)
        (self._mark,) = self._axp.plot([], [], "o", ms=11, mfc="none", mec=CURRENT_POLE, mew=2.2, zorder=9)
        self._mark.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
        self._axp.set_title("Pole in this frame (circle), earlier frames (dots)", fontsize=9, loc="left")
        what = (f"the measured chi, phi of run {track.runs} in the order taken" if track.measured
                else "the predicted chi, phi grid")  # fmt: skip
        c = self._center
        fig.suptitle(f"{mesh.name} turned through {what} (cradle frame, phi axis up, beams fixed). "
                     f"Gauge at vx {c[0]:g}, vy {c[1]:g}, vz {c[2]:g}. Slits {beams.incident_width:g} x "
                     f"{beams.incident_height:g} in, {beams.diffracted_width:g} out (mm)", fontsize=8.5, x=0.01,
                     ha="left", y=0.985, wrap=True)  # fmt: skip
        fig.set_dpi(dpi)
        # the parts that never change (pole-figure cells, color bar, title) are drawn once and kept as a background;
        # each frame pastes it back and draws only the 3D view and the pole markers
        self._ax3.set_visible(False)
        self._canvas.draw()
        self._background = self._canvas.copy_from_bbox(fig.bbox)
        self._ax3.set_visible(True)
        self.frame(0, 0, [0])  # sets the shared palette from the first orientation (the same in every process)

    def _draw_beams(self, R: FloatArray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """The beam centerlines (thick in the sample, dashed in air), arrows and labels. Returns the segments."""
        ax3, lim, ki, kf, m = self._ax3, self._lim, self._ki, self._kf, self._track.mount
        seg_in = self._mesh.ray_segments(self._center, R.T @ (-m.ki))
        seg_out = self._mesh.ray_segments(self._center, R.T @ m.kf)
        for segs, kv, col, sgn in ((seg_in, ki, INCIDENT, -1), (seg_out, kf, DIFFRACTED, 1)):
            ax3.plot(*np.array([[0, 0, 0], sgn * lim * kv]).T, color=col, lw=1.2, ls=(0, (3, 2)), zorder=5)
            for a0, b0 in segs:
                ax3.plot(*np.array([sgn * a0 * kv, sgn * b0 * kv]).T, color=col, lw=4.5, solid_capstyle="butt",
                         zorder=6)  # fmt: skip
        ax3.quiver(*(-lim * ki), *(0.22 * lim * ki), color=INCIDENT, lw=1.8, arrow_length_ratio=0.35)
        ax3.quiver(*(0.75 * lim * kf), *(0.22 * lim * kf), color=DIFFRACTED, lw=1.8, arrow_length_ratio=0.35)
        ax3.text(*(-lim * ki), " incident", color=INCIDENT_TEXT, fontsize=8)
        ax3.text(*(0.97 * lim * kf), " to detector", color=DIFFRACTED_TEXT, fontsize=8)
        ax3.quiver(0, 0, 0, *(0.3 * lim * self._q), color=Q_COLOR, lw=1.8, arrow_length_ratio=0.25)
        ax3.text(*(0.32 * lim * self._q), "Q", color=Q_TEXT, fontsize=8)
        return seg_in, seg_out

    def _draw_gauge(self, R: FloatArray) -> float:
        """The part of the gauge inside the sample (hull or points). Returns its volume (mm3)."""
        inside = self._mesh.trace(self._gauge_points @ R + self._center, PROBE)[0] > 0.5
        v_in = self._gauge_volume * float(inside.mean())
        G = self._gauge_points[inside] @ CRADLE_TO_PLOT
        if len(G) >= MIN_HULL_POINTS:
            try:
                hull = ConvexHull(G)
                convex = hull.volume <= CONVEX_TOLERANCE * v_in
            except QhullError:
                convex = False
            if convex:  # the part in the sample is convex: shade its hull
                self._ax3.add_collection3d(Poly3DCollection(list(G[hull.simplices]), facecolors=GAUGE_FILL,
                                                            edgecolors="none", zorder=9))  # fmt: skip
            else:  # not convex (hole, edge, concave part): show the points
                self._ax3.scatter(*G.T, s=1, color=GAUGE, alpha=0.35, depthshade=False, zorder=9)
        return v_in

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
        t, ax3, lim = self._track, self._ax3, self._lim
        chi_offset_deg = t.mount.conv.chi_offset_deg
        chi, phi = float(t.chi[i]), float(t.phi[i])
        R = t.mount.R(chi, phi)
        ax3.cla()
        ax3.computed_zorder = False  # beams drawn over the translucent sample
        T = ((self._tris - self._center) @ R.T) @ CRADLE_TO_PLOT
        shade = 0.5 + 0.5 * np.clip(((self._normals @ R.T) @ CRADLE_TO_PLOT) @ LIGHT, 0, 1)
        fc = np.c_[_BASE[None, :] * shade[:, None], np.full(len(T), 0.5)]
        ax3.add_collection3d(Poly3DCollection(list(T), facecolors=fc, edgecolors="none", zorder=1))
        seg_in, seg_out = self._draw_beams(R)
        v_in = self._draw_gauge(R)
        o = ((np.zeros(3) - self._center) @ R.T) @ CRADLE_TO_PLOT  # STL origin (vx 0, vy 0, vz 0)
        for j, key in enumerate(("vx", "vy", "vz")):
            v = ((np.eye(3)[j] * 0.22 * lim) @ R.T) @ CRADLE_TO_PLOT
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
        ign = bool(self._ignored[i])
        head = f"{{{t.hkl}}}: chi {chi - chi_offset_deg:g}, phi {phi:g}   (frame {k + 1} of {len(order)})"
        if t.measured:
            head = (
                f"{{{t.hkl}}} run {t.runs}, point {i + 1} of {len(t.chi)}: chi {chi - chi_offset_deg:.4g}, "
                f"phi {phi:.4g}, {t.count_s[i]:.0f} s   (frame {k + 1} of {len(order)})"
            )
        ax3.set_title(f"{head}\n"
                      f"centerline in sample: incident {L_in:.1f} mm ({len(seg_in)} seg.), diffracted {L_out:.1f} mm "
                      f"({len(seg_out)} seg.)\ngauge in sample (red) {v_in:.1f} of {self._gauge_volume:.1f} mm3, "
                      f"mean path {t.mean_path[i]:.1f} mm, correction {t.correction[i]:.2f}"
                      f"{'  IGNORED' if ign else ''}", fontsize=9, loc="left",
                      color=WARNING_TEXT if ign else "black")  # fmt: skip
        upto = order[: k + 1]
        self._visited.set_data(self._xy_all[upto, 0], self._xy_all[upto, 1])
        self._mark.set_data([self._xy_all[i, 0]], [self._xy_all[i, 1]])
        self._canvas.restore_region(self._background)
        self._fig.draw_artist(ax3)
        self._axp.draw_artist(self._visited)
        self._axp.draw_artist(self._mark)
        w, h = self._canvas.get_width_height()
        img = Image.frombuffer("RGBA", (w, h), bytes(self._canvas.buffer_rgba()), "raw", "RGBA", 0, 1).convert("RGB")
        if self._palette is None:  # first frame: its adaptive palette is used for all frames (no color flicker)
            self._palette = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        return img.quantize(palette=self._palette, dither=Image.Dither.NONE)
