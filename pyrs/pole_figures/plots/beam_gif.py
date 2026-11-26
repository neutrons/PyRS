"""
Beam-path animation: one frame per chi, phi orientation (the reference script's ``beam_path_gif``).

- ``predicted_beam_gif``: a predicted plane's grid.
- ``measured_beam_gif``: a measured pole figure's sub-runs in the order they were taken, with its own beams and
  cutoff.

Each frame is drawn by ``beam_frame.FrameRenderer``. Long GIFs are drawn in worker processes (``gif_worker``), each
with its own renderer. The GIF is written whole or not at all (``io.retry.write_atomic``).

Example:
    >>> n = predicted_beam_gif("111.gif", mesh, plane=pred.planes[0], center=center, beams=pred.beams, cutoff=3.0,
    ...                        settings=GifSettings(fps=8.0, every=2))
"""

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt
from PIL import Image

from ..core.correction import Beams
from ..core.parallel import Workers, n_workers, run_chunks
from ..core.polefig import PolePeak
from ..core.prediction import PlanePrediction
from ..core.projection import ViewFrom, check_view
from ..core.stl_mesh import SampleMesh
from ..core.types import Progress, no_progress
from ..io.retry import write_atomic
from . import gif_worker
from .beam_frame import FIGSIZE, FrameRenderer, Track

MIN_FRAMES_PARALLEL = 24  # shorter GIFs: starting processes costs more than it saves
FRAMES_DONE = 0.97  # progress fraction when every frame is drawn (the rest is writing the file)
WRITING = 0.98  # progress fraction reported while the GIF is written
MAX_FPS = 100.0  # GIF frame delays are whole hundredths of a second in most viewers
MIN_DPI, MAX_DPI = 20.0, 300.0
#: frames x pixels per frame above this are refused before drawing (the frames are held in memory, 1 byte a pixel)
MAX_GIF_PIXELS = 3_000_000_000


@dataclass(frozen=True)
class GifSettings:
    """
    Beam-path GIF options (the reference script's ``beam_path_gif`` settings).

    Attributes:
        fps: Frames per second, above 0 and at most ``MAX_FPS`` (100).
        every: Use every n-th orientation, at least 1.
        dpi: Resolution of each frame (the figure is 12.5 x 5.8 in), from ``MIN_DPI`` (20) to ``MAX_DPI`` (300).
        workers: Processes that draw the frames in parallel, a positive whole number or "auto". "auto" follows
            ``core.parallel.n_workers``: 1 for short GIFs, else up to its maximum, at low priority.

    Raises:
        ValueError: If a value is out of its range.

    Example:
        >>> GifSettings(fps=10.0, every=2).dpi
        80.0
    """

    fps: float = 8.0
    every: int = 1
    dpi: float = 80.0
    workers: Workers = "auto"

    def __post_init__(self) -> None:
        """Refuse values out of range, naming the value."""
        if not (math.isfinite(self.fps) and 0 < self.fps <= MAX_FPS):
            raise ValueError(f"GIF fps must be above 0 and at most {MAX_FPS:g}, got {self.fps!r}")
        if not (math.isfinite(self.dpi) and MIN_DPI <= self.dpi <= MAX_DPI):
            raise ValueError(f"GIF dpi must be from {MIN_DPI:g} to {MAX_DPI:g}, got {self.dpi!r}")
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError(f"GIF every must be a whole number of at least 1, got {self.every!r}")
        if self.workers != "auto" and (
            isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1
        ):
            raise ValueError(f'GIF workers must be "auto" or a whole number of at least 1, got {self.workers!r}')

    def frame_pixels(self) -> int:
        """
        Pixels in one frame at this dpi.

        Returns:
            Width times height of a frame.
        """
        return round(FIGSIZE[0] * self.dpi) * round(FIGSIZE[1] * self.dpi)


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


def predicted_beam_gif(
    path: str | Path,
    mesh: SampleMesh,
    *,
    plane: PlanePrediction,
    center: npt.ArrayLike,
    beams: Beams,
    cutoff: float,
    settings: GifSettings = GifSettings(),
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
    progress: Progress = no_progress,
) -> int:
    """
    Write the animation of a predicted plane, one frame per grid point (every ``settings.every``-th).

    Args:
        path: Output GIF file.
        mesh: Sample.
        plane: Predicted plane (``core.prediction.Prediction.planes``).
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        beams: Slit widths and height (mm) of the gauge drawn in red (``Prediction.beams``).
        cutoff: Correction above which a point is ignored (hatched, title marked).
        settings: GIF options (default: ``GifSettings()``).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
        progress: Called as ``progress(fraction, message)`` as frames finish. It may raise to stop the job (the
            processes are then stopped) (default: no reporting).

    Returns:
        The number of frames written.

    Raises:
        ValueError: If the plane has no orientations, the view is unknown or the GIF would be too large.
    """
    return _write_gif(Path(path), mesh, track=Track.from_plane(plane), center=center, beams=beams, cutoff=cutoff,
                      settings=settings, view=view, vz_name=vz_name, progress=progress)  # fmt: skip


def measured_beam_gif(
    path: str | Path,
    mesh: SampleMesh,
    *,
    peak: PolePeak,
    center: npt.ArrayLike,
    settings: GifSettings = GifSettings(),
    view: ViewFrom = "+vz",
    vz_name: str = "vz",
    progress: Progress = no_progress,
) -> int:
    """
    Write the animation of a measured pole figure, point by point in the order of its sub-runs.

    The gauge is drawn with the pole figure's own beams, and its own cutoff marks the ignored points. The titles name
    the runs and the count time of each point.

    Args:
        path: Output GIF file.
        mesh: Sample.
        peak: Measured pole figure (``core.polefig.PolePeak``).
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        settings: GIF options (default: ``GifSettings()``).
        view: Pole-figure view, "+vz", "-vz" or "+vx" (default: "+vz").
        vz_name: Label of +vz, for example "BD" (default: "vz").
        progress: Called as ``progress(fraction, message)`` as frames finish. It may raise to stop the job
            (default: no reporting).

    Returns:
        The number of frames written.

    Raises:
        ValueError: If the pole figure has no points, the view is unknown or the GIF would be too large.
    """
    return _write_gif(Path(path), mesh, track=Track.from_peak(peak), center=center, beams=peak.beams,
                      cutoff=peak.cutoff, settings=settings, view=view, vz_name=vz_name,
                      progress=progress)  # fmt: skip


def _write_gif(path: Path, mesh: SampleMesh, *, track: Track, center: npt.ArrayLike, beams: Beams, cutoff: float,
               settings: GifSettings, view: ViewFrom, vz_name: str, progress: Progress) -> int:  # fmt: skip
    """Draw the frames of ``track`` (in worker processes for long GIFs) and write the GIF."""
    check_view(view)
    order = frame_indices(len(track.chi), settings.every)
    n = len(order)
    if n == 0:
        raise ValueError(f"{{{track.hkl}}}: no chi, phi orientations to animate")
    pixels = n * settings.frame_pixels()
    if pixels > MAX_GIF_PIXELS:
        raise ValueError(f"{{{track.hkl}}}: {n} frames at dpi {settings.dpi:g} hold {pixels:,} pixels, more than "
                         f"{MAX_GIF_PIXELS:,}. Lower the dpi or use every n-th orientation.")  # fmt: skip
    hkl = track.hkl
    renderer_kw: dict[str, Any] = {"center": center, "track": track, "beams": beams, "cutoff": cutoff,
                                   "dpi": settings.dpi, "view": view, "vz_name": vz_name}  # fmt: skip
    jobs = list(enumerate(order))
    n_proc = gif_workers(n, settings.workers)
    frames: list[Image.Image | None] = [None] * n
    if n_proc <= 1:
        r = FrameRenderer(mesh, **renderer_kw)
        for k, i in jobs:
            frames[k] = r.frame(k, i, order)
            progress((k + 1) / n * FRAMES_DONE, f"{{{hkl}}} GIF: frame {k + 1} of {n}")
    else:
        # frames are independent: render chunks in worker processes (each builds the figure once)
        size = max(2, min(25, n // (4 * n_proc) or 2))
        chunks = [jobs[j : j + size] for j in range(0, n, size)]
        done = 0

        def on_done(_chunk: Any, result: list[tuple[int, Image.Image]]) -> None:
            nonlocal done
            for k, img in result:
                frames[k] = img
            done += len(result)
            progress(done / n * FRAMES_DONE, f"{{{hkl}}} GIF: {done} of {n} frames ({n_proc} processes)")

        run_chunks(chunks, init=gif_worker.init, task=gif_worker.render,
                   payload=pickle.dumps((mesh, renderer_kw, order)),
                   workers=n_proc, on_done=on_done)  # fmt: skip
    progress(WRITING, "Writing the GIF...")
    _save_gif(path, [f for f in frames if f is not None], settings.fps)
    return n


def _save_gif(path: Path, frames: list[Image.Image], fps: float) -> None:
    """Write the frames as a looping GIF, whole or not at all (``io.retry.write_atomic``)."""
    kw: dict[str, Any] = {"format": "GIF", "save_all": True, "append_images": frames[1:],
                          "duration": int(round(1000 / fps)), "loop": 0, "optimize": True}  # fmt: skip
    write_atomic(path, lambda tmp: frames[0].save(tmp, **kw))
