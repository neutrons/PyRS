"""
Predict presenter: run the prediction and the GIF off the GUI thread and show the chosen plot.

Inputs come from the Sample and Experiment setup tabs and from the grid and sampling on this tab.

Typical workflow:

1. The plane table follows the reflections chosen on Experiment setup (omega = theta, editable).
2. :meth:`PredictPresenter.run` starts the prediction in a worker. ``predicted`` is emitted when it ends.
3. The plot panel shows the summary, per-plane and correction vs phi plots. Changing the cutoff redraws them
   without recomputing.
4. :meth:`PredictPresenter.make_gif` or :meth:`PredictPresenter.make_all_gifs` writes beam-path GIFs to a temporary
   folder, and :meth:`PredictPresenter.save_all_gifs` copies them out.

Example:
    >>> presenter = PredictPresenter(PredictTab(), sample_presenter, material_presenter)  # doctest: +SKIP
    >>> presenter.predicted.connect(lambda pred: print(len(pred.planes)))  # doctest: +SKIP
    >>> presenter.run()  # doctest: +SKIP
"""

import atexit
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from pydantic import ValidationError
from qtpy.QtCore import QObject, QThreadPool, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLabel, QProgressBar, QPushButton

from pyrs.pole_figures.core.prediction import Plane, Prediction, predict
from pyrs.pole_figures.io.predicted_csv import write_predicted_csv
from pyrs.pole_figures.plots.beam_gif import GifSettings, frame_indices, predicted_beam_gif
from pyrs.pole_figures.plots.correction_profile import correction_profile_figure
from pyrs.pole_figures.plots.predicted import plane_figure, predicted_figure
from pyrs.pole_figures.text import format_duration
from ..figure_cache import FigureCache
from ..figure_jobs import BackgroundFigures
from ..models.predict import PredictModel
from ..tabs.predict_tab import PredictTab
from ..worker import Worker
from .material import MaterialPresenter
from .sample import SamplePresenter

SECONDS_PER_FRAME = 0.1  # per frame at dpi 80 (serial, Windows dev machine). Replaced by the measured speed.

TYPE_SUMMARY = "Summary (all planes)"
TYPE_PLANE = "Path and correction (one plane)"
TYPE_PROFILE = "Correction vs phi"
TYPE_GIF = "Beam-path GIF preview"
PLOT_TYPES = [TYPE_SUMMARY, TYPE_PLANE, TYPE_PROFILE, TYPE_GIF]
PER_PLANE = {TYPE_PLANE, TYPE_PROFILE, TYPE_GIF}


def plane_label(hkl: str) -> str:
    """The plane label shown in the dropdowns: "111" -> "{111}"."""
    return f"{{{hkl}}}"


def hkl_of(label: str) -> str:
    """The hkl of a plane label: "{111}" -> "111"."""
    return label.strip("{}")


@dataclass
class _Target:
    """Widgets that report one kind of background job."""

    bar: QProgressBar
    label: QLabel
    cancel: QPushButton


class CancelledError(Exception):
    """Raised inside a job when the user presses Cancel."""


class PredictPresenter(QObject):
    """
    Logic of the Predict tab.

    Args:
        view: The Predict tab.
        sample: The Sample presenter (mesh, gauge center, mount).
        material: The Experiment setup presenter (phase, mu, beams, reflections).
        model: Starting inputs (``PredictModel()`` for the defaults).

    Attributes:
        status: Signal with a short message for the status bar.
        predicted: Signal with the Prediction when a prediction finished.
        gif_made: Signal with the path of a beam-path GIF written to the session's temporary folder.
        inputs_changed: Signal: readiness or staleness may have changed (an input changed or a job ended).
        view: The Predict tab.
        sample: The Sample presenter.
        material: The Experiment setup presenter.
        model: The current grid and sampling inputs.
        result: The last prediction, or None.
        gifs: GIF made from the current result, by hkl.
    """

    status = Signal(str)
    predicted = Signal(object)
    gif_made = Signal(str)
    inputs_changed = Signal()

    def __init__(
        self,
        view: PredictTab,
        sample: SamplePresenter,
        *,
        material: MaterialPresenter,
        model: PredictModel,
    ) -> None:
        super().__init__(view)
        self.view, self.sample, self.material = view, sample, material
        self.model = model
        self.result: Prediction | None = None
        self.gifs: dict[str, Path] = {}  # hkl -> GIF made from the current result
        self._figs = FigureCache()  # figures shown from the current result
        self._bg = BackgroundFigures(view.plot_panel, self._figs, parent=self)  # builds them off the GUI thread
        self._result_inputs: tuple[Any, ...] | None = None
        self._labels: list[str] = []
        self._plane_key: tuple[Any, ...] | None = None
        self._worker: Worker | None = None
        self._cancel = False
        self._last_dir = str(Path.cwd())
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="hidra_pf_"))  # GIFs until saved
        atexit.register(shutil.rmtree, self._tmp_dir, ignore_errors=True)
        self._gif_count = 0
        self._sec_per_frame_80 = SECONDS_PER_FRAME  # seconds per frame at dpi 80, updated after each GIF
        self._partial_gifs: dict[str, Path] = {}  # GIFs finished by a make-all job (kept on cancel)
        self._run_target = _Target(view.progress, view.progress_label, view.cancel_button)
        self._gif_target = _Target(view.gif_progress, view.gif_progress_label, view.gif_cancel_button)
        self._target = self._run_target

        view.predict_button.clicked.connect(self.run)
        view.cancel_button.clicked.connect(self.cancel)
        view.gif_cancel_button.clicked.connect(self.cancel)
        view.reset_planes_button.clicked.connect(lambda: self.sync_planes(force=True))
        view.cutoff_spin.valueChanged.connect(self._on_cutoff)
        view.save_csv_button.clicked.connect(self._on_save_csv)
        view.plot_panel.selected.connect(self.show_plot)
        view.plot_panel.saved.connect(lambda p: self.status.emit(f"Saved {p}"))
        for box in view.part_boxes.values():
            box.toggled.connect(self._on_parts)
        view.gif_button.clicked.connect(lambda: self.make_gif(hkl_of(view.gif_plane.currentText())))
        view.gif_all_button.clicked.connect(self.make_all_gifs)
        view.gif_save_all_button.clicked.connect(self._on_save_all_gifs)
        view.gif_every.valueChanged.connect(self._update_gif_label)
        view.gif_dpi.valueChanged.connect(self._update_gif_label)
        material.changed.connect(self.sync_planes)
        material.changed.connect(self.update_readiness)
        sample.changed.connect(self.update_readiness)
        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        self._show_grid()
        self.sync_planes(force=True)
        self._update_gif_label()
        view.parts_box.setVisible(False)
        self.update_readiness()

    def _on_form_changed(self) -> None:
        vals: dict[str, Any] = {}
        for form in self.view.forms:
            vals.update(form.values())
        try:
            self.model = PredictModel.model_validate({**self.model.model_dump(), **vals})
        except ValidationError as exc:
            self.status.emit(f"Invalid input: {exc.errors()[0]['msg']}")
            return
        self._show_grid()
        self.update_readiness()

    def _show_grid(self) -> None:
        n = len(self.model.grid().points()[0])
        self.view.grid_label.setText(f"{n} orientations per plane")

    def sync_planes(self, force: bool = False) -> None:
        """
        Rebuild the plane table from the selected reflections (omega = theta) when they change.

        Args:
            force: Rebuild even when the reflections and their 2theta did not change (default: False).
        """
        refl = self.material.selected_reflections()
        key = tuple((r.label, round(r.two_theta, 4)) for r in refl)
        if not force and key == self._plane_key:
            return
        self._plane_key = key
        self._labels = [r.label for r in refl]
        self.view.set_planes([Plane.symmetric(r.label, r.two_theta) for r in refl])
        for spin in self.view.plane_spins():
            spin.valueChanged.connect(self.update_readiness)
        self.update_readiness()

    def planes(self) -> list[Plane]:
        """The planes with the 2theta and omega (deg) shown in the table."""
        return self.view.planes(self._labels)

    def cutoff(self) -> float:
        """The correction cutoff on this tab. Points with a larger correction are ignored."""
        return float(self.view.cutoff_spin.value())

    def missing(self) -> list[str]:
        """What is still needed before a prediction can run."""
        out = []
        if self.sample.mesh is None:
            out.append("load an STL on the Sample tab")
        elif not self.sample.gauge_inside():
            out.append("move the gauge center inside the sample (Sample tab)")
        if not self.material.has_material():
            out.append("choose a material on the Experiment setup tab")
        elif not self._labels:
            out.append("select at least one reflection on the Experiment setup tab")
        mu = self.material.mu()
        if self.material.has_material() and (not np.isfinite(mu) or mu < 0):
            out.append("fix the material so mu can be computed (Experiment setup tab)")
        return out

    def _inputs(self) -> tuple[Any, ...]:
        s, i, b = self.sample.model, self.model, self.material.beams()
        return (s.stl_path, tuple(s.gauge_center()), s.conventions(), tuple(self.planes()), b, i.grid(),
                i.mc_points, round(self.material.mu(), 8), i.best_phi)  # fmt: skip

    def stale(self) -> bool:
        """True when the inputs changed since the shown prediction was made."""
        return self.result is not None and self._result_inputs != self._inputs()

    def update_readiness(self) -> None:
        """Enable the buttons that can run now and say what is missing or stale. Emits ``inputs_changed``."""
        self.inputs_changed.emit()
        missing = self.missing()
        busy = self._worker is not None
        self.view.predict_button.setEnabled(not missing and not busy)
        can_gif = self.result is not None and not busy and self.sample.mesh is not None
        self.view.gif_button.setEnabled(can_gif)
        self.view.gif_all_button.setEnabled(can_gif)
        self.view.gif_save_all_button.setEnabled(bool(self.gifs) and not busy)
        if missing:
            self.view.ready_label.setText("<b>Not ready:</b> " + "; ".join(missing) + ".")
            return
        s, i, b = self.sample.model, self.model, self.material.beams()
        c = s.gauge_center()
        n = len(i.grid().points()[0]) * len(self._labels)
        txt = (f"<b>Ready.</b> {Path(s.stl_path).name}, gauge at ({c[0]:g}, {c[1]:g}, {c[2]:g}) mm, "
               f"mu {self.material.mu():.4f} /mm, {b.incident_width:g} x {b.incident_height:g} mm incident, "
               f"{b.diffracted_width:g} mm receiving slit. {n} orientations x {i.mc_points:,} points.")  # fmt: skip
        if self.stale():
            txt += "<br><span style='color:#b00020'>Inputs changed since the plots were made. Predict again.</span>"
        self.view.ready_label.setText(txt)

    def _start(self, job: Callable[[Callable[[float, str], None]], Any], *, on_done: Callable[[Any], None],
               first_msg: str, target: _Target) -> None:  # fmt: skip
        self._cancel = False
        self._target = target

        def wrapped(progress: Any) -> Any:
            def tick(frac: float, msg: str) -> None:
                if self._cancel:
                    raise CancelledError
                progress(frac, msg)

            return job(tick)

        worker = Worker(wrapped)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(on_done)
        worker.signals.failed.connect(self._on_failed)
        self._worker = worker
        target.cancel.setEnabled(True)
        target.bar.setValue(0)
        target.label.setText(first_msg)
        self.update_readiness()
        QThreadPool.globalInstance().start(worker)

    def run(self) -> None:
        """Start the prediction in the background. Does nothing while inputs are missing or a job runs."""
        if self.missing() or self._worker is not None:
            return
        mesh = self.sample.mesh
        assert mesh is not None
        s, i = self.sample.model, self.model
        inputs = self._inputs()
        args: dict[str, Any] = {
            "mesh": mesh,
            "center": s.gauge_center(),
            "conv": s.conventions(),
            "planes": self.planes(),
            "beams": self.material.beams(),
            "mu": self.material.mu(),
            "grid": i.grid(),
            "cutoff": self.cutoff(),
            "n_points": i.mc_points,
            "best_phi": i.best_phi,
        }
        self._start(lambda tick: predict(**args, progress=tick), on_done=lambda res: self._on_predicted(res, inputs),
                    first_msg="Starting...", target=self._run_target)  # fmt: skip

    def cancel(self) -> None:
        """Ask the running job to stop at its next progress report."""
        if self._worker is None:
            return
        self._cancel = True
        self._target.label.setText("Cancelling...")

    def _on_progress(self, frac: float, msg: str) -> None:
        self._target.bar.setValue(int(round(1000 * frac)))
        self._target.label.setText(msg)
        if self._target is self._gif_target:
            self.status.emit(f"{msg} ({100 * frac:.0f}% done)")

    def _done(self, msg: str, ok: bool = True) -> None:
        self._worker = None
        self._target.cancel.setEnabled(False)
        self._target.bar.setValue(1000 if ok else 0)
        self._target.label.setText(msg)
        self.update_readiness()

    def _on_predicted(self, pred: Prediction, inputs: tuple[Any, ...]) -> None:
        self.result, self._result_inputs = pred, inputs
        self.gifs.clear()
        self._bg.clear()
        self._done("Done.")
        self.view.save_csv_button.setEnabled(True)
        hkls = [q.plane.hkl for q in pred.planes]
        self.view.gif_plane.blockSignals(True)
        self.view.gif_plane.clear()
        self.view.gif_plane.addItems([plane_label(h) for h in hkls])
        self.view.gif_plane.blockSignals(False)
        self.view.gif_progress.setValue(0)
        self.view.gif_progress_label.setText("")
        self._update_gif_label()
        panel = self.view.plot_panel
        keep_type, keep_plane = panel.current()
        panel.set_choices(PLOT_TYPES, [plane_label(h) for h in hkls], PER_PLANE)
        panel.select(keep_type or TYPE_SUMMARY, keep_plane)
        self.predicted.emit(pred)
        self.status.emit("Prediction finished")

    def _on_failed(self, tb: str) -> None:
        last = tb.strip().splitlines()[-1]
        kept = self._take_partial_gifs()
        if "CancelledError" in last:
            self._done("Cancelled." + (f" Kept the GIFs already made: {kept}." if kept else ""), ok=False)
            self.status.emit("Cancelled")
            return
        self._done(f"<span style='color:#b00020'>Failed: {last}</span>", ok=False)
        self.status.emit(f"Failed: {last}")

    def _index(self, hkl: str) -> int:
        assert self.result is not None
        return [q.plane.hkl for q in self.result.planes].index(hkl)

    def figure(self, plot_type: str, plane: str = "") -> Figure | None:
        """
        Build a plot of the current prediction at the current cutoff.

        Args:
            plot_type: One of PLOT_TYPES.
            plane: Plane label such as "{111}" for the per-plane types (default: "").

        Returns:
            The figure, or None without a prediction, for the GIF type, or for an unknown plane.
        """
        return self._builder(plot_type, plane)()

    def _builder(self, plot_type: str, plane: str) -> Callable[[], Figure | None]:
        """A figure builder with its inputs taken now (on the GUI thread), safe to run on a worker thread."""
        pred = self.result
        if pred is None:
            return lambda: None
        s, cut = self.sample.model, self.cutoff()
        view, vz_name = s.view_from, s.vz_name
        if plot_type == TYPE_SUMMARY:
            return lambda: predicted_figure(pred, cutoff=cut, view=view, vz_name=vz_name)
        hkl = hkl_of(plane)
        if hkl not in [q.plane.hkl for q in pred.planes]:
            return lambda: None
        k = self._index(hkl)
        if plot_type == TYPE_PLANE:
            return lambda: plane_figure(pred, k, cutoff=cut, view=view, vz_name=vz_name)
        if plot_type == TYPE_PROFILE:
            parts = self.view.parts()
            return lambda: correction_profile_figure(pred, k, cutoff=cut, parts=parts)
        return lambda: None

    def show_plot(self, plot_type: str, plane: str = "") -> None:
        """
        Show a plot type (and plane) in the plot panel.

        Figures are built in the background (the panel says "Drawing ..." meanwhile) and cached.

        Args:
            plot_type: One of PLOT_TYPES.
            plane: Plane label such as "{111}" for the per-plane types (default: "").
        """
        self.view.parts_box.setVisible(plot_type == TYPE_PROFILE)
        if self.result is None or not plot_type:
            return
        panel = self.view.plot_panel
        if plot_type == TYPE_GIF:
            self._bg.cancel()  # a figure still being drawn must not replace the GIF
            hkl = hkl_of(plane)
            if hkl in self.gifs:
                panel.show_gif(self.gifs[hkl])
            else:
                self.view.gif_plane.setCurrentText(plane)
                panel.show_message(f"No beam-path GIF for {plane} yet.\n\nMake it in the Beam-path GIF section on the "
                                   f"left: the plane is set to {plane}. Press Make beam-path GIF.")  # fmt: skip
            return
        s = self.sample.model
        key = (plot_type, plane, self.cutoff(), tuple(self.view.parts()) if plot_type == TYPE_PROFILE else (),
               s.view_from, s.vz_name, s.chi_offset_deg)  # fmt: skip
        what = "the summary" if plot_type == TYPE_SUMMARY else f"{plane} {plot_type.lower()}"
        self._bg.show(key, self._builder(plot_type, plane), name=what)
        self._update_sphere_label()

    def _redraw_current(self) -> None:
        t, plane = self.view.plot_panel.current()
        if self.result is not None and t != TYPE_GIF:
            self.show_plot(t, plane)
        self._update_sphere_label()

    def _on_cutoff(self) -> None:
        self._redraw_current()

    def _on_parts(self) -> None:
        if self.view.plot_panel.current()[0] == TYPE_PROFILE:
            self._redraw_current()

    def sphere_fractions(self) -> dict[str, float]:
        """Solid-angle (grid weight) fraction of the points above the current cutoff, by hkl."""
        if self.result is None:
            return {}
        g = self.result.grid
        out = {}
        for q in self.result.planes:
            w = g.weights(q.chi)
            out[q.plane.hkl] = float((w * (q.correction > self.cutoff())).sum() / w.sum())
        return out

    def _update_sphere_label(self) -> None:
        fr = self.sphere_fractions()
        if fr:
            self.view.sphere_label.setText(
                "Ignored (solid-angle weighted): " + ", ".join(f"{{{h}}} {100 * f:.0f}%" for h, f in fr.items())
            )

    def gif_settings(self) -> GifSettings:
        """GIF frames per second, point step and dpi from the tab."""
        v = self.view
        return GifSettings(fps=float(v.gif_fps.value()), every=int(v.gif_every.value()), dpi=float(v.gif_dpi.value()))

    def _frames(self) -> int:
        assert self.result is not None
        return len(frame_indices(len(self.result.planes[0].chi), self.view.gif_every.value()))

    def gif_seconds(self, n_planes: int = 1) -> float:
        """Estimated time (s) for n_planes GIFs at the current settings (from the last measured speed)."""
        if self.result is None:
            return 0.0
        return n_planes * self._frames() * self._sec_per_frame_80 * (self.view.gif_dpi.value() / 80) ** 2

    def _update_gif_label(self) -> None:
        if self.result is None:
            self.view.gif_label.setText("Predict first. The GIF uses the predicted grid and corrections.")
            return
        n_pl = len(self.result.planes)
        basis = "measured on the last GIF" if self._sec_per_frame_80 != SECONDS_PER_FRAME else "typical speed"
        self.view.gif_label.setText(
            f"{self._frames()} frames per plane. About {format_duration(max(self.gif_seconds(), 1))} for one plane, "
            f"{format_duration(max(self.gif_seconds(n_pl), 1))} for all {n_pl} planes ({basis})."
        )

    def _gif_job_parts(self) -> tuple[Any, ...]:
        assert self.result is not None and self.sample.mesh is not None
        s = self.sample.model
        return (self.sample.mesh, self.result.center, self.result.beams, self.cutoff(), self.gif_settings(), s)

    def _new_gif_path(self, hkl: str) -> Path:
        self._gif_count += 1
        return self._tmp_dir / f"beam_paths_{hkl}_{self._gif_count}.gif"

    def make_gif(self, hkl: str) -> None:
        """
        Make the beam-path GIF for one plane in the background, then show it.

        Does nothing without a prediction and an STL, or while a job runs.

        Args:
            hkl: The plane, for example "200" (the GIF section's choice is
                ``hkl_of(view.gif_plane.currentText())``).
        """
        if self.result is None or self.sample.mesh is None or self._worker is not None:
            return
        q = self.result.planes[self._index(hkl)]
        path = self._new_gif_path(hkl)
        mesh, center, beams, cut, st, s = self._gif_job_parts()

        def job(tick: Callable[[float, str], None]) -> tuple[Path, float, int]:
            t0 = time.perf_counter()
            n = predicted_beam_gif(path, mesh, plane=q, center=center, beams=beams, cutoff=cut, settings=st,
                                   view=s.view_from, vz_name=s.vz_name, progress=tick)  # fmt: skip
            return path, time.perf_counter() - t0, n

        est = format_duration(max(self.gif_seconds(), 1))
        self._start(
            job,
            on_done=lambda r: self._on_gif(hkl, r),
            first_msg=f"{{{hkl}}} GIF: starting (about {est})...",
            target=self._gif_target,
        )

    def make_all_gifs(self) -> None:
        """Make the beam-path GIF of every plane, one after the other, in one background job."""
        if self.result is None or self.sample.mesh is None or self._worker is not None:
            return
        planes = list(self.result.planes)
        n_pl = len(planes)
        paths = {q.plane.hkl: self._new_gif_path(q.plane.hkl) for q in planes}
        mesh, center, beams, cut, st, s = self._gif_job_parts()
        done: dict[str, Path] = {}
        self._partial_gifs = done
        est = format_duration(max(self.gif_seconds(n_pl), 1))

        def job(tick: Callable[[float, str], None]) -> tuple[dict[str, Path], float, int]:
            t0, frames = time.perf_counter(), 0
            for k, q in enumerate(planes):
                h = q.plane.hkl

                def sub(frac: float, msg: str, *, k: int = k) -> None:
                    tick((k + frac) / n_pl, f"Plane {k + 1} of {n_pl}, {msg}")

                frames += predicted_beam_gif(paths[h], mesh, plane=q, center=center, beams=beams, cutoff=cut,
                                             settings=st, view=s.view_from, vz_name=s.vz_name,
                                             progress=sub)  # fmt: skip
                done[h] = paths[h]
            return dict(done), time.perf_counter() - t0, frames

        self._start(
            job,
            on_done=self._on_all_gifs,
            first_msg=f"All {n_pl} GIFs: starting (about {est})...",
            target=self._gif_target,
        )

    def _learn_speed(self, seconds: float, frames: int) -> None:
        if frames > 0 and seconds > 0:
            self._sec_per_frame_80 = seconds / frames / (self.view.gif_dpi.value() / 80) ** 2

    def _take_partial_gifs(self) -> str:
        """Keep GIFs that a cancelled or failed make-all job finished. Returns their planes as text."""
        kept, self._partial_gifs = dict(self._partial_gifs), {}
        self.gifs.update(kept)
        return ", ".join(plane_label(h) for h in kept)

    def _on_gif(self, hkl: str, result: tuple[Path, float, int]) -> None:
        path, seconds, frames = result
        self._learn_speed(seconds, frames)
        self.gifs[hkl] = path
        self._done(f"{{{hkl}}} GIF done in {format_duration(seconds)}. Use Save plot... to keep it.")
        self._update_gif_label()
        self.view.plot_panel.select(TYPE_GIF, plane_label(hkl))
        self.gif_made.emit(str(path))
        self.status.emit(f"Beam-path GIF for {{{hkl}}} ready")

    def _on_all_gifs(self, result: tuple[dict[str, Path], float, int]) -> None:
        made, seconds, frames = result
        self._partial_gifs = {}
        self._learn_speed(seconds, frames)
        self.gifs.update(made)
        self._done(f"All {len(made)} GIFs done in {format_duration(seconds)}. Use Save all GIFs... to keep them.")
        self._update_gif_label()
        first = next(iter(made))
        self.view.plot_panel.select(TYPE_GIF, plane_label(first))
        for path in made.values():
            self.gif_made.emit(str(path))
        self.status.emit(f"All {len(made)} beam-path GIFs ready")

    def save_all_gifs(self, folder: str | Path) -> list[Path]:
        """
        Copy every GIF made from the current prediction to ``folder`` as beam_paths_<hkl>.gif.

        Args:
            folder: Destination folder, created if needed. Files of the same name are replaced.

        Returns:
            The files written.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for hkl, src in self.gifs.items():
            dst = folder / f"beam_paths_{hkl}.gif"
            shutil.copyfile(src, dst)
            out.append(dst)
        return out

    def _on_save_all_gifs(self) -> None:
        folder = self.view.ask_folder("Folder for the GIFs", self._last_dir)
        if folder:
            paths = self.save_all_gifs(folder)
            self._last_dir = folder
            self.status.emit(f"Saved {len(paths)} GIFs to {folder}")

    def save_csv(self, path: str | Path) -> None:
        """Write the predicted corrections at the current cutoff to a CSV file. Needs a prediction."""
        assert self.result is not None
        write_predicted_csv(path, self.result, cutoff=self.cutoff())

    def _on_save_csv(self) -> None:
        start = str(Path(self._last_dir) / "predicted_correction.csv")
        path = self.view.ask_save_path("Save CSV", start, "CSV (*.csv)")
        if path:
            self.save_csv(path)
            self._last_dir = str(Path(path).parent)
            self.status.emit(f"Saved {path}")
