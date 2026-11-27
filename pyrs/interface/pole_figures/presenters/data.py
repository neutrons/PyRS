"""
Data and Corrections presenter: load, correct, plot and export runs, and make beam-path GIFs of the measured points.

Works without a prediction. It needs the sample (Sample tab), the material (Experiment setup) and the gauge position
during the runs (vx, vy, vz on this tab). These fill from the runs' vx, vy, vz logs when the files have them, from
the prediction on request, or by hand, and stay blank until then. Each run's wavelength, incident slits and motor angles
come from its file. The receiving slit is an input.

Typical workflow:

1. :meth:`DataPresenter.add_files` reads the reduced ``HB2B_<run>.h5`` files and fills vx, vy, vz from their logs.
2. :meth:`DataPresenter.run` analyzes every run in a worker: peaks, intensities, path-length corrections.
3. :meth:`DataPresenter.pole_figures` gives the pole figures (combined by hkl when asked) to the plots, the export
   and the MTEX tab. Changing the cutoff or the combine setting updates them without a new analysis.
4. :meth:`DataPresenter.export` writes PF's CSVs and figures. Beam-path GIFs show the measured points.

Example:
    >>> presenter = DataPresenter(DataTab(), sample_presenter, material=material_presenter,  # doctest: +SKIP
    ...                           predict=predict_presenter, model=DataModel())
    >>> errors = presenter.add_files(["HB2B_1234.h5"])  # doctest: +SKIP
    >>> presenter.run()  # doctest: +SKIP
"""

import atexit
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from pydantic import ValidationError
from qtpy.QtCore import QObject, QThreadPool, Signal  # type: ignore[attr-defined]

from pyrs.pole_figures.core.combine import combine_by_hkl
from pyrs.pole_figures.core.peaks import find_peaks
from pyrs.pole_figures.core.polefig import PolePeak, analyze_run
from pyrs.pole_figures.core.protocols import PoleFigureLike
from pyrs.pole_figures.core.runs import RunData, read_run
from pyrs.pole_figures.io.polefig_csv import write_all
from pyrs.pole_figures.plots.beam_gif import GifSettings, frame_indices, measured_beam_gif
from pyrs.pole_figures.plots.polefigures import pole_figures_figure
from pyrs.pole_figures.plots.save import save_figure
from pyrs.pole_figures.text import format_duration
from ..figure_cache import FigureCache
from ..figure_jobs import BackgroundFigures
from ..models.data import GAUGE_FIELDS, DataModel
from ..run_checks import MAP_NOTE, MARGIN_NOTE, NEGATIVE_NOTE, inside_with_margin, is_spatial_map, mostly_negative
from ..tabs.data_tab import DataTab
from ..worker import Worker
from .material import MaterialPresenter
from .predict import SECONDS_PER_FRAME, PredictPresenter
from .sample import SamplePresenter

TYPE_ALL = "All pole figures"
TYPE_ONE = "One pole figure"
TYPE_GIF = "Beam-path GIF preview"
PLOT_TYPES = [TYPE_ALL, TYPE_ONE, TYPE_GIF]
PER_PF = {TYPE_ONE, TYPE_GIF}
POSITION_TOL_MM = 0.01  # logged gauge positions closer than this are the same position
RED = "<span style='color:#b00020'>{}</span>"
STYLED = {TYPE_ALL, TYPE_ONE}  # plot types the Point cells / Interpolated toggle changes


def gif_name(label: str) -> str:
    """
    beam_paths_<label>.gif with the label made safe for a file name.

    Args:
        label: Pole-figure label, for example "1234 {222}".

    Returns:
        The file name. Runs of characters other than letters, digits, "." and "-" become "_".

    Example:
        >>> gif_name("1234 {222}")
        'beam_paths_1234_222.gif'
    """
    return "beam_paths_" + re.sub(r"[^A-Za-z0-9.-]+", "_", label).strip("_") + ".gif"


def _same(a: float | None, b: float | None) -> bool:
    """Both blank, or both set to the same value."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 1e-9


def _xyz(p: Any) -> str:
    return ", ".join(f"{float(x):g}" for x in p)


def _runs(names: list[str]) -> str:
    return ("runs " if len(names) > 1 else "run ") + ", ".join(names)


def _moves(run: RunData) -> bool:
    """True when the run's logged vx, vy or vz changed by more than POSITION_TOL_MM during the run."""
    return run.position_spread > POSITION_TOL_MM


def _position_text(run: RunData) -> str:
    """Runs-table cell: the logged vx, vy, vz, or 'not logged'."""
    p = run.logged_position
    note = " (incomplete)" if run.position_note else ""
    if p is None:
        return "not logged" + note
    moves = f" (moves {run.position_spread:.2f})" if run.position_spread > POSITION_TOL_MM else ""
    return _xyz(p) + moves + note


class CancelledError(Exception):
    """Raised inside the job when the user presses Cancel."""


class GaugeSource(str, Enum):
    """Where the gauge position (vx, vy, vz) on the Data and Corrections tab came from."""

    BLANK = "blank"  # not set yet
    FILES = "files"  # the logs of a loaded run
    PREDICTION = "prediction"  # the gauge center of the last prediction
    USER = "user"  # typed by hand


class DataPresenter(QObject):
    """
    Logic of the Data and Corrections tab.

    Args:
        view: The Data and Corrections tab.
        sample: The Sample presenter (mesh and mount conventions).
        material: The Experiment setup presenter (phase, mu, reflections).
        predict: The Predict presenter, whose gauge center can be copied.
        model: Starting inputs (``DataModel()`` for the defaults).

    Attributes:
        status: Signal with a short message for the status bar.
        analyzed: Signal with the PolePeak results of a finished analysis (one per run and peak).
        pole_figures_changed: Signal with the pole figures to use downstream (combined when asked), sent after a new
            analysis or a change of cutoff or combine setting.
        view: The Data and Corrections tab.
        model: The current inputs.
        runs: The loaded runs, sorted by run number.
        results: The pole figures of the last analysis, one per run and peak.
        gifs: GIF made from the current pole figures, by pole-figure label.
    """

    status = Signal(str)
    analyzed = Signal(list)
    pole_figures_changed = Signal(list)

    def __init__(self, view: DataTab, sample: SamplePresenter, *, material: MaterialPresenter,
                 predict: PredictPresenter, model: DataModel) -> None:  # fmt: skip
        super().__init__(view)
        self.view, self.sample, self.material, self.predict = view, sample, material, predict
        self._gauge_source = GaugeSource.BLANK  # where vx, vy, vz came from
        self._gauge_run = ""  # the run whose logged position filled vx, vy, vz (source "files")
        self._gauge_error = ""  # why the last fill from the files or the prediction was refused
        self.model = model
        self.runs: list[RunData] = []
        self.results: list[PolePeak] = []
        self._result_inputs: tuple[Any, ...] | None = None
        self._worker: Worker | None = None
        self._cancel = False
        self._last_dir = str(Path.cwd())
        self.gifs: dict[str, Path] = {}  # pole-figure label -> GIF made from the current pole figures
        self._figs = FigureCache()  # figures shown from the current pole figures
        self._bg = BackgroundFigures(view.plot_panel, self._figs, parent=self)  # builds them off the GUI thread
        self._table_inputs: tuple[Any, ...] = ()  # settings the runs table depends on (peak window, slit)
        self._partial_gifs: dict[str, Path] = {}  # GIFs finished by a make-all job (kept on cancel)
        self._gif_job = False  # the running job makes GIFs (its progress goes to the GIF section)
        self._gif_count = 0
        self._sec_per_frame_80 = SECONDS_PER_FRAME
        self._sec_per_point_tri = 0.0  # analysis speed, measured on the last analysis
        self._t_run, self._run_work = 0.0, 0
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="hidra_pf_data_"))  # GIFs until saved
        atexit.register(shutil.rmtree, self._tmp_dir, ignore_errors=True)

        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        view.gauge_from_prediction_button.clicked.connect(self.inherit_prediction_center)
        if predict is not None:
            predict.predicted.connect(lambda _p: self.update_readiness())
        view.add_button.clicked.connect(self._on_add)
        view.remove_button.clicked.connect(self._on_remove)
        view.clear_button.clicked.connect(self.clear_runs)
        view.analyze_button.clicked.connect(self.run)
        view.cancel_button.clicked.connect(self.cancel)
        view.export_button.clicked.connect(self._on_export)
        view.style_toggle.toggled.connect(lambda _on: self.redraw())
        view.plot_panel.selected.connect(lambda t, label: self.show_plot(t, label))
        view.plot_panel.saved.connect(lambda p: self.status.emit(f"Saved {p}"))
        view.plot_panel.plane_label.setText("Pole figure:")
        view.gif_button.clicked.connect(lambda: self.make_gif(view.gif_pf.currentText()))
        view.gif_all_button.clicked.connect(self.make_all_gifs)
        view.gif_save_all_button.clicked.connect(self._on_save_all_gifs)
        view.gif_cancel_button.clicked.connect(self.cancel)
        view.gif_every.valueChanged.connect(self._update_gif_label)
        view.gif_dpi.valueChanged.connect(self._update_gif_label)
        view.gif_pf.currentTextChanged.connect(lambda _t: self._update_gif_label())
        sample.changed.connect(self.update_readiness)
        material.changed.connect(self._on_material)
        self._update_gif_label()
        view.style_toggle.setVisible(False)  # shown with a pole-figure plot
        self.update_readiness()

    def add_files(self, paths: list[str | Path]) -> list[str]:
        """
        Read reduced files. Files already loaded are skipped. vx, vy, vz fill from the logs when not set otherwise.

        Args:
            paths: The reduced HB-2B files.

        Returns:
            The error messages of files that could not be used.
        """
        errors = []
        have = {r.path.resolve() for r in self.runs}
        for p in paths:
            p = Path(p)
            if p.resolve() in have:
                continue
            try:
                self.runs.append(read_run(p))
            except ValueError as exc:
                errors.append(str(exc))
            else:
                have.add(p.resolve())
                self._last_dir = str(p.parent)
        self.runs.sort(key=lambda r: (r.run.isdigit(), int(r.run) if r.run.isdigit() else 0, r.run))
        self._gauge_from_files()
        self._show_runs()
        self.update_readiness()
        n = len(paths) - len(errors)
        self.status.emit(f"Loaded {n} run{'s' if n != 1 else ''}" + (f", {len(errors)} failed" if errors else ""))
        return errors

    def pf_runs(self) -> list[RunData]:
        """The loaded runs that are pole-figure scans: every run except spatial maps (see ``is_spatial_map``)."""
        return [r for r in self.runs if not is_spatial_map(r)]

    def _on_add(self) -> None:
        paths = self.view.ask_files(self._last_dir)
        if paths:
            errors = self.add_files(list(paths))
            self.view.files_note.setText("<span style='color:#b00020'>" + "<br>".join(errors) + "</span>"
                                         if errors else "")  # fmt: skip

    def _on_remove(self) -> None:
        rows = set(self.view.selected_rows())
        self.runs = [r for i, r in enumerate(self.runs) if i not in rows]
        self._gauge_from_files()
        self._show_runs()
        self.update_readiness()

    def clear_runs(self) -> None:
        """Remove every run."""
        self.runs = []
        self._gauge_from_files()
        self._show_runs()
        self.update_readiness()

    def _mu(self, run: RunData) -> float:
        """Mu at the run's wavelength (or the override from Experiment setup)."""
        m = self.material.model
        if m.mu_override:
            return float(m.mu_value)
        return float(self.material.phase().mu_per_mm(run.wavelength_A))

    def _candidates(self, run: RunData) -> list[tuple[str, float]]:
        return [(r.label, r.two_theta) for r in self.material.phase().reflections(run.wavelength_A)]

    def _show_runs(self) -> None:
        self._table_inputs = (self.model.central_hw, self.model.receiving_slit_mm)
        rows, maps = [], []
        have_material = self.material.has_material()
        for i, r in enumerate(self.runs):
            mu, peaks = "", ""
            if is_spatial_map(r):
                maps.append(i)
            elif have_material:
                try:
                    mu = f"{self._mu(r):.4f}"
                    found = find_peaks(r, self._candidates(r), central_hw=self.model.central_hw)
                    peaks = ", ".join(f"{{{p.hkl}}}" + ("" if p.central else " (edge)") for p in found) or "none"
                except (ValueError, RuntimeError, KeyError):
                    mu, peaks = "?", "?"
            n_text = f"{r.n_subruns}" + (" (live)" if r.live else "") + (" (spatial map)" if i in maps else "")
            rows.append([r.run, n_text, f"{r.wavelength_A:.4f}", f"{r.omega_median:.2f}", f"{r.arm_two_theta:.2f}",
                         f"{r.incident_width:g} x {r.incident_height:g}", mu, peaks, _position_text(r),
                         r.title])  # fmt: skip
        self.view.set_runs(rows, flagged=maps)
        self.view.map_note.setText(
            RED.format("<br>".join(f"Run {self.runs[i].run}: {MAP_NOTE}" for i in maps)) if maps else ""
        )
        reasons = (("live_skipped", "still counting"), ("short_skipped", "too short"),
                   ("invalid_skipped", "with a non-finite duration or motor log"))  # fmt: skip
        skipped = []
        for r in self.runs:
            parts = [f"{getattr(r, k)} {why}" for k, why in reasons if getattr(r, k)]
            if parts:
                skipped.append(f"run {r.run}: {', '.join(parts)}")
        lines = [f"Sub-runs left out: {'; '.join(skipped)}."] if skipped else []
        lines += [f"Run {r.run}: {n}." for r in self.runs for n in r.notes]
        if lines and not self.view.files_note.text():
            self.view.files_note.setText(" ".join(lines))

    def _on_material(self) -> None:
        self._show_runs()
        self.update_readiness()

    def gauge_center(self) -> np.ndarray | None:
        """Gauge position during the runs (mm, STL frame), or None until vx, vy and vz are all entered."""
        return self.model.gauge_center()

    def gauge_inside(self) -> bool:
        """
        True when the gauge position is set, an STL is loaded and the position is inside it with a margin.

        The rule is ``run_checks.inside_with_margin``: the center and the six points 0.1 mm from it along vx, vy and
        vz are all inside, so a point on a face, edge or corner of the sample is refused.
        """
        c = self.gauge_center()
        mesh = self.sample.mesh
        return c is not None and mesh is not None and inside_with_margin(mesh, c)

    def _set_gauge(self, center: Any, *, source: GaugeSource, run: str = "") -> bool:
        """
        Show ``center`` (three values, or None for blank) in vx, vy, vz and remember where it came from.

        Returns False, and leaves the model, the boxes and the source as they were, when the values are refused
        (for example out of range). The reason is shown in the gauge note.
        """
        vals = [None] * 3 if center is None else [float(x) for x in center]
        self.view.gauge_form.set_values(dict(zip(GAUGE_FIELDS, vals, strict=True)))
        old_source = self._gauge_source
        old = {k: getattr(self.model, k) for k in GAUGE_FIELDS}
        ok = self._on_form_changed()
        got = (self.model.gauge_vx, self.model.gauge_vy, self.model.gauge_vz)
        shown = [None if v is None else round(v, self.view.gauge_form.decimals) for v in vals]
        if not ok or not all(map(_same, got, shown)):  # refused, for example out of range
            self.model = self.model.model_copy(update=old)  # nothing of a refused position is kept
            self.view.gauge_form.set_values(old)  # the boxes back in step with the model
            self._gauge_source = old_source
            what = f"run {run}" if run else source.value
            self._gauge_error = f"The position from {what} ({_xyz(vals)}) was refused: out of range."
            self.update_readiness()
            return False
        self._gauge_error = ""
        self._gauge_source = source if center is not None else GaugeSource.BLANK
        self._gauge_run = run if center is not None else ""
        self.update_readiness()
        return True

    def _gauge_from_files(self) -> None:
        """
        Fill vx, vy, vz from the runs' logs, unless the user or the prediction set them.

        A position already taken from a run that is still loaded is kept, so adding a run never swaps it. Spatial
        maps and runs whose position moved during the run do not fill it: their median is not where the gauge was.
        """
        if self._gauge_source in (GaugeSource.USER, GaugeSource.PREDICTION) and self.gauge_center() is not None:
            return
        logged = [
            (r.run, r.logged_position) for r in self.pf_runs() if r.logged_position is not None and not _moves(r)
        ]
        if self._gauge_source == GaugeSource.FILES and any(run == self._gauge_run for run, _ in logged):
            return
        if logged:
            self._set_gauge(logged[0][1], source=GaugeSource.FILES, run=logged[0][0])
        elif self._gauge_source == GaugeSource.FILES:
            self._set_gauge(None, source=GaugeSource.BLANK)  # the runs that gave the values were removed

    def inherit_prediction_center(self) -> None:
        """Fill vx, vy, vz with the gauge center of the last prediction."""
        if self.predict.result is not None:
            self._set_gauge(self.predict.result.center, source=GaugeSource.PREDICTION)

    def _on_form_changed(self) -> bool:
        """Read the forms into the model. Returns False (model unchanged) when the values are refused."""
        vals: dict[str, Any] = {}
        for form in self.view.forms:
            vals.update(form.values())
        old_cut, old_comb = self.model.cutoff, (self.model.combine, self.model.combine_scale)
        old_gauge = (self.model.gauge_vx, self.model.gauge_vy, self.model.gauge_vz)
        old_table = self._table_inputs
        try:
            self.model = DataModel.model_validate({**self.model.model_dump(), **vals})
        except ValidationError as exc:
            self.status.emit(f"Invalid input: {exc.errors()[0]['msg']}")
            return False
        if (self.model.gauge_vx, self.model.gauge_vy, self.model.gauge_vz) != old_gauge:
            self._gauge_source = GaugeSource.USER  # typed by hand (_set_gauge sets its own source after this)
            self._gauge_error = ""  # a refused fill is no longer what the boxes show
        if self.model.cutoff != old_cut and self.results:
            self.results = [r.with_cutoff(self.model.cutoff) for r in self.results]
        if self.results and (
            self.model.cutoff != old_cut or (self.model.combine, self.model.combine_scale) != old_comb
        ):
            self._show_results()
            self._new_pole_figures()
            self.pole_figures_changed.emit(self.pole_figures())
        if (self.model.central_hw, self.model.receiving_slit_mm) != old_table:
            self._show_runs()  # re-read the runs only when a setting of the table changed, not on gauge edits
        self.update_readiness()
        return True

    def missing(self) -> list[str]:
        """What is still needed before the analysis can run (empty when ready)."""
        out = []
        if not self.runs:
            out.append("add reduced runs")
        elif not self.pf_runs():
            out.append("add pole-figure runs (the loaded runs are spatial maps)")
        if self.sample.mesh is None:
            out.append("load the STL on the Sample tab")
        bad = self.view.gauge_form.invalid()
        if bad:
            out.append(f"{', '.join(n.removeprefix('gauge_') for n in bad)} is not a valid number")
        elif self.gauge_center() is None:
            out.append("enter the gauge position vx, vy, vz during the runs")
        elif self.sample.mesh is not None and not self.gauge_inside():
            out.append("the gauge position vx, vy, vz is outside the sample")
        if not self.material.has_material():
            out.append("choose a material on the Experiment setup tab")
        return out

    def _inputs(self) -> tuple[Any, ...]:
        s, m = self.sample.model, self.model
        try:
            mus = tuple(round(self._mu(r), 8) for r in self.pf_runs()) if self.material.has_material() else ()
            phase = repr(self.material.phase()) if self.material.has_material() else ""
        except (ValueError, RuntimeError, KeyError):
            mus, phase = (), ""
        c = self.gauge_center()
        return (tuple(str(r.path) for r in self.pf_runs()), s.stl_path, () if c is None else tuple(c), s.conventions(),
                m.receiving_slit_mm, m.mc_points, m.central_hw, m.method, mus, phase)  # fmt: skip

    def stale(self) -> bool:
        """True when the inputs changed since the shown analysis was made."""
        return bool(self.results) and self._result_inputs != self._inputs()

    def update_readiness(self) -> None:
        """Enable the buttons that can run now, update the gauge note, and say what is missing or stale."""
        missing = self.missing()
        busy = self._worker is not None
        v = self.view
        v.analyze_button.setEnabled(not missing and not busy)
        v.export_button.setEnabled(bool(self.results) and not busy)
        can_gif = bool(self.results) and not busy and self.sample.mesh is not None and self.gauge_inside()
        v.gif_button.setEnabled(can_gif)
        v.gif_all_button.setEnabled(can_gif)
        v.gif_save_all_button.setEnabled(bool(self.gifs) and not busy)
        v.remove_button.setEnabled(bool(self.runs) and not busy)
        v.clear_button.setEnabled(bool(self.runs) and not busy)
        v.add_button.setEnabled(not busy)
        v.gauge_from_prediction_button.setEnabled(self.predict.result is not None and not busy)
        self._show_gauge_note()
        if missing:
            v.ready_label.setText("<b>Not ready:</b> " + "; ".join(missing) + ".")
            return
        runs = self.pf_runs()
        n = sum(r.n_subruns for r in runs)
        m = self.model
        mesh = self.sample.mesh
        assert mesh is not None
        est = ""
        if self._sec_per_point_tri:  # measured on the last analysis: time ~ sub-runs x points x triangles
            sec = self._sec_per_point_tri * n * m.mc_points * len(mesh.n)
            est = f" About {format_duration(max(sec, 1.0))} (from the speed of the last analysis)."
        left_out = len(self.runs) - len(runs)
        skip = f" ({left_out} spatial map{'s' if left_out > 1 else ''} left out)" if left_out else ""
        txt = (f"<b>Ready.</b> {len(runs)} runs{skip}, {n} sub-runs, receiving slit {m.receiving_slit_mm:g} mm, "
               f"{m.mc_points:,} Monte Carlo points per sub-run traced through {mesh.name} "
               f"({len(mesh.n):,} triangles).{est}")  # fmt: skip
        if self.stale():
            txt += "<br><span style='color:#b00020'>Inputs changed since the analysis. Analyze again.</span>"
        v.ready_label.setText(txt)

    def gauge_flags(self) -> list[str]:
        """Warnings about vx, vy, vz: not set, outside the sample, or not what the pole-figure runs logged."""
        c, out, runs = self.gauge_center(), [], self.pf_runs()
        if self._gauge_error:
            out.append(self._gauge_error)
        bad = self.view.gauge_form.invalid()
        if bad:
            names = ", ".join(n.removeprefix("gauge_") for n in bad)
            out.append(f"{names}: not a valid number (mm, -10000 to 10000, a point for decimals).")
        elif c is None:
            out.append("Not set. Enter vx, vy and vz.")
        elif self.sample.mesh is not None and not self.gauge_inside():
            out.append(f"This point is outside the sample or too close to its surface. {MARGIN_NOTE}")
        logged = [(r.run, r.logged_position) for r in runs if r.logged_position is not None]
        if logged:
            first = logged[0][1]
            if any(np.abs(p - first).max() > POSITION_TOL_MM for _, p in logged):
                listing = "; ".join(f"{run} ({_xyz(p)})" for run, p in logged)
                out.append(f"The runs logged different positions: {listing}. "
                           "Every run is corrected at the one position above.")  # fmt: skip
            unlogged = [r.run for r in runs if r.logged_position is None]
            if unlogged:
                out.append(f"No vx, vy, vz logged in {_runs(unlogged)}.")
            if c is not None and self._gauge_source != GaugeSource.FILES:
                off = [run for run, p in logged if np.abs(p - c).max() > POSITION_TOL_MM]
                if off:
                    out.append(f"Differs from the position logged in {_runs(off)}.")
        moving = [f"{r.run} ({r.position_spread:.2f} mm)" for r in runs if _moves(r)]
        if moving:
            out.append(f"vx, vy or vz changed during {_runs(moving)}. A moving position does not fill vx, vy, vz. "
                       "Enter the gauge position by hand for a pole-figure scan.")  # fmt: skip
        out += [f"Run {r.run}: {r.position_note}." for r in runs if r.position_note]
        return out

    def _show_gauge_note(self) -> None:
        source = {GaugeSource.FILES: f"From the logs of run {self._gauge_run}.",
                  GaugeSource.PREDICTION: "From the prediction."}.get(self._gauge_source, "")  # fmt: skip
        parts = [source] if source and self.gauge_center() is not None else []
        parts += [f"<span style='color:#b00020'>{f}</span>" for f in self.gauge_flags()]
        self.view.gauge_note.setText(" ".join(parts))

    def run(self) -> None:
        """Analyze every loaded run in the background. Does nothing while inputs are missing or a job runs."""
        if self.missing() or self._worker is not None:
            return
        mesh = self.sample.mesh
        assert mesh is not None
        s, m = self.sample.model, self.model
        runs = self.pf_runs()
        jobs = [(r, self._candidates(r), self._mu(r)) for r in runs]
        center, conv = self.gauge_center(), s.conventions()
        assert center is not None
        inputs = self._inputs()
        self._cancel = False
        self._t_run = time.perf_counter()
        self._run_work = sum(r.n_subruns for r in runs) * m.mc_points * len(mesh.n)

        def job(progress: Callable[[float, str], None]) -> list[PolePeak]:
            out: list[PolePeak] = []
            for k, (r, cands, mu) in enumerate(jobs):

                def tick(frac: float, msg: str, *, k: int = k) -> None:
                    if self._cancel:
                        raise CancelledError
                    progress((k + frac) / len(jobs), f"Run {k + 1} of {len(jobs)}, {msg}")

                out += analyze_run(r, cands, mesh=mesh, center=center, conv=conv, receiving_slit_mm=m.receiving_slit_mm,
                                   mu=mu, cutoff=m.cutoff, n_points=m.mc_points, central_hw=m.central_hw,
                                   method=m.core_method, progress=tick)  # fmt: skip
            return out

        worker = Worker(job)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda res: self._on_done(res, inputs))
        worker.signals.failed.connect(self._on_failed)
        self._launch(worker, first_msg="Starting...", gif=False)

    def _launch(self, worker: Worker, *, first_msg: str, gif: bool) -> None:
        self._worker, self._gif_job = worker, gif
        bar, label, cancel = self._widgets()
        cancel.setEnabled(True)
        bar.setValue(0)
        label.setText(first_msg)
        self.update_readiness()
        QThreadPool.globalInstance().start(worker)

    def _widgets(self) -> tuple[Any, Any, Any]:
        """Progress bar, label and Cancel button of the running job (Analysis or the GIF section)."""
        v = self.view
        if self._gif_job:
            return v.gif_progress, v.gif_progress_label, v.gif_cancel_button
        return v.progress, v.progress_label, v.cancel_button

    def cancel(self) -> None:
        """Ask the running analysis or GIF job to stop at its next progress report."""
        if self._worker is not None:
            self._cancel = True
            self._widgets()[1].setText("Cancelling...")

    def _on_progress(self, frac: float, msg: str) -> None:
        bar, label, _ = self._widgets()
        bar.setValue(int(round(1000 * frac)))
        label.setText(msg)
        self.status.emit(f"{msg} ({100 * frac:.0f}% done)")

    def _finish(self, msg: str, ok: bool) -> None:
        bar, label, cancel = self._widgets()
        self._worker, self._gif_job = None, False
        cancel.setEnabled(False)
        bar.setValue(1000 if ok else 0)
        label.setText(msg)
        self.update_readiness()

    def _on_done(self, results: list[PolePeak], inputs: tuple[Any, ...]) -> None:
        self.results, self._result_inputs = results, inputs
        if self._run_work > 0:
            self._sec_per_point_tri = (time.perf_counter() - self._t_run) / self._run_work
        self._finish("Done." if results else "Done, but no central peaks were found.", True)
        self._show_results()
        self._new_pole_figures()
        self.update_readiness()
        self.analyzed.emit(results)
        self.pole_figures_changed.emit(self.pole_figures())
        self.status.emit(f"Analysis finished: {len(results)} pole figures")

    def _on_failed(self, tb: str) -> None:
        last = tb.strip().splitlines()[-1]
        what = "GIF" if self._gif_job else "Analysis"
        kept = self._take_partial_gifs()
        if "CancelledError" in last:
            self._finish("Cancelled." + (f" Kept the GIFs already made: {kept}." if kept else ""), False)
            return
        self._finish(f"<span style='color:#b00020'>Failed: {last}</span>", False)
        self.status.emit(f"{what} failed: {last}")

    def pole_figures(self) -> list[PolePeak]:
        """The pole figures to plot, export and send to MTEX: per run, or combined by hkl when asked."""
        if self.model.combine and self.results:
            return combine_by_hkl(self.results, scale=self.model.core_scale)
        return list(self.results)

    def _new_pole_figures(self) -> None:
        """The pole figures changed (analysis, cutoff or combine): GIFs and figures are stale, refresh the choices."""
        self.gifs.clear()
        self._bg.clear()
        labels = [r.label for r in self.pole_figures()]
        g = self.view.gif_pf
        keep = g.currentText()
        g.blockSignals(True)
        g.clear()
        g.addItems(labels)
        if keep in labels:
            g.setCurrentText(keep)
        g.blockSignals(False)
        self.view.gif_progress.setValue(0)
        self.view.gif_progress_label.setText("")
        self._update_gif_label()
        self._refresh_choices()

    def _refresh_choices(self) -> None:
        pfs = self.pole_figures()
        panel = self.view.plot_panel
        keep_type, keep = panel.current()
        panel.set_choices(PLOT_TYPES if pfs else [], [r.label for r in pfs], PER_PF)
        if pfs:
            panel.select(keep_type or TYPE_ALL, keep)

    def _show_results(self) -> None:
        lines = [f"{r.label}: 2theta {r.two_theta:.3f}, FWHM {r.fwhm:.2f}, {int(r.ignored.sum())} of {r.n_points} "
                 f"points ignored (correction > {r.cutoff:g}), correction {np.nanmin(r.correction):.2f} to "
                 f"{np.nanmax(r.correction[np.isfinite(r.correction)]):.1f}" for r in self.results]  # fmt: skip
        lines += [RED.format(f"{r.label}: {NEGATIVE_NOTE}") for r in self.results if mostly_negative(r.rate)]
        if self.model.combine:
            lines += [f"<b>{r.label}</b>: {r.notes}" for r in self.pole_figures() if r.source_runs]
        self.view.results_label.setText("<br>".join(lines))

    @property
    def interpolated(self) -> bool:
        """True when the toggle asks for interpolated pole figures instead of point cells."""
        return self.view.style_toggle.isChecked()

    def figure(self, plot_type: str, *, label: str = "", interpolated: bool) -> Figure | None:
        """
        Build a pole-figure plot of the current pole figures.

        Args:
            plot_type: TYPE_ALL or TYPE_ONE.
            label: Pole-figure label for TYPE_ONE (default: "").
            interpolated: Interpolated (True) or point cells (False). ``self.interpolated`` is the toggle's choice.

        Returns:
            The figure, or None without results or for an unknown label.
        """
        if not self.results:
            return None
        return self._builder(plot_type, label, interpolated)()

    def _builder(self, plot_type: str, label: str, interp: bool) -> Callable[[], Figure | None]:
        """A figure builder with its inputs taken now (on the GUI thread), safe to run on a worker thread."""
        s = self.sample.model
        view, vz_name = s.view_from, s.vz_name
        pfs = self.pole_figures()
        chosen = pfs if plot_type == TYPE_ALL else [r for r in pfs if r.label == label]
        return lambda: pole_figures_figure(chosen, interpolated=interp, view=view, vz_name=vz_name) if chosen else None

    def show_plot(self, plot_type: str, label: str = "") -> None:
        """
        Show a plot type (and pole figure) in the plot panel.

        Figures are built in the background (the panel says "Drawing ..." meanwhile) and cached.

        Args:
            plot_type: One of PLOT_TYPES.
            label: Pole-figure label for the per-pole-figure types (default: "").
        """
        panel = self.view.plot_panel
        self.view.style_toggle.setVisible(plot_type in STYLED)  # the toggle only where it changes the plot
        if plot_type == TYPE_GIF:
            self._bg.cancel()  # a figure still being drawn must not replace the GIF
            if label in self.gifs:
                panel.show_gif(self.gifs[label])
            elif label:
                self.view.gif_pf.setCurrentText(label)
                panel.show_message(f"No beam-path GIF for {label} yet.\n\nMake it in the Beam-path GIF section on "
                                   f"the left: the pole figure is set to {label}. Press Make GIF for this pole "
                                   "figure.")  # fmt: skip
            return
        if not self.results:
            return
        s = self.sample.model
        key = (plot_type, label, self.interpolated, s.view_from, s.vz_name)
        what = "the pole figures" if plot_type == TYPE_ALL else label
        self._bg.show(key, self._builder(plot_type, label, self.interpolated), name=what)

    def redraw(self) -> None:
        """Show the current plot again (after the point cells / interpolated toggle changed)."""
        t, label = self.view.plot_panel.current()
        if t and t != TYPE_GIF:
            self.show_plot(t, label)

    def gif_settings(self) -> GifSettings:
        """GIF frames per second, point step and dpi from the tab."""
        v = self.view
        return GifSettings(fps=float(v.gif_fps.value()), every=int(v.gif_every.value()), dpi=float(v.gif_dpi.value()))

    def _frames(self, r: PoleFigureLike) -> int:
        return len(frame_indices(len(r.chi), self.view.gif_every.value()))

    def gif_seconds(self, pfs: list[PolePeak]) -> float:
        """Estimated time (s) for the GIFs of these pole figures (from the last measured speed)."""
        scale = self._sec_per_frame_80 * (self.view.gif_dpi.value() / 80) ** 2
        return sum(self._frames(r) for r in pfs) * scale

    def _chosen_gif_pf(self) -> PolePeak | None:
        label = self.view.gif_pf.currentText()
        return next((r for r in self.pole_figures() if r.label == label), None)

    def _update_gif_label(self) -> None:
        pfs = self.pole_figures()
        if not pfs:
            self.view.gif_label.setText("Analyze runs first. The GIF uses the measured points and their corrections.")
            return
        one = self._chosen_gif_pf() or pfs[0]
        basis = "measured on the last GIF" if self._sec_per_frame_80 != SECONDS_PER_FRAME else "typical speed"
        self.view.gif_label.setText(
            f"{self._frames(one)} frames for {one.label}. About {format_duration(max(self.gif_seconds([one]), 1))} "
            f"for it, {format_duration(max(self.gif_seconds(pfs), 1))} for all {len(pfs)} pole figures ({basis})."
        )

    def _new_gif_path(self, label: str) -> Path:
        self._gif_count += 1
        return self._tmp_dir / gif_name(f"{label}_{self._gif_count}")

    def _gif_job_parts(self) -> tuple[Any, ...]:
        assert self.sample.mesh is not None
        s = self.sample.model
        center = self.gauge_center()
        assert center is not None
        return self.sample.mesh, center, self.gif_settings(), s

    def _start_gif_job(self, job: Callable[[Callable[[float, str], None]], Any], *, on_done: Callable[[Any], None],
                       first_msg: str) -> None:  # fmt: skip
        self._cancel = False

        def wrapped(progress: Callable[[float, str], None]) -> Any:
            def tick(frac: float, msg: str) -> None:
                if self._cancel:
                    raise CancelledError
                progress(frac, msg)

            return job(tick)

        worker = Worker(wrapped)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(on_done)
        worker.signals.failed.connect(self._on_failed)
        self._launch(worker, first_msg=first_msg, gif=True)

    def make_gif(self, label: str) -> None:
        """
        Make the beam-path GIF of one pole figure in the background, then show it.

        The GIF uses the run's own slits, cutoff and chi offset. Does nothing without results and an STL, or while a
        job runs.

        Args:
            label: Pole-figure label (the GIF section's choice is ``view.gif_pf.currentText()``).
        """
        if not self.results or self.sample.mesh is None or self._worker is not None:
            return
        r = next((x for x in self.pole_figures() if x.label == label), None)
        if r is None:
            return
        path = self._new_gif_path(label)
        mesh, center, st, s = self._gif_job_parts()

        def job(tick: Callable[[float, str], None]) -> tuple[Path, float, int]:
            t0 = time.perf_counter()
            n = measured_beam_gif(path, mesh, peak=r, center=center, settings=st, view=s.view_from,
                                  vz_name=s.vz_name, progress=tick)  # fmt: skip
            return path, time.perf_counter() - t0, n

        est = format_duration(max(self.gif_seconds([r]), 1))
        self._start_gif_job(
            job, on_done=lambda res: self._on_gif(label, res), first_msg=f"{label} GIF: starting (about {est})..."
        )

    def make_all_gifs(self) -> None:
        """Make the beam-path GIF of every pole figure, one after the other, in one background job."""
        if not self.results or self.sample.mesh is None or self._worker is not None:
            return
        pfs = self.pole_figures()
        n_pf = len(pfs)
        paths = {r.label: self._new_gif_path(r.label) for r in pfs}
        mesh, center, st, s = self._gif_job_parts()
        done: dict[str, Path] = {}
        self._partial_gifs = done
        est = format_duration(max(self.gif_seconds(pfs), 1))

        def job(tick: Callable[[float, str], None]) -> tuple[dict[str, Path], float, int]:
            t0, frames = time.perf_counter(), 0
            for k, r in enumerate(pfs):

                def sub(frac: float, msg: str, *, k: int = k) -> None:
                    tick((k + frac) / n_pf, f"Pole figure {k + 1} of {n_pf}, {msg}")

                frames += measured_beam_gif(paths[r.label], mesh, peak=r, center=center, settings=st,
                                            view=s.view_from, vz_name=s.vz_name, progress=sub)  # fmt: skip
                done[r.label] = paths[r.label]
            return dict(done), time.perf_counter() - t0, frames

        self._start_gif_job(job, on_done=self._on_all_gifs, first_msg=f"All {n_pf} GIFs: starting (about {est})...")

    def _learn_speed(self, seconds: float, frames: int) -> None:
        if frames > 0 and seconds > 0:
            self._sec_per_frame_80 = seconds / frames / (self.view.gif_dpi.value() / 80) ** 2

    def _take_partial_gifs(self) -> str:
        kept, self._partial_gifs = dict(self._partial_gifs), {}
        self.gifs.update(kept)
        return ", ".join(kept)

    def _on_gif(self, label: str, result: tuple[Path, float, int]) -> None:
        path, seconds, frames = result
        self._learn_speed(seconds, frames)
        self.gifs[label] = path
        self._finish(f"{label} GIF done in {format_duration(seconds)}. Use Save plot... to keep it.", True)
        self._update_gif_label()
        self.view.plot_panel.select(TYPE_GIF, label)
        self.status.emit(f"Beam-path GIF for {label} ready")

    def _on_all_gifs(self, result: tuple[dict[str, Path], float, int]) -> None:
        made, seconds, frames = result
        self._partial_gifs = {}
        self._learn_speed(seconds, frames)
        self.gifs.update(made)
        self._finish(f"All {len(made)} GIFs done in {format_duration(seconds)}. Use Save all GIFs... to keep them.",
                     True)  # fmt: skip
        self._update_gif_label()
        self.view.plot_panel.select(TYPE_GIF, next(iter(made)))
        self.status.emit(f"All {len(made)} beam-path GIFs ready")

    def save_all_gifs(self, folder: str | Path) -> list[Path]:
        """
        Copy every GIF made from the current pole figures to ``folder`` as beam_paths_<run>_<hkl>.gif.

        Args:
            folder: Destination folder, created if needed. Files of the same name are replaced.

        Returns:
            The files written.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for label, src in self.gifs.items():
            dst = folder / gif_name(label)
            shutil.copyfile(src, dst)
            out.append(dst)
        return out

    def _on_save_all_gifs(self) -> None:
        folder = self.view.ask_folder("Folder for the GIFs", self._last_dir)
        if folder:
            paths = self.save_all_gifs(folder)
            self._last_dir = folder
            self.status.emit(f"Saved {len(paths)} GIFs to {folder}")

    def export(self, folder: str | Path) -> list[Path]:
        """
        Write PF's outputs: pole_figure_<run>_<hkl>.csv, pole_figures.png and pole_figures_interpolated.png.

        Args:
            folder: Destination folder.

        Returns:
            The files written.
        """
        folder = Path(folder)
        paths = write_all(folder, self.pole_figures())
        for name, interp in (("pole_figures.png", False), ("pole_figures_interpolated.png", True)):
            fig = self.figure(TYPE_ALL, interpolated=interp)
            assert fig is not None
            save_figure(fig, folder / name, dpi=130, bbox_inches="tight")
            paths.append(folder / name)
        return paths

    def _on_export(self) -> None:
        folder = self.view.ask_folder("Folder for the pole-figure CSVs and figures", self._last_dir)
        if folder:
            paths = self.export(folder)
            self.view.progress_label.setText(f"Exported {len(paths)} files to {folder}.")
            self.status.emit(f"Exported {len(paths)} files to {folder}")
