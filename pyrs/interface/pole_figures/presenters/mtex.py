"""
MTEX presenter: take the corrected pole figures from Data and Corrections, fit the ODF (MTEX for Python), show it.

Typical workflow:

1. The data presenter sends its pole figures. Those with at least two chi rings start checked.
2. :meth:`MtexPresenter.run` writes the inputs to a temporary work folder and runs the ODF fit and plots in a worker,
   with step-by-step progress and a time estimate.
3. The plot panel shows the MTEX pictures. :meth:`MtexPresenter.save_results` and
   :meth:`MtexPresenter.save_plot_data` copy the outputs out.

MTEX for Python (pymtex) is an optional dependency. Without it the tab shows ``PYMTEX_MISSING`` and Run stays off.

Example:
    >>> presenter = MtexPresenter(MtexTab(), data_presenter, sample=sample_presenter, material=material_presenter,
    ...                           model=MtexModel())  # doctest: +SKIP
    >>> presenter.run()  # doctest: +SKIP
"""

import atexit
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qtpy.QtCore import QObject, QThreadPool, QTimer, Signal  # type: ignore[attr-defined]

from pyrs.pole_figures.core import odf as core_odf
from pyrs.pole_figures.core.mtex import chi_rings, laue_group, plot_data_files, plot_files, usable
from pyrs.pole_figures.core.odf import STAGE_KEYS, STAGES, stage_progress
from pyrs.pole_figures.core.polefig import PolePeak
from pyrs.pole_figures.io.odf import run_odf
from pyrs.pole_figures.text import format_duration
from ..models.mtex import AUTO, MtexModel
from ..tabs.mtex_tab import MtexTab
from ..worker import Worker
from .data import DataPresenter
from .material import MaterialPresenter
from .sample import SamplePresenter

#: the note shown on the tab when MTEX for Python (pymtex) cannot be imported
PYMTEX_MISSING = (
    "MTEX for Python (pymtex) is not installed. It is installed separately on the analysis cluster, so the ODF "
    "step can run there. To use it elsewhere, install the PyPI package mtex with finufft and numba from "
    "conda-forge. The other tabs work without it."
)


class MtexPresenter(QObject):
    """
    Logic of the MTEX tab.

    Args:
        view: The MTEX tab.
        data: The Data and Corrections presenter (source of the pole figures).
        sample: The Sample presenter (pole-figure view direction and the name of +vz).
        material: The Experiment setup presenter (phase, space group, lattice).
        model: Starting inputs (``MtexModel()`` for the defaults).

    Attributes:
        status: Signal with a short message for the status bar.
        finished: Signal with the path of the output (mtex) folder when MTEX finished.
        view: The MTEX tab.
        model: The current inputs.
        pole_figures: The pole figures from Data and Corrections.
        out_dir: The mtex output folder of the last successful run, or None.
        pymtex: True when MTEX for Python (pymtex) can be imported. Without it the tab shows ``PYMTEX_MISSING``.
    """

    status = Signal(str)
    finished = Signal(str)

    def __init__(self, view: MtexTab, data: DataPresenter, *, sample: SamplePresenter, material: MaterialPresenter,
                 model: MtexModel) -> None:  # fmt: skip
        super().__init__(view)
        self.view, self.data, self.sample, self.material = view, data, sample, material
        self.model = model
        self.pole_figures: list[PolePeak] = []
        self.out_dir: Path | None = None  # <work>/mtex of the last successful run
        self._used: list[str] = []
        self._worker: Worker | None = None
        self._cancel = False
        self._last_dir = str(Path.cwd())
        self._tmp_root = Path(tempfile.mkdtemp(prefix="hidra_mtex_"))
        atexit.register(shutil.rmtree, self._tmp_root, ignore_errors=True)
        self._runs = 0
        self._stage: str | None = None
        self._t0 = self._t_stage = 0.0
        self._speed = 1.0  # actual / typical time of the steps done so far
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._show_progress)

        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        view.run_button.clicked.connect(self.run)
        view.cancel_button.clicked.connect(self.cancel)
        view.save_button.clicked.connect(self._on_save)
        view.save_data_button.clicked.connect(self._on_save_data)
        view.pf_table.itemChanged.connect(lambda _i: self.update_readiness())
        view.plot_panel.selected.connect(lambda t, _p: self.show_plot(t))
        view.white_toggle.toggled.connect(lambda _on: self.show_plot(view.plot_panel.current()[0]))
        view.plot_panel.saved.connect(lambda p: self.status.emit(f"Saved {p}"))
        view.plot_panel.plane_label.setText("")
        data.pole_figures_changed.connect(self.set_pole_figures)
        material.changed.connect(self.update_readiness)
        self.pymtex = core_odf.pymtex_installed()
        view.show_install_note(PYMTEX_MISSING if not self.pymtex else "")
        self.set_pole_figures(data.pole_figures())

    def _on_form_changed(self) -> None:
        vals: dict[str, Any] = {}
        for form in self.view.forms:
            vals.update(form.values())
        try:
            self.model = MtexModel.model_validate({**self.model.model_dump(), **vals})
        except ValidationError as exc:
            self.status.emit(f"Invalid input: {exc.errors()[0]['msg']}")
            return
        self.update_readiness()

    def set_pole_figures(self, pfs: list[PolePeak]) -> None:
        """
        List new corrected pole figures from Data and Corrections. Pole figures MTEX can use start checked.

        Args:
            pfs: The pole figures.
        """
        self.pole_figures = list(pfs)
        rows = []
        for r in self.pole_figures:
            rings = chi_rings(r)
            note = "" if usable(r) else "fewer than 2 chi rings: MTEX cannot use it"
            if r.source_runs:
                note = (note + "; " if note else "") + f"combined from {len(r.source_runs)} runs"
            rows.append((r.label, int((~r.ignored).sum()), r.n_points, rings, note, usable(r)))
        self.view.pf_table.blockSignals(True)
        self.view.set_pole_figures(rows)
        self.view.pf_table.blockSignals(False)
        self.update_readiness()

    def selected(self) -> list[PolePeak]:
        """The checked pole figures."""
        labels = set(self.view.checked_labels())
        return [r for r in self.pole_figures if r.label in labels]

    def crystal_symmetry(self) -> str:
        """
        The chosen Laue class, or (Auto) the Laue class of the phase's space group.

        Raises:
            ValueError: If Auto is chosen and the edited phase is not valid.
        """
        if self.model.crystal_symmetry != AUTO:
            return self.model.crystal_symmetry
        return laue_group(self.material.phase().space_group)

    def missing(self) -> list[str]:
        """What is still needed before MTEX can run (empty when ready)."""
        out = []
        if not self.pymtex:
            out.append("install MTEX for Python (pymtex)")
        if not self.pole_figures:
            out.append("analyze runs on the Data and Corrections tab")
        elif len(self.selected()) < 2:
            out.append("select at least two usable pole figures")
        if not self.material.has_material():
            out.append("choose a material on the Experiment setup tab")
        try:
            self.model.options("m-3m")
        except ValueError as exc:
            out.append(str(exc))
        return out

    def update_readiness(self) -> None:
        """Show the crystal symmetry in use, enable the buttons that can run now and say what is missing."""
        v = self.view
        busy = self._worker is not None
        if self.material.has_material():
            try:
                sg = self.material.phase().space_group
                auto = laue_group(sg)
                cs = self.crystal_symmetry()
                note = "" if cs == auto else (f" <span style='color:#b00020'>The space group gives {auto}; "
                                              f"{cs} is your override.</span>")  # fmt: skip
                v.symmetry_label.setText(f"Crystal symmetry {cs} (space group {sg}).{note}")
            except (ValueError, RuntimeError):
                v.symmetry_label.setText("")
        missing = self.missing()
        v.run_button.setEnabled(not missing and not busy)
        v.save_button.setEnabled(self.out_dir is not None and not busy)
        v.save_data_button.setEnabled(self.out_dir is not None and bool(plot_data_files(self.out_dir)) and not busy)
        if missing:
            v.ready_label.setText("<b>Not ready:</b> " + "; ".join(missing) + ".")
            return
        sel = self.selected()
        txt = f"<b>Ready.</b> {len(sel)} pole figures: {', '.join(r.label for r in sel)}. About 10 s"
        txt += " (MTEX for Python; the first run also compiles its kernels)."
        if self.out_dir is not None and self._used != [r.label for r in sel]:
            txt += "<br><span style='color:#b00020'>The pole figures changed since the last MTEX run.</span>"
        v.ready_label.setText(txt)

    def run(self) -> None:
        """Fit the ODF and make the plots in the background. Does nothing while inputs are missing or a job runs."""
        if self.missing() or self._worker is not None:
            return
        phase = self.material.phase()
        opts = self.model.options(self.crystal_symmetry())
        s = self.sample.model
        sel = self.selected()
        self._runs += 1
        work = self._tmp_root / f"run{self._runs}"
        name = self.material.model.name or phase.name
        self._cancel = False

        def job(progress: Any) -> Path:
            return run_odf(work, sel, phase_name=name, cell=phase.cell, view_from=s.view_from, vz_name=s.vz_name,
                           opts=opts, on_stage=lambda key: progress(float(STAGE_KEYS.index(key)), key),
                           should_stop=lambda: self._cancel)  # fmt: skip

        worker = Worker(job)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda out: self._on_done(out, [r.label for r in sel]))
        worker.signals.failed.connect(self._on_failed)
        self._worker = worker
        self.view.log.clear()
        self.view.cancel_button.setEnabled(True)
        self.view.set_progress(0.0)
        self.view.status_label.setText("Writing the CSVs and settings...")
        self._stage, self._speed = None, 1.0
        self.update_readiness()
        QThreadPool.globalInstance().start(worker)

    def cancel(self) -> None:
        """Ask the running job to stop. It stops after the current step."""
        if self._worker is not None:
            self._cancel = True
            self.view.status_label.setText("Cancelling (stops after the current step)...")

    def _on_progress(self, code: float, text: str) -> None:
        """A log line (code -1) or the start of a step (code = its index, text = its key)."""
        if code < 0:
            self.view.log.appendPlainText(text)
            return
        self._on_stage(text)

    def _on_stage(self, key: str) -> None:
        now = time.monotonic()
        if key == "start":
            self._t0 = now
        elif self._stage is not None:  # learn the speed from the steps done (the first run compiles kernels)
            typical = sum(t for _, _, t in STAGES[: STAGE_KEYS.index(key)])
            if typical > 0:
                self._speed = max((now - self._t0) / typical, 0.2)
        self._stage, self._t_stage = key, now
        self._show_progress()
        if not self._timer.isActive():
            self._timer.start()
        label = STAGES[STAGE_KEYS.index(key)][1]
        self.view.log.appendPlainText(f"{format_duration(now - self._t0):>8}  {label}")
        self.status.emit(f"MTEX: {label}")

    def _show_progress(self) -> None:
        """Bar and text for the current step (also called every 0.5 s while the ODF runs)."""
        if self._stage is None:
            return
        now = time.monotonic()
        i = STAGE_KEYS.index(self._stage)
        frac, left = stage_progress(self._stage, seconds_in_stage=now - self._t_stage, speed=self._speed)
        self.view.set_progress(frac)
        self.view.status_label.setText(
            f"Step {i + 1} of {len(STAGES)}: {STAGES[i][1]}... {format_duration(now - self._t0)} elapsed, "
            f"about {format_duration(max(left, 1.0))} left."
        )

    def _finish(self, msg: str, ok: bool = False) -> None:
        self._timer.stop()
        self._stage = None
        self._worker = None
        self.view.cancel_button.setEnabled(False)
        self.view.set_progress(1.0 if ok else 0.0)
        self.view.status_label.setText(msg)
        self.update_readiness()

    def _on_done(self, out: Path, used: list[str]) -> None:
        self.out_dir, self._used = Path(out), used
        files = plot_files(self.out_dir, phi2_deg=self.model.phi2_deg)
        summary = self.out_dir / "odf_summary.txt"
        self.view.summary.setPlainText(summary.read_text(encoding="utf-8") if summary.exists() else "")
        took = format_duration(time.monotonic() - self._t0) if self._t0 else ""
        n_csv = sum(p.suffix == ".csv" for p in plot_data_files(self.out_dir))
        data = f", {n_csv} plot-data CSVs" if n_csv else ""
        self._finish(f"MTEX finished: {len(files)} plots{data}" + (f" in {took}." if took else "."), ok=True)
        panel = self.view.plot_panel
        keep, _ = panel.current()
        names = [n for n, _, _ in files]
        panel.set_choices(names, [], set())
        if names:
            panel.select(keep if keep in names else names[0])
        self.finished.emit(str(self.out_dir))
        self.status.emit("MTEX finished")

    def _on_failed(self, tb: str) -> None:
        msg = tb.strip().splitlines()[-1]
        if "cancelled" in msg:
            self._finish("Cancelled.")
            return
        for line in tb.strip().splitlines()[-8:]:
            self.view.log.appendPlainText(line)
        self._finish(f"<span style='color:#b00020'>MTEX failed: {msg}</span>")
        self.status.emit(f"MTEX failed: {msg}")

    def show_plot(self, name: str) -> None:
        """Show an MTEX plot by name, in its white-below-random version when the toggle asks for it."""
        if self.out_dir is None:
            return
        for n, path, white in plot_files(self.out_dir, phi2_deg=self.model.phi2_deg):
            if n == name:
                # the white-below-random toggle only for plots that have that version
                self.view.white_toggle.setVisible(white is not None)
                use_white = white is not None and self.view.white_toggle.isChecked()
                self.view.plot_panel.show_image(white if use_white and white is not None else path)
                return

    def save_results(self, folder: str | Path) -> Path:
        """
        Copy the CSVs and the mtex folder (plots, summary, ODF table, settings) of the last run to folder.

        Args:
            folder: Destination folder. Files of the same name are replaced.

        Returns:
            The destination folder.
        """
        assert self.out_dir is not None
        dst = Path(folder)
        shutil.copytree(self.out_dir.parent, dst, dirs_exist_ok=True)
        return dst

    def save_plot_data(self, folder: str | Path) -> list[Path]:
        """
        Copy plot_data/*.csv and README.txt of the last run to folder.

        Args:
            folder: Destination folder, created if needed.

        Returns:
            The copied files.
        """
        assert self.out_dir is not None
        dst = Path(folder)
        dst.mkdir(parents=True, exist_ok=True)
        out = []
        for src in plot_data_files(self.out_dir):
            shutil.copyfile(src, dst / src.name)
            out.append(dst / src.name)
        return out

    def _on_save_data(self) -> None:
        folder = self.view.ask_folder("Folder for the MTEX plot data (CSV)", self._last_dir)
        if folder:
            paths = self.save_plot_data(folder)
            self._last_dir = folder
            self.status.emit(f"Saved {len(paths)} plot-data files to {folder}")
            self.view.status_label.setText(f"Saved {len(paths)} plot-data files to {folder} (see README.txt).")

    def _on_save(self) -> None:
        folder = self.view.ask_folder("Folder for the MTEX results", self._last_dir)
        if folder:
            self.save_results(folder)
            self._last_dir = folder
            self.status.emit(f"Saved the MTEX results to {folder}")
            self.view.status_label.setText(f"Saved to {folder} (CSVs, and plots and summary in mtex/).")
