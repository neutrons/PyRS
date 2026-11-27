"""
Table scan presenter: plan one table scan per predicted plane, preview it, and write the .xlsx files.

Typical workflow:

1. After each prediction the planes are listed and the cutoff is copied from the Predict tab.
2. The user sets the base count time per plane and the count-time rule. The plan and its totals update.
3. :meth:`TableScanPresenter.write` writes one workbook per plane.

Example:
    >>> presenter = TableScanPresenter(TableScanTab(), predict_presenter, model=TableScanModel())  # doctest: +SKIP
    >>> paths = presenter.write("scans")  # doctest: +SKIP
"""

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from pyrs.pole_figures.core.table_scan import PlaneScan, phi_travel, plan_scan
from pyrs.pole_figures.io.table_scan import scan_filename, write_table_scans
from pyrs.pole_figures.plots.table_scan import table_scan_figure
from pyrs.pole_figures.text import format_duration
from ..models.table_scan import TableScanModel
from ..tabs.table_scan_tab import TableScanTab
from .predict import PredictPresenter

PREVIEW = "Table scan points"


class TableScanPresenter(QObject):
    """
    Logic of the Table scan tab.

    Args:
        view: The Table scan tab.
        predict: The Predict presenter (prediction, sample and material).
        model: Starting inputs (``TableScanModel()`` for the defaults).

    Attributes:
        status: Signal with a short message for the status bar.
        written: Signal with the paths (str) of the table-scan files written.
        view: The Table scan tab.
        predict: The Predict presenter.
        model: The current inputs.
    """

    status = Signal(str)
    written = Signal(list)

    def __init__(self, view: TableScanTab, predict: PredictPresenter, *, model: TableScanModel) -> None:
        super().__init__(view)
        self.view, self.predict = view, predict
        self.model = model
        self._last_dir = str(Path.cwd())
        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        view.write_button.clicked.connect(self._on_write)
        view.plot_panel.selected.connect(lambda _t, plane: self.show_preview(plane))
        view.plot_panel.saved.connect(lambda p: self.status.emit(f"Saved {p}"))
        predict.predicted.connect(self._on_predicted)
        predict.inputs_changed.connect(self.refresh)
        self._update_cap_enabled()
        self.refresh()

    def _on_form_changed(self) -> None:
        vals: dict[str, Any] = {}
        for form in self.view.forms:
            vals.update(form.values())
        try:
            self.model = TableScanModel.model_validate({**self.model.model_dump(), **vals})
        except ValidationError as exc:
            self.status.emit(f"Invalid input: {exc.errors()[0]['msg']}")
            return
        self._update_cap_enabled()
        self.refresh()
        self._redraw()

    def _update_cap_enabled(self) -> None:
        self.view.settings_form.widgets["max_s"].setEnabled(self.model.cap_on)

    def _on_predicted(self, _pred: object) -> None:
        """New prediction: list its planes and start from the Predict cutoff."""
        res = self.predict.result
        assert res is not None
        self.model = self.model.model_copy(update={"cutoff": self.predict.cutoff()})
        self.view.settings_form.set_values({"cutoff": self.model.cutoff})
        self.view.set_planes([(q.plane.hkl, q.plane.two_theta, q.plane.omega) for q in res.planes])
        for spin in self.view.base_spins():
            spin.valueChanged.connect(self._on_base)
        panel = self.view.plot_panel
        _t, keep = panel.current()
        panel.set_choices([PREVIEW], [f"{{{q.plane.hkl}}}" for q in res.planes], {PREVIEW})
        panel.select(PREVIEW, keep)
        self.refresh()

    def _on_base(self) -> None:
        self.refresh()
        self._redraw()

    def scans(self) -> list[PlaneScan]:
        """The planned table scan per plane, in logged chi (the prediction's mount offset), or empty."""
        res = self.predict.result
        if res is None or len(self.view.bases()) != len(res.planes):
            return []
        m = self.model
        return [plan_scan(q, cutoff=m.cutoff, base_s=b, serpentine=m.serpentine, scaled=m.scaled, max_s=m.cap)
                for q, b in zip(res.planes, self.view.bases(), strict=True)]  # fmt: skip

    def refresh(self) -> None:
        """Update the summary and plan columns. Writing is off when the prediction is stale or no point is kept."""
        v = self.view
        res = self.predict.result
        if res is None:
            v.ready_label.setText("<b>Predict first.</b> The table scans use the predicted grid and corrections.")
            v.summary.setText("")
            v.write_button.setEnabled(False)
            return
        stale = self.predict.stale()
        g = res.grid
        txt = (f"From the prediction: {len(res.planes)} planes on a {g.describe()} chi/phi grid "
               f"({len(res.planes[0].chi)} points per plane).")  # fmt: skip
        if stale:
            txt += ("<br><span style='color:#b00020'>Inputs changed since the prediction. Predict again before "
                    "writing table scans.</span>")  # fmt: skip
        v.ready_label.setText(txt)
        scans = self.scans()
        for i, sc in enumerate(scans):
            v.set_row(
                i,
                kept=f"{len(sc.points)} of {sc.n_grid}",
                total=format_duration(sc.total_s),
                travel=f"{phi_travel(sc):,.0f}",
            )
        total = sum(sc.total_s for sc in scans)
        n = sum(len(sc.points) for sc in scans)
        capped = sum(sc.n_capped for sc in scans)
        msg = (f"<b>{len(scans)} files, {n} points, count time {format_duration(total)}</b> in all "
               "(without motor moves).")  # fmt: skip
        if capped:
            msg += f" {capped} points are at the cap."
        v.summary.setText(msg)
        v.write_button.setEnabled(not stale and n > 0)

    def show_preview(self, plane: str) -> None:
        """Show the planned points of a plane, given as "{hkl}" or "hkl". Unknown planes are ignored."""
        res = self.predict.result
        hkl = plane.strip("{}")
        hkls = [q.plane.hkl for q in res.planes] if res is not None else []
        if res is None or hkl not in hkls:
            return
        i = hkls.index(hkl)
        s = self.predict.sample.model
        self.view.plot_panel.show_figure(table_scan_figure(res, i, scan=self.scans()[i], view=s.view_from,
                                                           vz_name=s.vz_name))  # fmt: skip

    def _redraw(self) -> None:
        _t, plane = self.view.plot_panel.current()
        if plane:
            self.show_preview(plane)

    def write(self, folder: str | Path) -> list[Path]:
        """
        Write one table-scan workbook per plane into ``folder``, replacing files of the same name.

        Args:
            folder: Destination folder.

        Returns:
            The files written.
        """
        paths = write_table_scans(folder, self.scans(), prefix=self.model.run_title,
                                  phase_name=self.predict.material.model.name)  # fmt: skip
        self.written.emit([str(p) for p in paths])
        return paths

    def _on_write(self) -> None:
        folder = self.view.ask_folder("Folder for the table scans", self._last_dir)
        if not folder:
            return
        names = [scan_filename(self.model.run_title, sc.plane.hkl) for sc in self.scans()]
        existing = [n for n in names if (Path(folder) / n).exists()]
        if existing and not self.view.confirm(
            "Replace files?", "These files exist and will be replaced:\n" + "\n".join(existing)
        ):
            return
        paths = self.write(folder)
        self._last_dir = folder
        self.view.summary.setText(self.view.summary.text() + f"<br>Wrote {len(paths)} files to {folder}.")
        self.status.emit(f"Wrote {len(paths)} table scans to {folder}")
