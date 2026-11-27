"""
Predict tab: grid and sampling, planned planes, cutoff, run, plot options, beam-path GIF, and the plot panel.

The predict presenter lists the planes of the chosen reflections with :meth:`PredictTab.set_planes`, reads them back
(with any edited 2theta and omega) through :meth:`PredictTab.planes`, runs the prediction and shows the plots.

Example:
    >>> from pyrs.pole_figures.core.prediction import Plane
    >>> tab = PredictTab()  # doctest: +SKIP
    >>> tab.set_planes([Plane("111", 50.1, 25.05)])  # doctest: +SKIP
    >>> tab.planes(["111"])  # doctest: +SKIP
    [Plane(hkl='111', two_theta=50.1, omega=25.05)]
"""

from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrs.pole_figures.core.correction import CorrectionPart
from pyrs.pole_figures.core.hb2b import TWO_THETA_MAX, TWO_THETA_MIN
from pyrs.pole_figures.core.prediction import Plane
from ..models.predict import GRID_FIELDS, MC_FIELDS, PredictModel
from ..widgets.model_form import ModelForm
from ..widgets.plot_panel import PlotPanel

PLANE_COLUMNS = ["hkl", "2theta (deg)", "omega (deg)"]


def spin_box(value: float, *, lo: float, hi: float, decimals: int = 2) -> QDoubleSpinBox:
    """
    A float spin box that reports a value when the user finishes typing.

    Args:
        value: Start value.
        lo: Smallest value.
        hi: Largest value.
        decimals: Decimals shown (default: 2).

    Returns:
        The spin box.
    """
    s = QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


def int_spin_box(value: int, lo: int, hi: int) -> QSpinBox:
    """
    An integer spin box that reports a value when the user finishes typing.

    Args:
        value: Start value.
        lo: Smallest value.
        hi: Largest value.

    Returns:
        The spin box.
    """
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


class PredictTab(QWidget):
    """
    Widgets of the Predict tab.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        grid_form: chi and phi steps (deg) and equal spacing.
        mc_form: Monte Carlo points and the free-phi option.
        planes_table: One row per plane: hkl, then 2theta and omega spin boxes (deg).
        cutoff_spin: Points with a correction above this are ignored.
        predict_button: Starts the prediction.
        save_csv_button: Saves the predicted correction table.
        part_boxes: Which parts the correction vs phi plot shows: "total", "volume", "attenuation".
        gif_plane: Plane of the beam-path GIF.
        gif_fps: GIF frames per second.
        gif_every: Use every n-th grid point in the GIF.
        gif_dpi: GIF resolution (dpi).
        plot_panel: The plots.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid_form = ModelForm(PredictModel, GRID_FIELDS, decimals=2)
        self.mc_form = ModelForm(PredictModel, MC_FIELDS)
        self.forms = [self.grid_form, self.mc_form]
        self.grid_label = QLabel("")
        self.ready_label = QLabel("")
        self.ready_label.setWordWrap(True)

        self.planes_table = QTableWidget(0, len(PLANE_COLUMNS))
        self.planes_table.setHorizontalHeaderLabels(PLANE_COLUMNS)
        self.planes_table.verticalHeader().setVisible(False)
        self.planes_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.planes_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.planes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.planes_table.setMinimumHeight(160)
        self.reset_planes_button = QPushButton("Reset from Experiment setup (omega = theta)")

        self.cutoff_spin = spin_box(3.0, lo=1.01, hi=1000.0)
        self.cutoff_spin.setToolTip("Points whose correction is above this are ignored. Redraws without recomputing.")
        self.sphere_label = QLabel("")
        self.sphere_label.setWordWrap(True)

        self.predict_button = QPushButton("Predict corrections")
        self.predict_button.setStyleSheet("font-weight: bold")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.save_csv_button = QPushButton("Save CSV...")
        self.save_csv_button.setEnabled(False)

        self.part_boxes: dict[CorrectionPart, QCheckBox] = {
            "total": QCheckBox("Total correction (solid lines, points)"),
            "volume": QCheckBox("Gauge volume in the sample (dashed)"),
            "attenuation": QCheckBox("Attenuation, exp(-mu path) (dotted)"),
        }
        for box in self.part_boxes.values():
            box.setChecked(True)

        self.gif_plane = QComboBox()
        self.gif_fps = spin_box(8.0, lo=0.5, hi=30.0, decimals=1)
        self.gif_every = int_spin_box(1, 1, 100)
        self.gif_dpi = int_spin_box(80, 40, 200)
        self.gif_label = QLabel("")
        self.gif_label.setWordWrap(True)
        self.gif_button = QPushButton("Make GIF for this plane")
        self.gif_button.setEnabled(False)
        self.gif_all_button = QPushButton("Make GIFs for all planes")
        self.gif_all_button.setEnabled(False)
        self.gif_save_all_button = QPushButton("Save all GIFs...")
        self.gif_save_all_button.setEnabled(False)
        self.gif_cancel_button = QPushButton("Cancel")
        self.gif_cancel_button.setEnabled(False)
        self.gif_progress = QProgressBar()
        self.gif_progress.setRange(0, 1000)
        self.gif_progress_label = QLabel("")
        self.gif_progress_label.setWordWrap(True)

        self.plot_panel = PlotPanel()
        self._build_layout()

    def _build_layout(self) -> None:
        grid_box = QGroupBox("chi/phi grid and gauge sampling")
        gv = QVBoxLayout(grid_box)
        gv.addWidget(self.grid_form)
        gv.addWidget(self.mc_form)
        gv.addWidget(self.grid_label)

        planes_box = QGroupBox("Planes (from the selected reflections)")
        pv = QVBoxLayout(planes_box)
        pv.addWidget(self.planes_table)
        pv.addWidget(self.reset_planes_button)

        cut_box = QGroupBox("Correction cutoff")
        cf = QFormLayout(cut_box)
        cf.addRow("Ignore if correction >", self.cutoff_spin)
        cf.addRow(self.sphere_label)

        run_box = QGroupBox("Run")
        rv = QVBoxLayout(run_box)
        row = QHBoxLayout()
        row.addWidget(self.predict_button, stretch=1)
        row.addWidget(self.cancel_button)
        rv.addLayout(row)
        rv.addWidget(self.progress)
        rv.addWidget(self.progress_label)
        rv.addWidget(self.save_csv_button)

        self.parts_box = parts_box = QGroupBox("Correction vs phi plot: parts shown")
        av = QVBoxLayout(parts_box)
        for box in self.part_boxes.values():
            av.addWidget(box)
        note = QLabel("Total = gauge volume part x attenuation part.")
        note.setStyleSheet("color: #555")
        av.addWidget(note)

        gif_box = QGroupBox("Beam-path GIF")
        gf = QFormLayout(gif_box)
        gf.addRow("Plane", self.gif_plane)
        gf.addRow("Frames per second", self.gif_fps)
        gf.addRow("Use every n-th point", self.gif_every)
        gf.addRow("Resolution (dpi)", self.gif_dpi)
        gf.addRow(self.gif_label)
        gif_row = QHBoxLayout()
        gif_row.addWidget(self.gif_button, stretch=1)
        gif_row.addWidget(self.gif_all_button, stretch=1)
        gf.addRow(gif_row)
        gif_row2 = QHBoxLayout()
        gif_row2.addWidget(self.gif_save_all_button, stretch=1)
        gif_row2.addWidget(self.gif_cancel_button)
        gf.addRow(gif_row2)
        gf.addRow(self.gif_progress)
        gf.addRow(self.gif_progress_label)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        for w in (grid_box, self.ready_label, planes_box, cut_box, run_box, parts_box, gif_box):
            lv.addWidget(w)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(370)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.plot_panel)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def set_planes(self, planes: list[Plane]) -> None:
        """One row per plane, with 2theta and omega (deg) in spin boxes. omega may differ from theta."""
        self.planes_table.setRowCount(len(planes))
        for i, p in enumerate(planes):
            self.planes_table.setItem(i, 0, QTableWidgetItem(f"{{{p.hkl}}}"))
            self.planes_table.setCellWidget(i, 1, spin_box(p.two_theta, lo=TWO_THETA_MIN, hi=TWO_THETA_MAX))
            self.planes_table.setCellWidget(i, 2, spin_box(p.omega, lo=0.5, hi=179.0))  # omega may differ from theta

    def planes(self, labels: list[str]) -> list[Plane]:
        """
        The planes as edited.

        Args:
            labels: hkl of each row, in table order.

        Returns:
            One plane per row that has both spin boxes, with the shown 2theta and omega (deg).
        """
        out = []
        for i, hkl in enumerate(labels):
            tth, om = self.planes_table.cellWidget(i, 1), self.planes_table.cellWidget(i, 2)
            if isinstance(tth, QDoubleSpinBox) and isinstance(om, QDoubleSpinBox):
                out.append(Plane(hkl, tth.value(), om.value()))
        return out

    def plane_spins(self) -> list[QDoubleSpinBox]:
        """Every 2theta and omega spin box of the planes table."""
        out: list[QDoubleSpinBox] = []
        for i in range(self.planes_table.rowCount()):
            for j in (1, 2):
                w = self.planes_table.cellWidget(i, j)
                if isinstance(w, QDoubleSpinBox):
                    out.append(w)
        return out

    def parts(self) -> list[CorrectionPart]:
        """The checked parts of the correction vs phi plot: "total", "volume" and/or "attenuation"."""
        return [k for k, box in self.part_boxes.items() if box.isChecked()]

    def ask_folder(self, title: str, start: str) -> str:
        """
        Ask for a folder.

        Args:
            title: Dialog title.
            start: Folder the dialog starts in.

        Returns:
            The folder, or "" when the user cancels.
        """
        return str(QFileDialog.getExistingDirectory(self, title, start))

    def ask_save_path(self, title: str, start: str, filt: str) -> str:
        """
        Ask for a file to save.

        Args:
            title: Dialog title.
            start: Suggested path.
            filt: Qt file filter, for example "CSV (*.csv)".

        Returns:
            The path, or "" when the user cancels.
        """
        path, _ = QFileDialog.getSaveFileName(self, title, start, filt)
        return str(path)
