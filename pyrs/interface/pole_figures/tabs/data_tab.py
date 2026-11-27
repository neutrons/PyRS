"""
Data and Corrections tab: reduced runs, settings, analysis, export, beam-path GIFs and the pole-figure plots.

The data presenter adds the reduced files chosen with :meth:`DataTab.ask_files`, lists them with
:meth:`DataTab.set_runs`, analyzes them and shows the raw and corrected pole figures.

Example:
    >>> tab = DataTab()  # doctest: +SKIP
    >>> paths = tab.ask_files(start=".")  # doctest: +SKIP
"""

from collections.abc import Collection

from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models.data import COMBINE_FIELDS, GAUGE_FIELDS, SETTING_FIELDS, DataModel
from ..widgets.model_form import ModelForm
from ..widgets.plot_panel import PlotPanel
from ..widgets.toggle_switch import LabeledToggle
from .predict_tab import int_spin_box, spin_box

FLAG_COLOR = "#b00020"  # red of the warnings
RUN_COLUMNS = ["Run", "Sub-runs", "lambda (A)", "omega", "arm 2theta", "Slits (mm)", "mu (1/mm)", "Peaks",
               "vx, vy, vz (mm)", "Title"]  # fmt: skip


class DataTab(QWidget):
    """
    Widgets of the Data and Corrections tab.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        add_button: Adds reduced files.
        remove_button: Removes the selected runs.
        clear_button: Removes every run.
        runs_table: One row per run (columns in RUN_COLUMNS).
        files_note: Files that could not be read, sub-runs left out, and run notes.
        map_note: Runs left out of the analysis (spatial maps), in red.
        gauge_form: Gauge position during the runs (mm, STL frame).
        gauge_from_prediction_button: Copies the gauge center of the last prediction.
        settings_form: Receiving slit, cutoff, Monte Carlo points, peak window and intensity method.
        combine_form: Combine runs with the same hkl, and the scale between them.
        analyze_button: Starts the analysis.
        export_button: Exports the CSVs and figures.
        gif_pf: Pole figure of the beam-path GIF.
        plot_panel: The pole-figure plots.
        style_toggle: Point cells or interpolated pole figures.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.add_button = QPushButton("Add reduced files...")
        self.remove_button = QPushButton("Remove selected")
        self.clear_button = QPushButton("Clear")
        self.runs_table = QTableWidget(0, len(RUN_COLUMNS))
        self.runs_table.setHorizontalHeaderLabels(RUN_COLUMNS)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.setMinimumHeight(170)
        self.files_note = QLabel("")
        self.files_note.setWordWrap(True)
        self.map_note = QLabel("")  # runs left out of the analysis (spatial maps), in red
        self.map_note.setWordWrap(True)

        self.gauge_form = ModelForm(DataModel, GAUGE_FIELDS, decimals=4)
        self.gauge_from_prediction_button = QPushButton("Inherit vx, vy, vz from the prediction")
        self.gauge_from_prediction_button.setToolTip("Fill vx, vy, vz with the gauge center of the last prediction")
        self.gauge_from_prediction_button.setEnabled(False)
        self.gauge_note = QLabel("")
        self.gauge_note.setWordWrap(True)
        self.settings_form = ModelForm(DataModel, SETTING_FIELDS, decimals=2)
        self.combine_form = ModelForm(DataModel, COMBINE_FIELDS)
        self.forms = [self.gauge_form, self.settings_form, self.combine_form]
        self.ready_label = QLabel("")
        self.ready_label.setWordWrap(True)
        self.analyze_button = QPushButton("Analyze runs")
        self.analyze_button.setStyleSheet("font-weight: bold")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.export_button = QPushButton("Export CSVs and figures...")
        self.export_button.setEnabled(False)

        self.gif_pf = QComboBox()
        self.gif_fps = spin_box(8.0, lo=0.5, hi=30.0, decimals=1)
        self.gif_every = int_spin_box(1, 1, 100)
        self.gif_dpi = int_spin_box(80, 40, 200)
        self.gif_label = QLabel("")
        self.gif_label.setWordWrap(True)
        self.gif_button = QPushButton("Make GIF for this pole figure")
        self.gif_button.setEnabled(False)
        self.gif_all_button = QPushButton("Make GIFs for all pole figures")
        self.gif_all_button.setEnabled(False)
        self.gif_save_all_button = QPushButton("Save all GIFs...")
        self.gif_save_all_button.setEnabled(False)
        self.gif_cancel_button = QPushButton("Cancel")
        self.gif_cancel_button.setEnabled(False)
        self.gif_progress = QProgressBar()
        self.gif_progress.setRange(0, 1000)
        self.gif_progress_label = QLabel("")
        self.gif_progress_label.setWordWrap(True)

        files_box = QGroupBox("Reduced runs (HB2B_<run>.h5)")
        fv = QVBoxLayout(files_box)
        row = QHBoxLayout()
        for b in (self.add_button, self.remove_button, self.clear_button):
            row.addWidget(b)
        fv.addLayout(row)
        fv.addWidget(self.runs_table)
        fv.addWidget(self.files_note)
        fv.addWidget(self.map_note)

        gauge_box = QGroupBox("Gauge position during the runs")
        gv = QVBoxLayout(gauge_box)
        gtext = QLabel("Where the gauge center (the rotation point) was in the STL frame while these runs were "
                       "measured. Every run is corrected at this position. The values fill from the runs' vx, vy, "
                       "vz logs when the files have them. Otherwise enter all three.")  # fmt: skip
        gtext.setWordWrap(True)
        gtext.setStyleSheet("color: #555")
        gv.addWidget(gtext)
        gv.addWidget(self.gauge_form)
        gv.addWidget(self.gauge_from_prediction_button)
        gv.addWidget(self.gauge_note)

        settings_box = QGroupBox("Settings")
        sv = QVBoxLayout(settings_box)
        sv.addWidget(self.settings_form)
        note = QLabel(
            "Incident slits (ISR:X/Y:Gap), wavelength, chi, phi, omega and count times come from each file. "
            "mu is computed at each run's wavelength unless overridden on Experiment setup."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555")
        sv.addWidget(note)

        combine_box = QGroupBox("Combine runs")
        cv = QVBoxLayout(combine_box)
        cv.addWidget(self.combine_form)
        cnote = QLabel("For a pole figure split over several reduced files. Applied after the analysis, so no new "
                       "Monte Carlo is needed. Combined pole figures go to plots, export and MTEX.")  # fmt: skip
        cnote.setWordWrap(True)
        cnote.setStyleSheet("color: #555")
        cv.addWidget(cnote)

        run_box = QGroupBox("Analysis")
        rv = QVBoxLayout(run_box)
        rv.addWidget(self.ready_label)
        r2 = QHBoxLayout()
        r2.addWidget(self.analyze_button, stretch=1)
        r2.addWidget(self.cancel_button)
        rv.addLayout(r2)
        rv.addWidget(self.progress)
        rv.addWidget(self.progress_label)
        rv.addWidget(self.results_label)
        rv.addWidget(self.export_button)

        gif_box = QGroupBox("Beam-path GIF (measured points)")
        gf = QFormLayout(gif_box)
        gnote = QLabel("The sample turned through each measured chi, phi in the order of the sub-runs, with the "
                       "run's own slits and the corrections from the analysis.")  # fmt: skip
        gnote.setWordWrap(True)
        gnote.setStyleSheet("color: #555")
        gf.addRow(gnote)
        gf.addRow("Pole figure", self.gif_pf)
        gf.addRow("Frames per second", self.gif_fps)
        gf.addRow("Use every n-th point", self.gif_every)
        gf.addRow("Resolution (dpi)", self.gif_dpi)
        gf.addRow(self.gif_label)
        g1 = QHBoxLayout()
        g1.addWidget(self.gif_button, stretch=1)
        g1.addWidget(self.gif_all_button, stretch=1)
        gf.addRow(g1)
        g2 = QHBoxLayout()
        g2.addWidget(self.gif_save_all_button, stretch=1)
        g2.addWidget(self.gif_cancel_button)
        gf.addRow(g2)
        gf.addRow(self.gif_progress)
        gf.addRow(self.gif_progress_label)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        for w in (files_box, gauge_box, settings_box, run_box, combine_box, gif_box):
            lv.addWidget(w)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(480)

        self.plot_panel = PlotPanel()
        self.plot_panel.placeholder.setText("Add reduced runs and press Analyze runs.")
        self.style_toggle = LabeledToggle("Point cells", "Interpolated")
        self.style_toggle.setToolTip(
            "Point cells: one cell per measured chi, phi. Interpolated: linear between poles."
        )
        self.plot_panel.add_corner_widget(self.style_toggle)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.plot_panel)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def set_runs(self, rows: list[list[str]], flagged: Collection[int] = ()) -> None:
        """
        Fill the runs table, one list of cell texts per run. Each cell's tooltip repeats its text.

        Args:
            rows: Cell texts, one list per run.
            flagged: Rows shown in red: runs left out of the analysis (default: none).
        """
        self.runs_table.setRowCount(len(rows))
        for i, cells in enumerate(rows):
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if i in flagged:
                    item.setForeground(QBrush(QColor(FLAG_COLOR)))
                self.runs_table.setItem(i, j, item)

    def selected_rows(self) -> list[int]:
        """Indices of the selected rows, ascending."""
        return sorted({ix.row() for ix in self.runs_table.selectionModel().selectedRows()})

    def ask_files(self, start: str) -> list[str]:
        """Ask for reduced HB-2B files, starting in ``start``. Returns [] when the user cancels."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add reduced HB-2B runs", start,
                                                "Reduced runs (HB2B_*.h5 *.h5);;All files (*)")  # fmt: skip
        return [str(p) for p in paths]

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
