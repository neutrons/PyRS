"""
MTEX tab: pole figures to use, ODF settings, plot options, run log, summary, and the plots (MTEX for Python).

The MTEX presenter lists the corrected pole figures with :meth:`MtexTab.set_pole_figures`, fits the ODF with the
checked ones and shows the MTEX plots, log and summary.

Example:
    >>> tab = MtexTab()  # doctest: +SKIP
    >>> tab.set_pole_figures([("111", 600, 649, 10, "", True)])  # doctest: +SKIP
    >>> tab.checked_labels()  # doctest: +SKIP
    ['111']
"""

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models.mtex import ODF_FIELDS, PLOT_FIELDS, MtexModel
from ..widgets.model_form import ModelForm
from ..widgets.plot_panel import PlotPanel
from ..widgets.toggle_switch import LabeledToggle

PF_COLUMNS = ["Use", "Pole figure", "Points kept", "chi rings", "Note"]


def _group(title: str, *widgets: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    for w in widgets:
        lay.addWidget(w)
    return box


def _text_box(height: int) -> QPlainTextEdit:
    t = QPlainTextEdit()
    t.setReadOnly(True)
    t.setFont(QFont("Consolas", 8))
    t.setMinimumHeight(height)
    return t


class MtexTab(QWidget):
    """
    Widgets of the MTEX tab.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        pf_table: The pole figures, with a Use check box per row.
        odf_form: ODF settings.
        plot_form: Plot options.
        run_button: Starts MTEX.
        log: Run log.
        summary: ODF summary text.
        save_button: Saves all results.
        save_data_button: Saves the numbers behind each plot (CSV).
        plot_panel: The MTEX plots.
        white_toggle: Shows the full-scale or the white-below-random version of a plot.
        install_note: Says that MTEX for Python is missing (hidden while it is installed).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pf_table = QTableWidget(0, len(PF_COLUMNS))
        self.pf_table.setHorizontalHeaderLabels(PF_COLUMNS)
        self.pf_table.verticalHeader().setVisible(False)
        self.pf_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pf_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.pf_table.horizontalHeader().setStretchLastSection(True)
        self.pf_table.setMinimumHeight(140)
        self.install_note = QLabel("")
        self.install_note.setWordWrap(True)
        self.install_note.setStyleSheet("color: #b00020; font-weight: bold")
        self.install_note.setVisible(False)

        self.odf_form = ModelForm(MtexModel, ODF_FIELDS, decimals=1)
        self.symmetry_label = QLabel("")
        self.symmetry_label.setWordWrap(True)
        self.plot_form = ModelForm(MtexModel, PLOT_FIELDS, decimals=2)
        self.forms = [self.odf_form, self.plot_form]

        self.ready_label = QLabel("")
        self.ready_label.setWordWrap(True)
        self.run_button = QPushButton("Run MTEX")
        self.run_button.setStyleSheet("font-weight: bold")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.log = _text_box(120)
        self.summary = _text_box(150)
        self.save_button = QPushButton("Save all results...")
        self.save_button.setEnabled(False)
        self.save_data_button = QPushButton("Save plot data (CSV)...")
        self.save_data_button.setEnabled(False)
        self.save_data_button.setToolTip("The numbers behind each plot, with a README, to redraw them elsewhere")

        run_row = QWidget()
        rl = QHBoxLayout(run_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.run_button, stretch=1)
        rl.addWidget(self.cancel_button)
        save_row = QWidget()
        sr = QHBoxLayout(save_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(self.save_button, stretch=1)
        sr.addWidget(self.save_data_button, stretch=1)
        log_label = QLabel("Log")
        sum_label = QLabel("ODF summary (odf_summary.txt)")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        lv.addWidget(self.install_note)
        lv.addWidget(_group("Pole figures from Data and Corrections", self.pf_table))
        lv.addWidget(_group("ODF", self.odf_form, self.symmetry_label))
        lv.addWidget(_group("Plots", self.plot_form))
        lv.addWidget(_group("Run", self.ready_label, run_row, self.progress, self.status_label, log_label, self.log,
                            sum_label, self.summary, save_row))  # fmt: skip
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)

        self.plot_panel = PlotPanel()
        self.plot_panel.placeholder.setText("Run MTEX to make the ODF, pole figure and inverse pole figure plots.")
        self.white_toggle = LabeledToggle("Full scale", "White below random")
        self.white_toggle.setToolTip("Show the version with everything below the white level (1 mrd, random) white")
        self.plot_panel.add_corner_widget(self.white_toggle)
        self.white_toggle.setVisible(False)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.plot_panel)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def show_install_note(self, text: str) -> None:
        """
        Show a note about a missing dependency at the top of the tab.

        Args:
            text: The note, or "" to hide it.
        """
        self.install_note.setText(text)
        self.install_note.setVisible(bool(text))

    def set_pole_figures(self, rows: list[tuple[str, int, int, int, str, bool]]) -> None:
        """
        Fill the pole-figure table. Usable rows start checked, the others are disabled.

        Args:
            rows: (label, points kept, points total, chi rings, note, usable) per pole figure.
        """
        self.pf_table.setRowCount(len(rows))
        for i, (label, kept, total, rings, note, ok) in enumerate(rows):
            use = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled if ok else Qt.ItemFlag.NoItemFlags
            use.setFlags(flags)
            use.setCheckState(Qt.CheckState.Checked if ok else Qt.CheckState.Unchecked)
            use.setData(Qt.ItemDataRole.UserRole, label)
            self.pf_table.setItem(i, 0, use)
            for j, text in enumerate((label, f"{kept} of {total}", str(rings), note), start=1):
                self.pf_table.setItem(i, j, QTableWidgetItem(text))

    def checked_labels(self) -> list[str]:
        """Labels of the checked pole figures, in table order."""
        out = []
        for i in range(self.pf_table.rowCount()):
            item = self.pf_table.item(i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def set_progress(self, fraction: float) -> None:
        """Show progress, a fraction clipped to [0, 1]."""
        self.progress.setValue(int(round(1000 * min(max(fraction, 0.0), 1.0))))

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
