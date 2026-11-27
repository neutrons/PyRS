"""
Table scan tab: settings, the per-plane plan in the middle, and a preview of each plane's scan.

The table scan presenter lists the planes of the last prediction with :meth:`TableScanTab.set_planes`, reads each
plane's base time, fills the plan columns with :meth:`TableScanTab.set_row` and writes the files.

Example:
    >>> tab = TableScanTab()  # doctest: +SKIP
    >>> tab.set_planes([("111", 50.1, 25.0), ("200", 58.6, 29.3)])  # doctest: +SKIP
    >>> tab.bases()  # doctest: +SKIP
    [10.0, 10.0]
"""

from qtpy.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models.table_scan import SETTING_FIELDS, TableScanModel
from ..widgets.model_form import ModelForm
from ..widgets.plot_panel import PlotPanel

COLUMNS = ["hkl", "2theta", "omega", "Base time (s)", "Points kept", "Count time", "phi travel (deg)"]
BASE_COL = 3
DEFAULT_BASE_S = 10.0


class TableScanTab(QWidget):
    """
    Widgets of the Table scan tab.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        settings_form: Run title, cutoff, count-time rule and order.
        table: The plan, one row per plane, with a base time (s) spin box per row.
        summary: Totals of the plan.
        write_button: Writes the table scan files.
        plot_panel: Preview of each plane's planned points.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ready_label = QLabel("")
        self.ready_label.setWordWrap(True)
        self.settings_form = ModelForm(TableScanModel, SETTING_FIELDS, decimals=2)
        self.forms = [self.settings_form]
        rule = QLabel("Base time is the count time at correction 1 (scaled) or for every point (fixed). "
                      "Points above the cutoff are left out.")  # fmt: skip
        rule.setWordWrap(True)
        rule.setStyleSheet("color: #555")

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(180)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.write_button = QPushButton("Write table scans...")
        self.write_button.setStyleSheet("font-weight: bold")
        self.write_button.setEnabled(False)

        settings_box = QGroupBox("Settings")
        sv = QVBoxLayout(settings_box)
        sv.addWidget(self.settings_form)
        sv.addWidget(rule)
        plan_box = QGroupBox("Plan per plane (one file per plane)")
        pv = QVBoxLayout(plan_box)
        pv.addWidget(self.table, stretch=1)
        pv.addWidget(self.summary)
        pv.addWidget(self.write_button)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.addWidget(self.ready_label)
        cv.addWidget(settings_box)
        cv.addWidget(plan_box, stretch=1)
        center.setMinimumWidth(520)

        self.plot_panel = PlotPanel()
        self.plot_panel.placeholder.setText("Predict first. The preview shows each plane's planned points.")
        split = QSplitter()
        split.addWidget(center)
        split.addWidget(self.plot_panel)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def set_planes(self, rows: list[tuple[str, float, float]]) -> None:
        """
        One row per plane. Keeps the base time of planes already listed, new planes start at 10 s.

        Args:
            rows: (hkl, 2theta in deg, omega in deg) per plane.
        """
        old = dict(zip(self.hkls(), self.bases(), strict=True))
        self.table.setRowCount(len(rows))
        for i, (h, tth, om) in enumerate(rows):
            for j, text in ((0, f"{{{h}}}"), (1, f"{tth:.2f}"), (2, f"{om:.2f}"), (4, ""), (5, ""), (6, "")):
                self.table.setItem(i, j, QTableWidgetItem(text))
            spin = QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setRange(0.5, 3600.0)
            spin.setValue(old.get(h, DEFAULT_BASE_S))
            spin.setKeyboardTracking(False)
            self.table.setCellWidget(i, BASE_COL, spin)

    def hkls(self) -> list[str]:
        """The hkl of each row, without braces."""
        out = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            out.append(item.text().strip("{}") if item is not None else "")
        return out

    def base_spins(self) -> list[QDoubleSpinBox]:
        """The base-time spin box of each row."""
        out: list[QDoubleSpinBox] = []
        for i in range(self.table.rowCount()):
            w = self.table.cellWidget(i, BASE_COL)
            if isinstance(w, QDoubleSpinBox):
                out.append(w)
        return out

    def bases(self) -> list[float]:
        """The base time of each row (s)."""
        return [float(s.value()) for s in self.base_spins()]

    def set_row(self, i: int, *, kept: str, total: str, travel: str) -> None:
        """
        Fill the plan columns of one row.

        Args:
            i: Row index.
            kept: Points kept text.
            total: Total count time text.
            travel: phi travel text (deg).
        """
        for j, text in ((4, kept), (5, total), (6, travel)):
            item = self.table.item(i, j)
            if item is not None:
                item.setText(text)

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

    def confirm(self, title: str, text: str) -> bool:
        """
        Ask a yes/no question.

        Args:
            title: Dialog title.
            text: Question.

        Returns:
            True when the user answers Yes.
        """
        answer = QMessageBox.question(self, title, text)
        return bool(answer == QMessageBox.StandardButton.Yes)
