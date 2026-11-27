"""
Experiment setup tab: phase, wavelength, beams and slits, attenuation, and the reflection table.

The material presenter fills the preset list and the forms, and lists the allowed reflections with
:meth:`MaterialTab.set_reflections`. The checked rows are the reflections the other tabs use.

Example:
    >>> tab = MaterialTab()  # doctest: +SKIP
    >>> tab.set_reflections(reflections, checked={"111", "200"}, disabled=set())  # doctest: +SKIP
    >>> tab.checked_labels()  # doctest: +SKIP
    ['111', '200']
"""

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyrs.pole_figures.core.reflections import Reflection
from ..models.material import BEAM_FIELDS, MU_FIELDS, PHASE_FIELDS, WAVELENGTH_FIELDS, MaterialModel
from ..widgets.model_form import ModelForm

COLUMNS = ["Use", "hkl", "Multiplicity", "d (A)", "2theta (deg)", "|F|^2 x mult (rel.)"]


def _group(title: str, *widgets: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    for w in widgets:
        lay.addWidget(w)
    return box


class MaterialTab(QWidget):
    """
    Widgets of the Experiment setup tab (material, wavelength, beams).

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        preset_combo: Material presets.
        cif_button: Loads a CIF file.
        phase_form: Name, space group, lattice (Å, deg), composition and density.
        wavelength_form: Monochromator setting and wavelength (Å).
        mu_form: mu override (1/mm).
        beam_form: Incident and receiving slit sizes (mm).
        forms: The four forms above.
        table: Allowed reflections, with a Use check box per row.
        all_button: Checks every enabled reflection.
        none_button: Clears every check.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preset_combo = QComboBox()
        self.cif_button = QPushButton("Load CIF...")
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        self.phase_form = ModelForm(MaterialModel, PHASE_FIELDS, decimals=4)
        self.wavelength_form = ModelForm(MaterialModel, WAVELENGTH_FIELDS, decimals=4)
        self.mu_label = QLabel("")
        self.mu_label.setWordWrap(True)
        self.mu_form = ModelForm(MaterialModel, MU_FIELDS, decimals=4)
        self.beam_form = ModelForm(MaterialModel, BEAM_FIELDS, decimals=2)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020")
        self.forms = [self.phase_form, self.wavelength_form, self.beam_form, self.mu_form]

        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        top.addWidget(self.preset_combo, stretch=1)
        top.addWidget(self.cif_button)
        head = QWidget()
        head.setLayout(top)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)  # wrapped labels must not squash the forms
        lv.addWidget(_group("Phase", head, self.source_label, self.phase_form))
        lv.addWidget(_group("Wavelength", self.wavelength_form))
        lv.addWidget(_group("Beams and slits", self.beam_form))
        lv.addWidget(_group("Attenuation", self.mu_label, self.mu_form))
        lv.addWidget(self.error_label)
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.all_button = QPushButton("Use all")
        self.none_button = QPushButton("Use none")
        self.table_note = QLabel("")
        self.table_note.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.all_button)
        buttons.addWidget(self.none_button)
        buttons.addStretch(1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("<b>Allowed reflections</b> (space group and neutron structure factor)"))
        rv.addWidget(self.table, stretch=1)
        rv.addLayout(buttons)
        rv.addWidget(self.table_note)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def set_inputs_enabled(self, on: bool) -> None:
        """Gray out everything below the preset row until a material is chosen (Load CIF stays active)."""
        for w in (*self.forms, self.table, self.all_button, self.none_button):
            w.setEnabled(on)

    def ask_cif_path(self, start_dir: str) -> str:
        """Ask for a CIF file, starting in ``start_dir``. Returns "" when the user cancels."""
        path, _ = QFileDialog.getOpenFileName(self, "Open CIF", start_dir, "CIF files (*.cif);;All files (*)")
        return str(path)

    def set_reflections(self, refl: list[Reflection], *, checked: set[str], disabled: set[str]) -> None:
        """
        Fill the table without emitting itemChanged. Disabled rows are gray and cannot be checked.

        The intensity column is |F|^2 x multiplicity in percent of the strongest reflection.

        Args:
            refl: Reflections, in table order.
            checked: Labels (hkl) to check.
            disabled: Labels whose 2theta is outside the allowed range (gray, not checkable).
        """
        gray = QBrush(QColor("#9a9a9a"))
        self.table.blockSignals(True)
        self.table.setRowCount(len(refl))
        top = max((r.f2 * r.multiplicity for r in refl), default=1.0) or 1.0
        for i, r in enumerate(refl):
            off = r.label in disabled
            use = QTableWidgetItem()
            use.setFlags(
                Qt.ItemFlag.NoItemFlags if off else Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            use.setCheckState(Qt.CheckState.Checked if r.label in checked and not off else Qt.CheckState.Unchecked)
            use.setData(Qt.ItemDataRole.UserRole, r.label)
            tip = f"2theta {r.two_theta:.2f} deg is outside the allowed range" if off else ""
            cells = [f"{{{r.label}}}", str(r.multiplicity), f"{r.d_A:.4f}", f"{r.two_theta:.2f}",
                     f"{100 * r.f2 * r.multiplicity / top:.0f}"]  # fmt: skip
            self.table.setItem(i, 0, use)
            for j, text in enumerate(cells, start=1):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # type: ignore[arg-type]  # PyQt5 stubs want int, which PySide6 and PyQt6 deprecate
                if off:
                    item.setForeground(gray)
                    item.setToolTip(tip)
                self.table.setItem(i, j, item)
            use.setToolTip(tip)
        self.table.blockSignals(False)

    def checked_labels(self) -> list[str]:
        """Labels (hkl) of the checked rows, in table order."""
        out = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def set_all_checked(self, on: bool) -> None:
        """Check (True) or clear (False) every enabled row without emitting itemChanged."""
        self.table.blockSignals(True)
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
