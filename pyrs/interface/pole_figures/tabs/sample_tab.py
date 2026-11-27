"""
Sample tab: STL file, gauge center, the sample-to-stage matrix, stage conventions, preview, and the mount figure.

The sample presenter loads the STL chosen with :meth:`SampleTab.ask_stl_path`, keeps the forms and the matrix in
step with the sample model, and draws the mount figure on :attr:`SampleTab.canvas`.

Example:
    >>> tab = SampleTab()  # doctest: +SKIP
    >>> tab.set_matrix((1, 0, 0, 0, 1, 0, 0, 0, 1))  # doctest: +SKIP
    >>> tab.matrix_values()  # doctest: +SKIP
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..models.sample import GAUGE_FIELDS, MOUNT_FIELDS, PREVIEW_FIELDS, VIEW_FIELDS, SampleModel
from ..mount_matrix import STAGE_AXES
from ..widgets.model_form import ModelForm


def _group(title: str, inner: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    QVBoxLayout(box).addWidget(inner)
    return box


class SampleTab(QWidget):
    """
    Widgets of the Sample tab.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        open_button: Opens an STL file.
        center_button: Puts the gauge center at the center of the STL bounding box.
        gauge_form: Gauge center (mm, sample frame).
        mount_form: Stage conventions.
        view_form: Pole-figure view direction and the name of +vz.
        preview_form: Preview orientation (chi, phi, omega, 2theta in deg).
        forms: The four forms above.
        matrix_spins: The 3 x 3 sample-to-stage matrix. Rows: the phi, chi and Q stage axes. Columns: vx, vy, vz.
        rotate_angles: Angle (deg) for a turn about each stage axis.
        rotate_buttons: Apply that turn to the matrix.
        reset_mount_button: Resets the matrix to identity.
        canvas: The mount figure.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.open_button = QPushButton("Open STL...")
        self.center_button = QPushButton("Gauge at bounding-box center")
        self.path_label = QLabel("No STL loaded")
        self.path_label.setWordWrap(True)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)

        self.gauge_form = ModelForm(SampleModel, GAUGE_FIELDS)
        self.mount_form = ModelForm(SampleModel, MOUNT_FIELDS)
        self.view_form = ModelForm(SampleModel, VIEW_FIELDS)
        self.preview_form = ModelForm(SampleModel, PREVIEW_FIELDS, decimals=2)
        self.forms = [self.gauge_form, self.mount_form, self.view_form, self.preview_form]

        self.matrix_spins: list[list[QDoubleSpinBox]] = []
        grid = QGridLayout()
        for j, name in enumerate(("vx", "vy", "vz")):
            head = QLabel(f"<b>{name}</b>")
            grid.addWidget(head, 0, j + 1)
        for i, axis in enumerate(STAGE_AXES):
            grid.addWidget(QLabel(axis), i + 1, 0)
            row = []
            for j in range(3):
                sp = QDoubleSpinBox()
                sp.setRange(-1.0, 1.0)
                sp.setDecimals(4)
                sp.setSingleStep(0.01)
                sp.setKeyboardTracking(False)
                sp.setMaximumWidth(78)
                sp.setToolTip(f"Component of sample {'xyz'[j]} along the stage {axis}")
                grid.addWidget(sp, i + 1, j + 1)
                row.append(sp)
            self.matrix_spins.append(row)
        self.matrix_note = QLabel("")
        self.matrix_note.setWordWrap(True)
        self.rotate_angles: list[QDoubleSpinBox] = []
        self.rotate_buttons: list[QPushButton] = []
        rot_grid = QGridLayout()
        for i, axis in enumerate(STAGE_AXES):
            rot_grid.addWidget(QLabel(f"Rotate about the {axis.split(' (')[0]}"), i, 0)
            ang = QDoubleSpinBox()
            ang.setRange(-360.0, 360.0)
            ang.setDecimals(2)
            ang.setSuffix(" deg")
            ang.setValue(0.0)
            ang.setMaximumWidth(100)
            ang.setToolTip("Positive or negative, right-hand rule about the stage axis")
            btn = QPushButton("Rotate")
            rot_grid.addWidget(ang, i, 1)
            rot_grid.addWidget(btn, i, 2)
            self.rotate_angles.append(ang)
            self.rotate_buttons.append(btn)
        self.reset_mount_button = QPushButton("Reset to identity")
        mount_box = QGroupBox("Sample on the stage (sample-to-stage matrix at chi 0, phi 0)")
        mb = QVBoxLayout(mount_box)
        mb.addLayout(grid)
        mb.addWidget(self.matrix_note)
        mb.addLayout(rot_grid)
        mb.addWidget(self.reset_mount_button)
        hint = QLabel("Each column is where that sample axis points on the stage. Edit the values, or turn the sample "
                      "about a stage axis: each Rotate adds to the matrix. Every tab uses this matrix.")  # fmt: skip
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555")
        mb.addWidget(hint)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)  # wrapped labels must not squash the forms
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.center_button)
        lv.addLayout(buttons)
        lv.addWidget(self.path_label)
        lv.addWidget(self.info_label)
        lv.addWidget(_group("Gauge center (sample frame)", self.gauge_form))
        lv.addWidget(mount_box)
        lv.addWidget(_group("Stage conventions", self.mount_form))
        lv.addWidget(_group("Pole-figure view", self.view_form))
        lv.addWidget(_group("Preview orientation", self.preview_form))
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(16, 6.2)))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas, stretch=1)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QHBoxLayout(self).addWidget(split)

    def set_matrix(self, m: tuple[float, ...]) -> None:
        """Show the 9 values (rows: phi, chi, Q axes; columns: vx, vy, vz) without emitting changes."""
        for i in range(3):
            for j in range(3):
                sp = self.matrix_spins[i][j]
                sp.blockSignals(True)
                sp.setValue(float(m[3 * i + j]))
                sp.blockSignals(False)

    def matrix_values(self) -> tuple[float, ...]:
        """The 9 shown matrix values, row by row."""
        return tuple(float(self.matrix_spins[i][j].value()) for i in range(3) for j in range(3))

    def ask_stl_path(self, start_dir: str) -> str:
        """Ask for an STL file, starting in ``start_dir``. Returns "" when the user cancels."""
        path, _ = QFileDialog.getOpenFileName(self, "Open sample STL", start_dir, "STL files (*.stl);;All files (*)")
        return str(path)

    def show_error(self, title: str, text: str) -> None:
        """
        Show a warning dialog.

        Args:
            title: Dialog title.
            text: Message.
        """
        QMessageBox.warning(self, title, text)
