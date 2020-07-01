from mantidqt.widgets.sliceviewer.presenter import SliceViewer
from qtpy.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QWidget,
                            QLineEdit, QPushButton, QComboBox,
                            QGroupBox, QSplitter, QTabWidget,
                            QTableWidget, QTableWidgetItem,
                            QFormLayout, QFileDialog,
                            QStyledItemDelegate, QDoubleSpinBox,
                            QStackedWidget)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
import numpy as np
import functools


class FileLoad(QWidget):
    def __init__(self, name, fileType="HidraProjectFile (*.h5);;All Files (*)", parent=None):
        self.parent = parent
        super().__init__(parent)
        self.name = name
        self.fileType = fileType
        layout = QHBoxLayout()
        layout.addWidget(QLabel(name))
        self.lineEdit = QLineEdit()
        self.lineEdit.setReadOnly(True)
        layout.addWidget(self.lineEdit)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.openFileDialog)
        layout.addWidget(self.browse_button)
        self.setLayout(layout)

    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  self.name,
                                                  "",
                                                  self.fileType)
        if fileName:
            self.lineEdit.setText(fileName)
            self.parent.controller.fileSelected(self.name, fileName)
            self.parent.update_plot()


class DimSwitch(QWidget):
    dimChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.addStretch(1)
        self.button_2d = QPushButton("2D")
        self.button_2d.setCheckable(True)
        self.button_2d.setChecked(True)
        self.button_2d.clicked.connect(functools.partial(self.set_2D, True))
        self.button_3d = QPushButton("3D")
        self.button_3d.setCheckable(True)
        self.button_3d.clicked.connect(functools.partial(self.set_2D, False))
        layout.addWidget(self.button_2d)
        layout.addWidget(self.button_3d)
        layout.addStretch(1)
        self.setLayout(layout)

    def set_2D(self, bool2d):
        self.button_2d.setChecked(bool2d)
        self.button_3d.setChecked(not bool2d)
        self.dimChanged.emit(bool2d)


class StressCase(QGroupBox):
    dimChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Define Stress Case")
        layout = QVBoxLayout()
        self.switch = DimSwitch(self)
        self.switch.dimChanged.connect(self.set_2D)
        layout.addWidget(self.switch)
        self.combo = QComboBox()
        self.combo.addItems(["In-plain stress",
                             "In-plain strain"])
        layout.addWidget(self.combo)
        self.setLayout(layout)

    def set_2D(self, bool2d):
        self.combo.setDisabled(not bool2d)
        self.dimChanged.emit(bool2d)


class SpinBoxDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def createEditor(self, parent, option, index):
        editor = QDoubleSpinBox(parent)
        editor.setMinimum(0)
        editor.setDecimals(5)
        editor.setSingleStep(0.0001)
        return editor


class D0(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent)
        self.setTitle("Define d₀")
        layout = QVBoxLayout()
        d0_box = QWidget()
        d0_box_layout = QHBoxLayout()
        d0_box_layout.addWidget(QLabel("d₀"))
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.d0 = QLineEdit()
        self.d0.setValidator(validator)
        self.d0.editingFinished.connect(self.set_d0)
        d0_box_layout.addWidget(self.d0)
        d0_box.setLayout(d0_box_layout)
        layout.addWidget(d0_box)
        load_grid = FileLoad("d₀ Grid", parent=self)
        load_grid.setEnabled(False)
        layout.addWidget(load_grid)

        self.d0_grid = QTableWidget()
        self.d0_grid.setColumnCount(2)
        self.d0_grid.verticalHeader().setVisible(False)
        self.d0_grid.horizontalHeader().setStretchLastSection(True)
        self.d0_grid.setHorizontalHeaderLabels(['Sub-run', "d₀"])
        self.d0_grid.cellChanged.connect(self.cellChanged)
        self.d0_grid.setItemDelegateForColumn(1, SpinBoxDelegate())

        layout.addWidget(self.d0_grid)
        self.setLayout(layout)

    def set_sub_runs(self, sub_runs, d0_list):
        self.d0_grid.cellChanged.disconnect()
        self.d0_grid.setRowCount(len(sub_runs))
        for n, (sub_run, d0) in enumerate(zip(sub_runs, d0_list)):
            subrun_item = QTableWidgetItem(str(sub_run))
            subrun_item.setFlags(subrun_item.flags() ^ Qt.ItemIsEditable)
            d0_item = QTableWidgetItem()
            d0_item.setData(Qt.EditRole, float(d0))
            self.d0_grid.setItem(n, 0, QTableWidgetItem(subrun_item))
            self.d0_grid.setItem(n, 1, QTableWidgetItem(d0_item))
        self.d0_grid.cellChanged.connect(self.cellChanged)

    def set_d0(self, d0=None):
        self.d0_grid.cellChanged.disconnect()
        if d0 is None:
            d0 = self.d0.text()

        d0_item = QTableWidgetItem()
        d0_item.setData(Qt.EditRole, float(d0))
        for n in range(self.d0_grid.rowCount()):
            self.d0_grid.setItem(n, 1, QTableWidgetItem(d0_item))
        self.d0_grid.cellChanged.connect(self.cellChanged)
        self.d0_grid.cellChanged.emit(0, 0)
        self._parent.update_plot()

    def get_d0(self):
        return [float(self.d0_grid.item(n, 1).text()) for n in range(self.d0_grid.rowCount())]

    def cellChanged(self, row, column):
        self.d0.clear()
        self._parent.controller.update_d0(self.get_d0())


class FileLoading(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Define Project Files")
        layout = QVBoxLayout()
        self.file_load_e11 = FileLoad("e11", parent=parent)
        self.file_load_e22 = FileLoad("e22", parent=parent)
        self.file_load_e33 = FileLoad("e33", parent=parent)
        self.file_load_e33.setDisabled(True)
        layout.addWidget(self.file_load_e11)
        layout.addWidget(self.file_load_e22)
        layout.addWidget(self.file_load_e33)
        self.setLayout(layout)


class MechanicalConstants(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Mechanical Constants")

        layout = QFormLayout()
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.youngModulus = QLineEdit()
        self.youngModulus.setValidator(validator)
        self.poissonsRatio = QLineEdit()
        self.poissonsRatio.setValidator(validator)
        layout.addRow(QLabel("Young Modulus, E (GPa)"), self.youngModulus)
        layout.addRow(QLabel("Poisson's ratio, ν"), self.poissonsRatio)

        self.setLayout(layout)


class StrainSliceViewer(SliceViewer):
    def __init__(self, ws, parent=None):
        super().__init__(ws, parent=parent)
        self.view.data_view.mpl_toolbar.peaksOverlayClicked.disconnect()
        self.view.data_view.mpl_toolbar.peaksOverlayClicked.connect(self.overlay)
        self.overlay = False
        self.scatter = None

    def overlay(self):
        self.overlay = not self.overlay
        if self.overlay:
            X, Y = np.meshgrid(range(20), range(20))
            self.scatter = self.view.data_view.canvas.figure.axes[0].scatter(X, Y, c='black')
            self.view.data_view.canvas.draw_idle()
        else:
            self.scatter.remove()
            self.view.data_view.canvas.draw_idle()


class PlotSelect(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        self.plot_param = QComboBox()
        self.plot_param.addItems(["chi sqaured",
                                  "d spacing",
                                  "d referance",
                                  "fit status",
                                  "integrated intensity",
                                  "strain",
                                  "stress"])
        self.plot_param.setCurrentIndex(5)
        self.setStressEnabled(False)
        layout.addRow(QLabel("Plot"), self.plot_param)
        self.measure_dir = QComboBox()
        self.measure_dir.addItems(["11",
                                   "22",
                                   "33"])
        layout.addRow(QLabel("Measurement Direction "), self.measure_dir)
        self.setLayout(layout)

    def setStressEnabled(self, enabled=True):
        if enabled:
            self.plot_param.model().item(6).setFlags(Qt.ItemIsSelectable |
                                                     Qt.ItemIsEditable |
                                                     Qt.ItemIsDragEnabled |
                                                     Qt.ItemIsDropEnabled |
                                                     Qt.ItemIsUserCheckable |
                                                     Qt.ItemIsEnabled)
        else:
            self.plot_param.model().item(6).setFlags(Qt.NoItemFlags)

    def get_direction(self):
        return self.measure_dir.currentText()

    def get_plot_param(self):
        return self.plot_param.currentText()


class PeakSelection(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setTitle("Select peak")
        layout = QFormLayout()
        self.peak_select = QComboBox()
        layout.addRow(QLabel("Peak"), self.peak_select)
        self.setLayout(layout)

    def clear_peak_tags(self):
        for _ in range(self.peak_select.count()):
            self.peak_select.removeItem(0)

    def set_peak_tags(self, peak_tags):
        self.peak_select.addItems(peak_tags)


class VizTabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strainSliceViewer = None

        self.plot_2d = QStackedWidget()

        self.addTab(self.plot_2d, "2D")
        self.addTab(QWidget(), "3D")
        self.setTabEnabled(1, False)
        self.setCornerWidget(QLabel("Visualization Pane    "), corner=Qt.TopLeftCorner)

    def set_ws(self, ws):
        if self.strainSliceViewer:
            self.plot_2d.removeWidget(self.strainSliceViewer)
        if ws:
            self.strainSliceViewer = StrainSliceViewer(ws, parent=self).view
            self.plot_2d.addWidget(self.strainSliceViewer)
        else:
            self.strainSliceViewer = None


class StrainStressViewer(QSplitter):
    def __init__(self, model, ctrl, parent=None):
        self._model = model
        self._model.propertyUpdated.connect(self.updatePropertyFromModel)
        self._ctrl = ctrl

        super().__init__(parent)

        self.setWindowTitle("PyRS Strain-Stress Viewer")
        self.setHandleWidth(10)

        left = QWidget()
        left_layout = QVBoxLayout()

        self.stressCase = StressCase(self)
        self.stressCase.dimChanged.connect(self.dimChanged)
        left_layout.addWidget(self.stressCase)

        self.fileLoading = FileLoading(self)
        left_layout.addWidget(self.fileLoading)

        self.peak_selection = PeakSelection(self)
        self.peak_selection.peak_select.currentTextChanged.connect(self.controller.peakSelected)
        left_layout.addWidget(self.peak_selection)

        self.d0 = D0(self)
        left_layout.addWidget(self.d0)

        self.mechanicalConstants = MechanicalConstants(self)
        left_layout.addWidget(self.mechanicalConstants)

        self.calculate = QPushButton("Calculate Stress/Strain")
        left_layout.addWidget(self.calculate)
        left_layout.addStretch(0)

        left.setLayout(left_layout)

        self.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()

        self.plot_select = PlotSelect(self)
        self.plot_select.measure_dir.currentTextChanged.connect(self.update_plot)
        right_layout.addWidget(self.plot_select)

        self.viz_tab = VizTabs(self)
        right_layout.addWidget(self.viz_tab)

        right.setLayout(right_layout)

        self.addWidget(right)
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 5)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model

    def dimChanged(self, bool2d):
        self.fileLoading.file_load_e33.setDisabled(bool2d)
        self.viz_tab.setTabEnabled(1, not bool2d)

    def measure_dir_changed(self):
        self.update_plot()

    def update_plot(self):
        try:
            self.viz_tab.set_ws(self.model.get_field_md(direction=self.plot_select.get_direction(),
                                                        plot_param=self.plot_select.get_plot_param()))
        except KeyError:
            self.viz_tab.set_ws(None)

    def updatePropertyFromModel(self, name):
        getattr(self, name)(getattr(self.model, name))

    def peakTags(self, peak_tags):
        self.peak_selection.peak_select.currentTextChanged.disconnect()
        self.peak_selection.clear_peak_tags()
        self.peak_selection.peak_select.currentTextChanged.connect(self.controller.peakSelected)
        self.peak_selection.set_peak_tags(peak_tags)

    def subruns(self, subruns):
        self.d0.set_sub_runs(subruns, self.model.d0[0])
