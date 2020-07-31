from mantidqt.widgets.sliceviewer.presenter import SliceViewer
from mantidqt.widgets.sliceviewer.model import SliceViewerModel
from mantidqt.icons import get_icon
from qtpy.QtWidgets import (QHBoxLayout, QVBoxLayout, QLabel, QWidget,
                            QLineEdit, QPushButton, QComboBox,
                            QGroupBox, QSplitter, QTabWidget,
                            QFormLayout, QFileDialog,
                            QStyledItemDelegate, QDoubleSpinBox,
                            QStackedWidget, QMessageBox)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import functools
import traceback


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
            success = self.parent.controller.fileSelected(self.name, fileName)
            if success:
                self.lineEdit.setText(fileName)
            else:
                self.lineEdit.setText(None)
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
        self.combo.addItems(["In-plane stress",
                             "In-plane strain"])
        self.combo.currentTextChanged.connect(self.stress_case_changed)
        layout.addWidget(self.combo)
        self.setLayout(layout)

    def set_2D(self, bool2d):
        self.combo.setDisabled(not bool2d)
        self.dimChanged.emit(bool2d)

    def stress_case_changed(self):
        self.dimChanged.emit(True)

    def get_stress_case(self):
        if self.switch.button_3d.isChecked():
            return "diagonal"
        else:
            return self.combo.currentText()


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
        self.d0.editingFinished.connect(self.update_d0)
        d0_box_layout.addWidget(self.d0)
        d0_box.setLayout(d0_box_layout)
        layout.addWidget(d0_box)
        self.setLayout(layout)

    def update_d0(self):
        self._parent.controller.update_d0(float(self.d0.text()))
        self._parent.update_plot()

    def set_d0(self, d0):
        self.d0.setText(str(d0))


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
        self.overlay_visible = False
        self.scatter = None
        self.aspect_equal = False
        super().__init__(ws, parent=parent)

        self.view.data_view.mpl_toolbar.addSeparator()

        # Add aspect ratio button, use used nonOrthogonalClicked
        # signal, easier than creating a new one
        self.view.data_view.mpl_toolbar.nonOrthogonalClicked.disconnect()
        self.view.data_view.mpl_toolbar.addAction(get_icon('mdi.aspect-ratio'),
                                                  'aspect',
                                                  self.view.data_view.mpl_toolbar.nonOrthogonalClicked).setToolTip(
                                                      'Toggle aspect ratio')
        self.view.data_view.mpl_toolbar.nonOrthogonalClicked.connect(self.toggle_aspect)

        self.view.data_view.mpl_toolbar.peaksOverlayClicked.disconnect()
        self.view.data_view.mpl_toolbar.peaksOverlayClicked.connect(self.overlay)

    def new_plot_MDH(self):
        """redefine this function so we can change the default plot interpolation"""
        self.view.data_view.plot_MDH(self.model.get_ws(), slicepoint=self.get_slicepoint(), interpolation='bilinear')
        if self.overlay_visible:
            self.update_overlay()
        if self.aspect_equal:
            self.update_aspect()

    def update_plot_data_MDH(self):
        super().update_plot_data_MDH()
        if self.overlay_visible:
            self.update_overlay()

    def overlay(self):
        self.overlay_visible = not self.overlay_visible

        if self.overlay_visible:
            self.update_overlay()
        else:
            if self.scatter:
                self.scatter.remove()
                self.scatter = None
            self.view.data_view.canvas.draw_idle()

    def update_overlay(self):
        if self.scatter:
            self.scatter.remove()
            self.scatter = None
        slicepoint = self.view.data_view.dimensions.get_slicepoint()
        x, y, z = self.current_field.x, self.current_field.y, self.current_field.z

        xy = []
        for n, (point, values) in enumerate(zip(slicepoint, (x, y, z))):
            if point is not None:
                mask = np.isclose(values, point, atol=self.bin_widths[n]/2)
            else:
                xy.append(values)

        X = xy[0][mask]
        Y = xy[1][mask]

        if self.view.data_view.dimensions.transpose:
            X, Y = Y, X

        self.scatter = self.view.data_view.canvas.figure.axes[0].scatter(X, Y, c='black')
        self.view.data_view.canvas.draw_idle()

    def set_new_field(self, field, bin_widths):
        self.current_field = field
        self.bin_widths = bin_widths

    def set_new_workspace(self, ws):
        self.model = SliceViewerModel(ws)
        self.new_plot()

    def toggle_aspect(self, state):
        self.aspect_equal = not self.aspect_equal
        self.update_aspect()

    def update_aspect(self):
        if self.aspect_equal:
            self.view.data_view.ax.set_aspect('equal')
        else:
            self.view.data_view.ax.set_aspect('auto')
        self.view.data_view.canvas.draw_idle()


class PlotSelect(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(0)
        self.plot_param = QComboBox()
        self.plot_param.addItems(["dspacing_center",
                                  "d_reference",
                                  "Center",
                                  "Height",
                                  "FWHM",
                                  "Mixing",
                                  "Intensity",
                                  "strain",
                                  "stress"])
        self.plot_param.setCurrentIndex(self.plot_param.findText('strain'))
        layout.addRow(QLabel("Plot"), self.plot_param)
        self.measure_dir = QComboBox()
        self.measure_dir.addItems(["11",
                                   "22",
                                   "33"])
        layout.addRow(QLabel("Measurement Direction "), self.measure_dir)
        self.setLayout(layout)

    def get_direction(self):
        return self.measure_dir.currentText()

    def get_plot_param(self):
        return self.plot_param.currentText()


class PeakSelection(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setTitle("Select peak")
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(0)
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
        self._parent = parent
        super().__init__(parent)
        self.oneDViewer = None
        self.strainSliceViewer = None

        self.plot_1d = QStackedWidget()
        self.plot_2d = QStackedWidget()

        self.message = QLabel("Load project files")
        self.message.setAlignment(Qt.AlignCenter)
        font = self.message.font()
        font.setPointSize(20)
        self.message.setFont(font)
        self.plot_2d.addWidget(self.message)

        self.addTab(self.plot_1d, "1D")
        self.addTab(self.plot_2d, "2D")
        self.addTab(QWidget(), "3D")
        self.set_1d_mode(False)
        self.setCornerWidget(QLabel("Visualization Pane    "), corner=Qt.TopLeftCorner)

    def set_1d_mode(self, oned):
        self.setTabEnabled(0, oned)
        self.setTabEnabled(1, not oned)
        self.setTabEnabled(2, False)

    def set_ws(self, field):

        if field is not None:
            try:
                ws = field.to_md_histo_workspace()
            except Exception as e:
                self._parent.show_failure_msg("Failed to generate field",
                                              str(e),
                                              traceback.format_exc())
                ws = None
        else:
            ws = None

        if ws:
            if len(ws.getNonIntegratedDimensions()) == 1:
                self.set_1d_mode(True)
                if self.oneDViewer:
                    self.plot_1d.removeWidget(self.oneDViewer)
                fig = Figure()
                self.oneDViewer = FigureCanvas(fig)

                # get scan direction
                for d in ('x', 'y', 'z'):
                    dim = getattr(field, d)
                    if not np.allclose(dim, dim[0], atol=0.1):
                        scan_dir = d

                # create simple 1D plot
                ax = fig.add_subplot()
                ax.errorbar(getattr(field, scan_dir), field.values, field.errors, marker='o')
                ax.set_xlabel(f'{scan_dir} (mm)')

                self.plot_1d.addWidget(self.oneDViewer)
                self.plot_1d.setCurrentIndex(1)
            else:
                self.set_1d_mode(False)
                if self.strainSliceViewer:
                    self.strainSliceViewer.set_new_workspace(ws)
                else:
                    self.strainSliceViewer = StrainSliceViewer(ws, parent=self)
                    self.plot_2d.addWidget(self.strainSliceViewer.view)
                    self.plot_2d.setCurrentIndex(1)
                self.strainSliceViewer.set_new_field(field,
                                                     bin_widths=[ws.getDimension(n).getBinWidth() for n in range(3)])
        else:
            self.set_1d_mode(False)
            if self.oneDViewer is not None:
                self.plot_1d.removeWidget(self.oneDViewer)
                self.oneDViewer = None
            if self.strainSliceViewer is not None:
                self.plot_2d.removeWidget(self.strainSliceViewer.view)
                self.strainSliceViewer = None

    def set_message(self, text):
        self.message.setText(text)


class StrainStressViewer(QSplitter):
    def __init__(self, model, ctrl, parent=None):
        self._model = model
        self._model.propertyUpdated.connect(self.updatePropertyFromModel)
        self._model.failureMsg.connect(self.show_failure_msg)
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
        self.mechanicalConstants.youngModulus.editingFinished.connect(self.update_plot)
        self.mechanicalConstants.poissonsRatio.editingFinished.connect(self.update_plot)
        left_layout.addWidget(self.mechanicalConstants)

        left_layout.addStretch(0)

        left.setLayout(left_layout)

        self.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()

        self.plot_select = PlotSelect(self)
        self.plot_select.measure_dir.currentTextChanged.connect(self.update_plot)
        self.plot_select.plot_param.currentTextChanged.connect(self.update_plot)
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
        self.update_plot()

    def measure_dir_changed(self):
        self.update_plot()

    def update_plot(self):
        if self.plot_select.get_plot_param() == 'stress' or (self.plot_select.get_plot_param() == 'strain' and
                                                             self.plot_select.get_direction() == "33" and
                                                             self.stressCase.get_stress_case() == "In-plane stress"):
            validated = self.controller.validate_stress_selection(self.stressCase.get_stress_case(),
                                                                  self.mechanicalConstants.youngModulus.text(),
                                                                  self.mechanicalConstants.poissonsRatio.text())
        else:
            validated = self.controller.validate_selection(self.plot_select.get_direction(),
                                                           self.stressCase.get_stress_case() != 'diagonal')

        if validated is None:
            if self.plot_select.get_plot_param() == 'stress' or (self.plot_select.get_plot_param() == 'strain' and
                                                                 self.plot_select.get_direction() == "33" and
                                                                 self.stressCase.get_stress_case()
                                                                 == "In-plane stress"):
                self.controller.calculate_stress(self.stressCase.get_stress_case(),
                                                 self.mechanicalConstants.youngModulus.text(),
                                                 self.mechanicalConstants.poissonsRatio.text())

            self.viz_tab.set_ws(self.model.get_field(direction=self.plot_select.get_direction(),
                                                     plot_param=self.plot_select.get_plot_param(),
                                                     stress_case=self.stressCase.get_stress_case()))
        else:
            self.viz_tab.set_ws(None)
            self.viz_tab.set_message(validated)

    def updatePropertyFromModel(self, name):
        getattr(self, name)(getattr(self.model, name))

    def peakTags(self, peak_tags):
        self.peak_selection.peak_select.currentTextChanged.disconnect()
        self.peak_selection.clear_peak_tags()
        self.peak_selection.peak_select.currentTextChanged.connect(self.controller.peakSelected)
        self.peak_selection.set_peak_tags(peak_tags)

    def selectedPeak(self, peak):
        self.d0.set_d0(self.model.d0)
        self.update_plot()

    def show_failure_msg(self, msg, info, details):
        self.viz_tab.set_message(msg)
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(msg)
        msgBox.setInformativeText(info)
        msgBox.setDetailedText(details)
        msgBox.exec()
