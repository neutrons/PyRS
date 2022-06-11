from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton, QComboBox  # type:ignore
from qtpy.QtWidgets import QGroupBox, QSplitter  # type:ignore
from qtpy.QtWidgets import QRadioButton, QFileDialog, QCheckBox  # type:ignore
from qtpy.QtWidgets import QStyledItemDelegate, QDoubleSpinBox  # type:ignore
from qtpy.QtWidgets import QTableWidget  # type:ignore
from qtpy.QtWidgets import QGridLayout, QMessageBox  # type:ignore
from qtpy.QtWidgets import QMainWindow, QAction  # type:ignore
# QTableWidgetItem, QTabWidget
from matplotlib import rcParams

from qtpy.QtCore import Qt
# , Signal

# import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, tight_layout
# import traceback
import os

# Can't run VTK embedded in PyQT5 using VirtualGL
# See https://gitlab.kitware.com/vtk/vtk/-/issues/17338
USING_THINLINC = "TLSESSIONDATA" in os.environ

scale_factor = 1
marker_size = 5 * scale_factor

rcParams['axes.labelsize'] = 10 * scale_factor
rcParams['xtick.labelsize'] = 9 * scale_factor
rcParams['ytick.labelsize'] = 9 * scale_factor
rcParams['lines.linewidth'] = 2 * scale_factor
rcParams['axes.linewidth'] = 1.0 * scale_factor / 2.
rcParams['xtick.major.size'] = 3.5 * scale_factor
rcParams['xtick.minor.size'] = 2 * scale_factor
rcParams['xtick.major.width'] = 0.8 * scale_factor / 2.
rcParams['xtick.minor.width'] = 0.8 * scale_factor / 2.
rcParams['ytick.major.size'] = 3.5 * scale_factor
rcParams['ytick.minor.size'] = 2 * scale_factor
rcParams['ytick.major.width'] = 0.8 * scale_factor / 2.
rcParams['ytick.minor.width'] = 0.8 * scale_factor / 2.
rcParams["savefig.dpi"] = 200


class FileLoad(QWidget):
    def __init__(self, name=None, fileType="HidraProjectFile (*.h5);;All Files (*)", parent=None):
        self.parent = parent
        super().__init__(parent)
        self.name = name
        self.fileType = fileType
        layout = QHBoxLayout()
        if name is not None:
            label = QLabel(name)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(label)
            self.lineEdit = QLineEdit()
            self.lineEdit.setReadOnly(True)
            layout.addWidget(self.lineEdit)

        if name == "Run Number:":
            self.lineEdit.setReadOnly(False)
            self.lineEdit.setFixedWidth(50)
            self.browse_button = QPushButton("Load")
            self.browse_button.clicked.connect(self.loadRunNumber)
        else:
            if name is None:
                self.browse_button = QPushButton("Browse Exp Data")
            else:
                self.browse_button = QPushButton("Browse")

            self.browse_button.clicked.connect(self.openFileDialog)

        layout.addWidget(self.browse_button)
        self.setLayout(layout)

    def openFileDialog(self):
        fileNames, _ = QFileDialog.getOpenFileNames(self,
                                                    self.name,
                                                    "",
                                                    self.fileType,
                                                    options=QFileDialog.DontUseNativeDialog)
        if fileNames:
            success = self.parent.controller.filesSelected(self.name, fileNames)
            if success:
                self.lineEdit.setText(', '.join(os.path.basename(filename) for filename in fileNames))
            else:
                self.lineEdit.setText(None)
            self.parent.update_plot()

    def loadRunNumber(self):
        fileNames = False
        print(self.lineEdit.text())
        if fileNames:
            success = self.parent.controller.filesSelected(self.name, fileNames)
            if success:
                self.lineEdit.setText(', '.join(os.path.basename(filename) for filename in fileNames))
            else:
                self.lineEdit.setText(None)
            self.parent.update_plot()

    def setFilenamesText(self, filenames):
        self.lineEdit.setText(filenames)


class SpinBoxDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def createEditor(self, parent, option, index):
        editor = QDoubleSpinBox(parent)
        editor.setMinimum(0)
        editor.setDecimals(5)
        editor.setSingleStep(0.0001)
        return editor


class FileLoading(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Define Fitting File")
        layout = QHBoxLayout()
        # self.file_load_e11 = FileLoad("e11", parent=parent)
        self.file_load_run_number = FileLoad(name="Run Number:", parent=parent)
        self.file_load_dilg = FileLoad(name=None, parent=parent)
        # layout.addWidget(self.file_load_e11)
        layout.addWidget(self.file_load_run_number)
        layout.addWidget(self.file_load_dilg)
        self.setLayout(layout)

    def set_text_values(self, direction, text):
        getattr(self, f"file_load_e{direction}").setFilenamesText(text)


class SetupViz(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QGridLayout()

        self.contour_bt = QRadioButton("Contour")
        self.lines_bt = QRadioButton("3D Lines")
        self.scatter_bt = QRadioButton("3D Scatter")
        self.contour_bt.setChecked(False)
        self.contour_bt.toggled.connect(self.btnstate(self.contour_bt))
        self.lines_bt.setChecked(True)
        self.lines_bt.toggled.connect(self.btnstate(self.lines_bt))
        self.scatter_bt.setChecked(True)
        self.scatter_bt.toggled.connect(self.btnstate(self.scatter_bt))

        layout.addWidget(self.contour_bt, 0, 3)
        layout.addWidget(self.lines_bt, 0, 4)
        layout.addWidget(self.scatter_bt, 0, 5)

        self.plot_paramX = QComboBox()
        self.plot_paramX.addItems(["sub_run"])
        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelX, 1, 0)
        layout.addWidget(self.plot_paramX, 1, 1)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems(["sub_run"])
        plot_labelY = QLabel("Y-axis")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelY, 1, 2)
        layout.addWidget(self.plot_paramY, 1, 3)
        self.plot_paramZ = QComboBox()
        self.plot_paramZ.addItems(["sub_run"])
        plot_labelZ = QLabel("Z-axis")
        plot_labelZ.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelZ, 1, 4)
        layout.addWidget(self.plot_paramZ, 1, 5)

        plot_label_sub = QLabel("List Sub Runs")
        self.lineEdit = QLineEdit()
        self.lineEdit.setReadOnly(False)
        self.lineEdit.setFixedWidth(50)
        example_label = QLabel('(ex: 1,2,3... or 3-5,8)')
        layout.addWidget(plot_label_sub, 2, 3)
        layout.addWidget(self.lineEdit, 2, 4)
        layout.addWidget(example_label, 2, 5)

        self.setLayout(layout)

    def btnstate(self, button):
        print(button.text())


class PlotSelect(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        # layout.setFieldGrowthPolicy(0)
        self.plot_paramX = QComboBox()
        self.plot_paramX.addItems(["sub_run"])
        self.plot_paramX.setCurrentIndex(self.plot_paramX.findText('sub_run'))
        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelX)
        layout.addWidget(self.plot_paramX)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems([])
        # layout.addCol(QLabel("Y-axis"), self.measure_dir)
        plot_labelY = QLabel("Y-axis")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelY)
        layout.addWidget(self.plot_paramY)
        self.setLayout(layout)

    def get_direction(self):
        return self.plot_paramY.currentText()

    def get_plot_param(self):
        return self.plot_paramX.currentText()


class FixFigureCanvas(FigureCanvas):
    def resizeEvent(self, event):
        if event.size().width() <= 0 or event.size().height() <= 0:
            return
        super(FixFigureCanvas, self).resizeEvent(event)


class PlotView(QWidget):
    def __init__(self, parent=None, fit_view=False, three_dim=False):
        super().__init__(parent)

        # self.fit_view = fit_view
        if fit_view:
            self.figure, self.ax = subplots(2, 1, sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1]})
            self.ax[1].set_xlabel(r"2$\theta$ ($deg.$)")
            self.ax[0].set_ylabel("Intensity (ct.)")
            self.ax[1].set_ylabel("Diff (ct.)")
        elif three_dim:
            self.figure, self.ax = subplots(1, 1)
        else:
            self.figure, self.ax = subplots(1, 1)

        tight_layout()
        grid = QVBoxLayout(self)
        self.canvas = self.getWidget()

        grid.addWidget(self.canvas)
        self.setLayout(grid)

    def update_diff_view(self, tth, i_data, fit=None):
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].plot(tth, i_data, 'k')
        if fit is not None:
            self.ax[1].plot(tth, fit - i_data, 'r')

    def getWidget(self):
        return FixFigureCanvas(self.figure)

    def update_param_view(self, xvalues, yvalues, xlabel, ylabel):
        self.ax.plot(xvalues, yvalues, color='k', marker='D', linestyle="--")
        self.canvas.draw()


class FitSetupView(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        layout = QHBoxLayout()

        self.splitter = QSplitter()

        self.fit_range = QGroupBox()
        self.fit_range_layout = QVBoxLayout()
        self.fit_range.setTitle("Peak Ranges")

        self.save_load_fit = QGroupBox()
        self.save_load_fit_layout = QHBoxLayout()

        self.save_fit_info = QPushButton("Save Peaks")
        self.save_fit_info.clicked.connect(self.save_json)

        self.load_fit_info = QPushButton("Load Peaks")
        self.load_fit_info.clicked.connect(self.load_json)

        self.save_load_fit_layout.addWidget(self.load_fit_info)
        self.save_load_fit_layout.addWidget(self.save_fit_info)
        self.save_load_fit.setLayout(self.save_load_fit_layout)

        self.table = QTableWidget(self)
        self.table.setRowCount(2)
        self.table.setColumnCount(3)

        self.fit_range_layout.addWidget(self.save_load_fit)
        self.fit_range_layout.addWidget(self.table)
        self.fit_range.setLayout(self.fit_range_layout)

        self.peak_setup = QGroupBox()
        self.peak_setup_layout = QVBoxLayout()

        self.sub_runs_select = QGroupBox()
        self.sub_runs_select_layout = QHBoxLayout()
        self.sub_runs_select.setTitle("Sub Runs")
        self.lineEdit = QLineEdit()
        self.lineEdit.setReadOnly(False)
        self.lineEdit.setFixedWidth(50)
        self.sub_runs_select_layout.addWidget(self.lineEdit)
        example_label = QLabel('(ex: 1,2,3... or 3-5,8)')
        example_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sub_runs_select_layout.addWidget(example_label)

        self.sub_runs_select.setLayout(self.sub_runs_select_layout)

        self.fit_setup = QGroupBox()
        self.fit_setup.setTitle("Fitting Functions")
        self.fit_setup_layout = QHBoxLayout()

        self.peak_model = QComboBox()
        self.peak_model.addItems(["Pseudo Voigt", "Gaussian", "Lorentez"])

        self.peak_back = QComboBox()
        self.peak_back.addItems(["Linear", "Quadraditc"])

        self.fit_peaks = QPushButton("Export Peak Information")
        self.fit_peaks.clicked.connect(self.fit)

        self.fit_setup_layout.addWidget(self.peak_model)
        self.fit_setup_layout.addWidget(self.peak_back)
        self.fit_setup.setLayout(self.fit_setup_layout)

        self.export = QPushButton("Export Peak Information")
        self.export.clicked.connect(self.save_CSV)

        self.peak_setup_layout.addWidget(self.sub_runs_select)
        self.peak_setup_layout.addWidget(self.fit_setup)
        self.peak_setup_layout.addWidget(self.export)

        self.peak_setup.setLayout(self.peak_setup_layout)

        self.splitter.addWidget(self.fit_range)
        self.splitter.addWidget(self.peak_setup)

        layout.addWidget(self.splitter)

        self.setLayout(layout)

    def save_CSV(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "CSV (*.csv);;All Files (*)")
        if not filename:
            return
        self._parent.controller.write_stress_to_csv(filename, self.detailed.isChecked())

    def save_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Fit Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "JSON (*.json);;All Files (*)")
        if not filename:
            return
        self._parent.controller.write_stress_to_csv(filename, self.detailed.isChecked())

    def load_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Fit Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "JSON (*.json);;All Files (*)")
        if not filename:
            return
        self._parent.controller.write_stress_to_csv(filename, self.detailed.isChecked())

    def fit(self):

        return


class FitSummaryView(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        layout = QVBoxLayout()

        self.summary_select = QGroupBox()
        self.summary_select_layout = QHBoxLayout()

        self.value_button = QPushButton('Value', self)
        self.error_button = QPushButton('Error', self)
        self.value_button.clicked.connect(self.btn_click)
        self.error_button.clicked.connect(self.btn_click)

        self.summary_select_layout.addWidget(self.value_button)
        self.summary_select_layout.addWidget(self.error_button)

        self.summary_select.setLayout(self.summary_select_layout)

        self.table = QTableWidget(self)
        self.table.setRowCount(2)
        self.table.setColumnCount(2)

        layout.addWidget(self.summary_select)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def btn_click(self):
        filename = None

        if not filename:
            return
        self._parent.controller.write_stress_to_csv(filename, self.detailed.isChecked())


class CSVExport(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)
        self.setTitle("CSV Export")
        layout = QHBoxLayout()
        self.export = QPushButton("Export Grid Information")
        self.export.clicked.connect(self.save_CSV)
        self.detailed = QCheckBox("Detailed")
        layout.addWidget(self.export)
        layout.addWidget(self.detailed)
        self.setLayout(layout)

        self.setEnabled(False)

    def setEnabled(self, enabled):
        self.export.setEnabled(enabled)
        self.detailed.setEnabled(enabled)

    def save_CSV(self):
        self._parent.calculate_stress()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Grid Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "CSV (*.csv);;All Files (*)")
        if not filename:
            return
        self._parent.controller.write_stress_to_csv(filename, self.detailed.isChecked())


class TextureFittingUI(QMainWindow):
    def __init__(self, fit_peak_model, fit_peak_ctrl, parent=None):

        self.fit_peak_core = fit_peak_model
        self.controller = fit_peak_ctrl

        super().__init__(parent)

        self.setWindowTitle("PyRS Texture Fitting Window")

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        self.saveAction = QAction('&Save', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Save project state')
        self.saveAction.triggered.connect(self.save)
        self.saveAction.setEnabled(False)
        fileMenu.addAction(self.saveAction)

        self.saveAction = QAction('&Save as', self)
        self.saveAction.setStatusTip('Save project state')
        self.saveAction.triggered.connect(self.saveas)
        self.saveAction.setEnabled(False)
        fileMenu.addAction(self.saveAction)

        self.loadAction = QAction('&Load state', self)
        self.loadAction.setStatusTip('Load application state')
        self.loadAction.triggered.connect(self.load)
        fileMenu.addAction(self.loadAction)

        fileMenu.addSeparator()
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        self.splitter = QSplitter()
        self.splitter.setHandleWidth(10)
        self.setCentralWidget(self.splitter)

        left = QWidget()
        left_layout = QVBoxLayout()

        self.fileLoading = FileLoading(self)
        left_layout.addWidget(self.fileLoading)

        self.fit_splitter = QSplitter(Qt.Vertical)
        self.fit_splitter.setHandleWidth(10)

        self.fit_window = PlotView(self, fit_view=True)
        self.fit_setup = FitSetupView(self)
        self.fit_summary = FitSummaryView(self)

        self.fit_splitter.addWidget(self.fit_window)
        self.fit_splitter.addWidget(self.fit_setup)
        self.fit_splitter.addWidget(self.fit_summary)

        left_layout.addWidget(self.fit_splitter)

        left_layout.addStretch(0)
        left.setLayout(left_layout)

        self.splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()

        self.plot_select = PlotSelect(self)
        self.plot_select.plot_paramX.currentTextChanged.connect(self.update_plot)
        self.plot_select.plot_paramY.currentTextChanged.connect(self.update_plot)
        right_layout.addWidget(self.plot_select)

        self.viz_splitter = QSplitter(Qt.Vertical)
        self.param_window = PlotView(self)
        self.compare_param_window = PlotView(self, three_dim=True)

        self.viz_splitter.addWidget(self.param_window)
        self.viz_splitter.addWidget(self.compare_param_window)

        right_layout.addWidget(self.viz_splitter)

        self.VizSetup = SetupViz(self)
        right_layout.addWidget(self.VizSetup)

        right.setLayout(right_layout)

        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 5)

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
                self.calculate_stress()
            print('plotting from here')

            self.viz_tab.set_ws(self.model.get_field(direction=self.plot_select.get_direction(),
                                                     plot_param=self.plot_select.get_plot_param(),
                                                     stress_case=self.stressCase.get_stress_case()))
        else:
            self.viz_tab.set_ws(None)
            self.viz_tab.set_message(validated)

        if self.controller.validate_stress_selection(self.stressCase.get_stress_case(),
                                                     self.mechanicalConstants.youngModulus.text(),
                                                     self.mechanicalConstants.poissonsRatio.text()) is None:
            self.calculate_stress()
            self.enable_stress_output(True)
        else:
            self.enable_stress_output(False)

    def enable_stress_output(self, enable):
        self.csvExport.setEnabled(enable)
        self.d0.setEnabled(enable)
        self.saveAction.setEnabled(enable)

    def updatePropertyFromModel(self, name):
        getattr(self, name)(getattr(self.model, name))

    def peakTags(self, peak_tags):
        self.peak_selection.peak_select.currentTextChanged.disconnect()
        self.peak_selection.clear_peak_tags()
        self.peak_selection.peak_select.currentTextChanged.connect(self.controller.peakSelected)
        self.peak_selection.set_peak_tags(peak_tags)

    def selectedPeak(self, peak):
        self.update_plot()

    def modelUpdated(self, modelUpdated):
        stressCase, selectedPeak, youngs_modulus, poisson_ratio = modelUpdated

        self.mechanicalConstants.set_values(youngs_modulus, poisson_ratio)

        self.peak_selection.peak_select.currentTextChanged.disconnect()
        self.peak_selection.set_selected_peak(selectedPeak)
        self.peak_selection.peak_select.currentTextChanged.connect(self.controller.peakSelected)

        self.stressCase.dimChanged.disconnect()
        self.stressCase.set_stress_case(stressCase)
        self.stressCase.dimChanged.connect(self.dimChanged)
        self.fileLoading.file_load_e33.setDisabled(stressCase.lower() != "diagonal")

        for d in ("11", "22", "33"):
            self.fileLoading.set_text_values(d,
                                             ", ".join(os.path.basename(filename)
                                                       for filename in self.model.get_filenames_for_direction(d)))
        self.update_d0_from_model()
        self.update_plot()

    def show_failure_msg(self, msg, info, details):
        self.viz_tab.set_message(msg)
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(msg)
        msgBox.setInformativeText(info)
        msgBox.setDetailedText(details)
        msgBox.exec()

    def calculate_stress(self):
        self.controller.calculate_stress(self.stressCase.get_stress_case(),
                                         self.mechanicalConstants.youngModulus.text(),
                                         self.mechanicalConstants.poissonsRatio.text(),
                                         self.d0.get_d0())
        self.update_d0_from_model()

    def update_d0_from_model(self):
        d0 = self.model.d0
        if d0 is None:
            self.d0.set_d0(None, None)
            self.d0.set_d0_field(None, None, None, None, None)
        else:
            self.d0.set_d0(d0.values[0], d0.errors[0])
            self.d0.set_d0_field(d0.x, d0.y, d0.z, d0.values, d0.errors)

    def save(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Stress state",
                                                  "",
                                                  "JSON (*.json);;All Files (*)")
        if filename:
            self.controller.save(filename)

    def saveas(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save HidraWorkspace",
                                                  "",
                                                  "HDF5 (*.h5);;All Files (*)")
        if filename:
            self.controller.save(filename)

    def load(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Load Stress state",
                                                  "",
                                                  "JSON (*.json);;All Files (*)")
        if filename:
            self.controller.load(filename)
