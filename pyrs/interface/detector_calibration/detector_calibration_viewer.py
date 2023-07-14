from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton, QComboBox  # type:ignore
from qtpy.QtWidgets import QGroupBox, QSplitter  # type:ignore
from qtpy.QtWidgets import QRadioButton, QFileDialog, QCheckBox  # type:ignore
from qtpy.QtWidgets import QStyledItemDelegate, QDoubleSpinBox  # type:ignore
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QSlider, QTabWidget  # type:ignore

#  from qtpy.QtWidgets import QGridLayout, QMessageBox, QMenu  # type:ignore
from qtpy.QtWidgets import QGridLayout, QMessageBox  # type:ignore
from qtpy.QtWidgets import QMainWindow, QAction  # type:ignore
from qtpy.QtGui import QColor  # type:ignore
from qtpy.QtCore import Qt  # type: ignore

from pyrs.interface.gui_helper import pop_message
from pyrs.utilities import get_input_project_file  # type: ignore

from matplotlib import rcParams

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# import traceback
import os

COLOR_FAILED_FITTING = QColor(247, 173, 13)  # orange
SUCCESS = "success"
MICROSTRAIN = u"\u00b5strain"


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
        self._parent = parent
        super().__init__(parent)
        self.name = name
        self.fileType = fileType
        layout = QHBoxLayout()
        if name == "Run Number:":
            label = QLabel(name)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(label)
            self.lineEdit = QLineEdit()
            self.lineEdit.setReadOnly(False)
            self.lineEdit.setFixedWidth(50)

            layout.addWidget(self.lineEdit)

            self.browse_button = QPushButton("Load")
            self.browse_button.clicked.connect(self.loadRunNumber)
        elif name == "Load Calibration":
            self.browse_button = QPushButton("Load Calibration")
            self.browse_button.clicked.connect(self.openFileDialog)
        else:
            if name is None:
                self.browse_button = QPushButton("Browse Exp Data")
            else:
                self.browse_button = QPushButton("Browse")

            self.browse_button.clicked.connect(self.openFileDialog)

        layout.addWidget(self.browse_button)
        self.setLayout(layout)

    def _reset_fit_data(self):
        self._parent.fit_summary.fit_table_operator.fits = None
        self._parent.fit_summary.fit_table_operator.fit_result = None

    def openFileDialog(self):
        self._parent._project_file, _ = QFileDialog.getOpenFileNames(self,
                                                                     self.name,
                                                                     "",
                                                                     self.fileType,
                                                                     options=QFileDialog.DontUseNativeDialog)

        if self._parent._project_file:
            if type(self._parent._project_file) is list:
                self._parent._project_file = self._parent._project_file[0]

            self.load_project_plot()

    def loadRunNumber(self):

        try:
            project_dir = get_input_project_file(int(self.lineEdit.text()),
                                                 preferredType=self._parent.fileLoading.run_location.currentText())
        except (FileNotFoundError, RuntimeError, ValueError) as run_err:
            pop_message(self, f'Failed to find run {self.lineEdit.text()}',
                        str(run_err), 'error')
            return

        self._parent._project_file = os.path.join(project_dir, f'HB2B_{self.lineEdit.text()}.h5')

        self.load_project_plot()

    def load_project_plot(self):
        self._parent.controller.load_nexus(self._parent._project_file,
                                           self._parent.peak_lines_setup.tthbin_lineEdit.text(),
                                           self._parent.peak_lines_setup.etabin_lineEdit.text())

        self._parent.compare_diff_data.sl.setMaximum(self._parent.model.sub_runs.size)
        self._parent.compare_diff_data.valueChanged()

        self._parent.peak_lines_setup.setup_calibration_table(self._parent._ctrl.get_powders())
        self._parent.calib_summary.set_wavelength(0, self._parent._model.get_wavelength)

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
        self.setTitle("Load Calibration Data")
        layout = QHBoxLayout()

        self.file_load_run_number = FileLoad(name="Run Number:", parent=parent)
        self.file_load_dilg = FileLoad(name=None, parent=parent,
                                       fileType="Nexus (*.nxs.h5)")
        self.file_load_calib = FileLoad(name="Load Calibration", parent=parent,
                                        fileType="json (*.json);;All Files (*)")
        layout.addWidget(self.file_load_run_number)
        layout.addWidget(self.file_load_dilg)
        layout.addWidget(self.file_load_calib)
        self.setLayout(layout)

    def set_text_values(self, direction, text):
        getattr(self, f"file_load_e{direction}").setFilenamesText(text)


class DiffractionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent

        panel_layout = QVBoxLayout()
        self.setLayout(panel_layout)

        self.tab_widgets = [PlotView(parent, True), PlotView(parent)]

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab_widgets[0], "2D")
        self.tabs.addTab(self.tab_widgets[1], "1D")
        panel_layout.addWidget(self.tabs)

        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(1)
        self.sl.setMaximum(2)
        self.sl.setValue(1)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(1)
        panel_layout.addWidget(self.sl)

        self.tabs.currentChanged.connect(self.valueChanged)
        self.sl.valueChanged.connect(self.valueChanged)

    def valueChanged(self):
        self.tabs.currentWidget().update_diff_view(self.sl.value())


class VisualizeResults(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent

        param_layout = QGridLayout()
        # panel_layout = QVBoxLayout()

        self.plot_paramX = QComboBox()
        self.plot_paramX.addItems(["iteration", 'shift x', 'shift y', 'distance',
                                   'rot x', 'rot y', 'rot z', 'wavelength'])

        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems(['shift x', 'shift y', 'distance',
                                   'rot x', 'rot y', 'rot z', 'wavelength'])

        self.plot_paramX.currentIndexChanged.connect(self.change_plot)
        self.plot_paramY.currentIndexChanged.connect(self.change_plot)

        plot_labelY = QLabel("Y-axis")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.param_vew = PlotView(parent=parent, param_view=True)

        param_layout.addWidget(plot_labelX, 0, 0)
        param_layout.addWidget(self.plot_paramX, 0, 1)
        param_layout.addWidget(plot_labelY, 0, 2)
        param_layout.addWidget(self.plot_paramY, 0, 3)

        param_layout.addWidget(self.param_vew, 1, 0, 5, 4)

        self.setLayout(param_layout)

    def change_plot(self):
        self.param_vew.update_param_view(self.plot_paramX.currentIndex(),
                                         self.plot_paramY.currentIndex(),
                                         self.plot_paramX.currentText(),
                                         self.plot_paramY.currentText())


class FixFigureCanvas(FigureCanvas):
    def resizeEvent(self, event):
        if event.size().width() <= 0 or event.size().height() <= 0:
            return
        super(FixFigureCanvas, self).resizeEvent(event)


class PlotView(QWidget):
    def __init__(self, parent=None, two_dim=False, param_view=False):
        super().__init__(parent)
        self._parent = parent

        self.figure, self.ax = plt.subplots(1, 1)

        plt.tight_layout()
        self.setLayout(QVBoxLayout())
        self.canvas = self.getWidget()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.toolbar)
        self.two_dim = two_dim
        if two_dim:
            self.ax.axis('off')
        elif param_view:
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
        else:
            self.ax.set_xlabel(r"2$\theta$ ($deg.$)")
            self.ax.set_ylabel("Intensity (ct.)")

        plt.tight_layout()

    def update_diff_view(self, sub_run):

        self._parent._ctrl.update_diffraction_view(self.ax, self._parent, sub_run, self.two_dim)

        plt.tight_layout()
        self.canvas.draw()

    def getWidget(self):
        return FixFigureCanvas(self.figure)

    def update_param_view(self, x_item, y_item, x_text, y_text):

        self._parent._ctrl.plot_2D_params(self.ax, x_item, y_item)

        self.ax.set_xlabel(x_text)
        self.ax.set_ylabel(y_text)

        plt.tight_layout()

        self.canvas.draw()


class PeakLinesSetupView(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        layout = QHBoxLayout()

        self.splitter = QSplitter()

        self.calibration_setup = QGroupBox()
        self.calibration_setup.setTitle("Calibration Powders")
        self.calibration_setup_layout = QVBoxLayout()

        self.calibrant_table = QTableWidget(self)
        self.calibrant_table.setColumnCount(2)
        self.calibrant_table.setHorizontalHeaderLabels(['Calibrant', 'exclude'])
        self.calibrant_table.resizeColumnsToContents()

        self.calibration_setup_layout.addWidget(self.calibrant_table)
        self.calibration_setup.setLayout(self.calibration_setup_layout)

        self.recipe_setup = QGroupBox()
        self.recipe_setup.setTitle("Define Calibration Recipe")
        recipe_setup_layout = QGridLayout()

        self.recipe_combos = {}
        recipe_setup_layout.addWidget(self.setup_label('Step'), 0, 0)
        recipe_setup_layout.addWidget(self.setup_label('Routine', Qt.AlignCenter), 0, 1)

        for i_row in range(8):
            self.recipe_combos[i_row] = self.setup_combo_box()
            recipe_setup_layout.addWidget(self.setup_label('{}'.format(i_row + 1)), i_row + 1, 0)
            recipe_setup_layout.addWidget(self.recipe_combos[i_row], i_row + 1, 1, 1, 3)


        tthbin_label = QLabel('2θ bin')
        tthbin_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tthbin_lineEdit = QLineEdit()
        self.tthbin_lineEdit.setText('512')
        self.tthbin_lineEdit.setReadOnly(False)
        self.tthbin_lineEdit.setFixedWidth(75)

        etabin_label = QLabel('η bin')
        etabin_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.etabin_lineEdit = QLineEdit()
        self.etabin_lineEdit.setText('3')
        self.etabin_lineEdit.setReadOnly(False)
        self.etabin_lineEdit.setFixedWidth(50)

        self.tthbin_lineEdit.textChanged.connect(self.set_reduction_param)
        self.etabin_lineEdit.textChanged.connect(self.set_reduction_param)

        self.load_info = QPushButton("Load Recipe")
        self.load_info.clicked.connect(self.load_json)
        self.export_recipe = QPushButton("Export Recipe")
        self.export_recipe.clicked.connect(self.save_json)
        self.fit = QPushButton("Fit Peaks")
        self.fit.clicked.connect(self.fit_peaks)
        self.calibrate = QPushButton("Calibrate")
        self.calibrate.clicked.connect(self.calibrate_detector)

        recipe_setup_layout.addWidget(tthbin_label, 9, 0)
        recipe_setup_layout.addWidget(self.tthbin_lineEdit, 9, 1)
        recipe_setup_layout.addWidget(etabin_label, 9, 2)
        recipe_setup_layout.addWidget(self.etabin_lineEdit, 9, 3)

        recipe_setup_layout.addWidget(self.load_info, 10, 0, 1, 2)
        recipe_setup_layout.addWidget(self.export_recipe, 10, 2, 1, 2)
        recipe_setup_layout.addWidget(self.fit, 11, 0, 1, 2)
        recipe_setup_layout.addWidget(self.calibrate, 11, 2, 1, 2)

        self.recipe_setup.setLayout(recipe_setup_layout)

        self.splitter.addWidget(self.calibration_setup)
        self.splitter.addWidget(self.recipe_setup)

        layout.addWidget(self.splitter)

        self.setLayout(layout)

        return

    def set_reduction_param(self):
        self._parent._ctrl.set_reduction_param(self.tthbin_lineEdit.text(),
                                               self.etabin_lineEdit.text())

    def setup_label(self, label_name, align=Qt.AlignRight):
        label = QLabel(label_name)
        label.setAlignment(align | Qt.AlignVCenter)

        return label

    def setup_combo_box(self):
        temp_combo_box = QComboBox(self)
        temp_combo_box.addItems(['', 'wavelength', 'wavelength+shift x', 'rotations', 'geometry',
                                 'shifts', 'shift x', 'shift y', 'distance', 'full'])

        return temp_combo_box

    def get_exclude_list(self):
        exclude_list = []
        for i_row in range(self.calibrant_table.rowCount()):
            exclude_list.append(self.calibrant_table.item(i_row, 1).checkState() != 0)

        return np.array(exclude_list)

    def fit_peaks(self):
        self.set_reduction_param()
        exclude_list = self.get_exclude_list()
        self._parent._ctrl.fit_diffraction_peaks(exclude_list)

    def calibrate_detector(self):
        self.set_reduction_param()
        exclude_list = self.get_exclude_list()
        fit_recipe = [self.recipe_combos[i_row].currentText() for i_row in range(8)]
        # fit_recipe = [self.recipe_combos[i_row].currentText() for i_row in range(8)]
        self._parent._ctrl.calibrate_detector(fit_recipe, exclude_list)

    def setup_calibration_table(self, powders):
        self.calibrant_table.setRowCount(len(powders))

        for row, string in enumerate(powders):
            powder_item = QTableWidgetItem(string)
            powder_item.setText(string)
            chkBoxItem = QTableWidgetItem('')
            chkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(0)
            self.calibrant_table.setItem(row, 0, powder_item)
            self.calibrant_table.setItem(row, 1, chkBoxItem)

    def save_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Fit Information",
                                                  "JSON (*.json);;All Files (*)",
                                                  options=QFileDialog.DontUseNativeDialog)
        if not filename:
            return
        self._parent.controller.save_fit_range(filename, self.fit_range_table)

    def load_json(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Export Peak Fit Information",
                                                  "JSON (*.json);;All Files (*)",
                                                  options=QFileDialog.DontUseNativeDialog)
        if not filename:
            return

        self.fit_range_table.setRowCount(0)
        self._parent.controller.load_fit_range(filename, self.fit_range_table)


class CalibrationSummaryView(QGroupBox):
    COL_SIZE = 100
    STATUS_COL_SIZE = 500  # last column

    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        layout = QVBoxLayout()
        # layout.addStretch(0, 10)
        layout.setStretch(0, 1)
        print(layout.maximumSize())
        self.setLayout(layout)

        self.tableView_CalibSummary = QTableWidget(self)
        self.tableView_CalibSummary.setColumnCount(1)
        # self.tableView_CalibSummary.setMinimumHeight(60)

        layout.addWidget(self.tableView_CalibSummary)

        labels = ['shift x', 'shift y', 'distance', 'rot x', 'rot y', 'rot z', 'wavelength']
        self.tableView_CalibSummary.setRowCount(len(labels))
        self.tableView_CalibSummary.setVerticalHeaderLabels(labels)

        self.tableView_CalibSummary.setHorizontalHeaderLabels(['Starting'])

        for i_row in range(7):
            self.tableView_CalibSummary.setItem(i_row, 0, QTableWidgetItem("0"))

    def set_wavelength(self, column, item):
        self.tableView_CalibSummary.setItem(6, column, QTableWidgetItem(str(item)))

    def initialize_table_column_size(self):
        nbr_column = self._parent.tableView_fitSummary.columnCount()
        for _col in np.arange(nbr_column):
            if _col < (nbr_column - 1):
                _col_size = self.COL_SIZE
            else:
                _col_size = self.STATUS_COL_SIZE
        self._parent.tableView_fitSummary.setColumnWidth(_col, _col_size)

    def _clear_rows(self):
        _nbr_row = self._parent.tableView_fitSummary.rowCount()
        for _ in np.arange(_nbr_row):
            self._parent.tableView_fitSummary.removeRow(0)

    def _clear_columns(self):
        _nbr_column = self.get_number_of_columns()
        for _ in np.arange(_nbr_column):
            self._parent.tableView_fitSummary.removeColumn(0)

    def get_number_of_columns(self):
        _nbr_column = self._parent.tableView_fitSummary.columnCount()
        return _nbr_column

    def _clear_table(self):
        self._clear_rows()
        self._clear_columns()


class DetectorCalibrationViewer(QMainWindow):
    def __init__(self, detector_calib_model, detector_calib_ctrl, parent=None):

        self._model = detector_calib_model
        self._ctrl = detector_calib_ctrl
        self._project_file = None

        super().__init__(parent)

        self.setWindowTitle("PyRS Detector Calibration Window")

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
        # self.setCentralWidget(self.fileLoading)
        left_layout.addWidget(self.fileLoading)

        self.fit_splitter = QSplitter(Qt.Vertical)
        self.fit_splitter.setHandleWidth(10)

        self.peak_lines_setup = PeakLinesSetupView(self)
        self.calib_summary = CalibrationSummaryView(self)

        self.fit_splitter.addWidget(self.peak_lines_setup)
        self.fit_splitter.addWidget(self.calib_summary)

        left_layout.addWidget(self.fit_splitter)

        left_layout.addStretch(10)
        left.setLayout(left_layout)
        self.splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()

        self.viz_splitter = QSplitter(Qt.Vertical)
        self.compare_diff_data = DiffractionWindow(self)
        self.param_window = VisualizeResults(self)

        self.viz_splitter.addWidget(self.compare_diff_data)
        self.viz_splitter.addWidget(self.param_window)

        # # add widgets to pannel layout
        # right_layout.addWidget(self.plot_select)
        right_layout.addWidget(self.viz_splitter)
        # right_layout.addWidget(self.VizSetup)
        right.setLayout(right_layout)

        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        self.resize(1024, 1080)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model

    def update_param_plots(self):
        self.update_2D_param_summary()
        self.update_3D_param_summary()

    def update_2D_param_summary(self):
        if self.model.ws is not None:
            if (self.plot_select.get_X != "") and (self.plot_select.get_Y != ""):
                self.param_window.update_param_view(self.plot_select.get_X,
                                                    self.plot_select.get_Y,
                                                    self.plot_select.get_PeakNum,
                                                    self.plot_select.get_out_of_plan_angle)

    def update_3D_param_summary(self):
        if self.model.ws is not None:
            if (self.VizSetup.get_X() != "") and (self.VizSetup.get_Y() != "") and (self.VizSetup.get_Z() != ""):
                self.compare_param_window.update_3D_view(self.VizSetup.get_X(),
                                                         self.VizSetup.get_Y(),
                                                         self.VizSetup.get_Z(),
                                                         self.plot_select.get_PeakNum,
                                                         self.plot_select.get_out_of_plan_angle,
                                                         self.VizSetup.get_sub_run_list())

    def update_peak_selection(self):
        self.fit_summary.fit_table_operator.change_fit(self.fit_summary.out_of_plan_angle)
        self.update_param_plots()

    def update_oop_select(self):
        self.plot_select.out_of_plane.setCurrentIndex(self.fit_summary.out_of_plane.currentIndex())
        self.fit_summary.fit_table_operator.change_fit(self.fit_summary.out_of_plan_angle)
        self.fit_window.update_diff_view(self.model.sub_runs[self.fit_setup.sl.value()])
        self.update_param_plots()

    def show_failure_msg(self, msg, info, details):
        self.viz_tab.set_message(msg)
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(msg)
        msgBox.setInformativeText(info)
        msgBox.setDetailedText(details)
        msgBox.exec()

    def save(self):
        if self._project_file is not None:
            self.controller.save(self._project_file,
                                 self._parent.fit_summary.fit_table_operator.fit_result)

    def saveas(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save HidraWorkspace",
                                                  "",
                                                  "HDF5 (*.h5);;All Files (*)")
        if filename:
            self.controller.save(filename, self._parent.fit_summary.fit_table_operator.fit_result)

    def load(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Load Stress state",
                                                  "",
                                                  "JSON (*.json);;All Files (*)")
        if filename:
            self.controller.load(filename)
