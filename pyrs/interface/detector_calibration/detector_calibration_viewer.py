from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton, QComboBox  # type:ignore
from qtpy.QtWidgets import QGroupBox, QSplitter, QFileDialog  # type:ignore
# from qtpy.QtWidgets import QStyledItemDelegate, QDoubleSpinBox  # type:ignore
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QSlider, QTabWidget  # type:ignore

from qtpy.QtWidgets import QGridLayout  # type:ignore
from qtpy.QtWidgets import QMainWindow  # type:ignore
from qtpy.QtCore import Qt  # type: ignore

from pyrs.interface.gui_helper import pop_message
from pyrs.utilities import get_input_project_file  # type: ignore
from pyrs.interface.threading.worker_pool import WorkerPool

from matplotlib import rcParams

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# import traceback
import os
import json

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
            self.browse_button.clicked.connect(self.openCalibFileDialog)
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

    def openCalibFileDialog(self):
        self._parent._calibration_input, _ = QFileDialog.getOpenFileNames(self,
                                                                          self.name,
                                                                          "",
                                                                          self.fileType,
                                                                          options=QFileDialog.DontUseNativeDialog)

        if self._parent._calibration_input:
            if type(self._parent._calibration_input) is list:
                self._parent._calibration_input = self._parent._calibration_input[0]

            self._parent.calib_summary._cal_summary_table._initalize_calibration()

    def openFileDialog(self):
        self._parent._nexus_file, _ = QFileDialog.getOpenFileNames(self,
                                                                   self.name,
                                                                   "",
                                                                   self.fileType,
                                                                   options=QFileDialog.DontUseNativeDialog)

        if self._parent._nexus_file:
            if type(self._parent._nexus_file) is list:
                self._parent._nexus_file = self._parent._nexus_file[0]

            self.load_nexus()

    def loadRunNumber(self):

        try:
            project_dir = get_input_project_file(int(self.lineEdit.text()),
                                                 preferredType=self._parent.fileLoading.run_location.currentText())
        except (FileNotFoundError, RuntimeError, ValueError) as run_err:
            pop_message(self, f'Failed to find run {self.lineEdit.text()}',
                        str(run_err), 'error')
            return

        self._parent._nexus_file = os.path.join(project_dir, f'HB2B_{self.lineEdit.text()}.h5')
        self._parent._run_number = int(self.lineEdit.text())

        self.load_nexus()

    def load_nexus(self):
        self._parent.controller.load_nexus(self._parent._nexus_file,
                                           self._parent.peak_lines_setup.tthbin_lineEdit.text(),
                                           self._parent.peak_lines_setup.etabin_lineEdit.text())

        self._parent.compare_diff_data.sl.setMaximum(self._parent.model.sub_runs.size)
        self._parent.compare_diff_data.valueChanged()

        self._parent.peak_lines_setup.setup_calibration_table(self._parent.controller.get_powders())
        self._parent.calib_summary._cal_summary_table.set_wavelength(0, self._parent.controller.get_wavelength())

    def setFilenamesText(self, filenames):
        self.lineEdit.setText(filenames)


# class SpinBoxDelegate(QStyledItemDelegate):
#     def __init__(self):
#         super().__init__()

#     def createEditor(self, parent, option, index):
#         editor = QDoubleSpinBox(parent)
#         editor.setMinimum(0)
#         editor.setDecimals(5)
#         editor.setSingleStep(0.0001)
#         return editor


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
                                   'rot x', 'rot y', 'rot z', 'tth 0', 'wavelength'])

        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems(['shift x', 'shift y', 'distance',
                                   'rot x', 'rot y', 'rot z', 'tth 0', 'wavelength'])

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
            return super(FixFigureCanvas, self).resizeEvent(event)


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

        self._parent.controller.update_diffraction_view(self.ax, self._parent, sub_run, self.two_dim)

        plt.tight_layout()
        self.canvas.draw()

    def getWidget(self):
        return FixFigureCanvas(self.figure)

    def update_param_view(self, x_item, y_item, x_text, y_text):

        self._parent.controller.plot_2D_params(self.ax, x_item, y_item)

        self.ax.set_xlabel(x_text)
        self.ax.set_ylabel(y_text)

        plt.tight_layout()

        self.canvas.draw()


class PeakLinesSetupView(QGroupBox):
    worker_pool = WorkerPool()

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
        self._parent.controller.set_reduction_param(self.tthbin_lineEdit.text(),
                                                    self.etabin_lineEdit.text())

    def setup_label(self, label_name, align=Qt.AlignRight):
        label = QLabel(label_name)
        label.setAlignment(align | Qt.AlignVCenter)

        return label

    def setup_combo_box(self):
        temp_combo_box = QComboBox(self)
        temp_combo_box.addItems(['', 'wavelength', 'wavelength_tth0', 'rotations', 'geometry',
                                 'shifts', 'shift x', 'shift y', 'distance', 'full', 'tth0'])

        return temp_combo_box

    def get_keep_list(self):
        keep_list = []
        for i_row in range(self.calibrant_table.rowCount()):
            keep_list.append(self.calibrant_table.item(i_row, 1).checkState() == 0)

        return np.array(keep_list)

    def fit_peaks(self):
        self.set_reduction_param()
        self.set_calibration_params()
        keep_list = self.get_keep_list()
        rmse, deltatth = self._parent.controller.fit_diffraction_peaks(keep_list)
        self._parent.compare_diff_data.valueChanged()  # auto plot data after fitting

        self._parent.calib_summary.rmse_lineEdit.setText('{0:3f}'.format(rmse))
        self._parent.calib_summary.delta_tth_lineEdit.setText('{0:4f}'.format(deltatth))

    def calibrate_detector(self):
        self.set_reduction_param()
        self.set_calibration_params()
        self.set_refinement_params()

        exclude_list = self.get_keep_list()
        fit_recipe = [self.recipe_combos[i_row].currentText() for i_row in range(8)]

        self.load_info.setEnabled(False)
        self.export_recipe.setEnabled(False)
        self.fit.setEnabled(False)
        self.calibrate.setEnabled(False)

        # do action
        args = [fit_recipe, exclude_list]
        self.worker = self.worker_pool.createWorker(target=self._parent.controller.calibrate_detector, args=(args))
        self.worker.finished.connect(lambda: self.load_info.setEnabled(True))
        self.worker.finished.connect(lambda: self.export_recipe.setEnabled(True))
        self.worker.finished.connect(lambda: self.fit.setEnabled(True))
        self.worker.finished.connect(lambda: self.calibrate.setEnabled(True))
        self.worker.result.connect(self._handleFailure)
        self.worker.success.connect(lambda success: self.calibration_success if success else None)

        self.worker_pool.submitWorker(self.worker)

    def calibration_success(self):
        print('Calibration was a success')

    def update_calibration(self, calibration, calibration_error, r_sum, rmse):

        self._parent.calib_summary._cal_summary_table.set_calibration(calibration, calibration_error)
        self._parent.compare_diff_data.valueChanged()  # auto plot data after fitting
        self._parent.param_window.change_plot()

        self._parent.calib_summary.rmse_lineEdit.setText('{0:3f}'.format(rmse))
        self._parent.calib_summary.delta_tth_lineEdit.setText('{0:4f}'.format(r_sum))

    def _handleFailure(self, result):

        if len(result[0]) == 1:
            calibration = result[0]
            calibration_error = result[1]
            r_sum = result[2]
            rmse = result[3]

            self.update_calibration(calibration, calibration_error, r_sum, rmse)

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

    def set_calibration_params(self):
        self._parent.controller.set_calibration_params(self._parent.calib_summary._cal_summary_table.get_calibration())

    def set_refinement_params(self):
        self._parent.controller.set_refinement_params(self._parent.calib_summary.method_combo_box.currentText(),
                                                      self._parent.calib_summary.neval_lineEdit.text())

    def save_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Calibration Recipe",
                                                  "JSON (*.json)",
                                                  options=QFileDialog.DontUseNativeDialog)
        if not filename:
            return

        output_dict = {}

        # write nexus file
        if self._parent._run_number is not None:
            output_dict['run_number'] = self._parent._run_number

        if self._parent._nexus_file is not None:
            output_dict['nexus_file'] = self._parent._nexus_file

        # write input calibration
        if self._parent._calibration_input is not None:
            output_dict['input_calibration'] = self._parent._calibration_input

        # write exclude list
        output_dict['keep'] = self.get_keep_list().tolist()

        # write routines
        Method = ''
        for i_item in range(8):
            if self.recipe_combos[i_item].currentText() != '':
                recipe = '{},{}'.format(Method, self.recipe_combos[i_item].currentText())

        output_dict['recipe'] = recipe[1:]

        # write tth bins
        output_dict['tth_bin'] = self.tthbin_lineEdit.text()

        # write eta bins
        output_dict['eta_bin'] = self.etabin_lineEdit.text()

        output_dict['method'] = self._parent.calib_summary.method_combo_box.currentText()
        output_dict['neval'] = self._parent.calib_summary.neval_lineEdit.text()

        print(output_dict)
        with open(filename, "w") as outfile:
            outfile.write(json.dumps(output_dict, indent=4))

    def load_json(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Load Calibration Recipe",
                                                  "JSON (*.json);;All Files (*)",
                                                  options=QFileDialog.DontUseNativeDialog)
        if not filename:
            return

        with open(filename, 'r') as openfile:
            # Reading from json file
            input_dict = json.load(openfile)

        # load new nexus data if data are not currently loaded
        if self._parent._nexus_file != input_dict['nexus_file']:
            self._parent._nexus_file = input_dict['nexus_file']
            self._parent.fileLoading.file_load_dilg.load_nexus()

        methods = ['', 'wavelength', 'wavelength_tth0', 'rotations', 'geometry',
                   'shifts', 'shift x', 'shift y', 'distance', 'full', 'tth0']

        if 'recipe' in input_dict.keys():
            for i_item, item in enumerate(input_dict['recipe'].split(',')):
                self.recipe_combos[i_item].setCurrentIndex(methods.index(item))

        # load input calibration if specified

        for key in list(input_dict.keys()):
            if key == 'input_calibration':
                self._parent._calibration_input = input_dict['input_calibration']
                self._parent.calib_summary._cal_summary_table._initalize_calibration(self._parent._calibration_input)
            elif key == 'keep':
                for i_keep, keep in enumerate(input_dict['keep']):
                    if keep is False:
                        self.calibrant_table.item(i_keep, 1).setCheckState(2)
            elif key == 'tth_bin':
                self.tthbin_lineEdit.setText(input_dict['tth_bin'])
            elif key == 'eta_bin':
                self.etabin_lineEdit.setText(input_dict['eta_bin'])
            elif key == 'method':
                self._parent.calib_summary.set_method(input_dict[key])
            elif key == 'neval':
                self._parent.calib_summary.neval_lineEdit.setText(str(input_dict[key]))


class CalibrationSummaryView(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        _layout = QHBoxLayout()

        self._cal_summary_table = CalibrationSummaryTable(self)
        _layout.addWidget(self._cal_summary_table)

        calib_summary = QGroupBox()
        calib_layout = QGridLayout()

        empty_label = QLabel('')
        rmse_label = QLabel('RMSE')
        rmse_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rmse_lineEdit = QLineEdit()
        self.rmse_lineEdit.setText('')
        self.rmse_lineEdit.setReadOnly(True)
        self.rmse_lineEdit.setFixedWidth(200)

        delta_tth_label = QLabel('Δ 2θ')
        delta_tth_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.delta_tth_lineEdit = QLineEdit()
        self.delta_tth_lineEdit.setText('')
        self.delta_tth_lineEdit.setReadOnly(True)
        self.delta_tth_lineEdit.setFixedWidth(200)

        neval_label = QLabel('max nfev')
        neval_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.neval_lineEdit = QLineEdit()
        self.neval_lineEdit.setText('300')
        self.neval_lineEdit.setFixedWidth(200)

        method_label = QLabel('refinement method')
        method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.method_combo_box = QComboBox(self)
        self.method_combo_box.addItems(['trf', 'dogbox', 'lm'])

        for i_row in range(3):
            calib_layout.addWidget(empty_label, i_row, 0)

        calib_layout.addWidget(method_label, i_row + 1, 0)
        calib_layout.addWidget(self.method_combo_box, i_row + 1, 1)
        calib_layout.addWidget(neval_label, i_row + 2, 0)
        calib_layout.addWidget(self.neval_lineEdit, i_row + 2, 1)
        calib_layout.addWidget(rmse_label, i_row + 3, 0)
        calib_layout.addWidget(self.rmse_lineEdit, i_row + 3, 1)
        calib_layout.addWidget(delta_tth_label, i_row + 4, 0)
        calib_layout.addWidget(self.delta_tth_lineEdit, i_row + 4, 1)

        self.export_calib_bttn = QPushButton("Export Calibration")
        self.export_calib_bttn.clicked.connect(self.export_calib)
        self.export_local_calib_bttn = QPushButton("Save Local Calibration")
        self.export_local_calib_bttn.clicked.connect(self.export_local_calib)

        calib_layout.addWidget(self.export_calib_bttn, i_row + 5, 0, 1, 2)
        calib_layout.addWidget(self.export_local_calib_bttn, i_row + 6, 0, 1, 2)
        calib_summary.setLayout(calib_layout)
        _layout.addWidget(calib_summary)

        self.setLayout(_layout)

    def set_method(self, method):
        methods = ['trf', 'dogbox', 'lm']
        self.method_combo_box.setCurrentIndex(methods.index(method))

    def export_calib(self, filename=None):
        self._parent.controller.export_calibration(filename=filename)

    def export_local_calib(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Calibration",
                                                  "JSON (*.json)",
                                                  options=QFileDialog.DontUseNativeDialog)

        if not filename:
            return

        self.export_calib(filename=filename)


class CalibrationSummaryTable(QTableWidget):
    COL_SIZE = 100
    STATUS_COL_SIZE = 500  # last column

    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        self.setColumnCount(1)

        self.labels = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'TTH_0', 'Lambda']

        self.setRowCount(len(self.labels))
        self.setVerticalHeaderLabels(self.labels)

        self.setHorizontalHeaderLabels(['Starting'])

        for i_row in range(8):
            self.setItem(i_row, 0, QTableWidgetItem("0"))

    def _initalize_calibration(self, json_input=None):
        if json_input is None:
            json_input = self._parent._parent._calibration_input

        if json_input is not None:
            with open(json_input, 'r') as openfile:
                # Reading from json file
                input_dict = json.load(openfile)

            for i_lable, label in enumerate(self.labels):
                print(i_lable, label)
                print(input_dict.keys())
                try:
                    self.setItem(i_lable, 0, QTableWidgetItem(str(input_dict[label])))
                except KeyError:
                    pass

    def set_calibration(self, calibration_list, calibration_error_list):

        for i_calibration in range(len(calibration_list)):
            calibration = calibration_list[i_calibration]
            col_index = self._add_column()
            for i_lable in range(len(self.labels)):
                self.setItem(i_lable, col_index, QTableWidgetItem('{0:3f}'.format(calibration[i_lable])))

    def get_calibration(self):
        col_nbr = self.get_number_of_columns()

        return [float(self.item(i_row, col_nbr - 1).text()) for i_row in range(8)]

    def set_wavelength(self, column, item):
        self.setItem(7, column, QTableWidgetItem(str(item)))

    def _add_column(self):
        current_cols = self.get_number_of_columns()
        self.setColumnCount(current_cols + 1)
        return current_cols

    def get_number_of_columns(self):
        return self.columnCount()


class DetectorCalibrationViewer(QMainWindow):
    def __init__(self, detector_calib_model, detector_calib_ctrl, parent=None):

        self._model = detector_calib_model
        self._ctrl = detector_calib_ctrl
        self._nexus_file = None
        self._run_number = None
        self._calibration_input = None

        super().__init__(parent)

        self.setWindowTitle("PyRS Detector Calibration Window")

        self.splitter = QSplitter()
        self.splitter.setHandleWidth(10)
        self.setCentralWidget(self.splitter)

        left = QWidget()
        left_layout = QVBoxLayout()

        self.fileLoading = FileLoading(self)
        self.peak_lines_setup = PeakLinesSetupView(self)
        self.calib_summary = CalibrationSummaryView(self)

        left_layout.addWidget(self.fileLoading)
        left_layout.addWidget(self.peak_lines_setup)
        left_layout.addWidget(self.calib_summary)

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
        right_layout.addWidget(self.viz_splitter)
        right.setLayout(right_layout)

        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model

    # def show_failure_msg(self, msg, info, details):
    #     self.viz_tab.set_message(msg)
    #     msgBox = QMessageBox()
    #     msgBox.setIcon(QMessageBox.Critical)
    #     msgBox.setText(msg)
    #     msgBox.setInformativeText(info)
    #     msgBox.setDetailedText(details)
    #     msgBox.exec()
