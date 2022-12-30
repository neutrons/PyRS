from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton, QComboBox  # type:ignore
from qtpy.QtWidgets import QGroupBox, QSplitter, QSpinBox  # type:ignore
from qtpy.QtWidgets import QRadioButton, QFileDialog, QCheckBox  # type:ignore
from qtpy.QtWidgets import QStyledItemDelegate, QDoubleSpinBox  # type:ignore
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QSlider  # type:ignore
from qtpy.QtWidgets import QGridLayout, QMessageBox, QMenu  # type:ignore
from qtpy.QtWidgets import QMainWindow, QAction, QTableWidgetSelectionRange  # type:ignore
from qtpy.QtGui import QColor  # type:ignore
# from qtpy.QtCore.Qt import CustomContextMenu  # type:ignore

from pyrs.interface.gui_helper import pop_message
from pyrs.utilities import get_input_project_file  # type: ignore

# QTableWidgetItem, QTabWidget
from matplotlib import rcParams

from qtpy.QtCore import Qt
# , Signal

import numpy as np
from scipy.interpolate import griddata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
# import subplots, tight_layout, figure
from matplotlib.backend_bases import MouseButton
from matplotlib.cm import coolwarm

# import traceback
import os
import copy

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
        self._reset_fit_data()
        self._parent.controller.load_projectfile(self._parent._project_file)
        self._parent.fit_window.update_diff_view(self._parent.model.sub_runs[0])
        self._parent.fit_setup.sl.setMaximum(self._parent.model.sub_runs.size - 1)
        self._parent.fit_summary.setup_out_of_plane_angle(self._parent.model.ws.reduction_masks)
        self._parent.plot_select.setup_out_of_plane_angle(self._parent.model.ws.reduction_masks)

        self._parent.VizSetup.enable_polar_plot(self._parent.model.ws.reduction_masks)

        self._parent.update_param_plots()

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
        self.run_location = QComboBox()
        self.run_location.addItems(["auto", "manual"])
        run_location_label = QLabel("Reduction")
        run_location_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(run_location_label)
        layout.addWidget(self.run_location)

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
        self._parent = parent

        layout = QGridLayout()

        self.shift_bt = QCheckBox("shift")
        self.shift_bt.setChecked(False)
        self.shift_bt.setVisible(False)
        self.shift_bt.stateChanged.connect(lambda: self.btnstate(self.shift_bt))

        self.polar_bt = QRadioButton("Polar")
        self.polar_bt.setChecked(False)
        self.polar_bt.toggled.connect(lambda: self.btnstate(self.polar_bt))
        self.polar_bt.setVisible(False)

        self.contour_bt = QRadioButton("Contour")
        self.contour_bt.setChecked(False)
        self.contour_bt.toggled.connect(lambda: self.btnstate(self.contour_bt))

        self.lines_bt = QRadioButton("3D Lines")
        self.lines_bt.setChecked(False)
        self.lines_bt.toggled.connect(lambda: self.btnstate(self.lines_bt))

        self.scatter_bt = QRadioButton("3D Scatter")
        self.scatter_bt.setChecked(True)
        self.scatter_bt.toggled.connect(lambda: self.btnstate(self.scatter_bt))

        layout.addWidget(self.shift_bt, 0, 1)
        layout.addWidget(self.polar_bt, 0, 2)
        layout.addWidget(self.contour_bt, 0, 3)
        layout.addWidget(self.lines_bt, 0, 4)
        layout.addWidget(self.scatter_bt, 0, 5)

        self.plot_paramX = QComboBox()
        self.plot_paramX.addItems(["sub-runs", "vx", "vy", "vz", "sx", "sy", "sz",
                                   "phi", "chi", "omega"])
        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelX, 1, 0)
        layout.addWidget(self.plot_paramX, 1, 1)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems(["sub-runs", "vx", "vy", "vz", "sx", "sy", "sz",
                                   "phi", "chi", "omega"])
        plot_labelY = QLabel("Y-axis")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelY, 1, 2)
        layout.addWidget(self.plot_paramY, 1, 3)
        self.plot_paramZ = QComboBox()
        self.plot_paramZ.addItems(["sub-runs", "vx", "vy", "vz", "sx", "sy", "sz",
                                   "phi", "chi", "omega"])
        plot_labelZ = QLabel("Z-axis")
        plot_labelZ.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelZ, 1, 4)
        layout.addWidget(self.plot_paramZ, 1, 5)

        plot_label_sub = QLabel("List Sub Runs")
        self.sub_runs_list = QLineEdit()
        self.sub_runs_list.setReadOnly(False)
        self.sub_runs_list.setFixedWidth(75)
        example_label = QLabel('(ex: 1,2,3... or 3-5,8)')
        layout.addWidget(plot_label_sub, 2, 3)
        layout.addWidget(self.sub_runs_list, 2, 4)
        layout.addWidget(example_label, 2, 5)

        self.setLayout(layout)

    def btnstate(self, button):
        self._parent.update_3D_param_summary()

    def get_X(self):
        return self.plot_paramX.currentText()

    def get_Y(self):
        return self.plot_paramY.currentText()

    def get_Z(self):
        return self.plot_paramZ.currentText()

    def get_sub_run_list(self):
        return self.sub_runs_list.text()

    def enable_polar_plot(self, dict_keys):
        if len(dict_keys) > 2:
            self.polar_bt.setVisible(True)
            self.shift_bt.setVisible(True)


class PlotSelect(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent)
        layout = QHBoxLayout()
        # layout.setFieldGrowthPolicy(0)
        self.plot_paramX = QComboBox()
        self.plot_paramX.addItems(["sub-runs", "vx", "vy", "vz", "sx", "sy", "sz",
                                   "phi", "chi", "omega"])
        plot_labelX = QLabel("X-axis")
        plot_labelX.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelX)
        layout.addWidget(self.plot_paramX)
        self.plot_paramY = QComboBox()
        self.plot_paramY.addItems(["sub-runs", "vx", "vy", "vz", "sx", "sy", "sz",
                                   "phi", "chi", "omega"])
        plot_labelY = QLabel("Y-axis")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(plot_labelY)
        layout.addWidget(self.plot_paramY)

        self.out_of_plane = QComboBox()
        self.oop_label = QLabel("out of plane")
        self.oop_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.oop_label.setVisible(False)
        self.out_of_plane.setVisible(False)
        layout.addWidget(self.oop_label)
        layout.addWidget(self.out_of_plane)

        self.plot_peakNum = QComboBox()
        peackNum_label = QLabel("Peak")
        peackNum_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(peackNum_label)
        layout.addWidget(self.plot_peakNum)

        self.setLayout(layout)

    @property
    def get_out_of_plan_angle(self):
        return self.out_of_plane.currentText()

    @property
    def get_Y(self):
        return self.plot_paramY.currentText()

    @property
    def get_X(self):
        return self.plot_paramX.currentText()

    @property
    def get_PeakNum(self):
        return self.plot_peakNum.currentText()

    def setup_out_of_plane_angle(self, dict_keys):
        if len(dict_keys) > 2:
            angles = []
            for key in list(dict_keys):
                if '_var' not in key:
                    angles.append(key)

            self.out_of_plane.clear()
            self.out_of_plane.addItems(angles)
            self.oop_label.setVisible(True)
            self.out_of_plane.setVisible(True)
        else:
            self.out_of_plane.clear()
            self.oop_label.setVisible(False)
            self.out_of_plane.setVisible(False)


class FixFigureCanvas(FigureCanvas):
    def resizeEvent(self, event):
        if event.size().width() <= 0 or event.size().height() <= 0:
            return
        super(FixFigureCanvas, self).resizeEvent(event)


class PlotView(QWidget):
    def __init__(self, parent=None, fit_view=False, three_dim=False):
        super().__init__(parent)
        self._parent = parent

        if fit_view:
            self.peak_entry_x0 = None

            self.figure, self.ax = plt.subplots(2, 1, sharex=True,
                                                gridspec_kw={'height_ratios': [3, 1]})
            plt.connect('button_press_event', self.define_peak_tabel)
            plt.connect('motion_notify_event', self.define_peak_tabel)
            plt.connect('button_release_event', self.define_peak_tabel)

        elif three_dim:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(1, 1, 1, projection='3d')

        else:
            self.figure, self.ax = plt.subplots(1, 1)

        plt.tight_layout()
        self.setLayout(QVBoxLayout())
        self.canvas = self.getWidget()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.toolbar)

    def update_diff_view(self, sub_run):
        self.ax[0].clear()
        self.ax[1].clear()
        tth, int_vec = self._parent.controller.get_reduced_diffraction_data(sub_run,
                                                                            self._parent.fit_summary.out_of_plan_angle)

        self.ax[0].plot(tth[1:], int_vec[1:], 'k')

        self.draw_fit_range(tth.min(), tth.max())

        if self._parent.fit_summary.fit_table_operator.fit_result is not None:
            sub_run_index = int(np.where(self._parent.model.sub_runs == sub_run)[0])

            fit_data = self._parent.controller.get_fitted_data(sub_run_index,
                                                               self._parent.fit_summary.out_of_plan_angle)

            fit_index = fit_data[1] > 0
            self.ax[0].plot(fit_data[0][fit_index], fit_data[1][fit_index], 'r')
            self.ax[1].plot(fit_data[2][fit_index], fit_data[3][fit_index], 'r')

        self.ax[1].set_xlabel(r"2$\theta$ ($deg.$)")
        self.ax[0].set_ylabel("Intensity (ct.)")
        self.ax[1].set_ylabel("Diff (ct.)")

        plt.tight_layout()

        self.canvas.draw()

    def draw_fit_range(self, tthmin, tthmax):

        colors = ['b', 'g', 'm']

        for i_entry in range(self._parent.fit_setup.fit_range_table.rowCount()):
            try:
                x0 = float(self._parent.fit_setup.fit_range_table.item(i_entry, 0).text())
                x1 = float(self._parent.fit_setup.fit_range_table.item(i_entry, 1).text())

                if (x0 > tthmin) and (x1 < tthmax):
                    self.ax[0].axvline(x=x0, color=colors[i_entry])
                    self.ax[0].axvline(x=x1, color=colors[i_entry])
            except AttributeError:
                pass

    def getWidget(self):
        return FixFigureCanvas(self.figure)

    def update_param_view(self, xlabel, ylabel, peak_number=1, out_of_plane=None):

        if peak_number == "":
            peak_number = 1

        self.ax.clear()

        xdata, ydata = self._parent._ctrl.get_log_plot(xlabel, ylabel, peak=int(peak_number),
                                                       fit_object=self._parent.fit_summary.fit_table_operator,
                                                       out_of_plane=out_of_plane)

        if len(xdata) != len(ydata):
            self.ax.errorbar(xdata, ydata[0], yerr=ydata[1], color='k', ls='None')
        else:
            self.ax.plot(xdata, ydata, color='k', marker='D', linestyle='None')

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        plt.tight_layout()

        self.canvas.draw()

    def update_3D_view(self, xlabel, ylabel, zlabel, peak_number=1, out_of_plane=None, sub_run_list=[]):

        if peak_number == "":
            peak_number = 1

        self.ax.clear()

        if peak_number == "":
            peak_number = 1

        xdata, ydata, zdata = self._parent._ctrl.get_log_plot(xlabel, ylabel, zname=zlabel, peak=int(peak_number),
                                                              fit_object=self._parent.fit_summary.fit_table_operator,
                                                              out_of_plane=out_of_plane, include_list=sub_run_list)

        if len(xdata) != len(zdata):
            zdata = zdata[0]

        plot_scatter = False
        if ((ydata.size == np.unique(ydata).size) or
                (xdata.size == np.unique(xdata).size)):

            plot_scatter = True

        if (self._parent.VizSetup.polar_bt.isChecked()):
            polar_data = self._parent._ctrl.extract_polar_projection(peak_number=int(peak_number))

            if polar_data is not None:
                norm = plt.Normalize(polar_data[:, 2].min(), polar_data[:, 2].max())
                colors = coolwarm(norm(polar_data[:, 2]))

                alpha_shift = 0
                if self._parent.VizSetup.shift_bt.isChecked():
                    alpha_shift = 1

                self.ax.scatter(alpha_shift - polar_data[:, 0], polar_data[:, 1], polar_data[:, 2],
                                marker='D', color=colors)

                R, P = np.meshgrid(np.unique(polar_data[:, 0]), np.unique(polar_data[:, 1]))
                Z = griddata(((polar_data[:, 0], polar_data[:, 1])), polar_data[:, 2], (R, P), method='nearest')

                print(np.unique(R), np.unique(P))

                # # Express the mesh in the cartesian system.
                # X = (alpha_shift - R) * np.cos(P)
                # Y = (alpha_shift - R) * np.sin(P)

                # self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm',
                #                      linewidth=0, antialiased=False)

                plot_scatter = False
            else:
                plot_scatter = True

        if (self._parent.VizSetup.contour_bt.isChecked()) and (not plot_scatter):
            X, Y = np.meshgrid(np.unique(xdata), np.unique(ydata))
            Z = griddata(((xdata, ydata)), zdata, (X, Y), method='nearest')

            self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm',
                                 linewidth=0, antialiased=False)

        elif (self._parent.VizSetup.lines_bt.isChecked()) and (not plot_scatter):

            X, Y = np.meshgrid(np.unique(xdata), np.unique(ydata))
            Z = griddata(((xdata, ydata)), zdata, (X, Y), method='nearest')

            norm = plt.Normalize(Z.min(), Z.max())
            colors = coolwarm(norm(Z))
            rcount, ccount, _ = colors.shape

            surf = self.ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                                        facecolors=colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))

        elif (self._parent.VizSetup.scatter_bt.isChecked()) or (plot_scatter):

            norm = plt.Normalize(zdata.min(), zdata.max())
            colors = coolwarm(norm(zdata))

            self.ax.scatter(xdata, ydata, zdata, marker='D', color=colors)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_zlabel(zlabel)

        plt.tight_layout()

        self.canvas.draw()

    def define_peak_tabel(self, event):
        if event.button is MouseButton.LEFT:
            if self.peak_entry_x0 is None:
                self.peak_entry_x0 = 1
                self._parent.fit_setup.add_peak_tabel_entry(event.xdata,
                                                            event.xdata)
            elif event.name == 'motion_notify_event':
                self._parent.fit_setup.update_peak_tabel_entry(event.xdata)
            elif event.name == 'button_release_event':
                self._parent.fit_setup.update_peak_tabel_entry(event.xdata)
                self.peak_entry_x0 = None

        return


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

        self.fit_range_table = QTableWidget(self)
        self.fit_range_table.setColumnCount(4)
        self.fit_range_table.setHorizontalHeaderLabels(['min 2theta', 'max 2theta', 'Peak Label',  'd0'])
        self.fit_range_table.resizeColumnsToContents()

        self.fit_range_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fit_range_table.customContextMenuRequested.connect(self.remove_peak_range_table_row)

        self.fit_range_layout.addWidget(self.save_load_fit)
        self.fit_range_layout.addWidget(self.fit_range_table)
        self.fit_range.setLayout(self.fit_range_layout)

        self.peak_setup = QGroupBox()
        self.peak_setup_layout = QVBoxLayout()

        self.sub_runs_select = QGroupBox()
        # self.sub_runs_select_layout = QHBoxLayout()
        sub_runs_select_layout = QGridLayout()
        self.sub_runs_select.setTitle("Sub Runs")
        self.lineEdit = QLineEdit()
        self.lineEdit.setReadOnly(False)
        self.lineEdit.setFixedWidth(50)
        example_label = QLabel('(ex: 1,2,3... or 3-5,8)')
        example_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(1)
        self.sl.setValue(0)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(1)

        self.sl.valueChanged.connect(self.valuechange)

        sub_runs_select_layout.addWidget(self.sl, 0, 1)
        sub_runs_select_layout.addWidget(self.lineEdit, 1, 0)
        sub_runs_select_layout.addWidget(example_label, 1, 1)
        self.sub_runs_select.setLayout(sub_runs_select_layout)

        self.fit_setup = QGroupBox()
        self.fit_setup.setTitle("Fitting Functions")
        self.fit_setup_layout = QHBoxLayout()

        self.peak_model = QComboBox()
        self.peak_model.addItems(["PseudoVoigt", "Gaussian"])

        self.peak_back = QComboBox()
        self.peak_back.addItems(["Linear"])

        self.fit_peaks = QPushButton("Fit")
        self.fit_peaks.clicked.connect(self.fit)

        self.fit_setup_layout.addWidget(self.peak_model)
        self.fit_setup_layout.addWidget(self.peak_back)
        self.fit_setup_layout.addWidget(self.fit_peaks)
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

    def add_peak_tabel_entry(self, x0, x1):

        self.fit_range_table.insertRow(self.fit_range_table.rowCount())
        self.fit_range_table.setItem(self.fit_range_table.rowCount() - 1, 0,
                                     QTableWidgetItem(str(x0)))
        self.fit_range_table.setItem(self.fit_range_table.rowCount() - 1, 1,
                                     QTableWidgetItem(str(x1)))
        self.fit_range_table.setItem(self.fit_range_table.rowCount() - 1, 2,
                                     QTableWidgetItem(str('Peak_{}'.format(self.fit_range_table.rowCount()))))
        self.fit_range_table.setItem(self.fit_range_table.rowCount() - 1, 3,
                                     QTableWidgetItem(str(1.0)))

        return

    def update_peak_tabel_entry(self, x1):

        row = self.fit_range_table.rowCount() - 1

        if float(self.fit_range_table.item(row, 0).text()) > x1:
            self.fit_range_table.setItem(row, 1, self.fit_range_table.item(row, 0).text())
            self.fit_range_table.setItem(row, 0, QTableWidgetItem(str(x1)))
        else:
            self.fit_range_table.setItem(row, 1, QTableWidgetItem(str(x1)))

        return

    def save_CSV(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "CSV (*.csv);;All Files (*)")
        if not filename:
            return

        self._parent.controller.export_peak_data(filename, self._parent.fit_summary.fit_table_operator.fits)

    def save_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Fit Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "JSON (*.json);;All Files (*)")
        if not filename:
            return
        self._parent.controller.save_fit_range(filename)

    def load_json(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Export Peak Fit Information",
                                                  self._parent.model.get_default_csv_filename(),
                                                  "JSON (*.json);;All Files (*)")
        if not filename:
            return
        self._parent.controller.load_fit_range(filename)

    def remove_peak_range_table_row(self, pos):
        it = self.fit_range_table.itemAt(pos)
        if it is None:
            return

        selection = it.row()
        item_range = QTableWidgetSelectionRange(0, selection, self.fit_range_table.rowCount() - 1, selection)
        self.fit_range_table.setRangeSelected(item_range, True)

        menu = QMenu()
        delete_column_action = menu.addAction("Delete entry")
        action = menu.exec_(self.fit_range_table.viewport().mapToGlobal(pos))
        if action == delete_column_action:
            self.fit_range_table.removeRow(selection)

    def setup_view_param(self, peaks=1):
        plot_selct_params = []
        plot_3d_params = []
        for _param in self._parent.fit_summary.fit_table_operator.clean_param_names:
            if (self._parent.plot_select.plot_paramY.findData(_param) == -1):
                plot_selct_params.append(_param)
            if (self._parent.VizSetup.plot_paramZ.findData(_param) == -1):
                plot_3d_params.append(_param)

        self._parent.plot_select.plot_paramY.addItems(plot_selct_params)
        for i_peak in range(peaks):
            self._parent.plot_select.plot_peakNum.addItems(["{}".format(i_peak + 1)])

        self._parent.VizSetup.plot_paramZ.addItems(plot_3d_params)

    def fit(self):

        peak_label = []
        tth_min = []
        tth_max = []

        for peak_row in range(self.fit_range_table.rowCount()):
            if (self.fit_range_table.item(peak_row, 0) is not None and
                    self.fit_range_table.item(peak_row, 1) is not None):

                tth_min.append(float(self.fit_range_table.item(peak_row, 0).text()))
                tth_max.append(float(self.fit_range_table.item(peak_row, 1).text()))
                if self.fit_range_table.item(peak_row, 2) is None:
                    peak_label.append('peak_{}'.format(peak_row + 1))
                    self.fit_range_table.setItem(peak_row, 2, QTableWidgetItem('peak_{}'.format(peak_row + 1)))
                else:
                    peak_label.append(self.fit_range_table.item(peak_row, 2).text())

            if self.fit_range_table.item(peak_row, 3) is None:
                self.fit_range_table.setItem(peak_row, 3, QTableWidgetItem('1.0'))

        fit_results = self._parent.controller.fit_peaks(tth_min, tth_max, peak_label,
                                                        self.peak_model.currentText(), self.peak_back.currentText())

        self._parent.fit_summary.fit_table_operator.set_fit_dict(fit_results)
        self._parent.fit_summary.fit_table_operator.initialize_fit_result_widgets()
        self._parent.fit_summary.fit_table_operator.populate_fit_result_table()
        self._parent.fit_window.update_diff_view(self._parent.model.sub_runs[0])
        self.setup_view_param(peaks=len(tth_min))

    def valuechange(self):
        self._parent.fit_window.update_diff_view(self._parent.model.sub_runs[self.sl.value()])


class FitSummaryView(QGroupBox):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent=parent)

        layout = QVBoxLayout()

        self.summary_select = QGroupBox()
        self.summary_select_layout = QHBoxLayout()

        self.spinBox_peak_index = QSpinBox()
        self.spinBox_peak_index.setRange(1, 1)

        plot_labelY = QLabel("Peak Index")
        plot_labelY.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.out_of_plane = QComboBox()
        self.oop_label = QLabel("out of plane")
        self.oop_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.oop_label.setVisible(False)
        self.out_of_plane.setVisible(False)

        self.radioButton_fit_value = QRadioButton('Fit Param Values')
        self.radioButton_fit_value.setChecked(True)
        self.radioButton_fit_error = QRadioButton('Fit Param Errors')
        self.radioButton_fit_value.clicked.connect(self.btn_click)
        self.radioButton_fit_error.clicked.connect(self.btn_click)

        self.summary_select_layout.addWidget(plot_labelY)
        self.summary_select_layout.addWidget(self.spinBox_peak_index)
        self.summary_select_layout.addWidget(self.oop_label)
        self.summary_select_layout.addWidget(self.out_of_plane)
        self.summary_select_layout.addWidget(self.radioButton_fit_value)
        self.summary_select_layout.addWidget(self.radioButton_fit_error)

        self.summary_select.setLayout(self.summary_select_layout)

        self.tableView_fitSummary = QTableWidget(self)
        self.tableView_fitSummary.setColumnCount(1)

        layout.addWidget(self.summary_select)
        layout.addWidget(self.tableView_fitSummary)

        self.setLayout(layout)

        self.fit_table_operator = FitTable(parent=self)

    @property
    def out_of_plan_angle(self):
        return self.out_of_plane.currentText()

    def btn_click(self):
        if self.fit_table_operator.fit_result is not None:
            self.fit_table_operator.initialize_fit_result_widgets()
            self.fit_table_operator.populate_fit_result_table()

    def setup_out_of_plane_angle(self, dict_keys):
        if len(dict_keys) > 2:
            angles = []
            for key in list(dict_keys):
                if '_var' not in key:
                    angles.append(key)

            self.out_of_plane.clear()
            self.out_of_plane.addItems(angles)
            self.oop_label.setVisible(True)
            self.out_of_plane.setVisible(True)
        else:
            self.out_of_plane.clear()
            self.oop_label.setVisible(False)
            self.out_of_plane.setVisible(False)


class FitTable:

    COL_SIZE = 100
    STATUS_COL_SIZE = 500  # last column

    def __init__(self, parent=None):
        self._parent = parent
        self.fits = None
        self.fit_result = None

    def initialize_fit_result_widgets(self):
        self.change_fit(self._parent.out_of_plan_angle)
        self._initialize_list_of_peaks()
        self.initialize_table()
        self.initialize_table_column_size()

    def set_fit_dict(self, fit_dictionary):
        self.fits = copy.copy(fit_dictionary)

    def change_fit(self, key):
        if self.fits is not None:
            self.fit_result = copy.copy(self.fits[key])
            self._clear_rows()
            self.populate_fit_result_table()

    def populate_fit_result_table(self):
        _peak_selected = self._parent.spinBox_peak_index.value()
        _peak_collection = self.fit_result.peakcollections[_peak_selected-1]  # peak 1 is at 0 index

        _value = self._get_value_to_display(peak_collection=_peak_collection)
        _chisq = _peak_collection.fitting_costs
        _status = _peak_collection.get_fit_status()

        _d_spacing = self._get_d_spacing_to_display(peak_selected=_peak_selected,
                                                    peak_collection=_peak_collection)

        _microstrain_mapping = self._get_microstrain_mapping_to_display(peak_collection=_peak_collection)

        def set_item(value='', fitting_worked=True):
            _item = QTableWidgetItem(value)
            if not fitting_worked:
                _item.setBackground(COLOR_FAILED_FITTING)
            return _item

        for _row, _row_value in enumerate(_value):
            self._parent.tableView_fitSummary.insertRow(_row)
            _global_col_index = 0

            _fitting_worked = True if _status[_row] == SUCCESS else False

            for _local_col_index, _col_value in enumerate(_row_value):
                _item = set_item(value=str(np.round(_col_value, 5)), fitting_worked=_fitting_worked)
                self._parent.tableView_fitSummary.setItem(_row, _global_col_index, _item)
                _global_col_index += 1

            # add chisq values (but forget when error is selected
            if self._parent.radioButton_fit_value.isChecked():
                _item = set_item(value=str(_chisq[_row]), fitting_worked=_fitting_worked)
                self._parent.tableView_fitSummary.setItem(_row, _global_col_index, _item)
                _global_col_index += 1

            # add d-spacing
            _item = set_item(value=str(_d_spacing[_row]), fitting_worked=_fitting_worked)
            self._parent.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

            # add strain calculation
            _microstrain = _microstrain_mapping[_row]
            if np.isnan(_microstrain):
                str_strain_value = "nan"
            else:
                str_strain_value = str(np.int(_microstrain))
            _item = set_item(value=str_strain_value, fitting_worked=_fitting_worked)
            self._parent.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

            # add status message
            _item = set_item(value=_status[_row], fitting_worked=_fitting_worked)
            self._parent.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

    def _get_d_spacing_to_display(self, peak_selected=1, peak_collection=None):
        _d_reference = np.float(str(self._parent._parent.fit_setup.fit_range_table.item(peak_selected-1, 3).text()))

        peak_collection.set_d_reference(values=_d_reference)
        values, error = peak_collection.get_dspacing_center()
        if self._parent.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def _get_microstrain_mapping_to_display(self, peak_collection=None):
        values, error = peak_collection.get_strain(units='microstrain')
        if self._parent.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def _get_value_to_display(self, peak_collection):
        values, error = peak_collection.get_effective_params()
        if self._parent.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def fit_value_error_changed(self):
        self._clear_rows()
        self.populate_fit_result_table()

    def _initialize_list_of_peaks(self):
        nbr_peaks = len(self.fit_result.peakcollections)
        self._parent.spinBox_peak_index.setRange(1, nbr_peaks)

    def initialize_table(self):
        self._clear_table()
        columns_names = self._get_list_of_columns()
        for _column in np.arange(len(columns_names)):
            self._parent.tableView_fitSummary.insertColumn(_column)
        self._parent.tableView_fitSummary.setHorizontalHeaderLabels(columns_names)
        self.clean_param_names = self._get_list_of_columns(True)

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

    def _get_list_of_columns(self, plotting=False):
        _peak_collection = self.fit_result.peakcollections[0]
        values, _ = _peak_collection.get_effective_params()
        column_names = values.dtype.names
        clean_column_names = []
        for _col_index, _col_value in enumerate(column_names):
            if (_col_index == 0) and (not plotting):
                # _col_value = 'Sub-run #'
                _col_value = 'Peak Center'
            clean_column_names.append(_col_value)

        if self._parent.radioButton_fit_value.isChecked():
            # also add chisq
            clean_column_names.append('chisq')

        # add d-spacing column
        clean_column_names.append("d spacing")

        if plotting:
            clean_column_names.append("microstrain")
        else:
            # add strain-mapping column
            clean_column_names.append("strain mapping (" + MICROSTRAIN + ")")

            # add a status column
            clean_column_names.append("Status message")

        return clean_column_names

    def select_first_row(self):
        _nbr_column = self.get_number_of_columns()
        selection_first_row = QTableWidgetSelectionRange(0, 0, 0, _nbr_column-1)
        self._parent.tableView_fitSummary.setRangeSelected(selection_first_row, True)


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


class TextureFittingViewer(QMainWindow):
    def __init__(self, fit_peak_model, fit_peak_ctrl, parent=None):

        self._model = fit_peak_model
        self._ctrl = fit_peak_ctrl
        self._project_file = None

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

        # Update plot if fitsetup is changed
        self.fit_setup.fit_range_table.itemChanged.connect(self.update_diffraction_data_plot)

        # UPDATE options to change summary table and plot data
        self.fit_summary.out_of_plane.currentTextChanged.connect(self.update_oop_select)
        self.fit_summary.spinBox_peak_index.valueChanged.connect(self.update_peak_selection)

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
        self.plot_select.plot_paramX.currentTextChanged.connect(self.update_2D_param_summary)
        self.plot_select.plot_paramY.currentTextChanged.connect(self.update_2D_param_summary)

        # Update 2D and 3D plots
        self.plot_select.plot_peakNum.currentTextChanged.connect(self.update_param_plots)
        self.plot_select.out_of_plane.currentTextChanged.connect(self.sync_oop)

        self.viz_splitter = QSplitter(Qt.Vertical)
        self.param_window = PlotView(self)
        self.compare_param_window = PlotView(self, three_dim=True)

        self.viz_splitter.addWidget(self.param_window)
        self.viz_splitter.addWidget(self.compare_param_window)

        # Lower pannel for controlling 3D plotting
        self.VizSetup = SetupViz(self)
        self.VizSetup.plot_paramX.currentTextChanged.connect(self.update_3D_param_summary)
        self.VizSetup.plot_paramY.currentTextChanged.connect(self.update_3D_param_summary)
        self.VizSetup.plot_paramZ.currentTextChanged.connect(self.update_3D_param_summary)
        self.VizSetup.sub_runs_list.editingFinished.connect(self.update_3D_param_summary)

        # add widgets to pannel layout
        right_layout.addWidget(self.plot_select)
        right_layout.addWidget(self.viz_splitter)
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

    def update_diffraction_data_plot(self):
        self.fit_window.update_diff_view(self.model.sub_runs[self.fit_setup.sl.value()])

    def sync_oop(self):
        self.fit_summary.out_of_plane.setCurrentIndex(self.plot_select.out_of_plane.currentIndex())

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
