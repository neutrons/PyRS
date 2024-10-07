try:  # Mantid 6.3 uses a different import pathway
    from mantidqt.widgets.sliceviewer.presenter import SliceViewer
    from mantidqt.widgets.sliceviewer.model import SliceViewerModel
except ImportError:
    from mantidqt.widgets.sliceviewer.presenters.presenter import SliceViewer
    from mantidqt.widgets.sliceviewer.models.model import SliceViewerModel

from mantidqt.icons import get_icon
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton, QComboBox  # type:ignore
from qtpy.QtWidgets import QGroupBox, QSplitter, QTabWidget  # type:ignore
from qtpy.QtWidgets import QFormLayout, QFileDialog, QCheckBox  # type:ignore
from qtpy.QtWidgets import QStyledItemDelegate, QDoubleSpinBox  # type:ignore
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem  # type:ignore
from qtpy.QtWidgets import QStackedWidget, QMessageBox  # type:ignore
from qtpy.QtWidgets import QMainWindow, QAction  # type:ignore

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtGui import QDoubleValidator  # type:ignore
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    # from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    # from vtkmodules.util.numpy_support import numpy_to_vtk, get_vtk_array_type
    from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type
    import vtk
    DISABLE_3D = False
except ImportError:
    # if we don't have vtk then disable the 3D Viewer
    DISABLE_3D = True
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import functools
import traceback
import os

# Can't run VTK embedded in PyQT5 using VirtualGL
# See https://gitlab.kitware.com/vtk/vtk/-/issues/17338
USING_THINLINC = "TLSESSIONDATA" in os.environ


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

    def setFilenamesText(self, filenames):
        self.lineEdit.setText(filenames)


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

    def set_stress_case(self, stressCase):
        if stressCase.lower() == "diagonal":
            self.switch.set_2D(False)
        else:
            self.switch.set_2D(True)
            if "stress" in stressCase.lower():
                self.combo.setCurrentIndex(0)
            else:
                self.combo.setCurrentIndex(1)


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
        self.d0_grid_switch = QComboBox()
        self.d0_grid_switch.addItems(["Constant", "Field"])
        self.d0_grid_switch.currentTextChanged.connect(self.set_case)
        layout.addWidget(self.d0_grid_switch)
        self.d0_box = QWidget()
        d0_box_layout = QHBoxLayout()
        d0_box_layout.addWidget(QLabel("d₀"))
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.d0 = QLineEdit()
        self.d0.setValidator(validator)
        self.d0.editingFinished.connect(self.update_d0)
        d0_box_layout.addWidget(self.d0)
        d0_box_layout.addWidget(QLabel("Δd₀"))
        self.d0e = QLineEdit()
        self.d0e.setValidator(validator)
        self.d0e.editingFinished.connect(self.update_d0)
        d0_box_layout.addWidget(self.d0e)
        self.d0_box.setLayout(d0_box_layout)
        layout.addWidget(self.d0_box)

        load_save = QWidget()
        load_save_layout = QHBoxLayout()
        self.load_grid = QPushButton("Load d₀ Grid")
        self.load_grid.clicked.connect(self.load_d0_field)
        load_save_layout.addWidget(self.load_grid)
        self.save_grid = QPushButton("Save d₀ Grid")
        self.save_grid.clicked.connect(self.save_d0_field)
        load_save_layout.addWidget(self.save_grid)
        load_save.setLayout(load_save_layout)
        layout.addWidget(load_save)
        self.d0_grid = QTableWidget()
        self.d0_grid.setColumnCount(5)
        self.d0_grid.setColumnWidth(0, 60)
        self.d0_grid.setColumnWidth(1, 60)
        self.d0_grid.setColumnWidth(2, 60)
        self.d0_grid.setColumnWidth(3, 60)
        self.d0_grid.verticalHeader().setVisible(False)
        self.d0_grid.horizontalHeader().setStretchLastSection(True)
        self.d0_grid.setHorizontalHeaderLabels(['vx', 'vy', 'vz', "d₀", "Δd₀"])
        spinBoxDelegate = SpinBoxDelegate()
        self.d0_grid.setItemDelegateForColumn(3, spinBoxDelegate)
        self.d0_grid.setItemDelegateForColumn(4, spinBoxDelegate)
        layout.addWidget(self.d0_grid)

        self.setLayout(layout)

        self.set_case('Constant')

    def set_case(self, case):
        if case == "Constant":
            self.d0_box.setEnabled(True)
            self.load_grid.setEnabled(False)
            self.save_grid.setEnabled(False)
            self.d0_grid.setEnabled(False)
        else:
            self.d0_box.setEnabled(False)
            self.load_grid.setEnabled(True)
            self.save_grid.setEnabled(True)
            self.d0_grid.setEnabled(True)

    def update_d0(self):
        self._parent.update_plot()

    def set_d0(self, d0, d0e):
        if d0 is None:
            self.d0.clear()
            self.d0e.clear()
        else:
            self.d0.setText(str(d0))
            self.d0e.setText(str(d0e))

    def set_d0_field(self, x, y, z, d0, d0e):
        if x is None:
            self.d0_grid.setRowCount(0)
        else:
            self.d0_grid.setRowCount(0)
            self.d0_grid.setRowCount(len(x))

            for n in range(len(x)):
                x_item = QTableWidgetItem(f'{x[n]: 7.2f}')
                x_item.setFlags(x_item.flags() ^ Qt.ItemIsEditable)
                y_item = QTableWidgetItem(f'{y[n]: 7.2f}')
                y_item.setFlags(y_item.flags() ^ Qt.ItemIsEditable)
                z_item = QTableWidgetItem(f'{z[n]: 7.2f}')
                z_item.setFlags(z_item.flags() ^ Qt.ItemIsEditable)
                d0_item = QTableWidgetItem()
                d0_item.setData(Qt.EditRole, float(d0[n]))
                d0e_item = QTableWidgetItem()
                d0e_item.setData(Qt.EditRole, float(d0e[n]))
                self.d0_grid.setItem(n, 0, QTableWidgetItem(x_item))
                self.d0_grid.setItem(n, 1, QTableWidgetItem(y_item))
                self.d0_grid.setItem(n, 2, QTableWidgetItem(z_item))
                self.d0_grid.setItem(n, 3, QTableWidgetItem(d0_item))
                self.d0_grid.setItem(n, 4, QTableWidgetItem(d0e_item))

    def get_d0_field(self):
        if self.d0_grid.rowCount() == 0:
            return None
        else:
            x = [float(self.d0_grid.item(row, 0).text()) for row in range(self.d0_grid.rowCount())]
            y = [float(self.d0_grid.item(row, 1).text()) for row in range(self.d0_grid.rowCount())]
            z = [float(self.d0_grid.item(row, 2).text()) for row in range(self.d0_grid.rowCount())]
            d0 = [float(self.d0_grid.item(row, 3).text()) for row in range(self.d0_grid.rowCount())]
            d0e = [float(self.d0_grid.item(row, 4).text()) for row in range(self.d0_grid.rowCount())]
            return (d0, d0e, x, y, z)

    def save_d0_field(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "Save d0 Grid",
                                                  "",
                                                  "CSV (*.csv);;All Files (*)")
        if filename:
            d0, d0e, x, y, z = self.get_d0_field()
            np.savetxt(filename, np.array([x, y, z, d0, d0e]).T,
                       fmt=['%.4g', '%.4g', '%.4g', '%.9g', '%.9g'],
                       header="vx, vy, vz, d0, d0_error",
                       delimiter=',')

    def load_d0_field(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Load d0 Grid",
                                                  "",
                                                  "CSV (*.csv);;All Files (*)")
        if filename:
            x, y, z, d0, d0e = np.loadtxt(filename, delimiter=',', unpack=True)
            valid, x_clean, y_clean, z_clean, d0_clean, d0e_clean = self._parent.controller \
                .validate_d0_grid_data(x, y, z, d0, d0e, float(self.d0.text()), float(self.d0e.text()))

            if (valid == 1):
                # all or excesss d0 grid coords
                self.set_d0_field(x_clean, y_clean, z_clean, d0_clean, d0e_clean)
            elif (valid == -1):
                # no matching d0 grid coords
                QMessageBox.information(self, 'Validation', "Grid was not loaded.\nNone of the coordinates in your \
                                        experimental data exist in the d0 grid provided. Choose a different d0 grid.",
                                        QMessageBox.Ok | QMessageBox.Ok)

            else:
                # some matching d0 grid coords
                valid = QMessageBox.question(self, 'Validation', "Some of your coordinates in your experimental \
                                             data do not exist in the d0 grid provided - do you wish to continue?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if (valid == QMessageBox.Yes):
                    self.set_d0_field(x_clean, y_clean, z_clean, d0_clean, d0e_clean)

    def get_d0(self):
        if self.d0_grid_switch.currentText() == "Constant":
            try:
                return (float(self.d0.text()), float(self.d0e.text()))
            except ValueError:
                return None
        else:
            return self.get_d0_field()


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

    def set_text_values(self, direction, text):
        getattr(self, f"file_load_e{direction}").setFilenamesText(text)


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

    def set_values(self, youngModulus, poissonsRatio):
        self.youngModulus.setText(str(youngModulus))
        self.poissonsRatio.setText(str(poissonsRatio))


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

    def new_plot_MDH(self, dimensions_transposing=False, dimensions_changing=False):
        """redefine this function so we can change the default plot interpolation"""
        if self.view is None:
            print('view is none')
        else:
            self.view.data_view.plot_MDH(self.model.get_ws(), slicepoint=self.get_slicepoint(),
                                         interpolation='bilinear')
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
        self.plot_param.addItems(["dspacing-center",
                                  "d-reference",
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

    def set_selected_peak(self, peak_tag):
        self.peak_select.setCurrentText(peak_tag)


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


class VizTabs(QTabWidget):
    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent)
        self.oneDViewer = None
        self.strainSliceViewer = None
        self.vtk3dviewer = None

        self.plot_1d = QStackedWidget()
        self.plot_2d = QStackedWidget()
        self.plot_3d = QStackedWidget()

        self.message = QLabel("Load project files")
        self.message.setAlignment(Qt.AlignCenter)
        font = self.message.font()
        font.setPointSize(20)
        self.message.setFont(font)

        self.message2 = QLabel("Load project files")
        self.message2.setAlignment(Qt.AlignCenter)
        self.message2.setFont(font)

        self.plot_2d.addWidget(self.message)
        self.plot_3d.addWidget(self.message2)

        self.addTab(self.plot_1d, "1D")
        self.addTab(self.plot_2d, "2D")
        self.addTab(self.plot_3d, "3D")

        self.set_1d_mode(False)

        self.setCornerWidget(QLabel("Visualization Pane    "), corner=Qt.TopLeftCorner)

    def set_1d_mode(self, oned):
        self.setTabEnabled(0, oned)
        self.setTabEnabled(1, not oned)
        self.setTabEnabled(2, not USING_THINLINC and not DISABLE_3D and not oned)

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
                ax = fig.add_subplot(111)
                ax.errorbar(getattr(field, scan_dir), field.values, field.errors, marker='o')
                ax.set_xlabel(f'{scan_dir} (mm)')

                self.plot_1d.addWidget(self.oneDViewer)
                self.plot_1d.setCurrentIndex(1)
            else:
                self.set_1d_mode(False)
                if self.strainSliceViewer:
                    if self.strainSliceViewer.view:
                        self.strainSliceViewer.set_new_workspace(ws)
                    else:
                        print('View needs redefined')
                        self.strainSliceViewer = StrainSliceViewer(ws, parent=self)
                        self.plot_2d.addWidget(self.strainSliceViewer.view)
                        self.plot_2d.setCurrentIndex(1)
                else:
                    self.strainSliceViewer = StrainSliceViewer(ws, parent=self)
                    self.plot_2d.addWidget(self.strainSliceViewer.view)
                    self.plot_2d.setCurrentIndex(1)

                self.strainSliceViewer.set_new_field(field,
                                                     bin_widths=[ws.getDimension(n).getBinWidth() for n in range(3)])

                if not USING_THINLINC and not DISABLE_3D:
                    if self.vtk3dviewer:
                        self.vtk3dviewer.set_ws(ws)
                    else:
                        self.vtk3dviewer = VTK3DView(ws)
                        self.plot_3d.addWidget(self.vtk3dviewer)
                        self.plot_3d.setCurrentIndex(1)

        else:
            self.set_1d_mode(False)
            if self.oneDViewer is not None:
                self.plot_1d.removeWidget(self.oneDViewer)
                self.oneDViewer = None
            if self.strainSliceViewer is not None:
                self.plot_2d.removeWidget(self.strainSliceViewer.view)
                self.strainSliceViewer = None
            if self.vtk3dviewer:
                self.plot_3d.removeWidget(self.vtk3dviewer)
                self.vtk3dviewer = None

    def set_message(self, text):
        self.message.setText(text)
        self.message2.setText(text)


class VTK3DView(QWidget):
    def __init__(self, ws, parent=None):
        super().__init__(parent)
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        vti = self.md_to_vti(ws)
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputData(vti)

        self.mapper.ScalarVisibilityOn()
        self.mapper.SetScalarModeToUseCellData()
        self.mapper.SetColorModeToMapScalars()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(self.mapper.GetLookupTable())
        scalarBar.SetNumberOfLabels(4)

        srange = vti.GetScalarRange()

        self.lut = vtk.vtkLookupTable()
        self.lut.SetTableRange(srange)
        self.lut.Build()

        self.mapper.UseLookupTableScalarRangeOn()
        self.mapper.SetLookupTable(self.lut)
        scalarBar.SetLookupTable(self.lut)

        self.renderer = vtk.vtkRenderer()
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.8, 0.8, 0.8)
        self.renderer.SetBackground2(0, 0, 0)

        axes = vtk.vtkCubeAxesActor()
        axes.SetUseTextActor3D(1)
        axes.SetBounds(vti.GetBounds())
        axes.SetCamera(self.renderer.GetActiveCamera())

        axes.DrawXGridlinesOn()
        axes.DrawYGridlinesOn()
        axes.DrawZGridlinesOn()
        axes.SetFlyModeToOuterEdges()

        self.renderer.AddActor(self.actor)
        self.renderer.AddActor(axes)
        self.renderer.AddActor2D(scalarBar)
        self.renderer.ResetCamera()

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        layout = QVBoxLayout()
        layout.addWidget(self.vtkWidget)
        self.setLayout(layout)

        camera = self.renderer.GetActiveCamera()
        assert camera is not None

        self.vtkWidget.show()

        self.iren.Initialize()

    def set_ws(self, ws):
        vti = self.md_to_vti(ws)
        self.mapper.SetInputData(vti)
        self.lut.SetTableRange(vti.GetScalarRange())
        self.vtkWidget.Render()

    def md_to_vti(self, md):
        array = md.getSignalArray()
        origin = [md.getDimension(n).getMinimum() for n in range(3)]
        spacing = [md.getDimension(n).getBinWidth() for n in range(3)]
        dimensions = [n+1 for n in array.shape]

        vtkArray = numpy_to_vtk(num_array=array.flatten('F'), deep=True,
                                array_type=get_vtk_array_type(array.dtype))

        imageData = vtk.vtkImageData()
        imageData.SetOrigin(origin)
        imageData.SetSpacing(spacing)
        imageData.SetDimensions(dimensions)
        imageData.GetCellData().SetScalars(vtkArray)

        return imageData


class StrainStressViewer(QMainWindow):
    def __init__(self, model, ctrl, parent=None):
        self._model = model
        self._model.propertyUpdated.connect(self.updatePropertyFromModel)
        self._model.failureMsg.connect(self.show_failure_msg)
        self._ctrl = ctrl

        super().__init__(parent)

        self.setWindowTitle("PyRS Strain-Stress Viewer")

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        self.saveAction = QAction('&Save state', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Save application state')
        self.saveAction.triggered.connect(self.save)
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

        self.csvExport = CSVExport(self)
        left_layout.addWidget(self.csvExport)

        left_layout.addStretch(0)

        left.setLayout(left_layout)

        self.splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout()

        self.plot_select = PlotSelect(self)
        self.plot_select.measure_dir.currentTextChanged.connect(self.update_plot)
        self.plot_select.plot_param.currentTextChanged.connect(self.update_plot)
        right_layout.addWidget(self.plot_select)

        self.viz_tab = VizTabs(self)
        right_layout.addWidget(self.viz_tab)

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

    def load(self):
        filename, _ = QFileDialog.getOpenFileName(self,
                                                  "Load Stress state",
                                                  "",
                                                  "JSON (*.json);;All Files (*)")
        if filename:
            self.controller.load(filename)
