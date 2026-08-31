"""
View for the peak-fitting interface.

This is the "V" of the Model-View-Presenter structure shared with the
texture-fitting UI. It owns the Qt widget tree (loaded from the
``peakfitwindow.ui`` Qt Designer file) and all signal/slot wiring. Slot bodies
delegate data access to :class:`PeakFittingModel` and plot preparation to
:class:`PeakFittingCrtl`; the widget-update orchestration that previously lived in
the ``EventHandler``/``Load``/``Fit``/``ExportCSV`` helper classes now lives here.

The pure-widget helper classes :class:`GuiUtilities` and :class:`FitTable` remain
view-side collaborators, instantiated with this window as their ``parent``.
"""

import os

import numpy as np
from qtpy import QtGui  # type:ignore
from qtpy.QtCore import Qt  # type: ignore
from qtpy.QtGui import QCursor  # type:ignore
from qtpy.QtWidgets import QApplication, QMainWindow, QMenu, QTableWidgetItem, QVBoxLayout  # type:ignore

import pyrs.icons
import pyrs.interface.gui_helper
from pyrs.interface.gui_helper import browse_dir, browse_file, parse_integers, pop_message
from pyrs.interface.peak_fitting.fit_table import FitTable
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.interface.ui import qt_util
from pyrs.interface.ui.diffdataviews import GeneralDiffDataView, PeakFitSetupView
from pyrs.interface.ui.rstables import FitResultTable
from pyrs.utilities import get_input_project_file  # type: ignore
from pyrs.utilities import load_ui  # type: ignore

# Splitter/icon images are loaded from disk rather than a compiled Qt resource
# module, since pyrcc-generated modules hard-code a single PyQt major version.
ICON_DIR = os.path.dirname(pyrs.icons.__file__).replace(os.sep, "/")

VERTICAL_SPLITTER = f"""QSplitter::handle {{image: url('{ICON_DIR}/vertical_splitter.png'); }}"""
VERTICAL_SPLITTER_SHORT = f"""QSplitter::handle {{image: url('{ICON_DIR}/vertical_splitter_short.png'); }}"""
HORIZONTAL_SPLITTER = f"""QSplitter::handle {{image: url('{ICON_DIR}/horizontal_splitter.png'); }}"""
HORIZONTAL_SPLITTER_SHORT = f"""QSplitter::handle {{image: url('{ICON_DIR}/horizontal_splitter_short.png'); }}"""

D0 = "d₀"
ANGSTROMS = "Å"


class PeakFittingViewer(QMainWindow):
    """GUI window for fitting peaks (Model-View-Presenter view component)."""

    def __init__(self, fit_peak_model, fit_peak_ctrl, parent=None):
        """
        Args:
            fit_peak_model: The :class:`PeakFittingModel` instance.
            fit_peak_ctrl: The :class:`PeakFittingCrtl` presenter instance.
            parent: Optional Qt parent window.
        """
        super(PeakFittingViewer, self).__init__(parent)

        self._model = fit_peak_model
        self._ctrl = fit_peak_ctrl

        # View-side state
        self.current_hidra_file_name = ""
        self.current_root_statusbar_message = ""
        self.list_peak_d0 = []
        self._sample_log_names = list()
        self._advanced_fit_dialog = None

        # set up UI: load_ui resolves by basename to pyrs/interface/designer/
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join("ui", "peakfitwindow.ui"))
        self.ui = load_ui(ui_path, baseinstance=self)

        self.setup_ui()

        # surface model failures without the model depending on any UI helper
        self._model.failureMsg.connect(self._on_failure)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model

    def _on_failure(self, title, message, detail):
        pop_message(self, title, detailed_message="{}\n{}".format(message, detail), message_type="error")

    def setup_ui(self):
        """Define the layout, widgets and signals."""

        # promote
        self.ui.graphicsView_fitResult = qt_util.promote_widget(
            self, self.ui.graphicsView_fitResult_frame, GeneralDiffDataView
        )
        self.ui.graphicsView_plot2D = qt_util.promote_widget(
            self, self.ui.graphicsView_2dPlot_frame, GeneralDiffDataView
        )
        self.ui.tableView_fitSummary = qt_util.promote_widget(self, self.ui.tableView_fitSummary_frame, FitResultTable)
        self._promote_peak_fit_setup()
        self._init_widgets()
        self.ui.graphicsView_plot2D.set_3Dview()

        # set up handling
        self.ui.lineEdit_expNumber.setValidator(QtGui.QIntValidator(1, 2147483647))
        self.ui.pushButton_expNumberLoad.clicked.connect(self.load_run_number)
        self.ui.pushButton_browseHDF.clicked.connect(self.browse_hdf)
        self.ui.lineEdit_listSubRuns.returnPressed.connect(self.plot_diff_data)
        self.ui.pushButton_FitPeaks.clicked.connect(self.fit_peaks)
        self.ui.horizontalScrollBar_SubRuns.valueChanged.connect(self.plot_scan)
        self.ui.radioButton_individualSubRuns.clicked.connect(self.individual_sub_runs)
        self.ui.radioButton_listSubRuns.clicked.connect(self.list_sub_runs)
        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionSave.triggered.connect(self.save)
        self.ui.actionSaveAs.triggered.connect(self.save_as)
        self.ui.pushButton_exportCSV.clicked.connect(self.export_csv)
        # self.ui.actionQuick_Fit_Result_Check.triggered.connect(self.do_make_movie)
        self.ui.lineEdit_subruns_2dplot.returnPressed.connect(self.list_subruns_2dplot_returned)
        self.ui.lineEdit_subruns_2dplot.textChanged.connect(self.list_subruns_2dplot_changed)
        self.ui.pushButton_save_peak_range.clicked.connect(self.clicked_save_peak_range)
        self.ui.pushButton_load_peak_range.clicked.connect(self.clicked_load_peak_range)
        self.ui.tableView_fitSummary.itemSelectionChanged.connect(self.fit_result_table_selection_changed)

        self.ui.radioButton_fit_value.clicked.connect(self.fit_table_radio_buttons)
        self.ui.radioButton_fit_error.clicked.connect(self.fit_table_radio_buttons)
        self.ui.spinBox_peak_index.valueChanged.connect(self.fit_table_radio_buttons)

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.axis_1d_changed)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.axis_1d_changed)
        self.ui.plot1d_xaxis_peak_label_comboBox.currentIndexChanged.connect(self.axis_1d_changed)
        self.ui.plot1d_yaxis_peak_label_comboBox.currentIndexChanged.connect(self.axis_1d_changed)

        self.ui.comboBox_xaxisNames_2dplot.currentIndexChanged.connect(self.axis_2d_changed)
        self.ui.comboBox_yaxisNames_2dplot.currentIndexChanged.connect(self.axis_2d_changed)
        self.ui.comboBox_zaxisNames_2dplot.currentIndexChanged.connect(self.axis_2d_changed)
        self.ui.plot2d_xaxis_peak_label_comboBox.currentIndexChanged.connect(self.axis_2d_changed)
        self.ui.plot2d_yaxis_peak_label_comboBox.currentIndexChanged.connect(self.axis_2d_changed)
        self.ui.plot2d_zaxis_peak_label_comboBox.currentIndexChanged.connect(self.axis_2d_changed)

        self.ui.radioButton_contour.clicked.connect(self.axis_2d_changed)
        self.ui.radioButton_3dline.clicked.connect(self.axis_2d_changed)
        self.ui.radioButton_3dscatter.clicked.connect(self.axis_2d_changed)

        self.ui.peak_range_table.cellChanged.connect(self.peak_range_table_changed)

        # tracker for sample log names and peak parameter names
        self._sample_log_name_set = set()
        self._function_param_name_set = set()

        # mutexes
        self._sample_log_names_mutex = False
        self.ui.tableView_fitSummary.setup(peak_param_names=list())

        o_gui = GuiUtilities(parent=self)
        o_gui.enabled_fitting_widgets(False)
        o_gui.enabled_1dplot_widgets(False)
        o_gui.check_axis1d_status()
        o_gui.enabled_2dplot_widgets(False)
        o_gui.check_axis2d_status()
        o_gui.make_visible_listsubruns_warning(False)
        o_gui.enabled_export_csv_widgets(False)
        o_gui.enabled_peak_ranges_widgets(False)
        o_gui.enabled_save_peak_range_widget(False)
        o_gui.enabled_sub_runs_interation_widgets(False)

        # for debugging only
        self.ui.radioButton_contour.setEnabled(True)
        self.ui.radioButton_3dline.setEnabled(True)

    # ------------------------------------------------------------------
    # Sub-run list helper (replaces Plot.__init__ caching)
    # ------------------------------------------------------------------
    def _sub_run_list(self):
        """Return the ordered list of sub-runs from the loaded workspace."""
        return self._model.get_subruns_limit()

    def _d_reference_list(self):
        """Return per-row d-reference values from the peak-range table."""
        table = self.ui.peak_range_table
        values = []
        for _row in range(table.rowCount()):
            _item = table.item(_row, 3)
            values.append(_item.text() if _item is not None else "1.0")
        return values

    # ------------------------------------------------------------------
    # Save / load files
    # ------------------------------------------------------------------
    def save(self):
        self._model.save_fit_result(self.current_hidra_file_name)

    def save_as(self):
        out_file_name = browse_file(
            self,
            caption="Choose a file to save fitted peaks to",
            default_dir=self._model.working_dir,
            file_filter="H5 (*.h5);;HDF (*.hdf5)",
            save_file=True,
        )

        try:
            if not (out_file_name.endswith(".hdf5") or out_file_name.endswith(".h5")):
                out_file_name += ".h5"
            self._model.save_fit_result(out_file_name)
        except AttributeError:
            pass

    def load_run_number(self):
        runs = parse_integers(self.ui.lineEdit_expNumber.text())

        try:
            project_dir = [
                get_input_project_file(run, preferredType=self.ui.comboBox_reduction.currentText()) for run in runs
            ]
        except (FileNotFoundError, RuntimeError, ValueError) as run_err:
            pop_message(self, f"Failed to find run {self.ui.lineEdit_expNumber.text()}", str(run_err), "error")
            return

        hidra_file_name = [os.path.join(project_dir[i_run], f"HB2B_{runs[i_run]}.h5") for i_run in range(len(runs))]

        self.current_hidra_file_name = hidra_file_name
        self.load_and_plot(hidra_file_name)
        try:
            self._do_plot_1d()
        except AttributeError:
            pass

    def browse_hdf(self):
        """Browse for a Hidra project HDF file, then load and plot it."""
        file_filter = "HDF (*.hdf);H5 (*.h5)"
        hidra_file_name = browse_file(
            self, "HIDRA Project File", os.getcwd(), file_filter, file_list=True, save_file=False
        )

        if hidra_file_name is None:
            return  # user clicked cancel

        self.current_hidra_file_name = hidra_file_name
        self.load_and_plot(hidra_file_name)
        try:
            self._do_plot_1d()
        except AttributeError:
            pass

    def load_and_plot(self, hidra_file_name):
        """Load a project file then refresh widgets, plots and the fit table."""
        o_gui = GuiUtilities(parent=self)

        # --- load the data into the model ---
        try:
            self._model.load_hidra_project(hidra_file_name)
        except (RuntimeError, TypeError) as run_err:
            pop_message(
                self, "Unable to load {}".format(hidra_file_name), detailed_message=str(run_err), message_type="error"
            )

        # --- set up sub-run range, comboboxes and 1D-plot widgets ---
        sub_run_list = self._model.get_subruns_limit()
        o_gui.initialize_fitting_slider(max=len(sub_run_list))
        o_gui.set_1D_2D_axis_comboboxes(with_clear=True, fill_raw=True)
        o_gui.enabled_1dplot_widgets(enabled=True)
        o_gui.initialize_combobox()
        self.ui.graphicsView_plot2D.reset_viewer()

        # --- status bar ---
        self.current_root_statusbar_message = "Working with: {} \t\t\t\t Project Name: {}".format(
            self._model.curr_file_name, self._model.project_name
        )
        self.ui.statusbar.showMessage(self.current_root_statusbar_message)

        # --- plot the diffraction data ---
        try:
            self.plot_diff_data()
            self.ui.graphicsView_fitResult.reset_viewer()
        except RuntimeError as run_err:
            pop_message(self, "Failed to plot {}".format(hidra_file_name), str(run_err), "error")

        # --- initialize the fit-result table and enable fitting widgets ---
        try:
            if self.ui.tableView_fitSummary.rowCount() > 0:
                self.ui.tableView_fitSummary.remove_all_rows()
            self.ui.tableView_fitSummary.init_exp(sub_run_list)

            o_gui.check_if_fitting_widgets_can_be_enabled()
            o_gui.enabled_sub_runs_interation_widgets(True)
            o_gui.enabled_data_fit_plot(True)
            o_gui.enabled_export_csv_widgets(False)
            o_gui.enabled_peak_ranges_widgets(True)
            o_gui.enabled_1dplot_widgets(True)
        except RuntimeError as run_err:
            pop_message(self, "Failed to initialize widgets for {}".format(hidra_file_name), str(run_err), "error")

    # ------------------------------------------------------------------
    # Diffraction-pattern plotting
    # ------------------------------------------------------------------
    def plot_diff_data(self):
        """Plot the diffraction data for the sub-runs listed in the text box."""
        try:
            scan_log_index_list = parse_integers(str(self.ui.lineEdit_listSubRuns.text()))
        except RuntimeError:
            pop_message(self, "Unable to parse the string", message_type="error")
            return

        if len(scan_log_index_list) == 0:
            pop_message(self, "There is not scan-log index input", "error")

        sub_run = self._ctrl.plot_diff_data(self._ui_graphicsView_fitSetup, scan_log_index_list, self._sub_run_list())
        if sub_run is not None:
            self.ui.label_SubRunsValue.setText("{}".format(sub_run))

    def plot_scan(self):
        """Plot the sub-run currently selected by the scroll bar."""
        scan_value = self.ui.horizontalScrollBar_SubRuns.value()
        sub_run = self._ctrl.plot_scan(self._ui_graphicsView_fitSetup, scan_value, self._sub_run_list())
        self.ui.label_SubRunsValue.setText("{}".format(sub_run))

    # ------------------------------------------------------------------
    # Peak fitting
    # ------------------------------------------------------------------
    def fit_peaks(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            peak_range_list = [tuple(_range) for _range in self._ui_graphicsView_fitSetup.list_peak_ranges]
            peak_center_list = [np.mean([left, right]) for (left, right) in peak_range_list]
            peak_tag_list = ["peak{}".format(_index) for _index, _ in enumerate(peak_center_list)]
            peak_function_name = str(self.ui.comboBox_peakType.currentText())
            peak_background_name = str(self.ui.comboBox_backgroundType.currentText())

            peak_xmin_list = [left for (left, _) in peak_range_list]
            peak_xmax_list = [right for (_, right) in peak_range_list]

            fit_result = self._model.fit_diff_peaks(
                peak_tag_list, peak_xmin_list, peak_xmax_list, peak_function_name, peak_background_name
            )
            if fit_result is None:
                return

            self.populate_fit_result_table(fit_result=fit_result)

            o_gui = GuiUtilities(parent=self)
            o_gui.set_1D_2D_axis_comboboxes(with_clear=True, fill_raw=True, fill_fit=True)
            o_gui.initialize_combobox()
            o_gui.enabled_export_csv_widgets(enabled=True)
            o_gui.enabled_2dplot_widgets(enabled=True)

            self._do_plot_2d()

            self.individual_or_list_sub_runs()
        finally:
            QApplication.restoreOverrideCursor()

    def individual_or_list_sub_runs(self):
        if self.ui.radioButton_individualSubRuns.isChecked():
            self.individual_sub_runs()
        else:
            self.list_sub_runs()

    def individual_sub_runs(self):
        self.check_subRunsDisplayMode()
        self.plot_scan()

    def list_sub_runs(self):
        self.check_subRunsDisplayMode()
        self.plot_diff_data()

    def check_subRunsDisplayMode(self):
        GuiUtilities(parent=self).check_subRuns_display_mode()

    # ------------------------------------------------------------------
    # 1D / 2D parameter plots
    # ------------------------------------------------------------------
    def _do_plot_1d(self):
        o_gui = GuiUtilities(parent=self)
        x_axis_name = str(self.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.ui.comboBox_yaxisNames.currentText())
        peak_index = o_gui.get_plot1d_axis_peak_label_index(is_xaxis=False)

        self._ctrl.plot_1d(
            self.ui.graphicsView_fitResult,
            x_axis_name,
            y_axis_name,
            peak_index,
            d_reference_list=self._d_reference_list(),
        )

    def _do_plot_2d(self, sub_run_list=None):
        o_gui = GuiUtilities(parent=self)
        x_axis_name = str(self.ui.comboBox_xaxisNames_2dplot.currentText())
        y_axis_name = str(self.ui.comboBox_yaxisNames_2dplot.currentText())
        z_axis_name = str(self.ui.comboBox_zaxisNames_2dplot.currentText())

        x_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis="x")
        y_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis="y")
        z_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis="z")

        if self.ui.radioButton_contour.isChecked():
            mode = "contour"
        elif self.ui.radioButton_3dline.isChecked():
            mode = "lines"
        else:
            mode = "scatter"

        self._ctrl.plot_2d(
            self.ui.graphicsView_plot2D,
            x_axis_name,
            y_axis_name,
            z_axis_name,
            x_peak_index,
            y_peak_index,
            z_peak_index,
            mode=mode,
            sub_run_list=sub_run_list,
            d_reference_list=self._d_reference_list(),
        )

    def axis_1d_changed(self):
        GuiUtilities(parent=self).check_axis1d_status()
        self._do_plot_1d()

    def axis_2d_changed(self, **kwargs):
        GuiUtilities(parent=self).check_axis2d_status()
        self._do_plot_2d(**kwargs)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def export_csv(self):
        out_folder = browse_dir(
            self, caption="Choose where to create the CSV file", default_dir=self._model.working_dir
        )
        csv_file_name = os.path.join(out_folder, self._model.project_name + ".csv")

        try:
            self._model.export_csv(csv_file_name)
            new_message = self.current_root_statusbar_message + "\t\t\t\t Last Exported CSV: {}".format(csv_file_name)
            self.ui.statusbar.showMessage(new_message)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Sub-run list for 2D plots
    # ------------------------------------------------------------------
    def list_subruns_2dplot_returned(self):
        sub_run_list = self._list_subruns_2dplot()
        self.axis_2d_changed(sub_run_list=sub_run_list)

    def list_subruns_2dplot_changed(self):
        self._list_subruns_2dplot()

    def _list_subruns_2dplot(self):
        raw_input = str(self.ui.lineEdit_subruns_2dplot.text())
        o_gui = GuiUtilities(parent=self)

        try:
            parse_input = parse_integers(raw_input)
            o_gui.make_visible_listsubruns_warning(False)
        except RuntimeError:
            parse_input = []
            o_gui.make_visible_listsubruns_warning(True)

        return parse_input

    # ------------------------------------------------------------------
    # Peak-range table and fit-setup plot
    # ------------------------------------------------------------------
    def _promote_peak_fit_setup(self):
        # 2D detector view
        curr_layout = QVBoxLayout()
        self.ui.frame_PeakView.setLayout(curr_layout)
        self._ui_graphicsView_fitSetup = PeakFitSetupView(self)
        self._ui_graphicsView_fitSetup.setEnabled(False)
        curr_layout.addWidget(self._ui_graphicsView_fitSetup)

    def update_peak_ranges_table(self, **kwargs):
        self.ui.peak_range_table.blockSignals(True)

        list_fit_peak_ranges = kwargs.get("list_fit_peak_ranges", [])
        list_fit_peak_labels = kwargs.get("list_fit_peak_labels", [])
        list_fit_peak_d0 = kwargs.get("list_fit_peak_d0", [])

        o_gui = GuiUtilities(parent=self)
        o_gui.reset_peak_range_table()
        o_gui.fill_peak_range_table(
            list_fit_peak_ranges=list_fit_peak_ranges,
            list_fit_peak_labels=list_fit_peak_labels,
            list_fit_peak_d0=list_fit_peak_d0,
        )

        self.ui.peak_range_table.blockSignals(False)
        o_gui.check_if_fitting_widgets_can_be_enabled()

    def peak_range_table_changed(self, row=0, column=0):
        self._update_fit_peak_ranges_plot()
        if column == 3:
            self._update_fit_result_table()

    def _update_fit_peak_ranges_plot(self):
        # retrieve all peaks and labels from table
        table_ui = self.ui.peak_range_table
        table_ui.blockSignals(True)

        nbr_row = table_ui.rowCount()

        list_peak_ranges = []
        list_fit_peak_labels = []
        list_fit_peak_d0 = []
        for _row in np.arange(nbr_row):
            _value1 = GuiUtilities.get_item_value(table_ui, _row, 0)
            _value2 = GuiUtilities.get_item_value(table_ui, _row, 1)

            try:
                _value1_float = np.float64(_value1)
                _value2_float = np.float64(_value2)
                _array = [_value1_float, _value2_float]

                _value1 = np.nanmin(_array)
                _value2 = np.nanmax(_array)

                _item0 = QTableWidgetItem("{:.3f}".format(_value1))
                self.ui.peak_range_table.setItem(_row, 0, _item0)

                _item1 = QTableWidgetItem("{:.3f}".format(_value2))
                self.ui.peak_range_table.setItem(_row, 1, _item1)

                list_peak_ranges.append([_value1, _value2])

            except ValueError:
                continue

            _label = GuiUtilities.get_item_value(table_ui, _row, 2)
            list_fit_peak_labels.append(_label)

            _d0 = np.float64(str(GuiUtilities.get_item_value(table_ui, _row, 3)))
            list_fit_peak_d0.append(_d0)

        # replace the list_peak_ranges and list_fit_peak_labels from mplfitplottingwidget.py
        self._ui_graphicsView_fitSetup.list_peak_ranges = list_peak_ranges
        self._ui_graphicsView_fitSetup.list_fit_peak_labels = list_fit_peak_labels
        self._ui_graphicsView_fitSetup.list_fit_peak_d0 = list_fit_peak_d0
        self._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

        table_ui.blockSignals(False)

    def _update_fit_result_table(self):
        if self._model.fit_result:
            self.populate_fit_result_table(fit_result=self._model.fit_result)

    def update_save_peak_range_widget(self):
        GuiUtilities(parent=self).update_save_peak_range_widget_status()

    def _browse_json_file(self, save_file=True):
        file_filter = "JSON (*.json)"
        return browse_file(self, "Peak Range File", os.getcwd(), file_filter, file_list=False, save_file=save_file)

    def clicked_save_peak_range(self):
        json_file_name = self._browse_json_file(save_file=True)
        if json_file_name is None:
            return  # user clicked cancel

        list_peak_ranges = self._ui_graphicsView_fitSetup.list_peak_ranges
        list_peak_labels = self._ui_graphicsView_fitSetup.list_fit_peak_labels
        list_peak_d0 = self.list_peak_d0

        self._model.save_peak_range_json(json_file_name, list_peak_ranges, list_peak_labels, list_peak_d0)

    def clicked_load_peak_range(self):
        json_file_name = self._browse_json_file(save_file=False)
        if json_file_name is None:
            return  # user clicked cancel

        peak_range, peak_label, peak_d0 = self._model.load_peak_range_json(json_file_name)

        self._ui_graphicsView_fitSetup.list_peak_ranges = peak_range
        self._ui_graphicsView_fitSetup.list_fit_peak_labels = peak_label
        self.list_peak_d0 = peak_d0

        self.update_peak_ranges_table(
            release=True,
            list_fit_peak_labels=peak_label,
            list_fit_peak_ranges=peak_range,
            list_fit_peak_d0=peak_d0,
            list_fit_peak_ranges_matplotlib_id=[],
            list_fit_peak_labels_matplotlib_id=[],
        )
        self._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

    def peak_range_table_right_click(self, position=-1):
        nbr_row = self.ui.peak_range_table.rowCount()
        if nbr_row == 0:
            return

        menu = QMenu(self)
        _remove_row = menu.addAction("Remove")
        action = menu.exec_(QCursor.pos())

        if action == _remove_row:
            self.remove_peak_range_table_row()

        GuiUtilities(parent=self).check_if_fitting_widgets_can_be_enabled()

    def remove_peak_range_table_row(self):
        row_selected = self.ui.peak_range_table.selectedRanges()[0]
        row_to_remove = row_selected.topRow()
        self.ui.peak_range_table.removeRow(row_to_remove)

        new_list_peak_ranges = []
        new_list_peak_labels = []
        new_list_matplotlib_id = []
        old_list_peak_label = self._ui_graphicsView_fitSetup.list_fit_peak_labels
        old_list_matplotlib_id = self._ui_graphicsView_fitSetup.list_peak_labels_matplotlib_id
        for _row, peak_range in enumerate(self._ui_graphicsView_fitSetup.list_peak_ranges):
            if _row == row_to_remove:
                _peak_label_id = old_list_matplotlib_id[_row]
                _peak_label_id.remove()
                continue

            new_list_peak_ranges.append(peak_range)
            new_list_peak_labels.append(old_list_peak_label[_row])
            new_list_matplotlib_id.append(old_list_matplotlib_id[_row])

        self._ui_graphicsView_fitSetup.list_fit_peak_labels = new_list_peak_labels
        self._ui_graphicsView_fitSetup.list_peak_ranges = new_list_peak_ranges
        self._ui_graphicsView_fitSetup.list_peak_labels_matplotlib_id = new_list_matplotlib_id

        self._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

    # ------------------------------------------------------------------
    # Fit-result table
    # ------------------------------------------------------------------
    def populate_fit_result_table(self, fit_result=None):
        self._model.fit_result = fit_result
        o_table = FitTable(parent=self, fit_result=fit_result)
        o_table.initialize_fit_result_widgets()
        o_table.populate_fit_result_table()
        o_table.select_first_row()

    def fit_table_radio_buttons(self):
        o_table = FitTable(parent=self, fit_result=self._model.fit_result)
        o_table.initialize_table()
        o_table.initialize_table_column_size()
        o_table.fit_value_error_changed()

    def fit_result_table_selection_changed(self):
        """When a row is selected, switch to the slider view and go to that sub-run."""
        row_selected = GuiUtilities.get_row_selected(table_ui=self.ui.tableView_fitSummary)
        if row_selected is None:
            return
        self.ui.radioButton_individualSubRuns.setChecked(True)
        self.check_subRunsDisplayMode()
        self.ui.horizontalScrollBar_SubRuns.setValue(row_selected + 1)
        self.plot_scan()

    # ------------------------------------------------------------------
    # Misc widget setup and file actions
    # ------------------------------------------------------------------
    def _init_widgets(self):
        """Initialize some widgets."""
        self.ui.actionSave.setEnabled(True)
        self.ui.actionSaveAs.setEnabled(True)

        self.ui.splitter.setStyleSheet(VERTICAL_SPLITTER_SHORT)
        self.ui.splitter_2.setStyleSheet(HORIZONTAL_SPLITTER)
        self.ui.splitter_4.setStyleSheet(HORIZONTAL_SPLITTER)
        self.ui.splitter_5.setStyleSheet(HORIZONTAL_SPLITTER)
        self.ui.splitter_3.setStyleSheet(VERTICAL_SPLITTER)
        self.ui.splitter_3.setSizes([100, 5])

        # status bar
        self.setStyleSheet("QStatusBar{padding-left:8px;color:green;}")

        # warning icon
        self.ui.listsubruns_warning_icon.setPixmap(QtGui.QPixmap(f"{ICON_DIR}/warning_icon.png"))

        # width of peak region table
        peak_table_col_width = [100, 100, 150, 200]
        for _col_index, _width in enumerate(peak_table_col_width):
            self.ui.peak_range_table.setColumnWidth(_col_index, _width)

        peak_range_table_labels = ["x_left", "x_right", "Label", D0 + " (" + ANGSTROMS + ")"]
        self.ui.peak_range_table.setHorizontalHeaderLabels(peak_range_table_labels)

    def do_save_fit(self):
        """Save fit result through a file dialog."""
        file_name = pyrs.interface.gui_helper.browse_file(
            self,
            "Select file to save fit result",
            default_dir=self._model.working_dir,
            file_filter="HDF (*.hdf5);;CSV (*.csv)",
            file_list=False,
            save_file=True,
        )

        if file_name.lower().endswith("hdf5") or file_name.lower().endswith("hdf") or file_name.lower().endswith("h5"):
            self._model.save_fit_result(out_file_name=file_name)
        elif file_name.lower().endswith("csv") or file_name.endswith("dat"):
            self.export_fit_result(file_name)
        else:
            pyrs.interface.gui_helper.pop_message(
                self,
                message="Input file {} has an unsupported posfix.".format(file_name),
                detailed_message="Supported are hdf5, h5, hdf, csv and dat",
                message_type="error",
            )

    def do_quit(self):
        """Close the window and quit."""
        self.close()

    def export_fit_result(self, file_name):
        """Export fit result to a CSV file."""
        self.ui.tableView_fitSummary.export_table_csv(file_name)

    def save_data_for_mantid(self, data_key, file_name):
        """Save data to a Mantid-compatible NeXus file."""
        self._model.save_nexus(data_key, file_name)
