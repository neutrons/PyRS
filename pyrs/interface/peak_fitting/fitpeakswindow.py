import os
from qtpy.QtWidgets import QVBoxLayout, QFileDialog, QMainWindow
from qtpy import QtGui

from pyrs.utilities import load_ui
from pyrs.interface.ui import qt_util
from pyrs.interface.ui.diffdataviews import GeneralDiffDataView
from pyrs.interface.ui.rstables import FitResultTable
from pyrs.interface.ui.diffdataviews import PeakFitSetupView
import pyrs.interface.advpeakfitdialog
import pyrs.interface.gui_helper
from pyrs.interface.peak_fitting.event_handler import EventHandler
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.fit import Fit
from pyrs.interface.peak_fitting.export import ExportCSV
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.icons import icons_rc  # noqa: F401

VERTICAL_SPLITTER = """QSplitter::handle {image: url(':/fitting/vertical_splitter.png'); }"""
HORIZONTAL_SPLITTER = """QSplitter::handle {image: url(':/fitting/horizontal_splitter.png'); }"""

MICROSTRAIN = u"\u212B"
D0 = u"d\u2080"


class FitPeaksWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """

    def __init__(self, parent, fit_peak_core=None):
        """
        initialization
        :param parent:
        """
        super(FitPeaksWindow, self).__init__(parent)

        # class variables
        self._core = fit_peak_core
        self._project_name = None
        self.hidra_workspace = None
        # current/last loaded data
        self._curr_file_name = None

        # a copy of sample logs
        self._sample_log_names = list()  # a copy of sample logs' names that are added to combo-box

        # sub windows
        self._advanced_fit_dialog = None

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'peakfitwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)

        self.setup_ui()

    def setup_ui(self):
        """define the layout, widgets and signals"""

        # promote
        self.ui.graphicsView_fitResult = qt_util.promote_widget(self, self.ui.graphicsView_fitResult_frame,
                                                                GeneralDiffDataView)
        self.ui.graphicsView_fitResult.setEnabled(False)
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self.ui.graphicsView_plot2D = qt_util.promote_widget(self, self.ui.graphicsView_2dPlot_frame,
                                                             GeneralDiffDataView)
        self.ui.tableView_fitSummary = qt_util.promote_widget(self, self.ui.tableView_fitSummary_frame,
                                                              FitResultTable)
        self._promote_peak_fit_setup()
        self._init_widgets()

        # set up handling
        # self.ui.pushButton_loadHDF.clicked.connect(self.load_hidra_file)
        self.ui.pushButton_browseHDF.clicked.connect(self.browse_hdf)
        self.ui.lineEdit_listSubRuns.returnPressed.connect(self.plot_diff_data)
        self.ui.pushButton_FitPeaks.clicked.connect(self.fit_peaks)
        self.ui.horizontalScrollBar_SubRuns.valueChanged.connect(self.plot_scan)
        self.ui.radioButton_individualSubRuns.clicked.connect(self.individual_sub_runs)
        self.ui.radioButton_listSubRuns.clicked.connect(self.list_sub_runs)
        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionSave.triggered.connect(self.save)
        self.ui.actionSaveAs.triggered.connect(self.save_as)
        self.ui.actionAdvanced_Peak_Fit_Settings.triggered.connect(self.do_launch_adv_fit)
        self.ui.pushButton_exportCSV.clicked.connect(self.export_csv)
        self.ui.actionQuick_Fit_Result_Check.triggered.connect(self.do_make_movie)
        self.ui.lineEdit_subruns_2dplot.returnPressed.connect(self.list_subruns_2dplot_returned)
        self.ui.lineEdit_subruns_2dplot.textChanged.connect(self.list_subruns_2dplot_changed)

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.xaxis_1d_changed)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.yaxis_1d_changed)

        self.ui.comboBox_xaxisNames_2dplot.currentIndexChanged.connect(self.xaxis_2d_changed)
        self.ui.comboBox_yaxisNames_2dplot.currentIndexChanged.connect(self.yaxis_2d_changed)
        self.ui.comboBox_zaxisNames_2dplot.currentIndexChanged.connect(self.zaxis_2d_changed)

        self.ui.peak_range_table.cellChanged.connect(self.peak_range_table_changed)

        # tracker for sample log names and peak parameter names
        self._sample_log_name_set = set()
        self._function_param_name_set = set()

        # mutexes
        self._sample_log_names_mutex = False

        # TODO - 20181124 - New GUI parameters (After FitPeaks)
        # checkBox_showFitError
        # checkBox_showFitValue
        # others
        # TODO - 20181124 - Make this table's column flexible!
        self.ui.tableView_fitSummary.setup(peak_param_names=list())

        o_gui = GuiUtilities(parent=self)
        o_gui.enabled_fitting_widgets(False)
        o_gui.enabled_1dplot_widgets(False)
        o_gui.enabled_2dplot_widgets(False)
        o_gui.make_visible_listsubruns_warning(False)
        o_gui.enabled_export_csv_widgets(False)
        o_gui.enabled_peak_ranges_widgets(False)

    # Menu event handler
    def browse_hdf(self):
        """ Browse Hidra project HDF file
        """
        o_handler = EventHandler(parent=self)
        o_handler.browse_load_plot_hdf()
        o_plot = Plot(parent=self)
        o_plot.plot_1d()

    def load_hidra_file(self):
        o_handler = EventHandler(parent=self)
        o_handler.load_hidra_file()

    def plot_diff_data(self):
        o_plot = Plot(parent=self)
        o_plot.plot_diff_data()

    def fit_peaks(self):
        o_fit = Fit(parent=self)
        o_fit.fit_multi_peaks()
        self.individual_or_list_sub_runs()

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
        o_gui = GuiUtilities(parent=self)
        o_gui.check_subRuns_display_mode()

    def plot_scan(self):
        o_plot = Plot(parent=self)
        o_plot.plot_scan()

    def list_subruns_2dplot_returned(self):
        o_handle = EventHandler(parent=self)
        o_handle.list_subruns_2dplot_returned()

    def list_subruns_2dplot_changed(self):
        o_handle = EventHandler(parent=self)
        o_handle.list_subruns_2dplot_changed()

    def xaxis_1d_changed(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_axis1d_status()
        o_plot = Plot(parent=self)
        o_plot.plot_1d()

    def yaxis_1d_changed(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_axis1d_status()
        o_plot = Plot(parent=self)
        o_plot.plot_1d()

    def xaxis_2d_changed(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_axis2d_status()

    def yaxis_2d_changed(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_axis2d_status()

    def zaxis_2d_changed(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_axis2d_status()

    def export_csv(self):
        o_export = ExportCSV(parent=self)
        o_export.select_output_folder()
        o_export.create_csv()

    def update_peak_ranges_table(self, **kwargs):
        o_handle = EventHandler(parent=self)
        o_handle.update_fit_peak_ranges_table(**kwargs)

    def _promote_peak_fit_setup(self):
        # 2D detector view
        curr_layout = QVBoxLayout()
        self.ui.frame_PeakView.setLayout(curr_layout)
        self._ui_graphicsView_fitSetup = PeakFitSetupView(self)
        self._ui_graphicsView_fitSetup.setEnabled(False)
        curr_layout.addWidget(self._ui_graphicsView_fitSetup)

    def peak_range_table_changed(self, row, column):
        print("tbale changed at row:{} and column:{}".format(row, column))

    def _init_widgets(self):
        """
        initialize the some widgets
        :return:
        """
        self.ui.actionSave.setEnabled(False)
        self.ui.actionSaveAs.setEnabled(False)

        self.ui.splitter_4.setStyleSheet(VERTICAL_SPLITTER)
        self.ui.splitter_4.setSizes([100, 0])
        self.ui.splitter_2.setStyleSheet(HORIZONTAL_SPLITTER)
        self.ui.splitter_2.setSizes([100, 0])
        self.ui.splitter_3.setStyleSheet(HORIZONTAL_SPLITTER)
        # self.ui.splitter.setStyleSheet(HORIZONTAL_SPLITTER)

        # status bar
        self.setStyleSheet("QStatusBar{padding-left:8px;color:green;}")

        # warning icon
        self.ui.listsubruns_warning_icon.setPixmap(QtGui.QPixmap(":/fitting/warning_icon.png"))

        # hide and initialize d0 and microstrength widgets
        self.ui.label_d0units1d.setText(MICROSTRAIN)
        self.ui.label_d0units2d.setText(MICROSTRAIN)
        self.ui.label_d01d.setText(D0)
        self.ui.label_d02d.setText(D0)

        o_gui = GuiUtilities(parent=self)
        o_gui.make_visible_d01d_widgets(visible=False)
        o_gui.make_visible_d02d_widgets(visible=False)

    def do_launch_adv_fit(self):
        """
        launch the dialog window for advanced peak fitting setup and control
        :return:
        """
        if self._advanced_fit_dialog is None:
            self._advanced_fit_dialog = pyrs.interface.advpeakfitdialog.SmartPeakFitControlDialog(self)

        self._advanced_fit_dialog.show()

    def do_make_movie(self):
        """
        plot all the fitted data for each scan log index and save the figure to PNG files
        in order to make a movie for quick fit result check
        :return:
        """
        # get target directory to save all the PNG files
        target_dir = QFileDialog.getExistingDirectory(self, 'Select the directory to save PNGs for quick '
                                                            'fit result checking movie',
                                                      self._core.working_dir)
        target_dir = str(target_dir)
        if len(target_dir) == 0:
            return

        # plot
        scan_log_indexes = self._core.get_peak_fit_scan_log_indexes(self._curr_data_key)
        for sample_log_index in scan_log_indexes:
            # reset the canvas
            self._ui_graphicsView_fitSetup.reset_viewer()
            # plot
            self.plot_diff_data(sample_log_index, True)
            png_name_i = os.path.join(target_dir, '{}_fit.png'.format(sample_log_index))
            self._ui_graphicsView_fitSetup.canvas().save_figure(png_name_i)
        # END-FOR

        # TODO - 20180809 - Pop the following command
        # TODO - continue - command to pop: ffmpeg -r 24 -framerate 8 -pattern_type glob -i '*_fit.png' out.mp4

    def do_plot_2d_data(self):
        # TODO - #84 - Implement this method
        return

    def save(self):
        pass

    def save_as(self):
        """ export the peaks to another file
        :return:
        """
        out_file_name = pyrs.interface.gui_helper.browse_file(self,
                                                              caption='Choose a file to save fitted peaks to',
                                                              default_dir=self._core.working_dir,
                                                              file_filter='HDF (*.hdf5)',
                                                              save_file=True)

        self.save_fit_result(out_file_name)

    def do_save_fit(self):
        """
        save fit result
        :return:
        """
        file_name = pyrs.interface.gui_helper.browse_file(self, 'Select file to save fit result',
                                                          default_dir=self._core.working_dir,
                                                          file_filter='HDF (*.hdf5);;CSV (*.csv)',
                                                          file_list=False,
                                                          save_file=True)

        if file_name.lower().endswith('hdf5') or file_name.lower().endswith('hdf') or file_name.lower().endswith('h5'):
            self.save_fit_result(out_file_name=file_name)
        elif file_name.lower().endswith('csv') or file_name.endswith('dat'):
            self.export_fit_result(file_name)
        else:
            pyrs.interface.gui_helper.pop_message(self,
                                                  message='Input file {} has an unsupported posfix.'.format(file_name),
                                                  detailed_message='Supported are hdf5, h5, hdf, csv and dat',
                                                  message_type='error')

    # def do_save_fit_result(self):
    #     """
    #     save fit result
    #     :return:
    #     """
    #     # get file name
    #     csv_filter = 'CSV Files(*.csv);;DAT Files(*.dat);;All Files(*.*)'
    #     # with filter, the returned will contain 2 values
    #     user_input = QFileDialog.getSaveFileName(self, 'CSV file for peak fitting result', self._core.working_dir,
    #                                              csv_filter)
    #     if isinstance(user_input, tuple) and len(user_input) == 2:
    #         file_name = str(user_input[0])
    #     else:
    #         file_name = str(user_input)
    #
    #     if file_name == '':
    #         # user cancels
    #         return
    #
    #     self.export_fit_result(file_name)

    def do_quit(self):
        """
        close the window and quit
        :return:
        """
        self.close()

    def export_fit_result(self, file_name):
        """
        export fit result to a csv file
        :param file_name:
        :return:
        """
        self.ui.tableView_fitSummary.export_table_csv(file_name)

    def fit_peaks_smart(self, peak_profiles_order):
        """
        fit peaks with a "smart" algorithm
        :param peak_profiles_order: a list for peak profile to fit in specified order
        :return:
        """
        try:
            self._core.fit_peaks_smart_alg(self._curr_data_key, peak_profiles_order)
        except RuntimeError as run_err:
            err_msg = 'Smart peak fitting with order {} failed due to {}' \
                      ''.format(peak_profiles_order, run_err)
            pyrs.interface.gui_helper.pop_message(self, err_msg, 'error')

    def save_data_for_mantid(self, data_key, file_name):
        """
        save data to Mantid-compatible NeXus
        :param data_key:
        :param file_name:
        :return:
        """
        self._core.save_nexus(data_key, file_name)

    def save_fit_result(self, out_file_name):
        """
        make a copy of the input file and add the fit result into it
        :param out_file_name:
        :return:
        """
        print('Plan to copy {} to {} and insert fit result'.format(self._curr_file_name,
                                                                   out_file_name))
        # TODO FIXME - TONIGHT NOW - Fit the following method!
        # FIXME Temporarily disabled:
        # self._core.save_peak_fit_result(self._curr_data_key, self._curr_file_name, out_file_name)
