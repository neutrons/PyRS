import numpy
import os
from qtpy.QtWidgets import QVBoxLayout, QFileDialog, QMainWindow


from pyrs.utilities import load_ui
from pyrs.interface.ui import qt_util
from pyrs.interface.ui.diffdataviews import GeneralDiffDataView, DiffContourView
from pyrs.interface.ui.rstables import FitResultTable
from pyrs.interface.ui.diffdataviews import PeakFitSetupView
# from pyrs.utilities import checkdatatypes
from pyrs.utilities.rs_project_file import HidraConstants
import pyrs.interface.advpeakfitdialog
import pyrs.interface.gui_helper
from pyrs.interface.peak_fitting.event_handler import EventHandler
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.fit import Fit
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities


class FitPeaksWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(FitPeaksWindow, self).__init__(parent)

        # class variables
        self._core = None
        self._project_name = None
        # current/last loaded data
        self._curr_file_name = None

        # a copy of sample logs
        self._sample_log_names = list()  # a copy of sample logs' names that are added to combo-box

        # sub windows
        self._advanced_fit_dialog = None

        # set up UI
        # self.ui = ui.ui_peakfitwindow.Ui_MainWindow()
        # self.ui.setupUi(self)

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'peakfitwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)

        # promote
        self.ui.graphicsView_fitResult = qt_util.promote_widget(self, self.ui.graphicsView_fitResult_frame,
                                                                GeneralDiffDataView)
        self.ui.graphicsView_fitResult.setEnabled(False)
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self.ui.graphicsView_contourView = qt_util.promote_widget(self, self.ui.graphicsView_contourView_frame,
                                                                  DiffContourView)
        self.ui.graphicsView_contourView.setEnabled(False)
        self.ui.tableView_fitSummary = qt_util.promote_widget(self, self.ui.tableView_fitSummary_frame,
                                                              FitResultTable)
        self._promote_peak_fit_setup()

        self._init_widgets()

        # set up handling
        self.ui.pushButton_loadHDF.clicked.connect(self.load_hidra_file)
        # self.ui.lineEdit_scanNumbers.returnPressed.connect(self.plot_diff_data)
        self.ui.pushButton_browseHDF.clicked.connect(self.browse_hdf)
        # self.ui.pushButton_plotPeaks.clicked.connect(self.plot_diff_data)
        self.ui.lineEdit_listSubRuns.returnPressed.connect(self.plot_diff_data)
        self.ui.pushButton_fitPeaks.clicked.connect(self.fit_peaks)
        self.ui.horizontalScrollBar_SubRuns.valueChanged.connect(self.plot_scan)
        self.ui.pushButton_saveFitResult.clicked.connect(self.do_save_fit)
        self.ui.radioButton_individualSubRuns.clicked.connect(self.individualSubRuns)
        self.ui.radioButton_listSubRuns.clicked.connect(self.listSubRuns)
        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionSave_As.triggered.connect(self.do_save_as)
        self.ui.actionSave_Fit_Result.triggered.connect(self.do_save_fit_result)
        self.ui.actionAdvanced_Peak_Fit_Settings.triggered.connect(self.do_launch_adv_fit)
        self.ui.actionQuick_Fit_Result_Check.triggered.connect(self.do_make_movie)

        # TODO - 20180805 - Implement : pushButton_plotLogs, comboBox_detectorI

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)
        self.ui.comboBox_2dPlotChoice.currentIndexChanged.connect(self.do_plot_2d_data)

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

    # Menu event handler
    def browse_hdf(self):
        """ Browse Hidra project HDF file
        """
        o_handler = EventHandler(parent=self)
        o_handler.browse_and_load_hdf()

    def load_hidra_file(self):
        o_handler = EventHandler(parent=self)
        o_handler.load_hidra_file()

    def plot_diff_data(self):
        o_plot = Plot(parent=self)
        o_plot.plot_diff_data()

    def fit_peaks(self):
        o_fit = Fit(parent=self)
        o_fit.fit_peaks()

    def individualSubRuns(self):
        self.check_subRunsDisplayMode()
        self.plot_scan()

    def listSubRuns(self):
        self.check_subRunsDisplayMode()
        self.plot_diff_data()

    def check_subRunsDisplayMode(self):
        o_gui = GuiUtilities(parent=self)
        o_gui.check_subRuns_display_mode()

    def plot_scan(self):
        o_plot = Plot(parent=self)
        o_plot.plot_scan()

    def _promote_peak_fit_setup(self):
        # 2D detector view
        curr_layout = QVBoxLayout()
        self.ui.frame_PeakView.setLayout(curr_layout)
        self._ui_graphicsView_fitSetup = PeakFitSetupView(self)
        self._ui_graphicsView_fitSetup.setEnabled(False)

        curr_layout.addWidget(self._ui_graphicsView_fitSetup)

    def _init_widgets(self):
        """
        initialize the some widgets
        :return:
        """
        self.ui.pushButton_loadHDF.setEnabled(False)

        # combo boxes
        self.ui.comboBox_2dPlotChoice.clear()
        self.ui.comboBox_2dPlotChoice.addItem('Raw Data')
        self.ui.comboBox_2dPlotChoice.addItem('Fitted')

        # check boxes
        self.ui.checkBox_autoSaveFitResult.setChecked(True)

    def do_launch_adv_fit(self):
        """
        launch the dialog window for advanced peak fitting setup and control
        :return:
        """
        if self._advanced_fit_dialog is None:
            self._advanced_fit_dialog = pyrs.interface.advpeakfitdialog.SmartPeakFitControlDialog(self)

        self._advanced_fit_dialog.show()

    def _set_sample_logs_for_plotting(self, sample_log_names):
        """ There are 2 combo boxes containing sample logs' names for plotting.  Clear the existing ones
        and add the sample log names specified to them
        :param sample_log_names:
        :return:
        """
        self._sample_log_names_mutex = True
        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()

        # Maintain a copy of sample logs!
        self._sample_log_names = list(set(sample_log_names))
        self._sample_log_names.sort()

        for sample_log in sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log)
            self.ui.comboBox_yaxisNames.addItem(sample_log)
            self._sample_log_name_set.add(sample_log)
        self._sample_log_names_mutex = False

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
        """

        :return:
        """
        # TODO - #84 - Implement this method
        return

    def do_plot_meta_data(self):
        """
        plot the meta/fit result data on the right side GUI
        :return:
        """
        if self._sample_log_names_mutex:
            return

        # if self.ui.checkBox_keepPrevPlotRight.isChecked() is False:
        # TODO - Shall be controlled by a more elegant mechanism
        self.ui.graphicsView_fitResult.reset_viewer()

        # get the sample log/meta data name
        x_axis_name = str(self.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.ui.comboBox_yaxisNames.currentText())

        # Return if sample logs combo box not set
        if x_axis_name == '' and y_axis_name == '':
            return

        if x_axis_name in self._function_param_name_set and y_axis_name == HidraConstants.SUB_RUNS:
            vec_y, vec_x = self.get_function_parameter_data(x_axis_name)
        elif y_axis_name in self._function_param_name_set and x_axis_name == HidraConstants.SUB_RUNS:
            vec_x, vec_y = self.get_function_parameter_data(y_axis_name)
        elif x_axis_name in self._function_param_name_set or y_axis_name in self._function_param_name_set:
            pyrs.interface.gui_helper.pop_message(self, 'It has not considered how to plot 2 function parameters '
                                                        '{} and {} against each other'
                                                        ''.format(x_axis_name, y_axis_name),
                                                        message_type='error')
            return
        else:
            vec_x = self.get_meta_sample_data(x_axis_name)
            vec_y = self.get_meta_sample_data(y_axis_name)
        # END-IF-ELSE

        if vec_x is None or vec_y is None:
            raise RuntimeError('{} or {} cannot be None ({}, {})'
                               ''.format(x_axis_name, y_axis_name, vec_x, vec_y))

        self.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)

        return

    def do_save_as(self):
        """ export the peaks to another file
        :return:
        """
        out_file_name = pyrs.interface.gui_helper.browse_file(self,
                                                              caption='Choose a file to save fitted peaks to',
                                                              default_dir=self._core.working_dir,
                                                              file_filter='HDF (*.hdf5)',
                                                              save_file=True)

        self.save_fit_result(out_file_name)

        return

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

        return

    def do_save_fit_result(self):
        """
        save fit result
        :return:
        """
        # get file name
        csv_filter = 'CSV Files(*.csv);;DAT Files(*.dat);;All Files(*.*)'
        # with filter, the returned will contain 2 values
        user_input = QFileDialog.getSaveFileName(self, 'CSV file for peak fitting result', self._core.working_dir,
                                                 csv_filter)
        if isinstance(user_input, tuple) and len(user_input) == 2:
            file_name = str(user_input[0])
        else:
            file_name = str(user_input)

        if file_name == '':
            # user cancels
            return

        self.export_fit_result(file_name)

        return

    def do_quit(self):
        """
        close the window and quit
        :return:
        """
        self.close()

        return

    def export_fit_result(self, file_name):
        """
        export fit result to a csv file
        :param file_name:
        :return:
        """
        self.ui.tableView_fitSummary.export_table_csv(file_name)

        return

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

        return

    def get_function_parameter_data(self, param_name):
        """ get the parameter function data
        :param param_name:
        :return:
        """
        # get data key
        if self._project_name is None:
            pyrs.interface.gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        param_names, param_data = self._core.get_peak_fitting_result(self._project_name, return_format=dict,
                                                                     effective_parameter=False)

        print('[DB...BAT] Param Names: {}'.format(param_names))
        sub_run_vec = param_data[HidraConstants.SUB_RUNS]
        param_value_2darray = param_data[param_name]
        print('[DB...BAT] 2D array shape: {}'.format(param_value_2darray.shape))

        return sub_run_vec, param_value_2darray[:, 0]

    def get_meta_sample_data(self, name):
        """
        get meta data to plot.
        the meta data can contain sample log data and fitted peak parameters
        :param name:
        :return:
        """
        # get data key
        if self._project_name is None:
            pyrs.interface.gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        sample_log_names = self._core.reduction_service.get_sample_logs_names(self._project_name, True)

        if name == HidraConstants.SUB_RUNS:
            # sub run vector
            value_vector = numpy.array(self._core.reduction_service.get_sub_runs(self._project_name))
        elif name in sample_log_names:
            # sample log but not sub-runs
            value_vector = self._core.reduction_service.get_sample_log_value(self._project_name, name)
        elif name == 'Center of mass':
            # center of mass is different????
            # TODO - #84 - Make sure of it!
            value_vector = self._core.get_peak_center_of_mass(self._project_name)
        else:
            value_vector = None

        return value_vector

    def save_data_for_mantid(self, data_key, file_name):
        """
        save data to Mantid-compatible NeXus
        :param data_key:
        :param file_name:
        :return:
        """
        self._core.save_nexus(data_key, file_name)

        return

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

        return

    def setup_window(self, pyrs_core):
        """ set up the window.  It must be called mandatory
        :param pyrs_core:
        :return:
        """
        from pyrs.core.pyrscore import PyRsCore
        # check
        assert isinstance(pyrs_core, PyRsCore), 'Controller core {0} must be a PyRSCore instance but not a {1}.' \
                                                ''.format(pyrs_core, pyrs_core.__class__.__name__)

        self._core = pyrs_core

        return
