from pyrs.utilities import load_ui
from qtpy.QtWidgets import QVBoxLayout, QFileDialog, QMainWindow

from pyrs.interface.ui import qt_util
from pyrs.interface.ui.diffdataviews import GeneralDiffDataView, DiffContourView
from pyrs.interface.ui.rstables import FitResultTable
from pyrs.utilities import hb2b_utilities
from pyrs.utilities import checkdatatypes
from pyrs.utilities.rs_project_file import HidraConstants
import advpeakfitdialog
import os
import gui_helper
import numpy


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
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self.ui.graphicsView_contourView = qt_util.promote_widget(self, self.ui.graphicsView_contourView_frame,
                                                                  DiffContourView)
        self.ui.tableView_fitSummary = qt_util.promote_widget(self, self.ui.tableView_fitSummary_frame,
                                                              FitResultTable)

        self._promote_peak_fit_setup()

        self._init_widgets()
        # init some widgets
        self.ui.checkBox_autoLoad.setChecked(True)

        # set up handling
        self.ui.pushButton_loadHDF.clicked.connect(self.do_load_hydra_file)
        self.ui.pushButton_browseHDF.clicked.connect(self.do_browse_hdf)
        self.ui.pushButton_plotPeaks.clicked.connect(self.do_plot_diff_data)
        self.ui.pushButton_plotPreviousScan.clicked.connect(self.do_plot_prev_scan)
        self.ui.pushButton_plotNextScan.clicked.connect(self.do_plot_next_scan)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_fit_peaks)
        self.ui.pushButton_saveFitResult.clicked.connect(self.do_save_fit)

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

        return

    def _promote_peak_fit_setup(self):
        """

        :return:
        """
        from ui.diffdataviews import PeakFitSetupView

        # 2D detector view
        curr_layout = QVBoxLayout()
        self.ui.frame_PeakView.setLayout(curr_layout)
        self._ui_graphicsView_fitSetup = PeakFitSetupView(self)

        curr_layout.addWidget(self._ui_graphicsView_fitSetup)

        return

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self._core is None:
            raise RuntimeError('Not set up yet!')

    def _get_default_hdf(self):
        """
        use IPTS and Exp to determine
        :return:
        """
        try:
            ipts_number = gui_helper.parse_integer(self.ui.lineEdit_iptsNumber)
            exp_number = gui_helper.parse_integer(self.ui.lineEdit_expNumber)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to parse IPTS or Exp due to {0}'.format(run_err))
            return None

        # Locate default saved HidraProject data
        archive_data = hb2b_utilities.get_hb2b_raw_data(ipts_number, exp_number)

        return archive_data

    def _init_widgets(self):
        """
        initialize the some widgets
        :return:
        """
        self.ui.pushButton_loadHDF.setEnabled(False)
        self.ui.checkBox_autoFit.setChecked(True)
        self.ui.checkBox_autoLoad.setChecked(True)

        # combo boxes
        self.ui.comboBox_2dPlotChoice.clear()
        self.ui.comboBox_2dPlotChoice.addItem('Raw Data')
        self.ui.comboBox_2dPlotChoice.addItem('Fitted')

        # check boxes
        self.ui.checkBox_autoSaveFitResult.setChecked(True)

        return

    def do_browse_hdf(self):
        """ Browse Hidra project HDF file
        :return:
        """
        # Check
        self._check_core()

        # Use IPTS and run number to get the default Hydra HDF
        hydra_file_name = self._get_default_hdf()
        if hydra_file_name is None:
            # No default Hidra file: browse the file
            file_filter = 'HDF (*.hdf);H5 (*.h5)'
            hydra_file_name = gui_helper.browse_file(self, 'HIDRA Project File', os.getcwd(), file_filter,
                                                     file_list=False, save_file=False)

            if hydra_file_name is None:
                # use cancel
                return
        # END-IF

        # Add file name to line edit to show
        self.ui.lineEdit_expFileName.setText(hydra_file_name)

        # Load file as an option
        if self.ui.checkBox_autoLoad.isChecked():
            try:
                self.do_load_hydra_file(from_browse=True)
            except RuntimeError as run_err:
                gui_helper.pop_message(self, 'Failed to load {}'.format(hydra_file_name),
                                       str(run_err), 'error')
        # END-IF

        return

    def do_launch_adv_fit(self):
        """
        launch the dialog window for advanced peak fitting setup and control
        :return:
        """
        if self._advanced_fit_dialog is None:
            self._advanced_fit_dialog = advpeakfitdialog.SmartPeakFitControlDialog(self)

        self._advanced_fit_dialog.show()

        return

    def do_load_hydra_file(self, hydra_project_file=None):
        """ Load Hidra project file
        :return: None
        """
        self._check_core()

        # Get file
        if hydra_project_file is None:
            hydra_project_file = str(self.ui.lineEdit_expFileName.text())
        else:
            checkdatatypes.check_string_variable(hydra_project_file)

        # load file
        try:
            self._project_name = os.path.basename(hydra_project_file).split('.')[0]
            self._core.load_hidra_project(hydra_project_file, project_name=self._project_name,
                                          load_detector_counts=False,
                                          load_diffraction=True)
            # Record data key and next
            self._curr_file_name = hydra_project_file
        except (RuntimeError, TypeError) as run_err:
            gui_helper.pop_message(self, 'Unable to load {}'.format(hydra_project_file), detailed_message=str(run_err),
                                   message_type='error')
            return

        # Edit information on the UI for user to visualize
        self.ui.label_loadedFileInfo.setText('Loaded {}; Project name: {}'
                                             ''.format(hydra_project_file, self._project_name))

        # Get the range of sub runs
        sub_run_list = self._core.reduction_manager.get_sub_runs(self._project_name)
        self.ui.label_logIndexMin.setText(str(sub_run_list[0]))
        self.ui.label_logIndexMax.setText(str(sub_run_list[-1]))

        # Set the widgets about viewer: get the sample logs and add the combo boxes for plotting
        sample_log_names = self._core.reduction_manager.get_sample_logs_names(self._project_name, can_plot=True)
        self._set_sample_logs_for_plotting(sample_log_names)

        # plot first peak for default peak range
        self.ui.lineEdit_scanNumbers.setText('1')
        self.do_plot_diff_data(plot_model=False)

        # reset the plot
        self.ui.graphicsView_fitResult.reset_viewer()

        # Set the table
        if self.ui.tableView_fitSummary.rowCount() > 0:
            self.ui.tableView_fitSummary.remove_all_rows()
        self.ui.tableView_fitSummary.init_exp(sub_run_list)

        # Auto fit for all the peaks
        if self.ui.checkBox_autoFit.isChecked():
            self.do_fit_peaks(all_sub_runs=True)

        return

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

        return

    def _parse_sub_runs(self):
        """ Parse sub run numbers specified in lineEdit_scanNumbers
        :return: List (of integers) or None
        """
        int_string_list = str(self.ui.lineEdit_scanNumbers.text()).strip()
        if len(int_string_list) == 0 or not self.ui.fit_selected.isChecked():
            sub_run_list = None  # not set and thus default for all
        else:
            sub_run_list = gui_helper.parse_integers(int_string_list)

        return sub_run_list

    def do_fit_peaks(self, all_sub_runs=False):
        """ Fit peaks either all peaks or selected peaks
        The workflow includes
        1. parse sub runs, peak and background type
        2. fit peaks
        3. show the fitting result in table
        4. plot the peak in first sub runs that is fit
        :param all_sub_runs: Flag to fit peaks of all sub runs without checking
        :return:
        """
        # Get the sub runs to fit or all the peaks
        if self.ui.checkBox_fitSubRuns.isChecked() and not all_sub_runs:
            # Parse sub runs specified in lineEdit_scanNumbers
            sub_run_list = self._parse_sub_runs()
        else:
            sub_run_list = None

        # Get peak function and background function
        peak_function = str(self.ui.comboBox_peakType.currentText())
        bkgd_function = str(self.ui.comboBox_backgroundType.currentText())

        # Get peak fitting range from the range of figure
        fit_range = self._ui_graphicsView_fitSetup.get_x_limit()
        print ('[INFO] Peak fit range: {0}'.format(fit_range))

        # Fit Peaks: It is better to fit all the peaks at the same time after testing
        guessed_peak_center = 0.5 * (fit_range[0] + fit_range[1])
        peak_info_dict = {'Peak 1': {'Center': guessed_peak_center, 'Range': fit_range}}
        self._core.fit_peaks(self._project_name, sub_run_list,
                             peak_type=peak_function,
                             background_type=bkgd_function,
                             peaks_fitting_setup=peak_info_dict)

        # Process fitted peaks
        # TEST - #84 - This shall be reviewed!
        function_params, fit_values = self._core.get_peak_fitting_result(self._project_name,
                                                                         return_format=dict,
                                                                         effective_parameter=False)
        # TODO - #84+ - Need to implement the option as effective_parameter=True

        print ('[DB...BAT...FITWINDOW....FIT] returned = {}, {}'.format(function_params, fit_values))

        self._sample_log_names_mutex = True
        curr_x_index = self.ui.comboBox_xaxisNames.currentIndex()
        curr_y_index = self.ui.comboBox_yaxisNames.currentIndex()
        # add fitted parameters by resetting and build from the copy of fit parameters
        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()
        # add sample logs (names)
        for sample_log_name in self._sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log_name)
            self.ui.comboBox_yaxisNames.addItem(sample_log_name)
        # add function parameters (names)
        for param_name in function_params:
            self.ui.comboBox_xaxisNames.addItem(param_name)
            self.ui.comboBox_yaxisNames.addItem(param_name)
            self._function_param_name_set.add(param_name)
        # add observed parameters
        self.ui.comboBox_xaxisNames.addItem('Center of mass')
        self.ui.comboBox_yaxisNames.addItem('Center of mass')
        # keep current selected item unchanged
        size_x = len(self._sample_log_names) + len(self._function_param_name_set) + 2  # log index and center of mass
        size_y = len(self._sample_log_names) + len(self._function_param_name_set) + 1  # center of mass
        if curr_x_index < size_x:
            # keep current set up
            self.ui.comboBox_xaxisNames.setCurrentIndex(curr_x_index)
        else:
            # original one does not exist: reset to first/log index
            self.ui.comboBox_xaxisNames.setCurrentIndex(0)

        # release the mutex: because re-plot is required anyway
        self._sample_log_names_mutex = False

        # plot Y
        if curr_y_index >= size_y:
            # out of boundary: use the last one (usually something about peak)
            curr_y_index = size_y - 1
        self.ui.comboBox_yaxisNames.setCurrentIndex(curr_y_index)

        # Show fitting result in Table
        # TODO - could add an option to show native or effective peak parameters
        self._show_fit_result_table(peak_function, function_params, fit_values, is_effective=False)

        # plot the model and difference
        if sub_run_list is None:
            scan_log_index = 0
            # FIXME This case is not likely to occur
        # FIXME - TODO - self.do_plot_diff_data()

        # plot the contour
        # TODO - #84+ - Implement this! Plot contour for what????
        # self.ui.graphicsView_contourView

        return

    def _show_fit_result_table(self, peak_function, peak_param_names, peak_param_dict, is_effective):
        """ Set up the table containing fit result
        :param peak_function: name of peak function
        :param peak_param_names: name of the parameters for the order of columns
        :param peak_param_dict: parameter names
        :param is_effective: Flag for the parameter to be shown as effective (True) or native (False)
        :return:
        """
        # Add peaks' centers of mass to the output table
        peak_param_names.append(HidraConstants.PEAK_COM)
        com_vec = self._core.get_peak_center_of_mass(self._project_name)
        peak_param_dict[HidraConstants.PEAK_COM] = com_vec

        # Initialize the table by resetting the column names
        self.ui.tableView_fitSummary.reset_table(peak_param_names)

        # Get sub runs for rows in the table
        sub_run_vec = peak_param_dict[HidraConstants.SUB_RUNS]

        # Add rows to the table for parameter information
        for row_index in range(sub_run_vec.shape[0]):
            # Set fit result
            self.ui.tableView_fitSummary.set_fit_summary(row_index, peak_param_names, peak_param_dict,
                                                         write_error=False,
                                                         peak_profile=peak_function)
        # END-FOR

        return

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

        return

    def do_plot_2d_data(self):
        """

        :return:
        """
        # TODO - #84 - Implement this method
        return

    def do_plot_diff_data(self, plot_model=True):
        """
        plot diffraction data
        :return:
        """
        # gather the information
        scan_log_index_list = gui_helper.parse_integers(str(self.ui.lineEdit_scanNumbers.text()))
        if len(scan_log_index_list) == 0:
            gui_helper.pop_message(self, 'There is not scan-log index input', 'error')

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self._ui_graphicsView_fitSetup.reset_viewer()

        # get data and plot
        err_msg = ''
        plot_model = len(scan_log_index_list) == 1 and plot_model
        for scan_log_index in scan_log_index_list:
            try:
                self.plot_diff_data(scan_log_index, plot_model)
            except RuntimeError as run_err:
                err_msg += '{0}\n'.format(run_err)
        # END-FOR

        if len(err_msg) > 0:
            gui_helper.pop_message(self, err_msg, message_type='error')

        return

    def do_plot_next_scan(self):
        """ plot the next scan (log index)
        It is assumed that al the scan log indexes are consecutive
        :return:
        """
        scan_log_index_list = gui_helper.parse_integers(str(self.ui.lineEdit_scanNumbers.text()))
        last_log_index = int(self.ui.label_logIndexMax.text())
        if len(scan_log_index_list) == 0:
            gui_helper.pop_message(self, 'There is not scan-log index input', 'error')
        elif len(scan_log_index_list) > 1:
            gui_helper.pop_message(self, 'There are too many scans for "next"', 'error')
        elif scan_log_index_list[0] == last_log_index:
            # last log index: no operation
            return

        next_scan_log = scan_log_index_list[0] + 1
        try:
            self._ui_graphicsView_fitSetup.reset_viewer()
            self.plot_diff_data(next_scan_log, True)
        except RuntimeError as run_err:
            # self.plot_diff_data(next_scan_log - 1, True)
            err_msg = 'Unable to plot next scan {} due to {}'.format(next_scan_log, run_err)
            gui_helper.pop_message(self, err_msg, message_type='error')
        else:
            self.ui.lineEdit_scanNumbers.setText('{}'.format(next_scan_log))

        return

    def do_plot_prev_scan(self):
        """ plot the previous scan (log index)
        It is assumed that al the scan log indexes are consecutive
        :return:
        """
        scan_log_index_list = gui_helper.parse_integers(str(self.ui.lineEdit_scanNumbers.text()))
        if len(scan_log_index_list) == 0:
            gui_helper.pop_message(self, 'There is not scan-log index input', 'error')
        elif len(scan_log_index_list) > 1:
            gui_helper.pop_message(self, 'There are too many scans for "next"', 'error')
        elif scan_log_index_list[0] == 0:
            # first one: no operation
            return

        prev_scan_log_index = scan_log_index_list[0] - 1
        try:
            self._ui_graphicsView_fitSetup.reset_viewer()
            self.plot_diff_data(prev_scan_log_index, True)
        except RuntimeError as run_err:
            # self.plot_diff_data(next_scan_log + 1, True)
            err_msg = 'Unable to plot previous scan {} due to {}'.format(prev_scan_log_index, run_err)
            gui_helper.pop_message(self, err_msg, message_type='error')
        else:
            self.ui.lineEdit_scanNumbers.setText('{}'.format(prev_scan_log_index))
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
            gui_helper.pop_message(self, 'It has not considered how to plot 2 function parameters '
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
        out_file_name = gui_helper.browse_file(self,
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
        file_name = gui_helper.browse_file(self, 'Select file to save fit result', default_dir=self._core.working_dir,
                                           file_filter='HDF (*.hdf5);;CSV (*.csv)', file_list=False,
                                           save_file=True)

        if file_name.lower().endswith('hdf5') or file_name.lower().endswith('hdf') or file_name.lower().endswith('h5'):
            self.save_fit_result(out_file_name=file_name)
        elif file_name.lower().endswith('csv') or file_name.endswith('dat'):
            self.export_fit_result(file_name)
        else:
            gui_helper.pop_message(self, message='Input file {} has an unsupported posfix.'.format(file_name),
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
            gui_helper.pop_message(self, err_msg, 'error')

        return

    def get_function_parameter_data(self, param_name):
        """ get the parameter function data
        :param param_name:
        :return:
        """
        # get data key
        if self._project_name is None:
            gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        sub_run_vec, chi2_vec, param_value_2darray = self._core.get_peak_fit_param_value(self._project_name,
                                                                                         param_name,
                                                                                         max_cost=1E5)
        print('DB...BAT: chi2 shape = {}, param values shape = {}'.format(chi2_vec.shape,
                                                                          param_value_2darray.shape))
        return sub_run_vec, param_value_2darray[0]

    def get_meta_sample_data(self, name):
        """
        get meta data to plot.
        the meta data can contain sample log data and fitted peak parameters
        :param name:
        :return:
        """
        # get data key
        if self._project_name is None:
            gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        sample_log_names = self._core.reduction_manager.get_sample_logs_names(self._project_name, True)

        if name == HidraConstants.SUB_RUNS:
            # sub run vector
            value_vector = numpy.array(self._core.reduction_manager.get_sub_runs(self._project_name))
        elif name in sample_log_names:
            # sample log but not sub-runs
            value_vector = self._core.reduction_manager.get_sample_log_values(self._project_name, name)
        elif name == 'Center of mass':
            # center of mass is different????
            # TODO - #84 - Make sure of it!
            value_vector = self._core.get_peak_center_of_mass(self._project_name)
        else:
            value_vector = None

        return value_vector

    def plot_diff_data(self, sub_run_number, plot_model):
        """ Plot a set of diffraction data (one scan log index) and plot its fitted data
        :return:
        """
        # TODO FIXME - #84: TypeError: get_diffraction_data() got an unexpected keyword argument 'data_key'
        # ..... UNIT TEST!
        # get experimental data and plot
        diff_data_set = self._core.get_diffraction_data(session_name=self._project_name,
                                                        sub_run=sub_run_number,
                                                        mask=None)

        data_set_label = 'Scan {0}'.format(sub_run_number)

        # Plot experimental data
        self._ui_graphicsView_fitSetup.plot_experiment_data(diff_data_set=diff_data_set, data_reference=data_set_label)

        # Plot fitted model data
        if plot_model:
            model_data_set = self._core.get_modeled_data(session_name=self._project_name,
                                                         sub_run=sub_run_number)
            if model_data_set is not None:
                residual_y_vec = diff_data_set[1] - model_data_set[1]
                residual_data_set = [diff_data_set[0], residual_y_vec]
                self._ui_graphicsView_fitSetup.plot_model_data(diff_data_set=model_data_set, model_label='',
                                                               residual_set=residual_data_set)
            # END-if
        # END-IF-ELSE

        return

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
        print ('Plan to copy {} to {} and insert fit result'.format(self._curr_file_name,
                                                                    out_file_name))
        # TODO FIXME - TONIGHT NOW - Fit the following method!
        # FIXME Temporarily disabled: self._core.save_peak_fit_result(self._curr_data_key, self._curr_file_name, out_file_name)

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
