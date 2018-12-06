try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog, QVBoxLayout
import ui.ui_peakfitwindow
import pyrs.utilities.hb2b_utilities as hb2bhb2b
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

        # sub windows
        self._advanced_fit_dialog = None

        # set up UI
        self.ui = ui.ui_peakfitwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self._promote_widgets()

        self._init_widgets()
        # init some widgets
        self.ui.checkBox_autoLoad.setChecked(True)

        # set up handling
        self.ui.pushButton_loadHDF.clicked.connect(self.do_load_scans)
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

        # current/last loaded data
        self._curr_data_key = None
        self._curr_file_name = None

        # a copy of sample logs
        self._sample_log_names = list()  # a copy of sample logs' names that are added to combo-box

        # TODO - 20181124 - New GUI parameters (After FitPeaks)
        # checkBox_showFitError
        # checkBox_showFitValue
        # others
        # TODO - 20181124 - Make this table's column flexible!
        self.ui.tableView_fitSummary.setup()

        return

    def _promote_widgets(self):
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

        archive_data = hb2b.get_hb2b_raw_data(ipts_number, exp_number)

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
        """
        browse HDF file
        :return:
        """
        self._check_core()

        default_dir = self._get_default_hdf()
        if default_dir is None:
            default_dir = self._core.working_dir

        file_filter = 'HDF(*.hdf5);;All Files(*.*)'
        open_value = QFileDialog.getOpenFileName(self, 'HB2B Raw HDF File', default_dir, file_filter)
        print open_value

        if isinstance(open_value, tuple):
            # PyQt5
            hdf_name = str(open_value[0])
        else:
            hdf_name = str(open_value)

        if len(hdf_name) == 0:
            # use cancel
            return

        if os.path.exists(hdf_name):
            self.ui.lineEdit_expFileName.setText(hdf_name)
        else:
            # pass
            raise RuntimeError('File {0} does not exist.'.format(hdf_name))

        if self.ui.checkBox_autoLoad.isChecked():
            self.do_load_scans(from_browse=True)

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

    def do_load_scans(self, from_browse=True):
        """
        load scan's reduced files
        :param from_browse: if True, then file will be read from lineEdit_expFileName.
        :return:
        """
        self._check_core()

        # get file
        if from_browse:
            rs_file_name = str(self.ui.lineEdit_expFileName.text())
        else:
            rs_file_name = self.set_file_from_archive()

        # load file
        try:
            data_key, message = self._core.load_rs_raw(rs_file_name)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to load {}'.format(rs_file_name), detailed_message=str(run_err),
                                   message_type='error')
            return

        # edit information
        self.ui.label_loadedFileInfo.setText(message)

        # get the range of log indexes
        log_range = self._core.data_center.get_scan_range(data_key)
        self.ui.label_logIndexMin.setText(str(log_range[0]))
        self.ui.label_logIndexMax.setText(str(log_range[-1]))

        # get the sample logs
        sample_log_names = self._core.data_center.get_sample_logs_list(data_key, can_plot=True)

        self._sample_log_names_mutex = True
        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()
        self.ui.comboBox_xaxisNames.addItem('Log Index')

        # Maintain a copy of sample logs!
        self._sample_log_names = sample_log_names[:]
        for sample_log in sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log)
            self.ui.comboBox_yaxisNames.addItem(sample_log)
            self._sample_log_name_set.add(sample_log)
        self._sample_log_names_mutex = False

        # reset the plot
        self.ui.graphicsView_fitResult.reset_viewer()

        # Record data key and next
        self._curr_data_key = data_key
        self._curr_file_name = rs_file_name

        # About table
        if self.ui.tableView_fitSummary.rowCount() > 0:
            self.ui.tableView_fitSummary.remove_all_rows()
        self.ui.tableView_fitSummary.init_exp(self._core.data_center.get_scan_range(data_key))

        # plot first peak for default peak range
        self.ui.lineEdit_scanNumbers.setText('0')
        self.do_plot_diff_data(plot_model=False)

        # auto fit
        if self.ui.checkBox_autoFit.isChecked():
            # auto fit: no need to plot anymore
            self.do_fit_peaks()

        # plot the contour
        # FIXME/TODO/ASAP3 self.ui.graphicsView_contourView.plot_contour(self._core.data_center.get_data_2d(data_key))

        return

    def do_fit_peaks(self):
        """
        Fit ALL peaks
        :return:
        """
        # int_string_list = str(self.ui.lineEdit_scanNumbers.text()).strip()
        # if len(int_string_list) == 0:
        #     scan_log_index = None
        # else:
        #     scan_log_index = gui_helper.parse_integers(int_string_list)
        data_key = self._core.current_data_reference_id
        if data_key != self._curr_data_key:
            raise RuntimeError('Core current data key {} shall be same as UI current data key {}'
                               ''.format(data_key, self._curr_data_key))

        peak_function = str(self.ui.comboBox_peakType.currentText())
        bkgd_function = str(self.ui.comboBox_backgroundType.currentText())

        fit_range = self._ui_graphicsView_fitSetup.get_x_limit()
        print ('[INFO] Peak fit range: {0}'.format(fit_range))

        # It is better to fit all the peaks at the same time after testing
        scan_log_index = None
        self._core.fit_peaks(data_key, scan_log_index, peak_function, bkgd_function, fit_range)

        function_params = self._core.get_peak_fit_parameter_names(data_key)
        self._sample_log_names_mutex = True
        curr_x_index = self.ui.comboBox_xaxisNames.currentIndex()
        curr_y_index = self.ui.comboBox_yaxisNames.currentIndex()
        # add fitted parameters by resetting and build from the copy of fit parameters
        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()
        self.ui.comboBox_xaxisNames.addItem('Log Index')
        for sample_log_name in self._sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log_name)
            self.ui.comboBox_yaxisNames.addItem(sample_log_name)
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

        # fill up the table
        self._set_fit_result_table(peak_function, data_key)

        # plot the model and difference
        if scan_log_index is None:
            scan_log_index = 0
            # FIXME This case is not likely to occur
        # FIXME - TODO - self.do_plot_diff_data()

        return

    def _set_fit_result_table(self, peak_function, data_key):
        """

        :param peak_function:
        :param data_key:
        :return:
        """
        table_param_names = ['Center', 'Intensity', 'FWHM', 'Height', 'Chi2']
        if peak_function == 'PseudoVoigt':
            # TODO - 20181210 - shall extending 'mixing' as a special case
            pass
        # param_names.extend(['A0', 'A1'])

        self.ui.tableView_fitSummary.reset_table(table_param_names)

        # get value
        not_used_vec, center_vec = self._core.get_peak_fit_param_value(data_key, 'centre', max_cost=None)
        not_used_vec, height_vec = self._core.get_peak_fit_param_value(data_key, 'height', max_cost=None)
        not_used_vec, fwhm_vec = self._core.get_peak_fit_param_value(data_key, 'width', max_cost=None)
        not_used_vec, chi2_vec = self._core.get_peak_fit_param_value(data_key, 'chi2', max_cost=None)
        not_used_vec, intensity_vec = self._core.get_peak_fit_param_value(data_key, 'intensity', max_cost=None)
        com_vec = self._core.get_peak_center_of_mass(data_key)
        #
        for row_index in range(len(center_vec)):
            self.ui.tableView_fitSummary.set_fit_summary(row_index, scan_index, param_dict, peak_name, com_vec)

            # self.ui.tableView_fitSummary.set_peak_params(row_index,
            #                                              center_vec[row_index],
            #                                              height_vec[row_index],
            #                                              fwhm_vec[row_index],
            #                                              intensity_vec[row_index],
            #                                              chi2_vec[row_index],
            #                                              peak_function)
            # self.ui.tableView_fitSummary.set_peak_center_of_mass(row_index, com_vec[row_index])
        # END-FOR (rows)

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

        if x_axis_name in self._function_param_name_set and y_axis_name == 'Log Index':
            vec_y, vec_x = self.get_function_parameter_data(x_axis_name)
        elif y_axis_name in self._function_param_name_set and x_axis_name == 'Log Index':
            vec_x, vec_y = self.get_function_parameter_data(y_axis_name)
        elif x_axis_name in self._function_param_name_set or y_axis_name in self._function_param_name_set:
            gui_helper.pop_message(self, 'It has not considered how to plot 2 function parameters against '
                                         'each other', message_type='error')
            return
        else:
            vec_x = self.get_meta_sample_data(x_axis_name)
            vec_y = self.get_meta_sample_data(y_axis_name)

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
        data_key = self._core.current_data_reference_id
        if data_key is None:
            gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        vec_log_index, vec_param_value = self._core.get_peak_fit_param_value(data_key, param_name, max_cost=1000)

        return vec_log_index, vec_param_value

    def get_meta_sample_data(self, name):
        """
        get meta data to plot.
        the meta data can contain sample log data and fitted peak parameters
        :param name:
        :return:
        """
        # get data key
        data_key = self._core.current_data_reference_id
        if data_key is None:
            gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        if name == 'Log Index':
            value_vector = numpy.array(self._core.data_center.get_scan_range(data_key))
        elif self._core.data_center.has_sample_log(data_key, name):
            value_vector = self._core.data_center.get_sample_log_values(data_key, name)
        elif name == 'Center of mass':
            value_vector = self._core.get_peak_center_of_mass(data_key)
        else:
            value_vector = None

        return value_vector

    def plot_diff_data(self, scan_log_index, plot_model):
        """
        plot a set of diffraction data (one scan log index) and plot its fitted data
        :return:
        """
        # get experimental data and plot
        diff_data_set = self._core.get_diffraction_data(data_key=None, scan_log_index=scan_log_index)
        data_set_label = 'Scan {0}'.format(scan_log_index)

        if plot_model:
            model_data_set = self._core.get_modeled_data(data_key=None, scan_log_index=scan_log_index)
        else:
            model_data_set = None

        # plot
        if model_data_set is None:
            # data only (no model or not chosen to)
            self._ui_graphicsView_fitSetup.plot_data(data_set=diff_data_set, color=None,
                                                     line_label=data_set_label)
        else:
            # plot with model
            residual_y_vec = diff_data_set[1] - model_data_set[1]
            residual_data_set = [diff_data_set[0], residual_y_vec]
            self._ui_graphicsView_fitSetup.plot_data_model(data_set=diff_data_set, data_label=data_set_label,
                                                           model_set=model_data_set, model_label='',
                                                           residual_set=residual_data_set)
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
        self._core.save_peak_fit_result(self._curr_data_key, self._curr_file_name, out_file_name)

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
