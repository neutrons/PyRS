try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_peakfitwindow
import pyrs.utilities.hb2b_utilities as hb2b
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

        # set up UI
        self.ui = ui.ui_peakfitwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self.ui.graphicsView_fitSetup.set_subplots(1, 1)

        # set up handling
        self.ui.pushButton_loadHDF.clicked.connect(self.do_load_scans)
        self.ui.pushButton_browseHDF.clicked.connect(self.do_browse_hdf)
        self.ui.pushButton_plotPeaks.clicked.connect(self.do_plot_diff_data)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_fit_peaks)

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionSave_As.triggered.connect(self.do_save_as)
        self.ui.actionSave_Fit_Result.triggered.connect(self.do_save_fit_result)

        # TODO - Implement : pushButton_plotLogs, comboBox_detectorID

        # others
        self.ui.tableView_fitSummary.setup()

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)

        # mutexes
        self._sample_log_names_mutex = False

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

        return

    def do_load_scans(self):
        """
        load scan's reduced files
        :return:
        """
        self._check_core()

        # get file
        rs_file_name = str(self.ui.lineEdit_expFileName.text())

        # load file
        data_key, message = self._core.load_rs_raw(rs_file_name)

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
        for sample_log in sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log)
            self.ui.comboBox_yaxisNames.addItem(sample_log)
        self._sample_log_names_mutex = False

        # TODO FIXME: how to record data key?

        # About table
        if self.ui.tableView_fitSummary.rowCount() > 0:
            self.ui.tableView_fitSummary.remove_all_rows()
        self.ui.tableView_fitSummary.init_exp(self._core.data_center.get_scan_range(data_key))

        # plot the first index
        self.ui.lineEdit_scanNUmbers.setText('0')
        self.do_plot_diff_data()

        # plot the contour
        # FIXME/TODO/ASAP3 self.ui.graphicsView_contourView.plot_contour(self._core.data_center.get_data_2d(data_key))

        return

    def do_fit_peaks(self):
        """
        Fit all peaks
        :return:
        """
        int_string_list = str(self.ui.lineEdit_scanNUmbers.text()).strip()
        if len(int_string_list) == 0:
            scan_log_index = None
        else:
            scan_log_index = gui_helper.parse_integers(int_string_list)
        data_key = self._core.current_data_reference_id

        peak_function = str(self.ui.comboBox_peakType.currentText())
        bkgd_function = str(self.ui.comboBox_backgroundType.currentText())

        # TODO .. TEST
        fit_range = self.ui.graphicsView_fitSetup.get_x_limit()
        print ('Fit range: {0}'.format(fit_range))

        # FIXME It is better to fit all the peaks at the same time!
        scan_log_index = None
        self._core.fit_peaks(data_key, scan_log_index, peak_function, bkgd_function, fit_range)

        function_params = self._core.get_fit_parameters(data_key)
        self._sample_log_names_mutex = True
        # TODO FIXME : add to X axis too
        curr_index = self.ui.comboBox_yaxisNames.currentIndex()
        # add fitted parameters
        for param_name in function_params:
            self.ui.comboBox_yaxisNames.addItem(param_name)
        # add observed parameters
        self.ui.comboBox_yaxisNames.addItem('Center of mass')
        # keep current selected item unchanged
        self.ui.comboBox_yaxisNames.setCurrentIndex(curr_index)
        self._sample_log_names_mutex = False

        # fill up the table
        center_vec = self._core.get_peak_fit_param_value(data_key, 'centre')
        height_vec = self._core.get_peak_fit_param_value(data_key, 'height')
        fwhm_vec = self._core.get_peak_fit_param_value(data_key, 'width')
        chi2_vec = self._core.get_peak_fit_param_value(data_key, 'chi2')
        intensity_vec = self._core.get_peak_fit_param_value(data_key, 'intensity')
        com_vec = self._core.get_peak_center_of_mass(data_key)

        for row_index in range(len(center_vec)):
            self.ui.tableView_fitSummary.set_peak_params(row_index,
                                                         center_vec[row_index],
                                                         height_vec[row_index],
                                                         fwhm_vec[row_index],
                                                         intensity_vec[row_index],
                                                         chi2_vec[row_index],
                                                         peak_function)
            self.ui.tableView_fitSummary.set_peak_center_of_mass(row_index, com_vec[row_index])

        # plot the model and difference
        if scan_log_index is None:
            scan_log_index = 0
            # FIXME This case is not likely to occur
        self.do_plot_diff_data()

        return

    def do_plot_diff_data(self):
        """
        plot diffraction data
        :return:
        """
        # gather the information
        scan_log_index_list = gui_helper.parse_integers(str(self.ui.lineEdit_scanNUmbers.text()))
        if len(scan_log_index_list) == 0:
            gui_helper.pop_message(self, 'There is not scan-log index input', 'error')

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self.ui.graphicsView_fitSetup.reset_viewer()

        # get data and plot
        err_msg = ''
        for scan_log_index in scan_log_index_list:
            try:
                diff_data_set = self._core.get_diff_data(data_key=None, scan_log_index=scan_log_index)
                self.ui.graphicsView_fitSetup.plot_diff_data(diff_data_set, 'Scan {0}'.format(scan_log_index))

                # more than 1 scan required to plot... no need to plot model and difference
                if len(scan_log_index_list) > 1:
                    continue

                model_data_set = self._core.get_modeled_data(data_key=None, scan_log_index=scan_log_index_list[0])
                if model_data_set is None:
                    continue
                # existing model
                self.ui.graphicsView_fitSetup.plot_model(model_data_set)
                self.ui.graphicsView_fitSetup.plot_fit_diff(diff_data_set, model_data_set)
            except RuntimeError as run_err:
                err_msg += '{0}\n'.format(run_err)
        # END-FOR

        if len(err_msg) > 0:
            gui_helper.pop_message(self, err_msg, message_type='error')

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

        vec_x = self.get_meta_sample_data(x_axis_name)
        vec_y = self.get_meta_sample_data(y_axis_name)

        self.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)

        return

    def do_save_as(self):
        """

        :return:
        """
        # TODO - 20180801 - Save as .. is used to save the fit result to original file
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
            # this is for fitted data parameters
            value_vector = self._core.get_peak_fit_param_value(data_key, name)

        return value_vector

    def save_data_for_mantid(self, data_key, file_name):
        """
        save data to Mantid-compatible NeXus
        :param data_key:
        :param file_name:
        :return:
        """
        self._core.save_nexus(data_key, file_name)

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
