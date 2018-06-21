try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_texturecalculationwindow
from pyrs.utilities import checkdatatypes
import os
import gui_helper
import numpy


class TextureAnalysisWindow(QMainWindow):
    """
    GUI window for texture analysis
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(TextureAnalysisWindow, self).__init__(parent)

        # class variables
        self._core = None

        # set up UI
        self.ui = ui.ui_texturecalculationwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self._init_widgets()

        # set up handling
        self.ui.pushButton_plotPeaks.clicked.connect(self.do_plot_diff_data)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_fit_peaks)
        self.ui.pushButton_calPoleFigure.clicked.connect(self.do_cal_pole_figure)

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionOpen_HDF5.triggered.connect(self.do_load_scans_hdf)
        self.ui.actionSave_as.triggered.connect(self.do_save_as)

        self.ui.actionSave_Diffraction_Data_For_Mantid.triggered.connect(self.do_save_pole_figure)

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)

        # mutexes
        self._sample_log_names_mutex = False

        return

    def _init_widgets(self):
        """
        initialize widgets
        :return:
        """
        # plots/graphics view
        self.ui.graphicsView_fitResult.set_subplots(1, 1)
        self.ui.graphicsView_fitSetup.set_subplots(1, 1)

        # table
        self.ui.tableView_poleFigureParams.setup()

        return

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self._core is None:
            raise RuntimeError('Not set up yet!')

    def _get_scan_log_indexes(self):
        """ from the line editor
        :return:
        """
        int_string_list = str(self.ui.lineEdit_scanNumbers.text()).strip()

        if len(int_string_list) == 0:
            scan_log_index = None
        else:
            scan_log_index = gui_helper.parse_integers(int_string_list)

        return scan_log_index

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

        # TODO - NEED TO FIND OUT HOW TO DEFINE hdf FROM IPTS and EXP

        return '/HFIR/HB2B/'

    def do_cal_pole_figure(self):
        # TODO/FIXME Det ID must be passed in!
        det_id = 1
        self._core.calculate_pole_figure(data_key_pair=(self._data_key, det_id))

        # get result out and show in table
        num_rows = self.ui.tableView_poleFigureParams.rowCount()
        for row_number in range(num_rows):
            det_id, log_index = self.ui.tableView_poleFigureParams.get_detector_log_index(row_number)
            alpha, beta = self._core.get_pole_figure_value((self._data_key, det_id), log_index)
            self.ui.tableView_poleFigureParams.set_pole_figure_projection(row_number, alpha, beta)

        return

    def do_load_scans_hdf(self):
        """
        load scan's reduced files
        :return: a list of tuple (detector ID, file name)
        """
        # check
        self._check_core()

        # browse file
        default_dir = self._get_default_hdf()
        if default_dir is None:
            default_dir = self._core.working_dir

        # FIXME : multiple file types seems not be supported on some MacOSX
        file_filter = 'HDF(*.hdf5);;All Files(*.*)'
        open_value = QFileDialog.getOpenFileNames(self, 'HB2B Raw HDF File', default_dir, file_filter)

        if isinstance(open_value, tuple):
            # PyQt5
            hdf_name_list = open_value[0]
        else:
            hdf_name_list = open_value

        if len(hdf_name_list) == 0:
            # use cancel
            return

        # convert the files
        new_file_list = list()
        for ifile, file_name in enumerate(hdf_name_list):
            det_id = int(file_name.split('[')[1].split(']')[0])
            new_file_list.append((det_id, file_name))
        # END-FOR

        self.load_h5_scans(new_file_list)

        return

    def load_h5_scans(self, rs_file_set):
        """
        load HDF5 for the reduced scans
        :param rs_file_set:
        :return:
        """
        # load file
        data_key, message = self._core.load_rs_raw_set(rs_file_set)
        self._data_key = data_key

        # edit information
        message = str(message)
        if len(message) > 80:
            message = message[:80]
        self.ui.label_loadedFileInfo.setText(message)

        # get the range of log indexes from detector 1 in order to set up the UI
        log_range = self._core.data_center.get_scan_range(data_key, 1)
        self.ui.label_logIndexMin.setText(str(log_range[0]))
        self.ui.label_logIndexMax.setText(str(log_range[-1]))

        # get the sample logs
        sample_log_names = self._core.data_center.get_sample_logs_list((data_key, 1), can_plot=True)

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
        if self.ui.tableView_poleFigureParams.rowCount() > 0:
            self.ui.tableView_poleFigureParams.remove_all_rows()
        self.ui.tableView_poleFigureParams.init_exp({1: self._core.data_center.get_scan_range(data_key, 1)})

        log_names = [('2theta', '2theta'),
                     ('omega', 'omega'),
                     ('chi', 'chi'),
                     ('phi', 'phi')]
        scan_log_dict = self._core.data_center.get_scan_index_logs_values((data_key, 1), log_names)
        for i_row in range(self.ui.tableView_poleFigureParams.rowCount()):
            det_id, scan_log_index = self.ui.tableView_poleFigureParams.get_detector_log_index(i_row)
            pole_figure_pos_dict = scan_log_dict[scan_log_index]
            self.ui.tableView_poleFigureParams.set_pole_figure_motors_position(i_row, pole_figure_pos_dict)

        # plot the first index
        self.ui.lineEdit_scanNumbers.setText('0')
        self.do_plot_diff_data()

        # plot the contour
        # FIXME/TODO/ASAP3 self.ui.graphicsView_contourView.plot_contour(self._core.data_center.get_data_2d(data_key))

        return

    def do_fit_peaks(self):
        """
        respond to the event triggered to fit all the peaks
        :return:
        """
        # get the data
        scan_log_index = None
        data_key = self._core.current_data_reference_id

        peak_function = str(self.ui.comboBox_peakType.currentText())
        bkgd_function = str(self.ui.comboBox_backgroundType.currentText())

        # get fit range
        fit_range = self.ui.graphicsView_fitSetup.get_x_limit()
        print ('[DB...BAT] Fit range: {0}'.format(fit_range))

        # call the core's method to fit peaks
        det_id = 1
        self._core.fit_peaks((data_key, det_id), scan_log_index, peak_function, bkgd_function, fit_range)

        # report fit result
        function_params = self._core.get_fit_parameters((data_key, det_id))
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
        data_key = data_key, det_id
        center_vec = self._core.get_peak_fit_param_value(data_key, 'centre')
        height_vec = self._core.get_peak_fit_param_value(data_key, 'height')
        fwhm_vec = self._core.get_peak_fit_param_value(data_key, 'width')
        chi2_vec = self._core.get_peak_fit_param_value(data_key, 'chi2')
        intensity_vec = self._core.get_peak_fit_param_value(data_key, 'intensity')
        com_vec = self._core.get_peak_center_of_mass(data_key)

        for row_index in range(len(center_vec)):
            det_id_i, log_index_i = self.ui.tableView_poleFigureParams.get_detector_log_index(row_index)
            # TODO: match the detector ID to current one!
            intensity_i = intensity_vec[log_index_i]
            self.ui.tableView_poleFigureParams.set_intensity(row_index, intensity_i)
        # END-FOR

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
        scan_log_index_list = gui_helper.parse_integers(str(self.ui.lineEdit_scanNumbers.text()))

        if len(scan_log_index_list) == 0:
            gui_helper.pop_message(self, 'There is not scan-log index input', 'error')
            return

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self.ui.graphicsView_fitSetup.reset_viewer()

        # get data and plot
        err_msg = ''
        detid = 1
        for scan_log_index in scan_log_index_list:
            try:
                diff_data_set = self._core.get_diff_data(data_key=(self._data_key, detid), scan_log_index=scan_log_index)
                self.ui.graphicsView_fitSetup.plot_diff_data(diff_data_set, 'Scan {0}'.format(scan_log_index))

                # more than 1 scan required to plot... no need to plot model and difference
                if len(scan_log_index_list) > 1:
                    continue

                # existing model
                if self._data_key is not None:
                    model_data_set = self._core.get_modeled_data(data_key=(self._data_key, detid),
                                                                 scan_log_index=scan_log_index_list[0])
                else:
                    model_data_set = None

                if model_data_set is not None:
                    self.ui.graphicsView_fitSetup.plot_model(model_data_set)
                    self.ui.graphicsView_fitSetup.plot_fit_diff(diff_data_set, model_data_set)
            except NotImplementedError as run_err:
                err_msg += '{0}\n'.format(run_err)
        # END-FOR

        if len(err_msg) > 0:
            raise RuntimeError(err_msg)
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
        self.ui.graphicsView_fitResult.clear_all_lines(include_right=False)

        # get the sample log/meta data name
        x_axis_name = str(self.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.ui.comboBox_yaxisNames.currentText())

        vec_x = self.get_meta_sample_data(x_axis_name)
        vec_y = self.get_meta_sample_data(y_axis_name)

        self.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)

        return

    def do_save_as(self):
        """
        save current pole figure to a text file
        :return:
        """
        file_info = QFileDialog.getSaveFileName(self, directory=self._core.working_dir,
                                                caption='Save Pole Figure To ASCII File')
        if isinstance(file_info, tuple):
            file_name = file_info[0]
        else:
            file_name = file_info
        file_name = str(file_name)

        if len(file_name) == 0:
            return

        raise NotImplementedError("I don't know what to do with save as ...")

    def do_save_pole_figure(self):
        """

        :return:
        """
        file_info = QFileDialog.getSaveFileName(self, directory=self._core.working_dir,
                                                caption='Save Pole Figure To ASCII File')
        if isinstance(file_info, tuple):
            file_name = file_info[0]
        else:
            file_name = file_info
        file_name = str(file_name)

        if len(file_name) == 0:
            return

        self._core.save_pole_figure(self._data_key, detector=1, file_name=file_name)

        return

    def do_save_workspace(self):
        """

        :return:
        """
        nxs_file_name = str(QFileDialog.getSaveFileName(self, 'Mantid Processed NeXus File Name',
                                                        self._core.working_dir))
        if len(nxs_file_name) == 0:
            return

        self._core.save_nexus((self._data_key, 1), nxs_file_name)

        return

    def do_quit(self):
        """
        close the window and quit
        :return:
        """
        self.close()

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
        # TODO
        self._core.save_nexus(data_key, file_name)

    def setup_window(self, pyrs_core):
        """

        :param pyrs_core:
        :return:
        """
        # check
        # blabla TODO

        self._core = pyrs_core

        # combo box
        self.ui.comboBox_peakType.clear()
        for peak_type in self._core.supported_peak_types:
            self.ui.comboBox_peakType.addItem(peak_type)

        return
