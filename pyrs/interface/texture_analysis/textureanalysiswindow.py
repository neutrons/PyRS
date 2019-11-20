from qtpy.QtWidgets import QMainWindow, QFileDialog
from pyrs.utilities import load_ui

from pyrs.interface.ui import qt_util
from pyrs.interface.ui.rstables import PoleFigureTable
from pyrs.interface.ui.diffdataviews import PeakFitSetupView, GeneralDiffDataView, Diffraction2DPlot
import pyrs.core.pyrscore
import os
import pyrs.interface.gui_helper
import numpy
import platform


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
        # self.ui = ui.ui_texturecalculationwindow.Ui_MainWindow()
        # self.ui.setupUi(self)

        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'texturecalculationwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)

        # promote widgets
        self.ui.graphicsView_fitSetup = qt_util.promote_widget(self, self.ui.graphicsView_fitSetup_frame,
                                                               PeakFitSetupView)
        self.ui.graphicsView_fitResult = qt_util.promote_widget(self, self.ui.graphicsView_fitResult_frame,
                                                                GeneralDiffDataView)
        self.ui.graphicsView_contour = qt_util.promote_widget(self, self.ui.graphicsView_contour_frame,
                                                              Diffraction2DPlot)
        self.ui.tableView_poleFigureParams = qt_util.promote_widget(self, self.ui.tableView_poleFigureParams_frame,
                                                                    PoleFigureTable)

        # init widgets
        self._init_widgets()

        # set up handling
        self.ui.pushButton_browseFile.clicked.connect(self.do_browse_load_file)
        self.ui.pushButton_plotPeaks.clicked.connect(self.do_plot_diff_data)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_fit_peaks)
        self.ui.pushButton_calPoleFigure.clicked.connect(self.do_cal_pole_figure)
        self.ui.pushButton_save_pf.clicked.connect(self.do_save_pole_figure)
        self.ui.pushButton_scanNumberForward.clicked.connect(self.do_forward_scan_log_index)
        self.ui.pushButton_scanNumberBackward.clicked.connect(self.do_rewind_scan_log_index)

        self.ui.pushButton_plotLogs.clicked.connect(self.do_plot_meta_data)
        self.ui.pushButton_plot_pf.clicked.connect(self.do_plot_pole_figure)
        self.ui.pushButton_clearPF.clicked.connect(self.do_clear_pole_figure_plot)

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.actionOpen_HDF5.triggered.connect(self.do_browse_load_file)
        self.ui.actionSave_as.triggered.connect(self.do_save_as)

        self.ui.actionSave_Diffraction_Data_For_Mantid.triggered.connect(self.do_save_workspace)

        self.ui.comboBox_xaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)
        self.ui.comboBox_yaxisNames.currentIndexChanged.connect(self.do_plot_meta_data)

        self.ui.actionExport_Table.triggered.connect(self.do_export_pole_figure_table)

        # mutexes
        self._sample_log_names_mutex = False

        # current data information
        self._data_key = None

        return

    def _init_widgets(self):
        """
        initialize widgets
        :return:
        """
        # plots/graphics view
        self.ui.graphicsView_fitResult.set_subplots(1, 1)

        # table
        self.ui.tableView_poleFigureParams.setup()

        # check boxes
        self.ui.checkBox_autoLoad.setChecked(True)
        self.ui.checkBox_autoFit.setChecked(True)

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
            scan_log_index = pyrs.interface.gui_helper.parse_integers(int_string_list)

        return scan_log_index

    def _get_default_hdf(self):
        """
        use IPTS and Exp to determine
        :return:
        """
        try:
            ipts_number = pyrs.interface.gui_helper.parse_integer(self.ui.lineEdit_iptsNumber)
            run_number = pyrs.interface.gui_helper.parse_integer(self.ui.lineEdit_expNumber)
        except RuntimeError as run_err:
            pyrs.interface.gui_helper.pop_message(self, 'Unable to parse IPTS or Exp due to {0}'.format(run_err))
            return None

        # TODO - NEED TO FIND OUT HOW TO DEFINE hdf FROM IPTS and EXP

        return '/HFIR/HB2B/IPTS-{}/shared/reduced/HB2B_{}.hdf'.format(ipts_number, run_number)

    def do_cal_pole_figure(self):
        """
        calculate pole figure
        :return:
        """
        det_id_list = None
        try:
            self._core.calculate_pole_figure(data_key=self._data_key, detector_id_list=det_id_list)
        except RuntimeError as run_err:
            pyrs.interface.gui_helper.pop_message(self, message='Failed to calculate pole figure',
                                                  detailed_message='{}'.format(run_err),
                                                  message_type='error')
            return

        # get result out and show in table
        num_rows = self.ui.tableView_poleFigureParams.rowCount()
        for row_number in range(num_rows):
            det_id, log_index = self.ui.tableView_poleFigureParams.get_detector_log_index(row_number)
            alpha, beta = self._core.get_pole_figure_value(self._data_key, det_id, log_index)
            # print ('[DB...BAT] row {0}:  alpha = {1}, beta = {2}'.format(row_number, alpha, beta))
            self.ui.tableView_poleFigureParams.set_pole_figure_projection(row_number, alpha, beta)

        # plot pole figure
        self.do_plot_pole_figure()

        return

    def do_clear_pole_figure_plot(self):
        """
        clear pole figure plot
        :return:
        """
        self.ui.graphicsView_contour.clear_image()

    def do_export_pole_figure_table(self):
        """
        export pole figure table
        :return:
        """
        # get inputs
        default_dir = self._core.working_dir
        if platform.system() == 'Darwin':
            file_filter = ''
        else:
            file_filter = 'CSV Files (*.csv);;All Files (*.*)'
        table_file_set = QFileDialog.getSaveFileName(self, caption='Select file name for pole figure calculation data',
                                                     directory=default_dir, filter=file_filter)
        if isinstance(table_file_set, tuple):
            table_file_name = str(table_file_set[0])
        else:
            table_file_name = str(table_file_set)

        self.ui.tableView_poleFigureParams.export_table_csv(table_file_name)

        return

    def do_browse_load_file(self):
        """
        load scan's reduced files and optionally load it!
        :return: a list of tuple (detector ID, file name)
        """
        # check
        self._check_core()

        # TESTME - 20180925 - Make it work!

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
        # TODO FIXME - 20180930 - count number of files loaded successfully and unsuccessfullly and decided
        # TODO                    fail or go on!
        error_msg = ''
        for ifile, file_name in enumerate(hdf_name_list):
            try:
                det_id = int(file_name.split('[')[1].split(']')[0])
            except IndexError as err:
                error_msg += 'Unable to retrieve detector ID from file {} due to {}\n' \
                             ''.format(file_name, err)
            else:
                new_file_list.append((det_id, str(file_name)))
        # END-FOR

        # auto load
        if self.ui.checkBox_autoLoad.isChecked():
            self.load_h5_scans(new_file_list)

        # Error message
        if error_msg != '':
            pyrs.interface.gui_helper.pop_message(self, 'Loading error', error_msg, 'error')

        return

    def load_h5_scans(self, rs_file_set):
        """
        load HDF5 for the reduced scans
        :param rs_file_set:
        :return:
        """
        # load file: the data key usually is based on the first file's name
        data_key, message = self._core.load_rs_raw_set(rs_file_set)
        self._data_key = data_key

        # edit information
        message = str(message)
        if len(message) > 80:
            message = message[:80]
        self.ui.label_loadedFileInfo.setText(message)

        # About table
        det_id_list = self._core.get_detector_ids(data_key)
        if self.ui.tableView_poleFigureParams.rowCount() > 0:
            self.ui.tableView_poleFigureParams.remove_all_rows()
        table_init_dict = dict()
        for det_id in sorted(det_id_list):
            table_init_dict[det_id] = self._core.data_center.get_scan_range(data_key, det_id)
            # self.ui.tableView_poleFigureParams.init_exp({1: self._core.data_center.get_scan_range(data_key, 1)})
        self.ui.tableView_poleFigureParams.init_exp(table_init_dict)

        # get the range of log indexes from detector 1 in order to set up the UI
        log_range = self._core.data_center.get_scan_range(data_key, det_id_list[0])
        self.ui.label_logIndexMin.setText(str(log_range[0]))
        self.ui.label_logIndexMax.setText(str(log_range[-1]))

        # Fill the combobox for detector IDs
        self.ui.comboBox_detectorIDsPlotPeak.clear()
        self.ui.comboBox_detectorID.clear()
        for det_id in sorted(det_id_list):
            self.ui.comboBox_detectorIDsPlotPeak.addItem(str(det_id))
            self.ui.comboBox_detectorID.addItem(str(det_id))

        # get the sample logs
        sample_log_names = self._core.data_center.get_sample_logs_names((data_key, det_id_list[0]),
                                                                        can_plot=True)

        self._sample_log_names_mutex = True
        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()
        self.ui.comboBox_xaxisNames.addItem('Log Index')
        for sample_log in sample_log_names:
            self.ui.comboBox_xaxisNames.addItem(sample_log)
            self.ui.comboBox_yaxisNames.addItem(sample_log)
        self._sample_log_names_mutex = False

        # set the pole figure related sample log values
        log_names = [('2theta', '2theta'),
                     ('omega', 'omega'),
                     ('chi', 'chi'),
                     ('phi', 'phi')]
        pole_figure_logs_dict = dict()
        for det_id in det_id_list:
            scan_log_dict = self._core.data_center.get_scan_index_logs_values((data_key, det_id), log_names)
            pole_figure_logs_dict[det_id] = scan_log_dict

        # set the pole figure motor positions to each row
        for i_row in range(self.ui.tableView_poleFigureParams.rowCount()):
            det_id, scan_log_index = self.ui.tableView_poleFigureParams.get_detector_log_index(i_row)
            pole_figure_pos_dict = pole_figure_logs_dict[det_id][scan_log_index]
            self.ui.tableView_poleFigureParams.set_pole_figure_motors_position(i_row, pole_figure_pos_dict)

        # plot the first index
        self.ui.lineEdit_scanNumbers.setText('0')
        self.do_plot_diff_data()

        # auto fit
        if self.ui.checkBox_autoFit.isChecked():
            # auto fit: no need to plot anymore
            self.do_fit_peaks()

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
        print('[DB...BAT] Fit range: {0}'.format(fit_range))

        # call the core's method to fit peaks
        det_id_list = self._core.get_detector_ids(data_key)
        print('[INFO] Detector ID list: {0}'.format(det_id_list))
        for det_id in det_id_list:
            self._core.fit_peaks((data_key, det_id), scan_log_index,
                                 peak_function, bkgd_function, fit_range)
        # END-FOR

        # report fit result... ...
        # add function parameters and detector IDs to UI
        function_params = self._core.get_peak_fit_parameter_names((data_key, det_id_list[0]))
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
        # END-BLOCK

        # fill up the table
        chi2_dict = dict()
        intensity_dict = dict()
        for det_id in det_id_list:
            data_key_pair = data_key, det_id
            chi2_vec = self._core.get_peak_fit_param_value(data_key_pair, 'chi2', max_cost=None)[1]
            intensity_vec = self._core.get_peak_fit_param_value(data_key_pair, 'intensity', max_cost=None)[1]
            chi2_dict[det_id] = chi2_vec
            intensity_dict[det_id] = intensity_vec
        # END-FOR

        for row_index in range(self.ui.tableView_poleFigureParams.rowCount()):
            det_id, scan_log_index = self.ui.tableView_poleFigureParams.get_detector_log_index(row_index)
            try:
                intensity_i = intensity_dict[det_id][scan_log_index]
                chi2_i = chi2_dict[det_id][scan_log_index]
                self.ui.tableView_poleFigureParams.set_intensity(row_index, intensity_i, chi2_i)
            except IndexError as index_error:
                print(intensity_dict[det_id])
                print(chi2_dict[det_id])
                raise RuntimeError('Unable to get intensity/chi2 of detector {} scan log index {} due to {}'
                                   ''.format(det_id, scan_log_index, index_error))

        # END-FOR

        # plot the model and difference
        self.do_plot_diff_data()

        return

    def do_plot_diff_data(self):
        """
        plot diffraction data
        :return:
        """
        # gather the information
        det_id = pyrs.interface.gui_helper.parse_integer(str(self.ui.comboBox_detectorIDsPlotPeak.currentText()))
        scan_log_index_list = pyrs.interface.gui_helper.parse_integers(str(self.ui.lineEdit_scanNumbers.text()))
        det_id_list = [det_id] * len(scan_log_index_list)
        # else:
        #     if len(det_id_list) != len(scan_log_index_list):
        #         gui_helper.pop_message('Number of detectors and scans do not match!', 'error')

        if len(scan_log_index_list) == 0:
            pyrs.interface.gui_helper.pop_message(self, 'There is not scan-log index input', 'error')
            return

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self.ui.graphicsView_fitSetup.reset_viewer()

        # get data and plot
        err_msg = ''
        for index in range(len(scan_log_index_list)):
            det_id = det_id_list[index]
            scan_log_index = scan_log_index_list[index]

            try:
                # get diffraction data
                diff_data_set = self._core.get_diffraction_data(data_key=(self._data_key, det_id),
                                                                scan_log_index=scan_log_index)
                self.ui.graphicsView_fitSetup.plot_diff_data(diff_data_set,
                                                             'Detector {0} Scan {1}'
                                                             ''.format(det_id, scan_log_index))

                # more than 1 scan required to plot... no need to plot model and difference
                if len(scan_log_index_list) > 1:
                    continue

                # existing model
                if self._data_key is not None:
                    model_data_set = self._core.get_modeled_data(session_name=(self._data_key, det_id),
                                                                 sub_run=scan_log_index_list[0])
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
            pyrs.interface.gui_helper.pop_message(self, err_msg, message_type='error')

        return

    def do_plot_meta_data(self):
        """
        plot the meta/fit result data on the right side GUI
        :return:
        """
        if self._sample_log_names_mutex:
            return

        # get the detector ID
        det_id = int(self.ui.comboBox_detectorID.currentText())
        # get the sample log/meta data name
        x_axis_name = str(self.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.ui.comboBox_yaxisNames.currentText())

        vec_x = self.get_meta_sample_data(det_id, x_axis_name, max_cost=70.)
        vec_y = self.get_meta_sample_data(det_id, y_axis_name, max_cost=70.)

        # clear whatever on the graph if the previous is not to be kept
        if not self.ui.checkBox_keepPrevPlotRight.isChecked():
            self.ui.graphicsView_fitResult.reset_viewer()
        elif self.ui.graphicsView_fitResult.current_x_name != x_axis_name:
            self.ui.graphicsView_fitResult.reset_viewer()

        # plot
        if isinstance(vec_x, tuple):
            # TODO - 20180820 - It is tricky to have selected log indexed X
            print('[CRITICAL/ERROR] Not Implemented Yet! Contact Developer!')
        elif isinstance(vec_y, tuple):
            # log indexes:
            vec_x, vec_y = vec_y
            self.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)
        else:
            self.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)

        return

    def do_save_as(self):
        """
        save current pole figure to a text file
        :return:
        """
        if platform.system() == 'Darwin':
            file_filter = ''
        else:
            file_filter = ''
        file_info = QFileDialog.getSaveFileName(self, directory=self._core.working_dir,
                                                caption='Save Pole Figure To ASCII File',
                                                filter=file_filter)
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
        save pole figure in both ascii and mtex format
        :return:
        """
        # get output file name
        if platform.system() == 'Darwin':
            file_filter = ''
        else:
            file_filter = 'MTEX (*.jul);;ASCII (*.dat);;All Files (*.*)'

        file_name = pyrs.interface.gui_helper.browse_file(self, caption='Save Pole Figure To ASCII File',
                                                          default_dir=self._core.working_dir, file_filter=file_filter,
                                                          file_list=False, save_file=True)

        # return/quit if action is cancelled
        if len(file_name) == 0:
            return

        # export 2 file types
        dir_name = os.path.dirname(file_name)
        base_name = os.path.basename(file_name).split('.')[0]
        for file_type, posfix in [('ascii', 'dat'), ('mtex', 'jul')]:
            file_name_i = os.path.join(dir_name, '{0}.{1}'.format(base_name, posfix))
            self._core.save_pole_figure(self._data_key, detectors=None, file_name=file_name_i,
                                        file_type=file_type)
        # END-FOR

        return

    def do_save_workspace(self):
        """ save workspace to NeXus file readable to Mantid
        :return:
        """
        if platform.system() == 'Darwin':
            file_filter = ''
        else:
            file_filter = 'NeXus Files (*.nxs);;All Files (*.*)'

        nxs_file_name_set = QFileDialog.getSaveFileName(self, caption='Mantid Processed NeXus File Name',
                                                        directory=self._core.working_dir,
                                                        filter=file_filter)

        if isinstance(nxs_file_name_set, tuple):
            nxs_file_name = str(nxs_file_name_set[0])
            print('[DB...BAT] Filter: {0}'.format(nxs_file_name_set[1]))
        else:
            nxs_file_name = str(nxs_file_name_set)

        if len(nxs_file_name) == 0:
            return
        else:
            self._core.save_nexus((self._data_key, 1), nxs_file_name)

        return

    def do_plot_pole_figure(self):
        """
        plot pole figure in the 2D
        :return:
        """
        # get pole figure from core
        max_cost_str = str(self.ui.lineEdit_maxCost.text()).strip()
        if len(max_cost_str) == 0:
            # empty.. non given
            max_cost = 100.
            self.ui.lineEdit_maxCost.setText('{}'.format(max_cost))
        else:
            try:
                max_cost = float(max_cost_str)
            except ValueError:
                max_cost = 100.
                self.ui.lineEdit_maxCost.setText('{}'.format(max_cost))
                pyrs.interface.gui_helper.pop_message(self, '{} is not a recognized float'.format(max_cost_str),
                                                      message_type='error')
                return
        # END-IF-ELSE

        vec_alpha, vec_beta, vec_intensity = self._core.get_pole_figure_values(data_key=self._data_key,
                                                                               detector_id_list=None,
                                                                               max_cost=max_cost)

        self.ui.graphicsView_contour.plot_pole_figure(vec_alpha, vec_beta, vec_intensity)

        return

    def do_quit(self):
        """
        close the window and quit
        :return:
        """
        self.close()

        return

    def do_forward_scan_log_index(self):
        """
        move scan log index (to plot) forward by 1
        :return:
        """
        try:
            current_log_index = int(self.ui.lineEdit_scanNumbers.text())
            current_log_index += 1
            # check whether the log index value exceeds the limit
            max_log_index = int(self.ui.label_logIndexMax.text())
            if current_log_index > max_log_index:
                current_log_index = 0
            # set value
            self.ui.lineEdit_scanNumbers.setText(str(current_log_index))
            # re-plot
            self.do_plot_diff_data()
        except ValueError:
            print('[WARNING] Current value {0} cannot be forwarded'.format(self.ui.lineEdit_scanNumbers.text()))

        return

    def do_rewind_scan_log_index(self):
        """
        move scan log index (to plot) backward by 1
        :return:
        """
        try:
            current_log_index = int(self.ui.lineEdit_scanNumbers.text())
            current_log_index -= 1
            # check whether the log index value exceeds the limit
            if current_log_index < 0:
                max_log_index = int(self.ui.label_logIndexMax.text())
                current_log_index = max_log_index
            # set value
            self.ui.lineEdit_scanNumbers.setText(str(current_log_index))
            # re-plot
            self.do_plot_diff_data()
        except ValueError:
            print('[WARNING] Current value {0} cannot be moved backward'.format(self.ui.lineEdit_scanNumbers.text()))

        return

    def get_meta_sample_data(self, det_id, name, max_cost=None):
        """
        get meta data to plot.
        the meta data can contain sample log data and fitted peak parameters
        :param det_id:
        :param name:
        :param max_cost
        :return:
        """
        # get data key
        data_key = self._core.current_data_reference_id
        if data_key is None:
            pyrs.interface.gui_helper.pop_message(self, 'No data loaded', 'error')
            return

        if name == 'Log Index':
            value_vector = numpy.array(self._core.data_center.get_scan_range(data_key, sub_key=det_id))
        elif self._core.data_center.has_sample_log((data_key, det_id), name):
            value_vector = self._core.data_center.get_sample_log_values((data_key, det_id), name)
        elif name == 'Center of mass':
            value_vector = self._core.get_peak_center_of_mass((data_key, det_id))
        else:
            # this is for fitted data parameters
            value_vector = self._core.get_peak_fit_param_value((data_key, det_id), name, max_cost=max_cost)

        return value_vector

    def save_data_for_mantid(self, data_key, file_name):
        """
        save the loaded diffraction data to Mantid processed NeXus file format
        :param data_key:
        :param file_name:
        :return:
        """
        self._core.save_nexus(data_key, file_name)

        return

    def setup_window(self, pyrs_core):
        """ set up the texture analysis window
        :param pyrs_core:
        :return:
        """
        # check
        assert isinstance(pyrs_core, pyrs.core.pyrscore.PyRsCore), 'PyRS core {0} of type {1} must be a PyRsCore ' \
                                                                   'instance.'.format(pyrs_core, type(pyrs_core))

        self._core = pyrs_core

        # combo box
        self.ui.comboBox_peakType.clear()
        for peak_type in self._core.supported_peak_types:
            self.ui.comboBox_peakType.addItem(peak_type)

        return
