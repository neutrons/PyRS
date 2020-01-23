import os
from qtpy.QtCore import Qt
from pyrs.interface.gui_helper import browse_file
from pyrs.interface.gui_helper import pop_message, parse_line_edit
from pyrs.interface.manual_reduction.pyrs_api import ReductionController


class EventHandler(object):
    """Class to handle the event sent from UI widget
    """
    def __init__(self, parent):
        """Init

        Parameters
        ----------
        parent : qtpy.QtWidgets.QMainWindow
            GUI maub Window
        """
        self.parent = parent
        self.ui = self.parent.ui

        # controller
        self._controller = ReductionController()

        return

    def _mask_state(self, state):
        if state != Qt.Unchecked:
            self.ui.lineEdit_maskFile.setText(self._controller.get_default_mask_dir() + 'HB2B_MASK_Latest.xml')
        self.ui.lineEdit_maskFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseMaskFile.setEnabled(state == Qt.Unchecked)

    def _calibration_state(self, state):
        if state != Qt.Unchecked:
            self.ui.lineEdit_calibrationFile.setText(self._controller.get_default_calibration_dir() +
                                                     'HB2B_Latest.json')
        self.ui.lineEdit_calibrationFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseCalibrationFile.setEnabled(state == Qt.Unchecked)

    def _output_state(self, state):
        if state != Qt.Unchecked:
            self._update_output_ipts(self.ui.spinBox_runNumber.value())
        self.ui.lineEdit_outputDirectory.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseOutputDirectory.setEnabled(state == Qt.Unchecked)

    def _update_output_ipts(self, run_number):
        project_dir = self._controller.get_default_output_dir(run_number)
        if project_dir is not None:
            self.ui.lineEdit_outputDirectory.setText(project_dir)

    def _parse_ipts_number(self):
        """Get IPTS number from

        Returns
        -------
        int or None
            IPTS number

        """
        run_number = parse_line_edit(self.ui.lineEdit_runNumbersList, int, False, default=None)
        if run_number is None:
            return None

        ipts_number = self._controller.get_ipts_from_run(run_number)

        return ipts_number

    def browse_load_nexus(self):
        """Allow users to browse for a nexus file to convert to project file

        Returns
        -------

        """
        # Get default directory
        ipts_number = self._parse_ipts_number()
        if ipts_number is None:
            default_dir = self._controller.working_dir
        else:
            default_dir = self._controller.get_nexus_dir(ipts_number)

        # If specified information is not sufficient, browse
        nexus_file = browse_file(self.parent,
                                 caption='Select a NeXus file',
                                 default_dir=default_dir,
                                 file_filter='NeXus (*.nxs.h5)',
                                 file_list=False, save_file=False)

        # Return if it is cancelled
        if nexus_file is None:
            return

        # Load Nexus
        self._controller.load_nexus_file(nexus_file)

        # sub runs
        sub_runs = self._controller.get_sub_runs()

        # set sub runs to (1) Table and (2) Combo box
        self._set_run_numbers(sub_runs)

    def browse_load_hidra(self):
        """Allow users to browse for a HiDRA project file

        Returns
        -------

        """
        # Get default directory
        ipts_number = self._parse_ipts_number()
        if ipts_number is None:
            default_dir = self._controller.working_dir
        else:
            default_dir = self._controller.get_hidra_project_dir(ipts_number)

        # Browse
        hidra_file = browse_file(self.parent,
                                 caption='Select a HIDRA project file',
                                 default_dir=default_dir,
                                 file_filter='HiDRA project (*.h5)',
                                 file_list=False, save_file=False)

        # Load Nexus
        self._controller.load_hidra_project(hidra_file)

        # sub runs
        sub_runs = self._controller.get_sub_runs()

        # set sub runs to (1) Table and (2) Combo box
        self._set_run_numbers(sub_runs)

    def browse_calibration_file(self):
        """

        Returns
        -------

        """
        calibration_file = browse_file(self, caption='Choose and set up the calibration file',
                                       default_dir=self._controller.get_default_calibration_dir(),
                                       file_filter='hdf5 (*hdf)', file_list=False, save_file=False)
        if calibration_file is None or calibration_file == '':
            # operation canceled
            return

        # set to the browser
        self.ui.lineEdit_calibratonFile.setText(calibration_file)

    def browse_idf(self):
        """

        Returns
        -------

        """
        idf_name = browse_file(self, 'Instrument definition file', os.getcwd(),
                               'Text (*.txt);;XML (*.xml)', False, False)
            if len(idf_name) == 0:
                return   # user cancels operation
            else:
                self.ui.lineEdit_idfName.setText(idf_name)
        # END-IF

    def browse_mask_file(self):
        mask_file_name = browse_file(
            self, 'Hidra Mask File', DEFAULT_MASK_DIRECTORY, 'hdf5 (*.h5);;xml (*.xml)', False, False)
        self.ui.lineEdit_maskFile.setText(mask_file_name)
        # self.load_hydra_file(project_file_name)
        return

    def browse_output_dir(self):
        output_dir = browse_dir(self, caption='Output directory for reduced data',
                                           default_dir=os.path.expanduser('~'))
        if output_dir != '':
            self.ui.lineEdit_outputDir.setText(output_dir)
            self._core.reduction_service.set_output_dir(output_dir)
            self._output_dir = output_dir

    def slice_nexus(self):
        """Slice NeXus file by arbitrary log sample parameter

        Returns
        -------

        """
        if self.ui.radioButton_chopByTime.isChecked():
            # set up slicers by time
            self.set_slicers_by_time()
        elif self.ui.radioButton_chopByLogValue.isChecked():
            # set up slicers by sample log value
            self.set_slicers_by_sample_log_value()
        else:
            # set from the table
            self.set_slicers_manually()
        # END-IF-ELSE

        try:
            data_key = self._core.reduction_service.chop_data()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Unable to slice data', detailed_message=str(run_err),
                                   message_type='error')
            return

        try:
            self._core.reduction_service.reduced_chopped_data(data_key)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Failed to reduce sliced data', detailed_message=str(run_err),
                                   message_type='error')
            return

        # fill the run numbers to plot selection
        self._setup_plot_selection(append_mode=False, item_list=self._core.reduction_service.get_chopped_names())

        # plot
        self._plot_data()

        return

    def plot_detector_counts(self):
        """

        Returns
        -------

        """
        sub_run = int(self.ui.comboBox_sub_runs.currentText())
        # FIXME - TODO - ValueError: invalid literal for int() with base 10: ''

        return

    def plot_powder_pattern(self):
        """

        Returns
        -------

        """
        sub_run = int(self.ui.comboBox_sub_runs.currentText())
        # FIXME - TODO - ValueError: invalid literal for int() with base 10: ''

        return

    def _set_sub_runs(self):
        """ set the sub runs to comboBox_sub_runs
        :return:
        """
        # sub_runs = self._core.reduction_manager.get_sub_runs(data_id)
        #
        # self.ui.comboBox_sub_runs.clear()
        # for sub_run in sorted(sub_runs):
        #     self.ui.comboBox_sub_runs.addItem('{:04}'.format(sub_run))
        #
        #
        # """
        # """
        # #
        sub_runs = self._core.reduction_service.get_sub_runs(self._project_data_id)
        sub_runs.sort()

        # set sub runs: lock and release
        self._mutexPlotRuns = True
        # clear and set
        self.ui.comboBox_sub_runs.clear()
        for sub_run in sub_runs:
            self.ui.comboBox_sub_runs.addItem('{:04}'.format(sub_run))
        self._mutexPlotRuns = False

        return

    def _setup_plot_runs(self, append_mode, run_number_list):
        """ set the runs (or sliced runs to plot)
        :param append_mode:
        :param run_number_list:
        :return:
        """
        checkdatatypes.check_list('Run numbers', run_number_list)

        # non-append mode
        self._plot_run_numbers_mutex = True
        if not append_mode:
            self.ui.comboBox_runs.clear()

        # add run numbers
        for run_number in run_number_list:
            self.ui.comboBox_runs.addItem('{}'.format(run_number))

        # open
        self._plot_run_numbers_mutex = False

        # if append-mode, then set to first run
        if append_mode:
            self.ui.comboBox_runs.setCurrentIndex(0)

        return

    def _setup_plot_selection(self, append_mode, item_list):
        """
        set up the combobox to select items to plot
        :param append_mode:
        :param item_list:
        :return:
        """
        checkdatatypes.check_bool_variable('Flag for appending the items to current combo-box or from start',
                                           append_mode)
        checkdatatypes.check_list('Combo-box item list', item_list)

        # turn on mutex lock
        self._plot_selection_mutex = True
        if append_mode is False:
            self.ui.comboBox_sampleLogNames.clear()
        for item in item_list:
            self.ui.comboBox_sampleLogNames.addItem(item)
        if append_mode is False:
            self.ui.comboBox_sampleLogNames.setCurrentIndex(0)
        self._plot_selection_mutex = False
        return

    def manual_reduce_run(self):
        """

        (simply) reduce a list of runs in same experiment in a batch

        Returns
        -------

        """
        # get (sub) run numbers
        sub_runs_str = str(self.ui.lineEdit_runNumbersList.text()).strip().lower()
        if sub_runs_str == 'all':
            sub_run_list = list()
        else:
            try:
                sub_run_list = gui_helper.parse_integers(sub_runs_str)
            except RuntimeError as run_err:
                gui_helper.pop_message(self, 'Failed to parse integer list',
                                       '{}'.format(run_err), 'error')
                return
        # Reduce data
        run_number = self.ui.spinBox_runNumber.text().strip()
        self._update_output_ipts(run_number)
        nexus_file = FileFinder.findRuns('HB2B'+run_number)[0]
        project_file = str(self.ui.lineEdit_outputDirectory.text().strip())
        mask_file = str(self.ui.lineEdit_maskFile.text().strip())
        calibration_file = str(self.ui.lineEdit_calibrationFile.text().strip())
        task = BlockingAsyncTaskWithCallback(reduce_hidra_workflow, args=(nexus_file, project_file,
                                                                          self.ui.progressBar),
                                             kwargs={'subruns': sub_run_list, 'mask': mask_file,
                                                     'calibration': calibration_file},
                                             blocking_cb=QApplication.processEvents)
        # TODO - catch RuntimeError! ...
        # FIXME - check output directory
        task.start()
        # <<<<

        # Update table
        # TODO - Need to fill the table!
        for sub_run in list():
            self.ui.rawDataTable.update_reduction_state(sub_run, True)

        return

    def save_project(self):
        self._event_handler.save_project()
        output_project_name = os.path.join(self._output_dir, os.path.basename(self._project_file_name))
        if output_project_name != self._project_file_name:
            import shutil
            shutil.copyfile(self._project_file_name, output_project_name)

        self._core.reduction_service.save_project(self._project_data_id, output_project_name)


    def plot_detector_counts(self, sub_run_number, mask_id):
        """
        Plot detector counts on the detector view
        :param sub_run_number:  sub run number (integer)
        :param mask_id: Mask ID (string) or None
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run_number, (0, None))

        # Get the detector counts
        detector_counts_array = self._core.reduction_service.get_detector_counts(self._project_data_id,
                                                                                 sub_run_number)

        # set information
        det_2theta = self._core.reduction_service.get_sample_log_value(self._project_data_id,
                                                                       HidraConstants.TWO_THETA,
                                                                       sub_run_number)
        info = 'sub-run: {}, 2theta = {}' \
               ''.format(sub_run_number, det_2theta)

        # If mask ID is not None
        if mask_id is not None:
            # Get mask in array and do a AND operation to detector counts (array)
            mask_array = self._core.reduction_service.get_mask_array(self._curr_project_name, mask_id)
            detector_counts_array *= mask_array
            info += ', mask ID = {}'.format(mask_id)

        # Set information
        self.ui.lineEdit_detViewInfo.setText(info)

        # Plot
        self.ui.graphicsView_detectorView.plot_detector_view(detector_counts_array, (sub_run_number, mask_id))

        return

    def plot_reduced_data(self, sub_run_number, mask_id):
        """
        Plot reduced data
        :param sub_run_number: sub run number (integer)
        :param mask_id: Mask ID (string) or None
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run_number, (0, None))

        try:
            two_theta_array, diff_array = self._core.reduction_service.get_reduced_diffraction_data(
                self._project_data_id, sub_run_number, mask_id)
            if two_theta_array is None:
                raise NotImplementedError('2theta array is not supposed to be None.')
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to retrieve reduced data',
                                   'For sub run {} due to {}'.format(sub_run_number, run_err),
                                   'error')
            return

        # set information
        det_2theta = self._core.reduction_service.get_sample_log_value(self._project_data_id,
                                                                       HidraConstants.TWO_THETA,
                                                                       sub_run_number)
        info = 'sub-run: {}, 2theta = {}' \
               ''.format(sub_run_number, det_2theta)

        # plot diffraction data
        self.ui.graphicsView_1DPlot.plot_diffraction(two_theta_array, diff_array, '2theta',
                                                     'intensity', line_label=info, keep_prev=False)

        return






