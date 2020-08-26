import os
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from pyrs.interface.gui_helper import pop_message, parse_combo_box
from mantidqt.utils.asynchronous import BlockingAsyncTaskWithCallback
from pyrs.interface.manual_reduction.pyrs_api import ReductionController
from pyrs.dataobjects.constants import HidraConstants  # type: ignore


class EventHandler:
    """Class to handle the event sent from UI widget
    """

    def __init__(self, parent):
        """Init

        Parameters
        ----------
        parent : qtpy.QtWidgets.QMainWindow
            GUI main Window
        """
        self.parent = parent
        self.ui = self.parent.ui

        # controller
        self._controller = ReductionController()
        self.__last_run_number = ''

    def _current_runnumber(self):
        run_number = str(self.ui.lineEdit_runNumber.text()).strip()
        if len(run_number) == 0:
            return None
        elif run_number.isdigit():
            return int(run_number)

        return None

    def _set_sub_run_numbers(self, sub_runs):
        """Set sub run numbers to (1) Table and (2) Combo box

        Parameters
        ----------
        sub_runs

        Returns
        -------

        """
        self.ui.comboBox_sub_runs.clear()

        for sub_run in sorted(sub_runs):
            self.ui.comboBox_sub_runs.addItem('{}'.format(sub_run))

    def plot_detector_counts(self):
        """

        Returns
        -------

        """
        # Get valid sub run
        sub_run = parse_combo_box(self.ui.comboBox_sub_runs, int)
        if sub_run is None:
            return

        # Get counts
        try:
            counts_matrix = self._controller.get_detector_counts(sub_run, output_matrix=True)
        except RuntimeError as run_err:
            pop_message(self.parent, 'Unable to plot sub run {} counts on detector view'.format(sub_run),
                        str(run_err), message_type='error')
            return

        # Plot
        # set information
        det_2theta = self._controller.get_sample_log_value(HidraConstants.TWO_THETA, sub_run)
        info = 'sub-run: {}, 2theta = {}'.format(sub_run, det_2theta)

        # mask ID is not None
        # if mask_id is not None:
        #     # Get mask in array and do a AND operation to detector counts (array)
        #     mask_array = self._core.reduction_service.get_mask_array(self._curr_project_name, mask_id)
        #     detector_counts_array *= mask_array
        #     info += ', mask ID = {}'.format(mask_id)

        # Set information
        self.ui.lineEdit_detViewInfo.setText(info)

        # Plot:
        self.ui.graphicsView_detectorView.plot_detector_view(counts_matrix, (sub_run, None))

    def plot_powder_pattern(self):
        """

        Returns
        -------

        """
        # Get valid sub run
        sub_run = parse_combo_box(self.ui.comboBox_sub_runs, int)
        print('[TEST-OUTPUT] sub run = {},  type = {}'.format(sub_run, type(sub_run)))
        if sub_run is None:
            return

        # Get diffraction pattern
        try:
            pattern = self._controller.get_powder_pattern(sub_run)
        except RuntimeError as run_err:
            pop_message(self.parent, 'Unable to plot sub run {} histogram/powder pattern'.format(sub_run),
                        str(run_err), message_type='error')
            return

        # Get detector 2theta of this sub run
        det_2theta = self._controller.get_sample_log_value(HidraConstants.TWO_THETA, sub_run)
        info = 'sub-run: {}, 2theta = {}'.format(sub_run, det_2theta)

        # Plot
        self.ui.graphicsView_1DPlot.plot_diffraction(pattern[0], pattern[1], '2theta', 'intensity',
                                                     line_label=info, keep_prev=False)

    def manual_reduce_run(self):
        """

        (simply) reduce a list of runs in same experiment in a batch

        Returns
        -------

        """
        # Files names: NeXus, (output) project, mask, calibration
        run_number = self._current_runnumber()
        if not isinstance(run_number, int):
            nexus_file = str(self.ui.lineEdit_runNumber.text()).strip()

            # quit if the input is not NeXus
            if not (os.path.exists(nexus_file) and nexus_file.endswith('.nxs.h5')):
                return
        # END-IF

        # Output HidraProject file
        project_file = str(self.ui.lineEdit_outputDirectory.text().strip())
        # mask file
        mask_file = str(self.ui.lineEdit_maskFile.text().strip())
        if mask_file == '':
            mask_file = None
        # calibration file
        calibration_file = str(self.ui.lineEdit_calibrationFile.text().strip())
        if calibration_file == '':
            calibration_file = None
        # vanadium file
        vanadium_file = str(self.ui.lineEdit_vanRunNumber.text().strip())
        if vanadium_file == '':
            vanadium_file = None

        # Start task
        if True:
            # single thread:
            try:
                hidra_ws = self._controller.reduce_hidra_workflow(nexus_file, project_file,
                                                                  self.ui.progressBar, mask=mask_file,
                                                                  calibration=calibration_file,
                                                                  vanadium_file=vanadium_file)
            except RuntimeError as run_err:
                pop_message(self.parent, 'Failed to reduce {}',
                            str(run_err), message_type='error')
                return

            # Update table
            # TODO - Need to fill the table!
            sub_runs = list(hidra_ws.get_sub_runs())
            # for sub_run in sub_runs:
            #     self.ui.rawDataTable.update_reduction_state(sub_run, True)

            # Set the sub runs combo box
            self._set_sub_run_numbers(sub_runs)

        else:
            task = BlockingAsyncTaskWithCallback(self._controller.reduce_hidra_workflow,
                                                 args=(nexus_file, project_file, self.ui.progressBar),
                                                 kwargs={'mask': mask_file, 'calibration': calibration_file},
                                                 blocking_cb=QApplication.processEvents)
            # TODO - catch RuntimeError! ...
            # FIXME - check output directory
            task.start()

        return

    def save_project(self):
        self._controller.save_project()

    def set_mask_file_widgets(self, state):
        """Set the default value of HB2B mask XML

        Parameters
        ----------
        state : Qt.State
            Qt state as unchecked or checked

        Returns
        -------
        None

        """

        if state != Qt.Unchecked:
            self.ui.lineEdit_maskFile.setText(self._controller.get_default_mask_dir() + 'HB2B_MASK_Latest.xml')
        self.ui.lineEdit_maskFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseMaskFile.setEnabled(state == Qt.Unchecked)

    def set_calibration_file_widgets(self, state):
        """Set the default value of HB2B geometry calibration file

        Parameters
        ----------
        state : Qt.State
            Qt state as unchecked or checked

        Returns
        -------

        """
        if state != Qt.Unchecked:
            self.ui.lineEdit_calibrationFile.setText(self._controller.get_default_calibration_dir() +
                                                     'HB2B_Latest.json')
        self.ui.lineEdit_calibrationFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseCalibrationFile.setEnabled(state == Qt.Unchecked)

    def update_run_changed(self, run_number):
        """Update widgets including output directory and etc due to change of run number

        Parameters
        ----------
        run_number : int
            run number

        Returns
        -------
        None

        """
        # don't do anything if the run number didn't change
        if run_number == self.__last_run_number:
            return
        elif not isinstance(run_number, int):
            return
