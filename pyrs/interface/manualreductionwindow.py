try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_manualreductionwindow
from pyrs.core.pyrscore import PyRsCore
from pyrs.utilities import hb2b_utilities
import os
import gui_helper
import numpy
from pyrs.utilities import checkdatatypes


class ManualReductionWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(ManualReductionWindow, self).__init__(parent)

        # class variables
        self._core = None
        self._currIPTSNumber = None
        self._currExpNumber = None

        # mutexes
        self._plot_run_numbers_mutex = False
        self._plot_sliced_mutex = False
        self._plot_selection_mutex = False

        # set up UI
        self.ui = ui.ui_manualreductionwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        # set up the event handling
        self.ui.pushButton_setIPTSNumber.clicked.connect(self.do_set_ipts_exp)
        self.ui.pushButton_browseOutputDir.clicked.connect(self.do_browse_output_dir)
        self.ui.pushButton_setCalibrationFile.clicked.connect(self.do_browse_calibration_file)

        self.ui.pushButton_batchReduction.clicked.connect(self.do_reduce_batch_runs)
        self.ui.pushButton_chopReduce.clicked.connect(self.do_chop_reduce_run)

        self.ui.pushButton_launchAdvSetupWindow.clicked.connect(self.do_launch_slice_setup)

        # radio button operation
        self.ui.radioButton_chopByTime.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopByLogValue.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopAdvanced.toggled.connect(self.event_change_slice_type)

        # event handling for combobox
        self.ui.comboBox_runs.currentIndexChanged.connect(self.event_new_run_to_plot)

        # init widgets
        self._init_widgets_setup()

        return

    def _init_widgets_setup(self):
        """
        init setup widgets
        :return:
        """
        self.ui.radioButton_chopByLogValue.setChecked(True)

    def do_browse_calibration_file(self):
        """ Browse and set up calibration file
        :return:
        """
        calibration_file = gui_helper.browse_file(self, caption='Choose and set up the calibration file',
                                                  default_dir=self._core.working_dir, file_filter='hdf5 (*hdf)',
                                                  file_list=False, save_file=False)
        if calibration_file is None or calibration_file == '':
            # operation canceled
            return

        # set to the browser
        self.ui.lineEdit_calibratonFile.setText(calibration_file)

        # set to core
        self._core.reduction_engine.set_calibration_file(calibration_file)

        return

    def do_browse_output_dir(self):
        """
        browse and set output directory
        :return:
        """
        output_dir = gui_helper.browse_dir(self, caption='Output directory for reduced data',
                                           default_dir=os.path.expanduser('~'))
        if output_dir is not None or output_dir != '':
            self.ui.lineEdit_outputDir.setText(output_dir)
            self._core.reduction_engine.set_output_dir(output_dir)

        return

    def do_chop_reduce_run(self):
        """
        chop and reduce the selected run
        :return:
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
            data_key = self._core.reduction_engine.chop_data()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Unable to slice data', detailed_message=str(run_err),
                                   message_type='error')
            return

        try:
            self._core.reduction_engine.reduced_chopped_data(data_key)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Failed to reduce sliced data', detailed_message=str(run_err),
                                   message_type='error')
            return

        # fill the run numbers to plot selection
        self._setup_plot_selection(append_mode=False, item_list=self._core.reduction_engine.get_chopped_names())

        # plot
        self._plot_data()

        return

    def do_launch_slice_setup(self):
        # TODO - 20181009 - Need to refine
        import slicersetupwindow
        self._slice_setup_window = slicersetupwindow.EventSlicerSetupWindow(self)
        self._slice_setup_window.show()
        return

    def _setup_plot_sliced_runs(self, run_number, sliced_):
        """

        :return:
        """

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

    def _plot_data(self):
        """

        :return:
        """
        print ('[WARNING] Not Implemented')
        # TODO - 20181006 - Implement ASAP

        # self.ui.comboBox_plotSelection
        #
        # self.ui.graphicsView_1DPlot.plot_

        return

    def do_reduce_batch_runs(self):
        """
        (simply) reduce a list of runs in same experiment in a batch
        :return:
        """
        # get run numbers
        # TODO - 20181113 - Entry point ... set up mock data
        try:
            run_number_list = gui_helper.parse_integers(str(self.ui.lineEdit_runNumbersList.text()))
        except RuntimeError as run_err:
            gui_helper.pop_message(blabla)
            return

        error_msg = ''
        for run_number in run_number_list:
            try:
                self._core.reduction_engine.reduce_rs_run(self._currIPTSNumber, self._expNumber, run_number)
            except RuntimeError as run_err:
                error_msg += 'Failed to reduce run {} due to {}\n'.format(run_number, run_err)
        # END-FOR

        # pop out error message if there is any
        if error_msg != '':
            error_msg = 'Reducing IPTS-{} Exp-{}:\n{}'.format(self._currIPTSNumber, self._currExpNumber, error_msg)
            gui_helper.pop_message(self, 'Batch reduction fails (partially or completely)', detailed_message=error_msg,
                                   message_type='error')
            return

        return

    def do_set_ipts_exp(self):
        """
        set IPTS number
        :return:
        """
        try:
            ipts_number = gui_helper.parse_integer(str(self.ui.lineEdit_iptsNumber.text()))
            exp_number = gui_helper.parse_integer(str(self.ui.lineEdit_expNumber.text()))
            self._currIPTSNumber = ipts_number
            self._currExpNumber = exp_number
            self._core.reduction_engine.set_ipts_exp_number(ipts_number, exp_number)
        except RuntimeError:
            gui_helper.pop_message(self, 'IPTS number shall be set to an integer.', message_type='error')

        return

    def event_change_slice_type(self):
        """ Handle the event as the event slicing type is changed
        :return:
        """
        disable_time_slice = True
        disable_value_slice = True
        disable_adv_slice = True

        # find out which group shall be enabled
        if self.ui.radioButton_chopByTime.isChecked():
            disable_time_slice = False
        elif self.ui.radioButton_chopByLogValue.isChecked():
            disable_value_slice = False
        else:
            disable_adv_slice = False

        # enable/disable group
        self.ui.groupBox_sliceByTime.setEnabled(not disable_time_slice)
        self.ui.groupBox_sliceByLogValue.setEnabled(not disable_value_slice)
        self.ui.groupBox_advancedSetup.setEnabled(not disable_adv_slice)

        return

    def event_new_run_to_plot(self):
        """ User selects a different run number to plot
        :return:
        """
        curr_run_number = int(str(self.ui.comboBox_runs.currentText()))
        is_chopped = self._core.reduction_engine.is_chopped_run(curr_run_number)

        # set the sliced box
        self._plot_sliced_mutex = True
        self.ui.comboBox_slicedRuns.clear()
        if is_chopped:
            sliced_segment_list = self._core.reduction_engine.get_chopped_names(curr_run_number)
            for segment in sorted(sliced_segment_list):
                self.ui.comboBox_slicedRuns.addItem('{}'.format(segment))
        else:
            pass

        # set the plot options
        # TODO - 20181008 - ASAP
        self._plot_selection_mutex = True
        if is_chopped:
            # set up with chopped data
            pass
        else:
            # set up with non-chopped data
            pass

        self._plot_sliced_mutex = False



    def setup_window(self, pyrs_core):
        """
        set up the manual reduction window from its parent
        :param pyrs_core:
        :return:
        """
        # check
        assert isinstance(pyrs_core, PyRsCore), 'Controller core {0} must be a PyRSCore instance but not a {1}.' \
                                                ''.format(pyrs_core, pyrs_core.__class__.__name__)

        self._core = pyrs_core

        return

