try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout
    from PyQt5.uic import loadUi as load_ui
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog, QVBoxLayout
    from PyQt4.uic import loadUi as load_ui
from pyrs.core.pyrscore import PyRsCore
from pyrs.core import calibration_file_io
from ui.diffdataviews import DetectorView, GeneralDiffDataView
import os
import gui_helper
import numpy
from pyrs.utilities import checkdatatypes
from pyrs.utilities.rs_project_file import HidraConstants
from pyrs.interface.ui import rstables


# TODO LIST - #84 - 1. UI: change segments to masks
# TODO              2. UI: add solid angle input
# TODO              3. UI: add option to use reduced data from input project file
# TODO              4. Implement plot method for reduced data
# TODO              5. Implement method to reduce data
# TODO              6. Add parameters for reducing data

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
        self._project_data_id = None
        self._output_dir = None
        self._curr_project_name = None   # last loaded project file

        # mutexes
        self._plot_run_numbers_mutex = False
        self._plot_sliced_mutex = False
        self._plot_selection_mutex = False

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'manualreductionwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)
        # promote some widgets
        self._promote_widgets()

        # set up the event handling
        self.ui.pushButton_loadProjectFile.clicked.connect(self.do_load_hidra_projec_file)
        self.ui.pushButton_browseOutputDir.clicked.connect(self.do_browse_output_dir)
        self.ui.pushButton_setCalibrationFile.clicked.connect(self.do_browse_calibration_file)

        self.ui.pushButton_setBrowseIDF.clicked.connect(self.do_browse_set_idf)

        self.ui.pushButton_batchReduction.clicked.connect(self.do_reduce_batch_runs)
        self.ui.pushButton_saveProject.clicked.connect(self.do_save_project)
        self.ui.pushButton_chopReduce.clicked.connect(self.do_chop_reduce_run)

        self.ui.pushButton_launchAdvSetupWindow.clicked.connect(self.do_launch_slice_setup)
        self.ui.pushButton_plotDetView.clicked.connect(self.do_plot)

        # radio button operation
        self.ui.radioButton_chopByTime.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopByLogValue.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopAdvanced.toggled.connect(self.event_change_slice_type)

        # event handling for combobox
        # self.ui.comboBox_sub_runs.currentIndexChanged.connect(self.event_new_run_to_plot)

        # TODO - ASAP - Use these 2 buttons to enable/disable write access to configuration
        # actionEdit_Calibrations
        # actionFix_Calibrations

        # TODO - ASAP - Load Instrument
        # actionLoad_Instrument

        # menu operation
        self.ui.actionLoad_Image.triggered.connect(self.event_load_image)
        # Load project file (*.h5)
        self.ui.actionLoad_Project_File.triggered.connect(self.do_load_project_h5)

        # init widgets
        self._init_widgets_setup()

        # mutexes
        self._mutexPlotRuns = False

        return

    def _init_widgets_setup(self):
        """
        init setup widgets
        :return:
        """
        self.ui.radioButton_chopByLogValue.setChecked(True)

        return

    def _promote_widgets(self):
        """
        promote widgets
        :return:
        """
        # 1D diffraction view
        curr_layout = QVBoxLayout()
        self.ui.frame_diffractionView.setLayout(curr_layout)
        self.ui.graphicsView_1DPlot = GeneralDiffDataView(self)
        curr_layout.addWidget(self.ui.graphicsView_1DPlot)

        # 2D detector view
        curr_layout = QVBoxLayout()
        self.ui.frame_detectorView.setLayout(curr_layout)
        self.ui.graphicsView_detectorView = DetectorView(self)
        curr_layout.addWidget(self.ui.graphicsView_detectorView)

        # Sub run information table
        self.ui.rawDataTable = rstables.RawDataTable(self)
        gui_helper.promote_widget(self.ui.frame_subRunInfoTable, self.ui.rawDataTable)

        return

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
        self._core.reduction_manager.set_calibration_file(calibration_file)

        return

    def do_browse_set_idf(self):
        """
        Browse (optonally) and set instrument definition file
        :return:
        """
        idf_name = str(self.ui.lineEdit_idfName.text()).strip()
        if idf_name == '' or not os.path.exists(idf_name):
            # browse IDF and set
            idf_name = gui_helper.browse_file(self, 'Instrument definition file', os.getcwd(),
                                              'Text (*.txt);;XML (*.xml)', False, False)
            if len(idf_name) == 0:
                return   # user cancels operation
            else:
                self.ui.lineEdit_idfName.setText(idf_name)
        # END-IF

        # set
        instrument = calibration_file_io.import_instrument_setup(idf_name)
        self._core.reduction_manager.set_instrument(instrument)

        return

    def do_browse_output_dir(self):
        """
        browse and set output directory
        :return:
        """
        output_dir = gui_helper.browse_dir(self, caption='Output directory for reduced data',
                                           default_dir=os.path.expanduser('~'))
        if output_dir != '':
            self.ui.lineEdit_outputDir.setText(output_dir)
            self._core.reduction_manager.set_output_dir(output_dir)
            self._output_dir = output_dir

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
            data_key = self._core.reduction_manager.chop_data()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Unable to slice data', detailed_message=str(run_err),
                                   message_type='error')
            return

        try:
            self._core.reduction_manager.reduced_chopped_data(data_key)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Failed to reduce sliced data', detailed_message=str(run_err),
                                   message_type='error')
            return

        # fill the run numbers to plot selection
        self._setup_plot_selection(append_mode=False, item_list=self._core.reduction_manager.get_chopped_names())

        # plot
        self._plot_data()

        return

    def do_launch_slice_setup(self):
        # TODO - 20181009 - Need to refine
        import slicersetupwindow
        self._slice_setup_window = slicersetupwindow.EventSlicerSetupWindow(self)
        self._slice_setup_window.show()
        return

    def do_load_project_h5(self):
        """ Load project file in HDF5 format
        :return:
        """
        project_h5_name = gui_helper.browse_file(self, 'HIDRA Project File', os.getcwd(),
                                                 file_filter='*.hdf5;;*.h5', file_list=False,
                                                 save_file=False)

        try:
            # TODO FIXME - #72 - Error!
            data_handler = self._core.reduction_manager.load_hidra_project(project_h5_name, blabla)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Failed to load project file {}: {}'.format(project_h5_name, run_err),
                                   None, 'error')
        else:
            print ('Loaded {} to {}'.format(project_h5_name, data_handler))

            # populate the sub-runs
            self._set_sub_runs(data_handler)
            self._project_data_id = data_handler
            self._curr_project_name = project_h5_name
        # END-TRY-EXCEPT

        return

    def do_plot(self):
        """ Plot detector counts as 2D detector view view OR reduced data according to the tab that is current on
        :return:
        """
        print ('[DB...BAT] Plotting tab index = {}'.format(self.ui.tabWidget_View.currentIndex()))

        current_tab_index = self.ui.tabWidget_View.currentIndex()
        sub_run = int(self.ui.comboBox_sub_runs.currentText())

        if current_tab_index == 0:
            # raw view
            self.plot_detector_counts(sub_run)
        elif current_tab_index == 1:
            # reduced view
            self.plot_reduced_data(sub_run)
        else:
            raise NotImplementedError('Tab {} with index {} is not defined'
                                      ''.format(self.ui.tabWidget_View.name(), current_tab_index))

        return

    def _set_sub_runs(self, data_id):
        """ set the sub runs to comboBox_sub_runs
        :param data_id:
        :return:
        """
        sub_runs = self._core.reduction_manager.get_sub_runs(data_id)

        self.ui.comboBox_sub_runs.clear()
        for sub_run in sorted(sub_runs):
            self.ui.comboBox_sub_runs.addItem('{:04}'.format(sub_run))

        self.ui.comboBox_sub_runs.setCurrentIndex(0)

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

    def do_reduce_batch_runs(self):
        """
        (simply) reduce a list of runs in same experiment in a batch
        :return:
        """
        # get (sub) run numbers
        sub_runs_str = str(self.ui.lineEdit_runNumbersList.text()).strip().lower()
        if sub_runs_str == 'all':
            sub_run_list = self._core.reduction_manager.get_sub_runs(self._project_data_id)
        else:
            try:
                sub_run_list = gui_helper.parse_integers(sub_runs_str)
            except RuntimeError as run_err:
                gui_helper.pop_message(self, 'Failed to parse integer list',
                                       '{}'.format(run_err), 'error')
                return
        # END-IF-ELSE

        # form a message
        message = 'Reduced.... \n'
        for sub_run_number in sub_run_list:
            tth_i = self._core.reduction_manager.get_sub_run_2theta(self._project_data_id, sub_run_number)
            try:
                self._core.reduction_manager.reduce_to_2theta_histogram(data_id=self._project_data_id,
                                                                        sub_run=sub_run_number,
                                                                        two_theta=tth_i,
                                                                        use_mantid_engine=False,
                                                                        mask=None)
            except RuntimeError as run_err:
                message += 'Sub-run: {}... Failed: {}\n'.format(sub_run_number, run_err)
            else:
                message += 'Sub-run: {}\n'.format(sub_run_number)

            """ This is for future IPTS/Run system
            message += 'Sub-run: {}\n'.format(self._currIPTSNumber, run_number,
                                              '/HFIR/HB2B/IPTS-{}/nexus/HB2B_{}.nxs.h5'
                                              ''.format(self._currIPTSNumber, run_number))
            """
        # END-FOR
        self.ui.plainTextEdit_message.setPlainText(message)

        return

    def do_save_project(self):
        """Save project
        :return:
        """
        output_project_name = os.path.join(self._output_dir, os.path.basename(self._curr_project_name))
        if output_project_name != self._curr_project_name:
            import shutil
            shutil.copyfile(self._curr_project_name, output_project_name)

        self._core.reduction_manager.save_project(self._project_data_id, output_project_name)

    def do_load_hidra_projec_file(self):
        """
        set IPTS number
        :return:
        """
        try:
            ipts_number = gui_helper.parse_integer(str(self.ui.lineEdit_iptsNumber.text()))
            exp_number = gui_helper.parse_integer(str(self.ui.lineEdit_expNumber.text()))
            self._currIPTSNumber = ipts_number
            self._currExpNumber = exp_number
            project_file_name = 'blabla.hdf5'
        except RuntimeError:
            gui_helper.pop_message(self, 'IPTS number shall be set to an integer.', message_type='error')
            project_file_name = gui_helper.browse_file(self, 'Hidra Project File', os.getcwd(), 'hdf5 (*.hdf5)', False, False)

        self.load_hydra_file(project_file_name)

        return

    def event_change_slice_type(self):
        """ Handle the event as the event slicing type is changed
        :return:
        """
        # TODO - ASAP - Clean
        # TODO - ASAP - Set default radio button
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
        # FIXME TODO - ASAP - use setTabEnabled(index, false)
        # self.ui.groupBox_sliceByTime.setEnabled(not disable_time_slice)
        # self.ui.groupBox_sliceByLogValue.setEnabled(not disable_value_slice)
        # self.ui.groupBox_advancedSetup.setEnabled(not disable_adv_slice)

        return

    def event_load_image(self):
        """
        Load image binary file (as HFIR SPICE binary standard)
        :return:
        """
        bin_file = gui_helper.browse_file(self, caption='Select a SPICE Image Binary File',
                                          default_dir=self._core.working_dir,
                                          file_filter='Binary (*.bin)', file_list=False, save_file=False)

        # return if user cancels operation
        if bin_file == '':
            return

        print ('[DB...BAT] Select {} to load: ... '.format(bin_file))

        # TODO - ASAP - Move the following to correct place
        import mantid

        # bin_file_name = '/home/wzz/Projects/PyRS/tests/testdata/LaB6_10kev_35deg-00004_Rotated.bin'

        ws_name = 'testws'

        LoadSpiceXML2DDet(Filename=bin_file_name, OutputWorkspace=ws_name, LoadInstrument=False)

        return

    def event_new_run_to_plot(self):
        """ User selects a different run number to plot
        :return:
        """
        if self._mutexPlotRuns:
            return

        curr_run_number = int(str(self.ui.comboBox_runs.currentText()))
        if not self._core.reduction_manager.has_run_reduced(curr_run_number):
            return

        is_chopped = self._core.reduction_manager.is_chopped_run(curr_run_number)

        # set the sliced box
        self._plot_sliced_mutex = True
        self.ui.comboBox_slicedRuns.clear()
        if is_chopped:
            sliced_segment_list = self._core.reduction_manager.get_chopped_names(curr_run_number)
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

        return

    def load_hydra_file(self, project_file_name):
        """
        Load Hidra project file to the core
        :param project_file_name:
        :return:
        """
        # TODO - #84 - Need try-catch
        # Load data file
        project_name = os.path.basename(project_file_name).split('.')[0]
        self._core.load_hidra_project(project_file_name, project_name=project_name)
        self._curr_project_name = project_file_name

        # Fill sub runs to self.ui.comboBox_sub_runs
        sub_runs = self._core.reduction_manager.get_sub_runs(self._curr_project_name)
        sub_runs.sort()

        # set sub runs: lock and release
        self._mutexPlotRuns = True
        # clear and set
        self.ui.comboBox_sub_runs.clear()
        for sub_run in sub_runs:
            self.ui.comboBox_sub_runs.addItem(sub_run)
        self._mutexPlotRuns = False

        # Set to first sub run and plot
        self.ui.comboBox_sub_runs.setCurrentIndex(0)

        # Fill in self.ui.frame_subRunInfoTable
        meta_data_array = self._core.reduction_manager.get_sample_logs_values([HidraConstants.SUB_RUNS,
                                                                               HidraConstants.TWO_THETA])
        self.ui.rawDataTable.add_subruns_info(meta_data_array, clear_table=True)

        return

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
        detector_counts_array = self._core.reduction_manager.get_detector_counts(self._curr_project_name,
                                                                                 sub_run_number)

        # set information
        det_2theta = self._core.reduction_manager.get_sample_log_value(self._project_data_id,
                                                                       sub_run_number,
                                                                       HidraConstants.TWO_THETA)
        info = 'sub-run: {}, 2theta = {}' \
               ''.format(sub_run_number, det_2theta)

        # If mask ID is not None
        if mask_id is not None:
            # Get mask in array and do a AND operation to detector counts (array)
            mask_array = self._core.reduction_manager.get_mask_array(self._curr_project_name, mask_id)
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
            two_theta_array, diff_array = self._core.reduction_manager.get_diffraction_pattern(self._project_data_id,
                                                                                               sub_run_number,
                                                                                               mask_id)
            if two_theta_array is None:
                raise NotImplementedError('2theta array is not supposed to be None.')
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to retrieve reduced data',
                                   'For sub run {} due to {}'.format(sub_run_number, run_err),
                                   'error')
            return

        # set information
        det_2theta = self._core.reduction_manager.get_sample_log_value(self._project_data_id,
                                                                       sub_run_number,
                                                                       HidraConstants.TWO_THETA)
        info = 'sub-run: {}, 2theta = {}' \
               ''.format(sub_run_number, det_2theta)

        # plot diffraction data
        self.ui.graphicsView_1DPlot.plot_diffraction(two_theta_array, diff_array, info)

        return

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

