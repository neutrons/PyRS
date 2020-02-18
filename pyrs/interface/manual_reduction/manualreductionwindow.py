from qtpy.QtWidgets import QMainWindow, QVBoxLayout
import os
from pyrs.utilities import load_ui
from pyrs.interface.gui_helper import promote_widget
from pyrs.interface.ui.diffdataviews import DetectorView, GeneralDiffDataView
from pyrs.interface.ui import rstables
from pyrs.interface.manual_reduction.event_handler import EventHandler


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
        self._currExpNumber = None
        self._hydra_workspace = None  # HiDRA worksapce instance if any file loaded
        self._project_data_id = None  # Project name for reference (str)
        self._project_file_name = None   # last loaded project file
        self._output_dir = None

        # mutexes
        self._plot_run_numbers_mutex = False
        self._plot_sliced_mutex = False
        self._plot_selection_mutex = False

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'manualreductionwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)
        # promote some widgets
        self._promote_widgets()

        # Hide some not-yet-implemented
        self.ui.tabWidget_reduceRuns.setTabEnabled(1, False)  # User specified instrument parameter (shifts)
        self.ui.tabWidget_reduceRuns.setTabEnabled(2, False)  # advanced slicing tab

        # Event handler: handler must be set up after UI is loaded
        self._event_handler = EventHandler(parent=self)

        # Mask file: check box and line edit
        # set default
        self._mask_state(self.ui.checkBox_defaultMaskFile.checkState())
        # link event handling
        self.checkBox_defaultMaskFile.stateChanged.connect(self._mask_state)
        self.ui.pushButton_browseMaskFile.clicked.connect(self.browse_mask_file)
        self.ui.pushButton_browseVanadium.clicked.connect(self.browse_vanadium_file)

        # Calibration file: check box and line edit
        self._calibration_state(self.ui.checkBox_defaultCalibrationFile.checkState())
        self.checkBox_defaultCalibrationFile.stateChanged.connect(self._calibration_state)
        self.ui.pushButton_browseCalibrationFile.clicked.connect(self.browse_calibration_file)

        # Output directory: check box, spin box and line edit
        # change of run number won't trigger the scan of NeXus file
        self.ui.lineEdit_runNumber.textChanged.connect(self._event_handler.update_run_changed)
        self.ui.pushButton_browseNeXus.clicked.connect(self.browse_nexus_file)
        # self._output_state(self.ui.checkBox_defaultOutputDirectory.checkState())
        self.checkBox_defaultOutputDirectory.stateChanged.connect(self._output_state)
        self.ui.pushButton_browseOutputDirectory.clicked.connect(self.browse_output_dir)

        # Push button for split, convert and save project file
        self.ui.pushButton_splitConvertSaveProject.clicked.connect(self.split_convert_save_nexus)
        # Next: self.ui.pushButton_chopReduce.clicked.connect(self.slice_nexus)

        # Plotting
        self.ui.pushButton_plotDetView.clicked.connect(self.plot_sub_runs)

        # radio button operation
        self.ui.radioButton_chopByTime.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopByLogValue.toggled.connect(self.event_change_slice_type)
        self.ui.radioButton_chopAdvanced.toggled.connect(self.event_change_slice_type)

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.progressBar.setVisible(False)
        # event handling for combobox
        # TODO - Make this work properly
        # self.ui.comboBox_sub_runs.currentIndexChanged.connect(self.event_different_sub_run_selected)

        # TODO - ASAP - Use these 2 buttons to enable/disable write access to configuration
        # actionEdit_Calibrations
        # actionFix_Calibrations

        # TODO - ASAP - Load Instrument
        # actionLoad_Instrument

        # Child windows
        self._slice_setup_window = None

        # menu operation

        # load NeXus
        self.ui.actionLoad_nexus_file.triggered.connect(self.load_nexus_file)

        # Load project file (*.h5)
        self.ui.actionLoad_Project_File.triggered.connect(self.load_hidra_project_file)

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
        self.ui.tabWidget_reduceRuns.setCurrentIndex(0)

        # Event slicer type is set to log value as default
        self.ui.radioButton_chopByLogValue.setChecked(True)

        # Set up data table
        self.ui.rawDataTable.setup()

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
        promote_widget(self.ui.frame_subRunInfoTable, self.ui.rawDataTable)

        return

    def _mask_state(self, state):
        """Set the default value of HB2B mask XML

        Parameters
        ----------
        state : Qt.State
            Qt state as unchecked or checked

        Returns
        -------
        None

        """
        self._event_handler.set_mask_file_widgets(state)

    def _calibration_state(self, state):
        """Set the default value of HB2B geometry calibration file

        Parameters
        ----------
        state : Qt.State
            Qt state as unchecked or checked

        Returns
        -------

        """
        self._event_handler.set_calibration_file_widgets(state)

    def _output_state(self, state):
        """Set the default value of directory for output files

        Parameters
        ----------
        state : Qt.State
            Qt state as unchecked or checked

        Returns
        -------
        None

        """
        self._event_handler.set_output_dir_widgets(state)

    # Menu event handler
    def load_nexus_file(self):
        """Browse NeXus file

        Returns
        -------

        """
        self._event_handler.browse_load_nexus()

    def load_hidra_project_file(self):
        """Browse and load Hidra project file

        Returns
        -------
        None

        """
        self._event_handler.browse_load_hidra()

    def browse_calibration_file(self):
        """ Browse and set up calibration file
        :return:
        """
        self._event_handler.browse_calibration_file()

    def browse_nexus_file(self):
        """Browse file system for NeXus file path

        Returns
        -------
        None

        """
        self._event_handler.browse_nexus_path()

    def browse_idf(self):
        """
        Browse (optonally) and set instrument definition file
        :return:
        """
        self._event_handler.browse_idf()

    def browse_output_dir(self):
        """
        browse and set output directory
        :return:
        """
        self._event_handler.browse_output_dir()

    def browse_vanadium_file(self):
        """Browse vanadium HiDRA file

        Returns
        -------
        None

        """
        self._event_handler.browse_vanadium_file()

    def browse_mask_file(self):
        """
        set IPTS number
        :return:
        """
        self._event_handler.browse_mask_file()

    def plot_sub_runs(self):
        """ Plot detector counts as 2D detector view view OR reduced data according to the tab that is current on
        :return:
        """
        current_tab_index = self.ui.tabWidget_View.currentIndex()

        if current_tab_index == 0:
            # raw view
            self._event_handler.plot_detector_counts()
        elif current_tab_index == 1:
            # reduced view
            self._event_handler.plot_powder_pattern()
        else:
            raise NotImplementedError('Tab {} with index {} is not defined'
                                      ''.format(self.ui.tabWidget_View.name(), current_tab_index))

        return

    def do_quit(self):
        """
        Quit manual reduction window

        Returns
        -------
        None
        """
        # Close child windows if exists
        if self._slice_setup_window:
            self._slice_setup_window.close()

        # This window
        self.close()

        return

    def split_convert_save_nexus(self):
        """Reduce (split sub runs, convert to powder pattern and save) manually

        Returns
        -------
        None

        """
        self._event_handler.manual_reduce_run()

    # Next: it is not implemented now
    def event_change_slice_type(self):
        """Handle the event as the event slicing type is changed

        Returns
        -------
        None
        """
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

        print('[DEBUG] Event filtering mode: Time slicer = {}, Value slicer = {}, Adv. Slicer = {}'
              ''.format(not disable_time_slice, not disable_value_slice, not disable_adv_slice))

        # enable/disable group
        # FIXME TODO - ASAP - use setTabEnabled(index, false)
        # self.ui.groupBox_sliceByTime.setEnabled(not disable_time_slice)
        # self.ui.groupBox_sliceByLogValue.setEnabled(not disable_value_slice)
        # self.ui.groupBox_advancedSetup.setEnabled(not disable_adv_slice)

        return
