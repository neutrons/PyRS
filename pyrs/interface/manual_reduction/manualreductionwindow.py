from qtpy.QtWidgets import QMainWindow, QVBoxLayout  # type: ignore
import os
from pyrs.utilities import load_ui  # type: ignore
from pyrs.interface.ui.diffdataviews import DetectorView, GeneralDiffDataView
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

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'manualreductionwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)
        # promote some widgets
        self._promote_widgets()

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

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.progressBar.setVisible(False)
        # event handling for combobox
        self.ui.comboBox_sub_runs.currentIndexChanged.connect(self.plot_sub_runs)

        # TODO - ASAP - Use these 2 buttons to enable/disable write access to configuration
        # actionEdit_Calibrations
        # actionFix_Calibrations

        # TODO - ASAP - Load Instrument
        # actionLoad_Instrument

        # Child windows
        self._slice_setup_window = None

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

        # raw view
        self._event_handler.plot_detector_counts()
        # reduced view
        self._event_handler.plot_powder_pattern()

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
