"""
View for the manual-reduction interface.

This is the "V" of the Model-View-Presenter structure shared with the
peak-fitting and texture-fitting UIs. It owns the Qt widget tree (loaded from
``manualreductionwindow.ui``) and all signal/slot wiring. Slot bodies delegate
data access to :class:`ManualReductionModel` and plot preparation to
:class:`ManualReductionCrtl`; the browse dialogs, widget-state toggles and
reduce-orchestration that previously lived in the ``EventHandler`` helper now
live here.
"""

import os

from qtpy.QtCore import Qt  # type: ignore
from qtpy.QtWidgets import QMainWindow, QVBoxLayout  # type: ignore

from pyrs.interface.gui_helper import browse_dir, browse_file, parse_combo_box, pop_message
from pyrs.interface.ui.diffdataviews import DetectorView, GeneralDiffDataView
from pyrs.utilities import get_default_output_dir, get_nexus_file, load_ui  # type: ignore


class ManualReductionViewer(QMainWindow):
    """GUI window for manual data reduction (Model-View-Presenter view component)."""

    def __init__(self, manual_reduction_model, manual_reduction_ctrl, parent=None):
        """
        Args:
            manual_reduction_model: The :class:`ManualReductionModel` instance.
            manual_reduction_ctrl: The :class:`ManualReductionCrtl` presenter.
            parent: Optional Qt parent window.
        """
        super(ManualReductionViewer, self).__init__(parent)

        self._model = manual_reduction_model
        self._ctrl = manual_reduction_ctrl

        # View-side state
        self.__last_run_number = ""
        self._slice_setup_window = None

        # set up UI: load_ui resolves by basename to pyrs/interface/designer/
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join("ui", "manualreductionwindow.ui"))
        self.ui = load_ui(ui_path, baseinstance=self)
        self._promote_widgets()

        # Mask file: check box and line edit
        self._mask_state(self.ui.checkBox_defaultMaskFile.checkState())
        self.ui.checkBox_defaultMaskFile.stateChanged.connect(self._mask_state)
        self.ui.pushButton_browseMaskFile.clicked.connect(self.browse_mask_file)
        self.ui.pushButton_browseVanadium.clicked.connect(self.browse_vanadium_file)

        # Calibration file: check box and line edit
        self._calibration_state(self.ui.checkBox_defaultCalibrationFile.checkState())
        self.ui.checkBox_defaultCalibrationFile.stateChanged.connect(self._calibration_state)
        self.ui.pushButton_browseCalibrationFile.clicked.connect(self.browse_calibration_file)

        # Output directory: check box, spin box and line edit
        self.ui.lineEdit_runNumber.textChanged.connect(self.update_run_changed)
        self.ui.pushButton_browseNeXus.clicked.connect(self.browse_nexus_file)
        self.ui.checkBox_defaultOutputDirectory.stateChanged.connect(self._output_state)
        self.ui.pushButton_browseOutputDirectory.clicked.connect(self.browse_output_dir)

        # Push button for split, convert and save project file
        self.ui.pushButton_splitConvertSaveProject.clicked.connect(self.split_convert_save_nexus)

        # Plotting
        self.ui.pushButton_plotDetView.clicked.connect(self.plot_sub_runs)

        self.ui.actionQuit.triggered.connect(self.do_quit)
        self.ui.progressBar.setVisible(False)
        self.ui.comboBox_sub_runs.currentIndexChanged.connect(self.plot_sub_runs)

        # surface model failures without the model depending on any UI helper
        self._model.failureMsg.connect(self._on_failure)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model

    def _on_failure(self, title, message, detail):
        pop_message(self, title, detailed_message="{}\n{}".format(message, detail), message_type="error")

    def _promote_widgets(self):
        """Promote the diffraction and detector view frames to plot widgets."""
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

    # ------------------------------------------------------------------
    # Default-file check-box state handlers
    # ------------------------------------------------------------------
    def _mask_state(self, state):
        """Toggle the default HB2B mask XML file."""
        if state != Qt.Unchecked:
            self.ui.lineEdit_maskFile.setText(self._model.get_default_mask_dir() + "HB2B_MASK_Latest.xml")
        self.ui.lineEdit_maskFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseMaskFile.setEnabled(state == Qt.Unchecked)

    def _calibration_state(self, state):
        """Toggle the default HB2B geometry calibration file."""
        if state != Qt.Unchecked:
            self.ui.lineEdit_calibrationFile.setText(self._model.get_default_calibration_dir() + "HB2B_Latest.json")
        self.ui.lineEdit_calibrationFile.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseCalibrationFile.setEnabled(state == Qt.Unchecked)

    def _output_state(self, state):
        """Toggle the default output directory."""
        if state != Qt.Unchecked:
            self.update_run_changed(self._current_runnumber())
        self.ui.lineEdit_outputDirectory.setEnabled(state == Qt.Unchecked)
        self.ui.pushButton_browseOutputDirectory.setEnabled(state == Qt.Unchecked)

    # ------------------------------------------------------------------
    # Browse dialogs
    # ------------------------------------------------------------------
    def browse_calibration_file(self):
        """Browse and set up the calibration file."""
        calibration_file = browse_file(
            self,
            caption="Choose and set up the calibration file",
            default_dir=self._model.get_default_calibration_dir(),
            file_filter="hdf5 (*hdf)",
            file_list=False,
            save_file=False,
        )
        if calibration_file is None or calibration_file == "":
            return  # operation canceled
        self.ui.lineEdit_calibrationFile.setText(calibration_file)

    def browse_mask_file(self):
        """Browse and set the masking file."""
        mask_file_name = browse_file(
            self, "Hidra Mask File", self._model.get_default_mask_dir(), "Mantid Mask(*.xml)", False, False
        )
        self.ui.lineEdit_maskFile.setText(mask_file_name)

    def browse_nexus_file(self):
        """Browse the NeXus file path and write it to the run-number line edit."""
        nexus_file_path = browse_file(
            self,
            "NeXus File",
            self._model.get_default_nexus_dir(ipts_number=None),
            "NeXus(*.nxs.h5)",
            False,
            False,
        )
        if nexus_file_path is not None:
            self.ui.lineEdit_runNumber.setText(nexus_file_path)

    def browse_output_dir(self):
        """Browse and set the output directory."""
        output_dir = browse_dir(self, caption="Output directory for reduced data", default_dir="/HFIR/HB2B/")
        if output_dir != "":
            self.ui.lineEdit_outputDirectory.setText(output_dir)

    def browse_vanadium_file(self):
        """Browse the vanadium HiDRA project file and set the line edit."""
        vanadium_file_name = browse_file(
            self,
            "HiDRA Vanadium File",
            self._model.get_default_mask_dir(),
            "HiDRA project(*.h5)",
            False,
            False,
        )
        if vanadium_file_name is not None and vanadium_file_name != "":
            self.ui.lineEdit_vanRunNumber.setText(vanadium_file_name)

    # ------------------------------------------------------------------
    # Run-number / output-directory helpers
    # ------------------------------------------------------------------
    def _current_runnumber(self):
        run_number = str(self.ui.lineEdit_runNumber.text()).strip()
        if len(run_number) == 0:
            return None
        elif run_number.isdigit():
            return int(run_number)
        return None

    def update_run_changed(self, run_number):
        """Update the output directory when the run number changes."""
        if run_number == self.__last_run_number:
            return
        elif not isinstance(run_number, int):
            return

        try:
            project_dir = get_default_output_dir(run_number)
            self.ui.lineEdit_outputDirectory.setText(project_dir)
            self.__last_run_number = run_number
        except RuntimeError as e:
            print("Failed to find project directory for {}".format(run_number))
            print(e)

    def _set_sub_run_numbers(self, sub_runs):
        """Populate the sub-run combo box."""
        self.ui.comboBox_sub_runs.clear()
        for sub_run in sorted(sub_runs):
            self.ui.comboBox_sub_runs.addItem("{}".format(sub_run))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_sub_runs(self):
        """Plot the selected sub-run as both a 2D detector view and a 1D pattern."""
        sub_run = parse_combo_box(self.ui.comboBox_sub_runs, int)
        if sub_run is None:
            return

        # raw 2D detector view
        try:
            info = self._ctrl.plot_detector_counts(self.ui.graphicsView_detectorView, sub_run)
            self.ui.lineEdit_detViewInfo.setText(info)
        except RuntimeError as run_err:
            pop_message(
                self,
                "Unable to plot sub run {} counts on detector view".format(sub_run),
                str(run_err),
                message_type="error",
            )

        # reduced 1D powder pattern
        try:
            self._ctrl.plot_powder_pattern(self.ui.graphicsView_1DPlot, sub_run)
        except RuntimeError as run_err:
            pop_message(
                self,
                "Unable to plot sub run {} histogram/powder pattern".format(sub_run),
                str(run_err),
                message_type="error",
            )

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------
    def split_convert_save_nexus(self):
        """Reduce (split sub runs, convert to powder pattern and save) manually."""
        run_number = self._current_runnumber()
        if isinstance(run_number, int):
            nexus_file = get_nexus_file(run_number)
        else:
            nexus_file = str(self.ui.lineEdit_runNumber.text()).strip()
            # quit if the input is not a NeXus file
            if not (os.path.exists(nexus_file) and nexus_file.endswith(".nxs.h5")):
                return

        output_dir = str(self.ui.lineEdit_outputDirectory.text().strip())

        mask_file = str(self.ui.lineEdit_maskFile.text().strip()) or None
        calibration_file = str(self.ui.lineEdit_calibrationFile.text().strip()) or None
        vanadium_file = str(self.ui.lineEdit_vanRunNumber.text().strip()) or None

        try:
            sub_runs = self._ctrl.reduce(
                nexus_file,
                output_dir,
                self.ui.progressBar,
                mask=mask_file,
                calibration=calibration_file,
                vanadium_file=vanadium_file,
            )
        except RuntimeError as run_err:
            pop_message(self, "Failed to reduce {}".format(nexus_file), str(run_err), message_type="error")
            return

        self._set_sub_run_numbers(sub_runs)

    def do_quit(self):
        """Quit the manual-reduction window."""
        if self._slice_setup_window:
            self._slice_setup_window.close()
        self.close()
