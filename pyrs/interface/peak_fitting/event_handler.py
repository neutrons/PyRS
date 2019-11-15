import os

from pyrs.interface.gui_helper import pop_message
from pyrs.interface.gui_helper import parse_integer
from pyrs.interface.gui_helper import browse_file
from pyrs.utilities import hb2b_utilities
from pyrs.utilities.check_data_types import check_string_variable
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.fit import Fit
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities


class EventHandler:

    def __init__(self, parent=None):
        self.parent = parent

    def browse_and_load_hdf(self):

        # Check
        self._check_core()

        # Use IPTS and run number to get the default Hydra HDF

        ## FIXME maybe recover auto name of hidra file
        # hydra_file_name = self._get_default_hdf()
        hidra_file_name = None
        if hidra_file_name is None:
            # No default Hidra file: browse the file
            file_filter = 'HDF (*.hdf);H5 (*.h5)'
            hidra_file_name = browse_file(self.parent, 'HIDRA Project File',
                                          os.getcwd(), file_filter,
                                          file_list=False, save_file=False)

            if hidra_file_name is None:
                # user clicked CANCEL
                return

        # Add file name to line edit to show
        self.parent.ui.lineEdit_expFileName.setText(hidra_file_name)

        # Load file as an option
        try:
            self.load_hidra_file(hydra_project_file=hidra_file_name)

            # enabled all fitting widgets
            o_gui = GuiUtilities(parent=self.parent)
            o_gui.enabled_fitting_widgets(True)
        except RuntimeError as run_err:
            pop_message(self, 'Failed to load {}'.format(hidra_file_name),
                                                  str(run_err), 'error')

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self.parent._core is None:
            raise RuntimeError('Not set up yet!')

    def _get_default_hdf(self):
        """
        use IPTS and Exp to determine
        :return:
        """
        try:
            ipts_number = parse_integer(self.parent.ui.lineEdit_iptsNumber)
            exp_number = parse_integer(self.parent.ui.lineEdit_expNumber)
        except RuntimeError as run_err:
            pop_message(self, 'Unable to parse IPTS or Exp due to {0}'.format(run_err))
            return None

        # Locate default saved HidraProject data
        archive_data = hb2b_utilities.get_hb2b_raw_data(ipts_number, exp_number)

        return archive_data

    def load_hidra_file(self, hydra_project_file=None):
        """ Load Hidra project file
        :return: None
        """

        # load file
        try:
            self.parent._project_name = os.path.basename(hydra_project_file).split('.')[0]
            self.parent._core.load_hidra_project(hydra_project_file,
                                                 project_name=self.parent._project_name,
                                                 load_detector_counts=False,
                                                 load_diffraction=True)
            # Record data key and next
            self.parent._curr_file_name = hydra_project_file
        except (RuntimeError, TypeError) as run_err:
            pop_message(self, 'Unable to load {}'.format(hydra_project_file),
                        detailed_message=str(run_err),
                        message_type='error')

        # Edit information on the UI for user to visualize
        self.parent.ui.label_loadedFileInfo.setText('Loaded {}; Project name: {}'
                                                    .format(hydra_project_file, self.parent._project_name))

        # Get the range of sub runs
        sub_run_list = self.parent._core.reduction_service.get_sub_runs(self.parent._project_name)
        self.parent.ui.label_logIndexMin.setText(str(sub_run_list[0]))
        self.parent.ui.label_logIndexMax.setText(str(sub_run_list[-1]))

        # Set the widgets about viewer: get the sample logs and add the combo boxes for plotting
        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name,
                                                                                     can_plot=True)
        self._set_sample_logs_for_plotting(sample_log_names)

        # plot first peak for default peak range
        self.parent.ui.lineEdit_scanNumbers.setText('1')

        o_plot = Plot(parent=self.parent)
        o_plot.plot_diff_data(plot_model=False)

        # reset the plot
        self.parent.ui.graphicsView_fitResult.reset_viewer()

        # Set the table
        if self.parent.ui.tableView_fitSummary.rowCount() > 0:
            self.parent.ui.tableView_fitSummary.remove_all_rows()
        self.parent.ui.tableView_fitSummary.init_exp(sub_run_list)

        # try:
        #     # Auto fit for all the peaks
        #     if self.parent.ui.checkBox_autoFit.isChecked():
        #         o_fit = Fit(parent=self.parent)
        #         o_fit.fit_peaks(all_sub_runs=True)
        # except (AttributeError) as err:
        #     pop_message(self, 'some errors during fitting!', detailed_message=str(err),
        #                 message_type='warning')

        # enabled all fitting widgets
        o_gui = GuiUtilities(parent=self.parent)
        o_gui.enabled_fitting_widgets(True)

    def _set_sample_logs_for_plotting(self, sample_log_names):
        """ There are 2 combo boxes containing sample logs' names for plotting.  Clear the existing ones
        and add the sample log names specified to them
        :param sample_log_names:
        :return:
        """
        self.parent._sample_log_names_mutex = True
        self.parent.ui.comboBox_xaxisNames.clear()
        self.parent.ui.comboBox_yaxisNames.clear()

        # Maintain a copy of sample logs!
        self.parent._sample_log_names = list(set(sample_log_names))
        self.parent._sample_log_names.sort()

        for sample_log in sample_log_names:
            self.parent.ui.comboBox_xaxisNames.addItem(sample_log)
            self.parent.ui.comboBox_yaxisNames.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)
        self.parent._sample_log_names_mutex = False


