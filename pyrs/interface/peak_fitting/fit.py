from pyrs.interface.peak_fitting.utilities import Utilities
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.gui_helper import pop_message
from pyrs.utilities.rs_project_file import HidraConstants


class Fit:

    def __init__(self, parent=None):
        self.parent = parent

    def fit_peaks(self, all_sub_runs=False):
        """ Fit peaks either all peaks or selected peaks
        The workflow includes
        1. parse sub runs, peak and background type
        2. fit peaks
        3. show the fitting result in table
        4. plot the peak in first sub runs that is fit
        :param all_sub_runs: Flag to fit peaks of all sub runs without checking
        :return:
        """
        # Get the sub runs to fit or all the peaks
        if self.parent.ui.checkBox_fitSubRuns.isChecked() and not all_sub_runs:
            # Parse sub runs specified in lineEdit_scanNumbers
            o_utilities = Utilities(parent=self.parent)
            sub_run_list = o_utilities.parse_sub_runs()
        else:
            sub_run_list = None

        # Get peak function and background function
        peak_function = str(self.parent.ui.comboBox_peakType.currentText())
        bkgd_function = str(self.parent.ui.comboBox_backgroundType.currentText())

        # Get peak fitting range from the range of figure
        fit_range = self.parent._ui_graphicsView_fitSetup.get_x_limit()
        print('[INFO] Peak fit range: {0}'.format(fit_range))

        # Fit Peaks: It is better to fit all the peaks at the same time after testing
        guessed_peak_center = 0.5 * (fit_range[0] + fit_range[1])
        peak_info_dict = {'Peak 1': {'Center': guessed_peak_center, 'Range': fit_range}}
        self.parent._core.fit_peaks(self.parent._project_name, sub_run_list,
                                    peak_type=peak_function,
                                    background_type=bkgd_function,
                                    peaks_fitting_setup=peak_info_dict)

        # Process fitted peaks
        # TEST - #84 - This shall be reviewed!
        try:
            function_params, fit_values = self.parent._core.get_peak_fitting_result(self.parent._project_name,
                                                                                    return_format=dict,
                                                                                    effective_parameter=False)
        except AttributeError as err:
            pop_message(self, 'Zoom in/out to only show peak to fit!', err, "error")
            return

        # TODO - #84+ - Need to implement the option as effective_parameter=True

        print('[DB...BAT...FITWINDOW....FIT] returned = {}, {}'.format(function_params, fit_values))

        self.parent._sample_log_names_mutex = True
        curr_x_index = self.parent.ui.comboBox_xaxisNames.currentIndex()
        curr_y_index = self.parent.ui.comboBox_yaxisNames.currentIndex()
        # add fitted parameters by resetting and build from the copy of fit parameters
        self.parent.ui.comboBox_xaxisNames.clear()
        self.parent.ui.comboBox_yaxisNames.clear()
        # add sample logs (names)
        for sample_log_name in self.parent._sample_log_names:
            self.parent.ui.comboBox_xaxisNames.addItem(sample_log_name)
            self.parent.ui.comboBox_yaxisNames.addItem(sample_log_name)
        # add function parameters (names)
        for param_name in function_params:
            self.parent.ui.comboBox_xaxisNames.addItem(param_name)
            self.parent.ui.comboBox_yaxisNames.addItem(param_name)
            self.parent._function_param_name_set.add(param_name)
        # add observed parameters
        self.parent.ui.comboBox_xaxisNames.addItem('Center of mass')
        self.parent.ui.comboBox_yaxisNames.addItem('Center of mass')
        # keep current selected item unchanged

        # log index and center of mass
        size_x = len(self.parent._sample_log_names) + len(self.parent._function_param_name_set) + 2

        # center of mass
        size_y = len(self.parent._sample_log_names) + len(self.parent._function_param_name_set) + 1

        if curr_x_index < size_x:
            # keep current set up
            self.parent.ui.comboBox_xaxisNames.setCurrentIndex(curr_x_index)
        else:
            # original one does not exist: reset to first/log index
            self.parent.ui.comboBox_xaxisNames.setCurrentIndex(0)

        # release the mutex: because re-plot is required anyway
        self.parent._sample_log_names_mutex = False

        # plot Y
        if curr_y_index >= size_y:
            # out of boundary: use the last one (usually something about peak)
            curr_y_index = size_y - 1
        self.parent.ui.comboBox_yaxisNames.setCurrentIndex(curr_y_index)

        # Show fitting result in Table
        # TODO - could add an option to show native or effective peak parameters
        self.show_fit_result_table(peak_function, function_params, fit_values, is_effective=False)

        # plot the model and difference
        if sub_run_list is None:
            o_plot = Plot(parent=self.parent)
            o_plot.plot_diff_and_fitted_data(1, True)

    def show_fit_result_table(self, peak_function, peak_param_names, peak_param_dict, is_effective):
        """ Set up the table containing fit result
        :param peak_function: name of peak function
        :param peak_param_names: name of the parameters for the order of columns
        :param peak_param_dict: parameter names
        :param is_effective: Flag for the parameter to be shown as effective (True) or native (False)
        :return:
        """
        # Add peaks' centers of mass to the output table
        peak_param_names.append(HidraConstants.PEAK_COM)
        com_vec = self.parent._core.get_peak_center_of_mass(self.parent._project_name)
        peak_param_dict[HidraConstants.PEAK_COM] = com_vec

        # Initialize the table by resetting the column names
        self.parent.ui.tableView_fitSummary.reset_table(peak_param_names)

        # Get sub runs for rows in the table
        sub_run_vec = peak_param_dict[HidraConstants.SUB_RUNS]

        # Add rows to the table for parameter information
        for row_index in range(sub_run_vec.shape[0]):
            # Set fit result
            self.parent.ui.tableView_fitSummary.set_fit_summary(row_index, peak_param_names,
                                                                peak_param_dict,
                                                                write_error=False,
                                                                peak_profile=peak_function)