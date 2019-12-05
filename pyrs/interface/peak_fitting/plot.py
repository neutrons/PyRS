import numpy as np

from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import pop_message
from pyrs.utilities.rs_project_file import HidraConstants

from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT


class Plot:

    def __init__(self, parent=None):
        self.parent = parent

    def plot_diff_data(self, plot_model=True):
        """
        plot diffraction data
        :return:
        """
        # gather the information
        try:
            scan_log_index_list = parse_integers(str(self.parent.ui.lineEdit_listSubRuns.text()))
        except RuntimeError as run_err:
            pop_message(self, "Unable to parse the string", message_type='error')

        if len(scan_log_index_list) == 0:
            pop_message(self, 'There is not scan-log index input', 'error')

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self.parent._ui_graphicsView_fitSetup.reset_viewer()

        if len(scan_log_index_list) == 1:
            self.plot_scan(value=np.int(scan_log_index_list[0]))
            return

        # get data and plot
        err_msg = ''
        plot_model = len(scan_log_index_list) == 1 and plot_model
        for scan_log_index in scan_log_index_list:
            try:
                self.plot_diff_and_fitted_data(scan_log_index, plot_model)
            except RuntimeError as run_err:
                err_msg += '{0}\n'.format(run_err)

        if len(err_msg) > 0:
            pop_message(self, err_msg, message_type='error')

    def reset_fitting_plot(self):
        """reset the fitting plots"""
        self.parent.ui.graphicsView_fitResult.reset_viewer()

    def plot_diff_and_fitted_data(self, sub_run_number, plot_model):
        """Plot a set of diffraction data (one scan log index) and plot its fitted data

        Parameters
        ----------
        sub_run_number: int
            sub run number
        plot_model: boolean
            Flag to plot model with diffraction data or not

        Returns
        -------
        None
        """
        # get experimental data and plot
        # diff data set is [x_axis, y axis]
        diff_data_set = self.parent._core.get_diffraction_data(session_name=self.parent._project_name,
                                                               sub_run=sub_run_number,
                                                               mask=None)

        data_set_label = 'Scan {0}'.format(sub_run_number)

        # Plot experimental data
        self.parent._ui_graphicsView_fitSetup.plot_experiment_data(diff_data_set=diff_data_set,
                                                                   data_reference=data_set_label)

        # Plot fitted model data
        model_data_set = None
        if plot_model:
            model_data_set = self.parent._core.get_modeled_data(session_name=self.parent._project_name,
                                                                sub_run=sub_run_number)

        if model_data_set is not None:
            residual_y_vec = diff_data_set[1] - model_data_set[1]
            residual_data_set = [diff_data_set[0], residual_y_vec]
            self.parent._ui_graphicsView_fitSetup.plot_model_data(diff_data_set=model_data_set,
                                                                  model_label='fit',
                                                                  residual_set=residual_data_set)

    def plot_scan(self, value=None):
        """ plot the scan defined by the scroll bar or the text line according to radio button selected
        """
        if (value is None):
            scan_value = self.parent.ui.horizontalScrollBar_SubRuns.value()
        else:
            scan_value = value

        try:
            self.parent._ui_graphicsView_fitSetup.reset_viewer()
            self.plot_diff_and_fitted_data(scan_value, True)
        except RuntimeError:
            pass

        self.parent.ui.label_SubRunsValue.setText('{}'.format(scan_value))

    def plot_1d(self):

        self.parent.ui.graphicsView_fitResult.reset_viewer()

        # get the sample log/meta data name
        x_axis_name = str(self.parent.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.parent.ui.comboBox_yaxisNames.currentText())

        print("x_axis_name: " + LIST_AXIS_TO_PLOT['full'][x_axis_name])
        print("y_axis_name: " + LIST_AXIS_TO_PLOT['full'][y_axis_name])

        print("self.parent._function_param_name_set): {}".format(self.parent._function_param_name_set))
        print("HidraConstants.SUB_RUNS: {}".format(HidraConstants.SUB_RUNS))

        param_names, param_data = self.parent._core.get_peak_fitting_result(self.parent._project_name,
                                                                            return_format=dict,
                                                                            effective_parameter=False)


        return

        # Return if sample logs combo box not set
        if x_axis_name == '' and y_axis_name == '':
            return

        if x_axis_name in self.parent._function_param_name_set and y_axis_name == HidraConstants.SUB_RUNS:
            vec_y, vec_x = self.get_function_parameter_data(x_axis_name)
        elif y_axis_name in self.parent._function_param_name_set and x_axis_name == HidraConstants.SUB_RUNS:
            vec_x, vec_y = self.get_function_parameter_data(y_axis_name)
        elif x_axis_name in self.parent._function_param_name_set or y_axis_name in \
                self.parent._function_param_name_set:
            pop_message(self, 'It has not considered how to plot 2 function parameters '
                              '{} and {} against each other'
                              ''.format(x_axis_name, y_axis_name),
                              message_type='error')
            return
        else:
            vec_x = self.get_meta_sample_data(x_axis_name)
            vec_y = self.get_meta_sample_data(y_axis_name)
        # END-IF-ELSE

        if vec_x is None or vec_y is None:
            raise RuntimeError('{} or {} cannot be None ({}, {})'
                               ''.format(x_axis_name, y_axis_name, vec_x, vec_y))

        self.parent.ui.graphicsView_fitResult.plot_scatter(vec_x, vec_y, x_axis_name, y_axis_name)

    def get_function_parameter_data(self, param_name):
        """ get the parameter function data
        :param param_name:
        :return:
        """
        # get data key
        if self.parent._project_name is None:
            pop_message(self, 'No data loaded', 'error')
            return

        param_names, param_data = self.parent._core.get_peak_fitting_result(self.parent._project_name,
                                                                            return_format=dict,
                                                                            effective_parameter=False)

        print('[DB...BAT] Param Names: {}'.format(param_names))
        sub_run_vec = param_data[HidraConstants.SUB_RUNS]
        param_value_2darray = param_data[param_name]
        print('[DB...BAT] 2D array shape: {}'.format(param_value_2darray.shape))

        return sub_run_vec, param_value_2darray[:, 0]

    def get_meta_sample_data(self, name):
        """
        get meta data to plot.
        the meta data can contain sample log data and fitted peak parameters
        :param name:
        :return:
        """
        # get data key
        if self.parent._project_name is None:
            pop_message(self, 'No data loaded', 'error')
            return

        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name, True)

        if name == HidraConstants.SUB_RUNS:
            # sub run vector
            value_vector = np.array(self.parent._core.reduction_service.get_sub_runs(self.parent._project_name))
        elif name in sample_log_names:
            # sample log but not sub-runs
            value_vector = self.parent._core.reduction_service.get_sample_log_value(self.parent._project_name, name)
        elif name == 'Center of mass':
            # center of mass is different????
            # TODO - #84 - Make sure of it!
            value_vector = self.parent._core.get_peak_center_of_mass(self.parent._project_name)
        else:
            value_vector = None

        return value_vector
