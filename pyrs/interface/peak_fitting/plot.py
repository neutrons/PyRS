import numpy as np

from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import pop_message
from pyrs.dataobjects import HidraConstants
from pyrs.interface.peak_fitting.config import fit_dict

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
            return

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

        # Plot experimental data
        data_set_label = 'Scan {0}'.format(sub_run_number)
        self.parent._ui_graphicsView_fitSetup.plot_experiment_data(diff_data_set=diff_data_set,
                                                                   data_reference=data_set_label)

        # plot fitted data
        fit_result = self.parent.fit_result
        if fit_result:
            sub_run_index = int(self.parent.fit_result.peakcollections[0].sub_runs.get_indices(sub_run_number)[0])
            x_array = self.parent.fit_result.fitted.readX(sub_run_index)
            y_array = self.parent.fit_result.fitted.readY(sub_run_index)
            self.parent._ui_graphicsView_fitSetup.plot_fitted_data(x_array, y_array)

        # # Plot fitted model data
        # model_data_set = None
        # if plot_model:
        #     model_data_set = self.parent._core.get_modeled_data(session_name=self.parent._project_name,
        #                                                         sub_run=sub_run_number)
        #
        # if model_data_set is not None:
        #     residual_y_vec = diff_data_set[1] - model_data_set[1]
        #     residual_data_set = [diff_data_set[0], residual_y_vec]
        #     self.parent._ui_graphicsView_fitSetup.plot_model_data(diff_data_set=model_data_set,
        #                                                           model_label='fit',
        #                                                           residual_set=residual_data_set)

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
        # x_axis_peak_index = self.parent.ui.plot1d_xaxis_peak_label_comboBox.currentIndex()
        y_axis_peak_index = self.parent.ui.plot1d_yaxis_peak_label_comboBox.currentIndex()

        hidra_workspace = self.parent.hidra_workspace
        if x_axis_name == 'Sub-runs':
            axis_x = np.array(hidra_workspace.get_sub_runs())
            if y_axis_name == 'Sub-runs':
                axis_y = np.array(hidra_workspace.get_sub_runs())
            elif y_axis_name in LIST_AXIS_TO_PLOT['raw'].keys():
                axis_y = hidra_workspace._sample_logs[y_axis_name]
            elif y_axis_name in LIST_AXIS_TO_PLOT['fit'].keys():
                value, error = self.get_fitted_value(peak=self.parent.fit_result.peakcollections[y_axis_peak_index],
                                                     value_to_display=y_axis_name)
                self.parent.ui.graphicsView_fitResult.plot_scatter_with_errors(vec_x=axis_x, vec_y=value,
                                                                               vec_y_error=error,
                                                                               x_label='sub_runs',
                                                                               y_label=y_axis_name)
                return
            else:
                raise NotImplementedError("y_axis choice not supported yet: {}".format(y_axis_name))
        elif x_axis_name in LIST_AXIS_TO_PLOT['raw'].keys():
            axis_x = hidra_workspace._sample_logs[x_axis_name]
            if y_axis_name == 'Sub-runs':
                axis_y = np.array(hidra_workspace.get_sub_runs())
            elif y_axis_name in LIST_AXIS_TO_PLOT['raw'].keys():
                axis_y = hidra_workspace._sample_logs[y_axis_name]
            elif y_axis_name in LIST_AXIS_TO_PLOT['fit'].keys():
                pass
            else:
                raise NotImplementedError("y_axis choice not supported yet: {}!".format(y_axis_name))
        elif x_axis_name in LIST_AXIS_TO_PLOT['fit'].keys():
            if y_axis_name == 'Sub-runs':
                axis_y = np.array(hidra_workspace.get_sub_runs())
            elif y_axis_name in LIST_AXIS_TO_PLOT['raw'].keys():
                axis_y = hidra_workspace._sample_logs[y_axis_name]
            elif y_axis_name in LIST_AXIS_TO_PLOT['fit'].keys():
                pass
            else:
                raise NotImplementedError("y_axis choice not supported yet: {}!".format(y_axis_name))
        else:
            raise NotImplementedError("x_axis choice not supported yet: {}!".format(x_axis_name))

        self.parent.ui.graphicsView_fitResult.plot_scatter(axis_x, axis_y,
                                                           'sub_runs', y_axis_name)

    def get_fitted_value(self, peak=None, value_to_display='Center'):
        """
        return the values and errors of the fitted parameters of the given peak
        :param peak:
        :param value_to_display:
        :return:
        """
        value, error = peak.get_effective_params()
        mantid_value_to_display = fit_dict[value_to_display]
        value_selected = value[mantid_value_to_display]
        error_selected = error[mantid_value_to_display]
        return value_selected, error_selected

    def get_function_parameter_data(self, param_name):
        """ get the parameter function data
        :param param_name:
        :return:
        """

        print(param_name)

        # get data key
        if self.parent._project_name is None:
            pop_message(self, 'No data loaded', 'error')
            return

        # fitted_peak = self.parent._core._peak_fitting_dict[self.parent._project_name]
        # from pyrs.core import peak_profile_utility

        # param_set = fitted_peak.get_effective_parameters_values()
        # eff_params_list, sub_run_array, fit_cost_array, eff_param_value_array, eff_param_error_array = param_set

        # retrieve Center: EFFECTIVE_PEAK_PARAMETERS = ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
        # i_center = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS.index('Center')
        # centers = eff_param_value_array[i_center]

        # retrieve Height
        # i_height = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS.index('Height')
        # heights = eff_param_value_array[i_height]

        # retrieve intensity
        # i_intensity = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS.index('Intensity')
        # intensities = eff_param_value_array[i_intensity]

        # retrieve FWHM
        # i_fwhm = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS.index('FWHM')
        # fwhms = eff_param_value_array[i_fwhm]

        # import pprint
        # pprint.pprint("fwhms: {}".format(fwhms))

        return

        param_names, param_data = self.parent._core.get_peak_fitting_result(self.parent._project_name,
                                                                            0,
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
