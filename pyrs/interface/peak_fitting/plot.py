from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import pop_message
from pyrs.dataobjects import HidraConstants
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.interface.peak_fitting.data_retriever import DataRetriever

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
        except RuntimeError:
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

    def plot_2d(self):

        o_gui = GuiUtilities(parent=self.parent)
        x_axis_name = str(self.parent.ui.comboBox_xaxisNames_2dplot.currentText())
        y_axis_name = str(self.parent.ui.comboBox_yaxisNames_2dplot.currentText())
        z_axis_name = str(self.parent.ui.comboBox_zaxisNames_2dplot.currentText())

        x_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='x')
        y_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='y')
        z_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='z')

        o_data_retriever = DataRetriever(parent=self.parent)

        axis_x_data, axis_x_error = o_data_retriever.get_data(name=x_axis_name, peak_index=x_axis_peak_index)
        axis_y_data, axis_y_error = o_data_retriever.get_data(name=y_axis_name, peak_index=y_axis_peak_index)
        axis_z_data, axis_z_error = o_data_retriever.get_data(name=z_axis_name, peak_index=z_axis_peak_index)

        str_axis_x_data = [str(_value) for _value in axis_x_data]
        str_axis_y_data = [str(_value) for _value in axis_y_data]
        str_axis_z_data = [str(_value) for _value in axis_z_data]

        debug_dic = {'axis_x_data': str_axis_x_data,
                     'axis_y_data': str_axis_y_data,
                     'axis_z_data': str_axis_z_data}

        array_dict = self.format_3D_axis_data(axis_x=axis_x_data, axis_y=axis_y_data, axis_z=axis_z_data)
        x_axis = array_dict['x_axis']
        y_axis = array_dict['y_axis']
        z_axis = array_dict['z_axis']

        self.parent.ui.graphicsView_plot2D.ax.clear()
        self.parent.ui.graphicsView_plot2D.ax.contourf(x_axis, y_axis, z_axis)
        # plt.colorbar(self.parent.ui.graphicsView_plot2D.ax)
        self.parent.ui.graphicsView_plot2D._myCanvas.draw()

    def format_3D_axis_data(self, axis_x=[], axis_y=[], axis_z=[]):

        set_axis_x_data = set(axis_x)
        set_axis_y_data = set(axis_y)

        size_set_x = len(set_axis_x_data)
        size_set_y = len(set_axis_y_data)

        set_x = list(set_axis_x_data)
        set_y = list(set_axis_y_data)

        set_x.sort()
        set_y.sort()

        array3d = np.zeros((size_set_x, size_set_y), dtype=np.float32).flatten()
        axis_xy_meshgrid = [[_x, _y] for _x in set_x for _y in set_y]
        axis_xy_zip = list(zip(axis_x, axis_y))

        for _xy in axis_xy_meshgrid:
            for _index, _xy_zip in enumerate(axis_xy_zip):
                if np.array_equal(_xy, _xy_zip):
                    array3d[_index] = axis_z[_index]
                    break

        # list_axis_x = list(set(axis_x))
        # list_axis_y = list(set(axis_y))
        #
        # list_axis_x.sort()
        # list_axis_y.sort()
        #
        # size_axis_x = len(list_axis_x)
        # size_axis_y = len(list_axis_y)
        #
        # array_3d = np.zeros((size_axis_x, size_axis_y), dtype=np.float32).flatten()
        # axis_xy_zip = list(zip(axis_x, axis_y))
        # axis_xy_meshgrid = [[_x, _y] for _x in list_axis_x for _y in list_axis_y]
        #
        # list_of_index = []
        # for _xy in axis_xy_meshgrid:
        #     for _index, _xy_zip in enumerate(axis_xy_zip):
        #         if np.array_equal(_xy, _xy_zip):
        #             list_of_index.append(_index)
        #             array_3d[_index] = axis_z[_index]
        #             break

        array_3d = np.reshape(array3d, (size_set_x, size_set_y))
        return {'x_axis': list(set_axis_x_data),
                'y_axis': list(set_axis_y_data),
                'z_axis': np.transpose(array_3d)}

    def plot_1d(self):

        self.parent.ui.graphicsView_fitResult.reset_viewer()

        # get the sample log/meta data name
        o_gui = GuiUtilities(parent=self.parent)
        x_axis_name = str(self.parent.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.parent.ui.comboBox_yaxisNames.currentText())
        x_axis_peak_index = o_gui.get_plot1d_axis_peak_label_index(is_xaxis=True)
        y_axis_peak_index = o_gui.get_plot1d_axis_peak_label_index(is_xaxis=False)

        o_data_retriever = DataRetriever(parent=self.parent)

        is_plot_with_error = False

        axis_x_data, axis_x_error = o_data_retriever.get_data(name=x_axis_name, peak_index=x_axis_peak_index)
        axis_y_data, axis_y_error = o_data_retriever.get_data(name=y_axis_name, peak_index=y_axis_peak_index)

        if ((x_axis_name in LIST_AXIS_TO_PLOT['fit'].keys()) or \
                (y_axis_name in LIST_AXIS_TO_PLOT['fit'].keys())):
            is_plot_with_error = True

        if is_plot_with_error:
            self.parent.ui.graphicsView_fitResult.plot_scatter_with_errors(vec_x=axis_x_data,
                                                                           vec_y=axis_y_data,
                                                                           vec_x_error=axis_x_error,
                                                                           vec_y_error=axis_y_error,
                                                                           x_label=x_axis_name,
                                                                           y_label=y_axis_name)
        else:
            self.parent.ui.graphicsView_fitResult.plot_scatter(axis_x_data,
                                                               axis_y_data,
                                                               x_axis_name,
                                                               y_axis_name)

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
