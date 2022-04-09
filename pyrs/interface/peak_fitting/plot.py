import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import pop_message
from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.interface.peak_fitting.data_retriever import DataRetriever

from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT


class Plot:

    def __init__(self, parent=None):
        self.parent = parent
        self.sub_run_list = self.parent.hidra_workspace._sample_logs._subruns

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

            err_x_array = self.parent.fit_result.difference.readX(sub_run_index)
            err_y_array = self.parent.fit_result.difference.readY(sub_run_index)
            self.parent._ui_graphicsView_fitSetup.plot_fitting_diff_data(x_axis=err_x_array, y_axis=err_y_array)
            # self.parent._ui_graphicsView_fitSetup.plot_model_data(diff_data_set=model_data_set,
            #                                                       model_label='fit',
            #                                                       residual_set=residual_data_set)

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

        # reference sub_run number from list of sub_runs in project file
        sub_run = self.sub_run_list[scan_value - 1]

        try:
            self.parent._ui_graphicsView_fitSetup.reset_viewer()
            self.plot_diff_and_fitted_data(sub_run, True)
        except RuntimeError:
            pass

        self.parent.ui.label_SubRunsValue.setText('{}'.format(sub_run))

    def parse_sub_run_list(self, sub_run_list, num_sub_runs):
        '''
        parse sub_run_list to ensure that the user inputs are within the bounds of the number of data
        Parameters
        ----------
        sub_run_list : list
            list of sublogs input by the user.
        num_sub_runs : int
            number of subruns in the project file.

        Returns
        -------
        sub_run_list : np.array
            array of input sub_runs that has been cropped to exlcude inputs outside of the number of sublogs

        '''
        sub_run_list = np.array(sub_run_list)
        sub_run_index = (sub_run_list >= 1) == (sub_run_list < num_sub_runs)

        return sub_run_list[sub_run_index]

    def plot_2d(self, sub_run_list=None):

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

        if sub_run_list is not None:
            sub_run_list = self.parse_sub_run_list(sub_run_list, len(axis_x_data))
            axis_x_data = axis_x_data[sub_run_list]
            axis_y_data = axis_y_data[sub_run_list]
            axis_z_data = axis_z_data[sub_run_list]

        if not self.parent.ui.radioButton_3dscatter.isChecked():

            array_dict = self.format_3D_axis_data(axis_x=axis_x_data, axis_y=axis_y_data, axis_z=axis_z_data)
            x_axis = array_dict['x_axis']
            y_axis = array_dict['y_axis']
            z_axis = array_dict['z_axis']

            if len(x_axis) < 2:
                return

        self.parent.ui.graphicsView_plot2D.ax.clear()
        if self.parent.ui.graphicsView_plot2D.colorbar:
            self.parent.ui.graphicsView_plot2D.colorbar.remove()
            self.parent.ui.graphicsView_plot2D.colorbar = None

        if self.parent.ui.radioButton_contour.isChecked():

            self.parent.ui.graphicsView_plot2D.ax = self.parent.ui.graphicsView_plot2D.figure.gca()
            my_plot = self.parent.ui.graphicsView_plot2D.ax.contourf(x_axis, y_axis, z_axis)

            self.parent.ui.graphicsView_plot2D.colorbar = \
                self.parent.ui.graphicsView_plot2D.figure.colorbar(my_plot)
            self.parent.ui.graphicsView_plot2D._myCanvas.draw()

            self.parent.ui.graphicsView_plot2D.ax.set_xlabel(x_axis_name)
            self.parent.ui.graphicsView_plot2D.ax.set_ylabel(y_axis_name)

        elif self.parent.ui.radioButton_3dline.isChecked():

            x, y = np.meshgrid(x_axis, y_axis)

            self.parent.ui.graphicsView_plot2D.ax = self.parent.ui.graphicsView_plot2D.figure.gca(projection='3d')
            my_plot = self.parent.ui.graphicsView_plot2D.ax.plot_surface(x, y, z_axis,
                                                                         cmap=cm.coolwarm,
                                                                         linewidth=0,
                                                                         antialiased=False)
            self.parent.ui.graphicsView_plot2D.colorbar = self.parent.ui.graphicsView_plot2D.figure.colorbar(my_plot)
            self.parent.ui.graphicsView_plot2D.ax.set_xlabel(x_axis_name)
            self.parent.ui.graphicsView_plot2D.ax.set_ylabel(y_axis_name)
            self.parent.ui.graphicsView_plot2D.ax.set_zlabel(z_axis_name)

            # maybe look at projections on the wall (matplotlib.org/gallery/mplot3d/contour3d_3.html)
            self.parent.ui.graphicsView_plot2D.ax.contour(x, y, z_axis, zdir='z', offset=np.nanmin(z_axis),
                                                          cmap=cm.coolwarm)
            self.parent.ui.graphicsView_plot2D.ax.contour(x, y, z_axis, zdir='y', offset=np.nanmax(y),
                                                          cmap=cm.coolwarm)
            self.parent.ui.graphicsView_plot2D.ax.contour(x, y, z_axis, zdir='x', offset=np.nanmax(x),
                                                          cmap=cm.coolwarm)

            self.parent.ui.graphicsView_plot2D._myCanvas.draw()

        else:

            self.parent.ui.graphicsView_plot2D.ax = self.parent.ui.graphicsView_plot2D.figure.gca(projection='3d')

            # only try plotting if the figure height is larger than 0
            fig_size = self.parent.ui.graphicsView_plot2D.figure.get_size_inches()
            if fig_size[1] > 0:
                self.parent.ui.graphicsView_plot2D.ax.scatter(axis_x_data, axis_y_data, axis_z_data)
                self.parent.ui.graphicsView_plot2D._myCanvas.draw()

                self.parent.ui.graphicsView_plot2D.ax.set_xlabel(x_axis_name)
                self.parent.ui.graphicsView_plot2D.ax.set_ylabel(y_axis_name)
                # self.parent.ui.graphicsView_plot2D.ax.set_zlabel(z_axis_name)

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

        if ((x_axis_name in LIST_AXIS_TO_PLOT['fit'].keys()) or
                (y_axis_name in LIST_AXIS_TO_PLOT['fit'].keys())):
            is_plot_with_error = True

        # is_plot_with_error = False  # REMOVE that line once the error bars are correctAT
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
