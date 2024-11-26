import numpy as np
from matplotlib.colormaps import coolwarm
from matplotlib.pyplot import Normalize
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.interpolate import griddata

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
            self.plot_scan(value=np.int32(scan_log_index_list[0]))
            return

        # get data and plot
        err_msg = ''
        plot_model = len(scan_log_index_list) == 1 and plot_model
        for scan_log_index in scan_log_index_list:
            try:
                self.plot_diff_and_fitted_data(scan_log_index)
            except RuntimeError as run_err:
                err_msg += '{0}\n'.format(run_err)

        if len(err_msg) > 0:
            pop_message(self, err_msg, message_type='error')

    def reset_fitting_plot(self):
        """reset the fitting plots"""
        self.parent.ui.graphicsView_fitResult.reset_viewer()

    def plot_diff_and_fitted_data(self, sub_run_number):
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
            self.plot_diff_and_fitted_data(sub_run)
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

        colors = None
        plot_scatter = False

        o_gui = GuiUtilities(parent=self.parent)
        x_axis_name = str(self.parent.ui.comboBox_xaxisNames_2dplot.currentText())
        y_axis_name = str(self.parent.ui.comboBox_yaxisNames_2dplot.currentText())
        z_axis_name = str(self.parent.ui.comboBox_zaxisNames_2dplot.currentText())

        x_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='x')
        y_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='y')
        z_axis_peak_index = o_gui.get_plot2d_axis_peak_label_index(axis='z')

        o_data_retriever = DataRetriever(parent=self.parent)

        try:
            axis_x_data, axis_x_error = o_data_retriever.get_data(name=x_axis_name, peak_index=x_axis_peak_index)
            axis_y_data, axis_y_error = o_data_retriever.get_data(name=y_axis_name, peak_index=y_axis_peak_index)
            axis_z_data, axis_z_error = o_data_retriever.get_data(name=z_axis_name, peak_index=z_axis_peak_index)

            if sub_run_list is not None:
                sub_run_list = self.parse_sub_run_list(sub_run_list, len(axis_x_data))
                axis_x_data = axis_x_data[sub_run_list]
                axis_y_data = axis_y_data[sub_run_list]
                axis_z_data = axis_z_data[sub_run_list]

            if ((axis_x_data.size == np.unique(axis_x_data).size) or
                    (axis_x_data.size == np.unique(axis_x_data).size)):

                plot_scatter = True

            if self.parent.ui.radioButton_contour.isChecked():
                vec_x, vec_y = np.meshgrid(np.unique(axis_x_data), np.unique(axis_y_data))
                vec_z = griddata(((axis_x_data, axis_y_data)), axis_z_data, (vec_x, vec_y), method='nearest')

            elif self.parent.ui.radioButton_3dline.isChecked():

                vec_x, vec_y = np.meshgrid(np.unique(axis_x_data), np.unique(axis_y_data))
                vec_z = griddata(((axis_x_data, axis_y_data)), axis_z_data, (vec_x, vec_y), method='nearest')

                norm = Normalize(vec_z.min(), vec_z.max())
                colors = coolwarm(norm(vec_z))

            else:
                plot_scatter = True
                try:
                    norm = Normalize(axis_z_data.min(), axis_z_data.max())
                    colors = coolwarm(norm(axis_z_data))
                except ValueError:
                    colors = None

                vec_x = np.copy(axis_x_data)
                vec_y = np.copy(axis_y_data)
                vec_z = np.copy(axis_z_data)

            self.parent.ui.graphicsView_plot2D.plot_3D_scatter(vec_x, vec_y, vec_z, plot_scatter, colors=colors,
                                                               x_label=x_axis_name,
                                                               y_label=y_axis_name,
                                                               z_label=z_axis_name)
        except RuntimeError:
            self.parent.ui.graphicsView_plot2D.reset_viewer()

    def plot_1d(self):

        self.parent.ui.graphicsView_fitResult.reset_viewer()

        # get the sample log/meta data name
        o_gui = GuiUtilities(parent=self.parent)
        x_axis_name = str(self.parent.ui.comboBox_xaxisNames.currentText())
        y_axis_name = str(self.parent.ui.comboBox_yaxisNames.currentText())
        x_axis_peak_index = o_gui.get_plot1d_axis_peak_label_index(is_xaxis=True)
        y_axis_peak_index = o_gui.get_plot1d_axis_peak_label_index(is_xaxis=False)

        o_data_retriever = DataRetriever(parent=self.parent)

        is_plot_with_error = True

        axis_x_data, axis_x_error = o_data_retriever.get_data(name=x_axis_name, peak_index=x_axis_peak_index)
        axis_y_data, axis_y_error = o_data_retriever.get_data(name=y_axis_name, peak_index=y_axis_peak_index)

        print(axis_x_data)
        print(axis_y_data)
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
