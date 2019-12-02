import numpy as np

from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import pop_message


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
