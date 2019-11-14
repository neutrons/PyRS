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
        scan_log_index_list = parse_integers(str(self.parent.ui.lineEdit_scanNumbers.text()))
        if len(scan_log_index_list) == 0:
            pop_message(self, 'There is not scan-log index input', 'error')

        # possibly clean the previous
        # keep_prev = self.ui.checkBox_keepPrevPlot.isChecked()
        # if keep_prev is False:
        self.parent._ui_graphicsView_fitSetup.reset_viewer()

        # get data and plot
        err_msg = ''
        plot_model = len(scan_log_index_list) == 1 and plot_model
        for scan_log_index in scan_log_index_list:
            try:
                self.plot_diff_and_fitted_data(scan_log_index, plot_model)
            except RuntimeError as run_err:
                err_msg += '{0}\n'.format(run_err)
        # END-FOR

        if len(err_msg) > 0:
            pop_message(self, err_msg, message_type='error')

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
        diff_data_set = self.parent._core.get_diffraction_data(session_name=self.parent._project_name,
                                                               sub_run=sub_run_number,
                                                               mask=None)

        data_set_label = 'Scan {0}'.format(sub_run_number)

        # Plot experimental data
        self.parent._ui_graphicsView_fitSetup.plot_experiment_data(diff_data_set=diff_data_set,
                                                                   data_reference=data_set_label)

        # Plot fitted model data
        if plot_model:
            model_data_set = self.parent._core.get_modeled_data(session_name=self.parent._project_name,
                                                                sub_run=sub_run_number)
            if model_data_set is not None:
                residual_y_vec = diff_data_set[1] - model_data_set[1]
                residual_data_set = [diff_data_set[0], residual_y_vec]
                self.parent._ui_graphicsView_fitSetup.plot_model_data(diff_data_set=model_data_set,
                                                                      model_label='',
                                                                      residual_set=residual_data_set)

    def plot_scan(self, is_next=True):
        """ plot the next or previous scan (log index)
        """
        scan_log_index_list = parse_integers(str(self.parent.ui.lineEdit_scanNumbers.text()))
        if len(scan_log_index_list) == 0:
            pop_message(self, 'There is not scan-log index input', 'error')
        elif len(scan_log_index_list) > 1:
            pop_message(self, 'There are too many scans for "next"', 'error')
        elif scan_log_index_list[0] == (int(self.parent.ui.label_logIndexMax.text()) if is_next else 0):
            # if we are trying to plot the next, we check relative to the last_log_index, otherwise 0
            return

        coeff = 1 if is_next else -1
        scan_log = scan_log_index_list[0] + coeff
        try:
            self.parent._ui_graphicsView_fitSetup.reset_viewer()
            self.plot_diff_and_fitted_data(scan_log, True)
        except RuntimeError as run_err:
            mess = "next" if is_next else "previous"
            err_msg = 'Unable to plot {} scan {} due to {}'.format(mess, scan_log, run_err)
            pop_message(self, err_msg, message_type='error')
        else:
            self.parent.ui.lineEdit_scanNumbers.setText('{}'.format(scan_log))


    def plot_next_scan(self):
        """ plot the next scan (log index)
        It is assumed that al the scan log indexes are consecutive
        """
        self.plot_scan(is_next=True)
        # scan_log_index_list = parse_integers(str(self.parent.ui.lineEdit_scanNumbers.text()))
        # last_log_index = int(self.parent.ui.label_logIndexMax.text())
        # if len(scan_log_index_list) == 0:
        #     pop_message(self, 'There is not scan-log index input', 'error')
        # elif len(scan_log_index_list) > 1:
        #     pop_message(self, 'There are too many scans for "next"', 'error')
        # elif scan_log_index_list[0] == last_log_index:
        #     # last log index: no operation
        #     return
        #
        # next_scan_log = scan_log_index_list[0] + 1
        # try:
        #     self.parent._ui_graphicsView_fitSetup.reset_viewer()
        #     self.plot_diff_and_fitted_data(next_scan_log, True)
        # except RuntimeError as run_err:
        #     # self.plot_diff_data(next_scan_log - 1, True)
        #     err_msg = 'Unable to plot next scan {} due to {}'.format(next_scan_log, run_err)
        #     pop_message(self, err_msg, message_type='error')
        # else:
        #     self.parent.ui.lineEdit_scanNumbers.setText('{}'.format(next_scan_log))

    def plot_prev_scan(self):
        """ plot the previous scan (log index)
        It is assumed that al the scan log indexes are consecutive
        """
        self.plot_scan(is_next=False)
        # scan_log_index_list = parse_integers(str(self.parent.ui.lineEdit_scanNumbers.text()))
        # if len(scan_log_index_list) == 0:
        #     pop_message(self, 'There is not scan-log index input', 'error')
        # elif len(scan_log_index_list) > 1:
        #     pop_message(self, 'There are too many scans for "next"', 'error')
        # elif scan_log_index_list[0] == 0:
        #     # first one: no operation
        #     return
        #
        # prev_scan_log_index = scan_log_index_list[0] - 1
        # try:
        #     self.parent._ui_graphicsView_fitSetup.reset_viewer()
        #     self.plot_diff_and_fitted_data(prev_scan_log_index, True)
        # except RuntimeError as run_err:
        #     # self.plot_diff_data(next_scan_log + 1, True)
        #     err_msg = 'Unable to plot previous scan {} due to {}'.format(prev_scan_log_index, run_err)
        #     pop_message(self, err_msg, message_type='error')
        # else:
        #     self.parent.ui.lineEdit_scanNumbers.setText('{}'.format(prev_scan_log_index))
        # return
