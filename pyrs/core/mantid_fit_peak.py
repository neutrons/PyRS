# Peak fitting engine by calling mantid
from mantid.simpleapi import FitPeaks, CreateWorkspace
import rshelper
import numpy as np


class MantidPeakFitEngine(object):
    """
    peak fitting engine class for mantid
    """
    def __init__(self, data_set_list, ref_id):
        """
        initialization
        :param data_set_list:
        :param ref_id:
        :param
        """
        # check
        # TODO rshelper.check_list('Data set list', data_set_list)
        rshelper.check_string_variable('Peak fitting reference ID', ref_id)

        self._workspace_name = self._get_matrix_name(ref_id)
        self._data_workspace = self.generate_matrix_workspace(data_set_list)

        return

    def _get_matrix_name(self, ref_id):
        # TODO
        return 'vulcan_test'

    @staticmethod
    def generate_matrix_workspace(data_set_list):
        # TODO
        vec_x_list = list()
        vec_y_list = list()
        for index in range(len(data_set_list)):
            diff_data = data_set_list[index]
            vec_x = diff_data[0]
            vec_y = diff_data[1]
            vec_x_list.append(vec_x)
            vec_y_list.append(vec_y)

        datax = np.concatenate(vec_x_list, axis=0)
        datay = np.concatenate(vec_y_list, axis=0)
        ws_full = CreateWorkspace(DataX=datax, DataY=datay, NSpec=len(vec_x_list))

        return ws_full

    def fit_peaks(self, peak_function_name, background_function_name, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :return:
        """
        rshelper.check_string_variable('Peak function name', peak_function_name)
        rshelper.check_string_variable('Background function name', background_function_name)
        if scan_index is not None:
            rshelper.check_int_variable('Scan (log) index', scan_index, value_range=[0, self.get_number_scans()])
            start = scan_index
            stop = scan_index
        else:
            start = 0
            stop = self.get_number_scans() - 1

        # fit
        # results = FitPeaks(InputWorkspace=self._workspace_name, OutputWorkspace=self._output_name,
        #                    StartWorkspaceIndex=start, StopWorkspaceIndex=stop,
        #                    PeakFunction=peak_function_name, BackgroundType=background_function_name,
        #                    PeakCenters=observed_peak_centers)

        print ('[DB...BAT] Data workspace # spec = {0}'.format(self._data_workspace.getNumberHistograms()))

        r = FitPeaks(InputWorkspace=self._data_workspace,
                 OutputWorkspace='full_fitted', PeakCenters='82', PeakFunction='Gaussian',
                 StartWorkspaceIndex=start, StopWorkspaceIndex=stop,
                 BackgroundType='Linear',
                 PositionTolerance=3, OutputPeakParametersWorkspace='param_m',
                 FittedPeaksWorkspace='model_full',
                 FitWindowBoundaryList='79, 85')

        # process output
        # TODO: Clean!
        self.peak_pos_ws = r[0]
        self.func_param_ws = r[1]
        self.fitted_ws = r[2]



        return

    def get_calculated_peak(self, log_index):
        # TODO
        vec_x = self.fitted_ws.readX(log_index)
        vec_y = self.fitted_ws.readY(log_index)

        return vec_x, vec_y

    def get_function_parameter_names(self):
        # TODO
        return self.func_param_ws.getColumnNames()

    def get_number_scans(self):
        """
        get number of scans in input data to fit
        :return:
        """
        if self._data_workspace is None:
            raise RuntimeError('No data is set up!')

        return self._data_workspace.getNumberHistograms()

    def get_fitted_params(self, param_name):
        """

        :return:
        """
        col_names = self.func_param_ws.getColumnNames()
        if param_name in col_names:
            col_index = col_names.index(param_name)
        else:
            raise RuntimeError('Function parameter {0} does not exist.'.format(param_name))

        param_vec = np.ndarray(shape=(self.func_param_ws.rowCount()), dtype='float')
        for row_index in range(self.func_param_ws.rowCount()):
            param_vec[row_index] = self.func_param_ws.cell(row_index, col_index)

        return param_vec




