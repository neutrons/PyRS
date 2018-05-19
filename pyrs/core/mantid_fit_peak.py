# Peak fitting engine by calling mantid
# Set up the testing environment for PyVDrive commands
import os
import sys
home_dir = os.path.expanduser('~')
if home_dir.startswith('/SNS/'):
    # analysis
    sys.path.insert(1, '/opt/mantidnightly/bin/')
from mantid.simpleapi import FitPeaks, CreateWorkspace
from mantid.api import AnalysisDataService
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

    def calculate_center_of_mass(self):
        # TODO

        # alculate center of mass and highest data point

        data_ws = AnalysisDataService.retrieve(self._workspace_name)
        num_spectra = data_ws.getNumberHistograms()

        peak_center_vec = np.ndarray(shape=(num_spectra, 2), dtype='float')

        for iws in range(num_spectra):
            vec_x = data_ws.readX(iws)
            vec_y = data_ws.readY(iws)
            com_i = np.sum(vec_x * vec_y) / np.sum(vec_y)
            peak_center_vec[iws, 0] = com_i
            imax_peak = np.argmax(vec_y, axis=0)
            peak_center_vec[iws, 1] = vec_x[imax_peak]

        # create 2 workspaces
        print ('[DB...BAT] Number of spec: {0}....\n{1}'.format(num_spectra, peak_center_vec[:, 0]))
        self._center_of_mass_ws = CreateWorkspace(DataX=peak_center_vec[:, 0], DataY=peak_center_vec[:, 0],
                                                  NSpec=num_spectra, OutputWorkspace='CenterOfMassWS')
        self._highest_point_ws = CreateWorkspace(DataX=peak_center_vec[:, 1], DataY=peak_center_vec[:, 1],
                                                 NSpec=num_spectra, OutputWorkspace='HighestPointWS')

        return

    def generate_matrix_workspace(self, data_set_list):
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
        ws_full = CreateWorkspace(DataX=datax, DataY=datay, NSpec=len(vec_x_list),
                                  OutputWorkspace=self._workspace_name)

        return ws_full

    def get_data_workspace_name(self):
        return self._workspace_name

    def fit_peaks(self, peak_function_name, background_function_name, peak_center, fit_range, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :param background_function_name:
        :param peak_center:
        :param fit_range:
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

        # check peak function name:
        if peak_function_name not in ['Gaussian', 'Voigt', 'PseudoVoigt']:
            raise RuntimeError('Peak function {0} is not supported yet.'.format(peak_function_name))
        if background_function_name not in ['Linear', 'Flat']:
            raise RuntimeError('Background type {0} is not supported yet.'.format(background_function_name))

        num_spectra = self._data_workspace.getNumberHistograms()
        peak_window_ws = CreateWorkspace(DataX=np.array([fit_range[0], fit_range[1]] * num_spectra),
                                         DataY=np.array([fit_range[0], fit_range[1]] * num_spectra),
                                         NSpec=num_spectra)

        # fit
        print ('[DB...BAT] Data workspace # spec = {0}. Fit range = {1}'
               ''.format(self._data_workspace.getNumberHistograms(), fit_range))
        # TODO FIXME - Fit range shall be determined by boundary of plot!
        # no pre-determined peak center: use center of mass
        r = FitPeaks(InputWorkspace=self._data_workspace,
                     OutputWorkspace='full_fitted',
                     # PeakCenters=peak_center,
                     # TODO FIXME - Need to make it more general for peak center workspace
                     PeakCentersWorkspace=self._center_of_mass_ws,
                     PeakFunction=peak_function_name,
                     BackgroundType=background_function_name,
                     StartWorkspaceIndex=start,
                     StopWorkspaceIndex=stop,
                     OutputPeakParametersWorkspace='param_m',
                     FittedPeaksWorkspace='model_full',
                     FindBackgroundSigma=1,
                     HighBackground=False,
                     ConstrainPeakPositions=False,
                     RawPeakParameters=True,
                     FitPeakWindowWorkspace=peak_window_ws)


        # process output
        # TODO: Clean!
        self.peak_pos_ws = r[0]
        self.func_param_ws = r[1]
        self.fitted_ws = r[2]

        print 'fitted workspace: {0}'.format(r[2])


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
        # init parameters
        param_vec = np.ndarray(shape=(self.func_param_ws.rowCount()), dtype='float')

        col_names = self.func_param_ws.getColumnNames()
        if param_name in col_names:
            col_index = col_names.index(param_name)
            for row_index in range(self.func_param_ws.rowCount()):
                param_vec[row_index] = self.func_param_ws.cell(row_index, col_index)
        else:
            err_msg = 'Function parameter {0} does not exist.'.format(param_name)
            # raise RuntimeError()
            print (err_msg)

        return param_vec




