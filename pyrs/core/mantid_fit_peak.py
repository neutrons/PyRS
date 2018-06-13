# Peak fitting engine by calling mantid
# Set up the testing environment for PyVDrive commands
import os
import sys
home_dir = os.path.expanduser('~')
if home_dir.startswith('/SNS/'):
    # analysis
    # nightly: sys.path.insert(1, '/opt/mantidnightly/bin/')
    # local build
    sys.path.insert(1, '/SNS/users/wzz/Mantid_Project/builds/debug/bin/')
elif home_dir.startswith('/Users/wzz'):
    # VZ local mac
    sys.path.append('/Users/wzz/MantidBuild/debug/bin')
elif home_dir.startswith('/home/wzz'):
    # VZ workstation
    sys.path.insert(1, '/home/wzz/Mantid_Project/builds/debug-master/bin')
import mantid
from mantid.simpleapi import FitPeaks, CreateWorkspace
from mantid.api import AnalysisDataService
import rshelper
import numpy as np

print ('[CHECK] Import Mantid from {0}'.format(mantid))


class MantidPeakFitEngine(object):
    """
    peak fitting engine class for mantid
    """
    def __init__(self, data_set_list, ref_id):
        """
        initialization
        :param data_set_list: list of data set
        :param ref_id: reference ID
        :param
        """
        # check
        rshelper.check_list('Data set list', data_set_list)
        rshelper.check_string_variable('Peak fitting reference ID', ref_id)

        self._workspace_name = self._get_matrix_name(ref_id)
        self._data_workspace = self.generate_matrix_workspace(data_set_list, matrix_ws_name=self._workspace_name)

        # some observed properties
        self._center_of_mass_ws = None
        self._highest_point_ws = None
        self._peak_center_vec = None  # 2D vector for observed center of mass and highest data point

        # fitting result
        self._fitted_peak_position_ws = None  # fitted peak position workspace
        self._fitted_function_param_table = None  # fitted function parameters table workspace
        self._model_matrix_ws = None  # MatrixWorkspace of the model from fitted function parameters

        return

    @staticmethod
    def _get_matrix_name(ref_id):
        """
        get the matrix workspace name hinted by a reference ID
        :param ref_id:
        :return:
        """
        return '{0}_workspace'.format(ref_id)

    def calculate_center_of_mass(self):
        """
        calculate center of mass of peaks in the workspace as class variable
        and highest data point
        :return:
        """
        # get the workspace
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
        self._center_of_mass_ws = CreateWorkspace(DataX=peak_center_vec[:, 0], DataY=peak_center_vec[:, 0],
                                                  NSpec=num_spectra, OutputWorkspace='CenterOfMassWS')
        self._highest_point_ws = CreateWorkspace(DataX=peak_center_vec[:, 1], DataY=peak_center_vec[:, 1],
                                                 NSpec=num_spectra, OutputWorkspace='HighestPointWS')

        self._peak_center_vec = peak_center_vec

        return

    @staticmethod
    def generate_matrix_workspace(data_set_list, matrix_ws_name):
        """
        convert data set of all scans to a multiple-spectra Mantid MatrixWorkspace
        :param data_set_list:
        :param matrix_ws_name
        :return:
        """
        # check input
        rshelper.check_list('Data set list', data_set_list)
        rshelper.check_string_variable('MatrixWorkspace name', matrix_ws_name)

        # convert input data set to list of vector X and vector Y
        vec_x_list = list()
        vec_y_list = list()
        for index in range(len(data_set_list)):
            diff_data = data_set_list[index]
            vec_x = diff_data[0]
            vec_y = diff_data[1]
            vec_x_list.append(vec_x)
            vec_y_list.append(vec_y)

        # create MatrixWorkspace
        datax = np.concatenate(vec_x_list, axis=0)
        datay = np.concatenate(vec_y_list, axis=0)
        ws_full = CreateWorkspace(DataX=datax, DataY=datay, NSpec=len(vec_x_list),
                                  OutputWorkspace=matrix_ws_name)

        return ws_full

    def get_observed_peaks_centers(self):
        """
        get center of mass vector and X value vector corresponding to maximum Y value
        :return:
        """
        return self._peak_center_vec

    def get_peak_intensities(self):
        """
        get peak intensities for each fitted peaks
        :return:
        """
        # get value
        scan_index_vector = self.get_scan_indexes()
        intensity_vector = self.get_fitted_params(param_name='Intensity')

        # check
        if len(scan_index_vector) != len(intensity_vector):
            raise RuntimeError('Scan indexes ({0}) and intensity ({1}) have different sizes.'
                               ''.format(len(scan_index_vector), len(intensity_vector)))

        # combine to dictionary
        intensity_dict = dict()
        for index in range(len(scan_index_vector)):
            intensity_dict[scan_index_vector[index]] = intensity_vector[index]

        return intensity_dict

    def get_data_workspace_name(self):
        """
        get the data workspace name
        :return:
        """
        return self._workspace_name

    def get_scan_indexes(self):
        """
        get a vector of scan indexes
        :return:
        """
        indexes_list = range(self._data_workspace.getNumberHistograms())

        return np.array(indexes_list)

    def fit_peaks(self, peak_function_name, background_function_name, fit_range, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :param background_function_name:
        :param fit_range:
        :param scan_index: single scan index to fit for.  If None, then fit for all spectra
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
        if peak_function_name not in ['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian']:
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

        # no pre-determined peak center: use center of mass
        r_positions_ws_name = 'fitted_peak_positions'
        r_param_table_name = 'param_m'
        r_model_ws_name = 'model_full'
        r = FitPeaks(InputWorkspace=self._data_workspace,
                     OutputWorkspace=r_positions_ws_name,
                     PeakCentersWorkspace=self._center_of_mass_ws,
                     PeakFunction=peak_function_name,
                     BackgroundType=background_function_name,
                     StartWorkspaceIndex=start,
                     StopWorkspaceIndex=stop,
                     OutputPeakParametersWorkspace=r_param_table_name,
                     FittedPeaksWorkspace=r_model_ws_name,
                     FindBackgroundSigma=1,
                     HighBackground=False,
                     ConstrainPeakPositions=False,
                     RawPeakParameters=False,
                     FitPeakWindowWorkspace=peak_window_ws)

        # process output
        self._fitted_peak_position_ws = AnalysisDataService.retrieve(r_positions_ws_name)
        self._fitted_function_param_table = AnalysisDataService.retrieve(r_param_table_name)
        self._model_matrix_ws = AnalysisDataService.retrieve(r_model_ws_name)

        return

    def get_calculated_peak(self, log_index):
        """
        get the model (calculated) peak of a certain scan
        :param log_index:
        :return:
        """
        if self._model_matrix_ws is None:
            raise RuntimeError('There is no fitting result!')

        rshelper.check_int_variable('Scan log index', log_index, (0, self._model_matrix_ws.getNumberHistograms()))

        vec_x = self._model_matrix_ws.readX(log_index)
        vec_y = self._model_matrix_ws.readY(log_index)

        return vec_x, vec_y

    def get_function_parameter_names(self):
        """
        get all the function parameters' names
        :return:
        """
        return self._fitted_function_param_table.getColumnNames()

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
        get the value of a fitted parameter
        :return:
        """
        # check
        rshelper.check_string_variable('Function parameter', param_name)

        # init parameters
        param_vec = np.ndarray(shape=(self._fitted_function_param_table.rowCount()), dtype='float')

        col_names = self._fitted_function_param_table.getColumnNames()
        if param_name in col_names:
            col_index = col_names.index(param_name)
            for row_index in range(self._fitted_function_param_table.rowCount()):
                param_vec[row_index] = self._fitted_function_param_table.cell(row_index, col_index)
        else:
            err_msg = 'Function parameter {0} does not exist.'.format(param_name)
            # raise RuntimeError()
            print (err_msg)

        return param_vec




