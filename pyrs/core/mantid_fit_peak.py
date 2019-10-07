# Peak fitting engine by calling mantid
from pyrs.core import mantid_helper
from pyrs.utilities import checkdatatypes
from pyrs.core import peak_fit_engine
import numpy as np
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateWorkspace, FitPeaks


class MantidPeakFitEngine(peak_fit_engine.PeakFitEngine):
    """
    peak fitting engine class for mantid
    """

    def __init__(self, workspace, mask_name):
        """ Initialization to set up the workspace for fitting
        :param workspace: Hidra workspace to get the diffraction peaks from
        :param mask_name: Mask acting on the detector (i.e., referring to a specific set of diffraction data)
        """
        # call parent
        super(MantidPeakFitEngine, self).__init__(workspace, mask_name)

        # sub-run, Mantid workspace index dictionary
        # self._ws_index_sub_run_dict = dict()

        # Create Mantid workspace: generate a workspace with all sub runs!!!
        mantid_workspace = mantid_helper.generate_mantid_workspace(workspace, mask_name)
        self._mantid_workspace_name = mantid_workspace.name()

        # wave length
        self._wavelength_vec = None

        # some observed properties
        self._center_of_mass_ws_name = None
        self._highest_point_ws_name = None

        # fitting result (Mantid specific)
        self._fitted_peak_position_ws = None  # fitted peak position workspace
        self._fitted_function_param_table = None  # fitted function parameters table workspace
        self._fitted_function_error_table = None  # fitted function parameters' fitting error table workspace
        self._model_matrix_ws = None  # MatrixWorkspace of the model from fitted function parameters

        return

    def calculate_center_of_mass(self):
        """ calculate center of mass of peaks in the Mantid MatrixWorkspace as class variable
        and highest data point
        :return:
        """
        # get the workspace
        data_ws = AnalysisDataService.retrieve(self._mantid_workspace_name)
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
        self._center_of_mass_ws_name = '{0}_COM'.format(self._mantid_workspace_name)
        com_ws = CreateWorkspace(DataX=peak_center_vec[:, 0], DataY=peak_center_vec[:, 0],
                                 NSpec=num_spectra, OutputWorkspace=self._center_of_mass_ws_name)
        print('[INFO] Center of Mass Workspace: {0} Number of spectra = {1}'
              ''.format(self._center_of_mass_ws_name, com_ws.getNumberHistograms()))

        self._highest_point_ws_name = '{0}_HighestPoints'.format(self._mantid_workspace_name)
        high_ws = CreateWorkspace(DataX=peak_center_vec[:, 1], DataY=peak_center_vec[:, 1],
                                  NSpec=num_spectra, OutputWorkspace=self._highest_point_ws_name)
        print('[INFO] Highest Point Workspace: {0} Number of spectra = {1}'
              ''.format(self._highest_point_ws_name, high_ws.getNumberHistograms()))

        self._peak_center_vec = peak_center_vec

        return

    def _create_peak_center_ws(self, peak_center):
        """ Create peak center workspace
        :param peak_center: float or numpy array
        :return:
        """
        mtd_ws = mantid_helper.retrieve_workspace(self._mantid_workspace_name)
        num_spectra = mtd_ws.getNumberHistograms()

        if isinstance(peak_center, float):
            peak_center_array = np.zeros(shape=(num_spectra,), dtype='float')
            peak_center_array += peak_center
        elif isinstance(peak_center, np.ndarray):
            peak_center_array = peak_center
            if peak_center_array.shape != (num_spectra,):
                raise RuntimeError('Peak center (array) must be a ndarray with shape ({}, ) '
                                   'but not {}'.format(num_spectra, peak_center_array.shape))
        else:
            raise NotImplementedError('Impossible situation: Peak center is specified as {} in type {}'
                                      ''.format(peak_center, type(peak_center)))

        # Create workspace
        peak_center_ws_name = 'Input_Center_{}'.format(self._mantid_workspace_name)
        center_ws = CreateWorkspace(DataX=peak_center_array,
                                    DataY=peak_center_array,
                                    NSpec=num_spectra, OutputWorkspace=peak_center_ws_name)

        return center_ws

    def fit_peaks(self, sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                  cal_center_d):
        """ Fit peaks with option to calculate peak center in d-spacing
        :param peak_function_name:
        :param background_function_name:
        :param peak_center: 1 of (1) string (center wksp), (2) wksp, (3) vector of float (4) single float
        :param peak_range:
        :param wave_length_array:
        :return:
        """
        # Check inputs
        checkdatatypes.check_string_variable('Peak function name', peak_function_name,
                                             ['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian'])
        checkdatatypes.check_string_variable('Background function name', background_function_name,
                                             ['Linear', 'Flat', 'Quadratic'])
        checkdatatypes.check_series('(To fit) sub runs range', sub_run_range, None, 2)

        self._fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                               cal_center_d)

        # Set class variable
        self._peak_function_name = peak_function_name

        # Get workspace and gather some information
        mantid_ws = mantid_helper.retrieve_workspace(self._mantid_workspace_name, True)
        num_spectra = mantid_ws.getNumberHistograms()

        # Get the sub run range
        start_sub_run, end_sub_run = sub_run_range
        if start_sub_run is None:
            start_spectrum = 0
        else:
            start_spectrum = self._hd_workspace.get_spectrum_index(start_sub_run)
        if end_sub_run is None:
            end_spectrum = num_spectra - 1
        else:
            end_spectrum = self._hd_workspace.get_spectrum_index(end_sub_run)

        # Create Peak range/window workspace
        peak_window_ws_name = 'fit_window_{0}'.format(self._mantid_workspace_name)
        CreateWorkspace(DataX=np.array([peak_range[0], peak_range[1]] * num_spectra),
                        DataY=np.array([peak_range[0], peak_range[1]] * num_spectra),
                        NSpec=num_spectra, OutputWorkspace=peak_window_ws_name)

        # Create peak center workspace
        if isinstance(peak_center, float) or isinstance(peak_center, np.ndarray) or isinstance(peak_center, list):
            # single value float or numpy array
            peak_center_ws = self._create_peak_center_ws(peak_center)
        elif isinstance(peak_center, str):
            # workspace name
            peak_center_ws = mantid_helper.retrieve_workspace(peak_center, True)
        elif mantid_helper.is_matrix_workspace(peak_center):  # TODO - #81 NOW - Implement!
            # already a workspace
            peak_center_ws = peak_center
            assert peak_center_ws.getNumberHistograms() == num_spectra
        else:
            # Nothing
            raise NotImplementedError('Peak center {} in format {} is not accepted.'
                                      ''.format(peak_center, type(peak_center)))
        # END-IF-ELSE

        # no pre-determined peak center: use center of mass
        r_positions_ws_name = 'fitted_peak_positions_{0}'.format(self._mantid_workspace_name)
        r_param_table_name = 'param_m_{0}'.format(self._mantid_workspace_name)
        r_error_table_name = 'param_e_{0}'.format(self._mantid_workspace_name)
        r_model_ws_name = 'model_full_{0}'.format(self._mantid_workspace_name)

        # TODO - #81 NOW - Requiring good estimation!!! - shall we use a dictionary to set up somewhere else?
        width_dict = {'Gaussian': ('Sigma', 0.36),
                      'PseudoVoigt': ('FWHM', 1.0),
                      'Voigt': ('LorentzFWHM, GaussianFWHM', '0.1, 0.7')}

        print('[DB...BAT] Peak function: {}'.format(peak_function_name))
        print('[DB...BAT] Param names:   {}'.format(width_dict[peak_function_name][0]))
        print('[DB...BAT] Param values:  {}'.format(width_dict[peak_function_name][1]))

        # Get peak center workspace
        self.calculate_center_of_mass()

        # SaveNexusProcessed(InputWorkspace=self._mantid_workspace_name, Filename='/tmp/data.nxs')
        # SaveNexusProcessed(InputWorkspace=peak_center_ws, Filename='/tmp/position.nxs')
        # SaveNexusProcessed(InputWorkspace=peak_window_ws_name, Filename='/tmp/peakwindow.nxs')

        # Fit peak by Mantid.FitPeaks
        r = FitPeaks(InputWorkspace=self._mantid_workspace_name,
                     PeakCentersWorkspace=peak_center_ws,
                     FitPeakWindowWorkspace=peak_window_ws_name,
                     PeakFunction=peak_function_name,
                     BackgroundType=background_function_name,
                     StartWorkspaceIndex=start_spectrum,
                     StopWorkspaceIndex=end_spectrum,
                     FindBackgroundSigma=1,
                     HighBackground=False,
                     ConstrainPeakPositions=False,
                     PeakParameterNames="{}, {}".format(width_dict[peak_function_name][0], 'Mixing'),
                     PeakParameterValues="{}, {}".format(width_dict[peak_function_name][1], '0.5'),
                     RawPeakParameters=True,
                     OutputWorkspace=r_positions_ws_name,
                     OutputPeakParametersWorkspace=r_param_table_name,
                     OutputParameterFitErrorsWorkspace=r_error_table_name,
                     FittedPeaksWorkspace=r_model_ws_name)

        # r is a class containing multiple outputs (workspaces)
        # print (r,  r.OutputParameterFitErrorsWorkspace.getColumnNames(), r.OutputPeakParametersWorkspace,
        #        r.FittedPeaksWorkspace)

        # Save all the workspaces automatically for further review
        if False:
            mantid_helper.study_mantid_peak_fitting()
        # END-IF-DEBUG (True)

        # process output
        self._peak_function_name = peak_function_name
        self._fitted_peak_position_ws = AnalysisDataService.retrieve(r_positions_ws_name)
        self._fitted_function_param_table = AnalysisDataService.retrieve(r_param_table_name)
        self._fitted_function_error_table = AnalysisDataService.retrieve(r_error_table_name)
        self._model_matrix_ws = AnalysisDataService.retrieve(r_model_ws_name)

        # Calculate d-spacing with wave length given
        if cal_center_d:
            # optionally to use calibrated wave length as default
            self.calculate_peak_position_d(wave_length=self._wavelength_vec)

        return

    def get_observed_peaks_centers(self):
        """
        get center of mass vector and X value vector corresponding to maximum Y value
        :return:
        """
        return self._peak_center_vec

    def get_calculated_peak(self, sub_run):
        """
        get the model (calculated) peak of a certain scan
        :param sub_run:
        :return:
        """
        if self._model_matrix_ws is None:
            raise RuntimeError('There is no fitting result!')

        checkdatatypes.check_int_variable('Scan log index', sub_run, (0, self._model_matrix_ws.getNumberHistograms()))

        vec_x = self._model_matrix_ws.readX(sub_run)
        vec_y = self._model_matrix_ws.readY(sub_run)

        return vec_x, vec_y

    def get_center_of_mass_workspace_name(self):
        """
        Get the center of mass workspace name
        :return:
        """
        return self._center_of_mass_ws_name

    def get_data_workspace_name(self):
        """
        get the data workspace name
        :return:
        """
        return self._mantid_workspace_name

    def _get_fitted_parameters_value(self, spec_index_vec, param_name_list, param_value_array):
        """
        Get fitted peak parameters' value
        :param spec_index_vec:
        :param param_name_list:
        :param param_value_array: a (p, s, e) array: p = param_name_list.size, s = sub runs size, e = 1 or 2
        :return:
        """
        # table column names
        col_names = self._fitted_function_param_table.getColumnNames()

        # get fitted parameter value
        for out_index, param_name in enumerate(param_name_list):
            # get value from column
            if param_name in col_names:
                param_col_index = col_names.index(param_name)
                param_vec = np.array(self._fitted_function_param_table.column(param_col_index))
            elif param_name == 'center_d':
                param_vec = self._peak_center_d_vec[:, 0]
            else:
                raise RuntimeError('Peak parameter {} does not exist. Available parameters are {} and center_d'
                                   ''.format(param_name, col_names))
            # set value
            param_value_array[out_index, :, 0] = param_vec[spec_index_vec]
        # END-FOR

        return

    def get_fit_cost(self, max_chi2):
        """ Get the peak function cost
        :param max_chi2:
        :return:
        """
        # Get chi2 column
        col_names = self._fitted_function_param_table.getColumnNames()
        chi2_col_index = col_names.index('chi2')

        # Get chi2 from table workspace (native return is List)
        chi2_vec = np.array(self._fitted_function_param_table.column(chi2_col_index))  # form to np.ndarray

        # Filter out the sub runs/spectra with large chi^2
        if max_chi2 is not None and max_chi2 < 1.E20:
            # selected
            good_fit_indexes = np.where(chi2_vec < max_chi2)
            chi2_vec = chi2_vec[good_fit_indexes]
            spec_vec = good_fit_indexes[0]
        else:
            # all
            print(chi2_vec)
            spec_vec = np.arange(chi2_vec.shape[0])

        return spec_vec, chi2_vec

    def get_fitted_params_x(self, param_name_list, including_error, max_chi2=1.E20):
        """ Get specified parameters' fitted value and optionally error with optionally filtered value
        :param param_name_list:
        :param including_error:
        :param max_chi2: Default is including all.
        :return: 2-tuple: (1) (n, ) vector for sub run number  (2) (p, n, 1) or (p, n, 2) vector for parameter values and
                 optionally fitting error: p = number of parameters , n = number of sub runs
        """
        # check
        checkdatatypes.check_list('Function parameters', param_name_list)
        checkdatatypes.check_bool_variable('Flag to output fitting error', including_error)
        checkdatatypes.check_float_variable('Maximum cost chi^2', max_chi2, (1, None))

        # get table information
        col_names = self._fitted_function_param_table.getColumnNames()
        chi2_col_index = col_names.index('chi2')
        ws_index_col_index = col_names.index('wsindex')

        # get the rows to survey
        if max_chi2 > 0.999E20:
            # all of sub runs / spectra
            row_index_list = range(self._fitted_function_param_table.rowCount())
        else:
            # need to filter: get a list of chi^2 and then filter
            chi2_vec = np.zeros(shape=(self._fitted_function_param_table.rowCount(),), dtype='float')
            for row_index in range(self._fitted_function_param_table.rowCount()):
                chi2_i = self._fitted_function_param_table.cell(row_index, chi2_col_index)
                chi2_vec[row_index] = chi2_i

            # filer chi2 against max
            filtered_indexes = np.where(chi2_vec < max_chi2)
            row_index_list = list(filtered_indexes)
        # END-IF-ELSE

        # init parameters
        num_params = len(param_name_list)
        if including_error:
            num_items = 2
        else:
            num_items = 1
        param_vec = np.zeros(shape=(num_params, len(row_index_list), num_items), dtype='float')
        sub_run_vec = np.zeros(shape=(len(row_index_list), ), dtype='int')

        # sub runs
        for vec_index, row_index in enumerate(row_index_list):
            # sub run
            ws_index_i = self._fitted_function_param_table.cell(row_index, ws_index_col_index)
            sub_run_i = self._ws_index_sub_run_dict[ws_index_i]
            sub_run_vec[vec_index] = sub_run_i
        # END-FOR

        # retrieve parameters
        for param_index, param_name_i in enumerate(param_name_list):
            # get the parameter column index
            if param_name_i in col_names:
                param_i_col_index = col_names.index(param_name_i)
            else:
                param_i_col_index = None

            # go through all the rows fitting the chi^2 requirement
            for vec_index, row_index in enumerate(row_index_list):
                # init error
                error_i = None

                if param_i_col_index is not None:
                    # native parameters
                    value_i = self._fitted_function_param_table.cell(row_index, param_i_col_index)
                    if including_error:
                        error_i = self._fitted_function_error_table.cell(row_index, param_i_col_index)
                elif param_name_i == 'center_d':
                    # special center in dSpacing
                    value_i = self._peak_center_d_vec[row_index]
                    if including_error:
                        error_i = self._peak_center_d_error_vec[row_index]
                elif param_name_i in self._effective_parameters:
                    # effective parameter
                    value_i = self.calculate_effective_parameter(param_name_i)
                    if including_error:
                        error_i = self.calculate_effective_parameter_error(param_name_i)
                else:
                    err_msg = 'Function parameter {0} does not exist. Supported parameters are {1}' \
                              ''.format(param_name_list, col_names)
                    # raise RuntimeError()
                    raise KeyError(err_msg)
                # END-IF-ELSE

                # set value
                param_vec[param_index, vec_index, 0] = value_i
                if including_error:
                    param_vec[param_index, vec_index, 1] = error_i
            # END-FOR (each row)
        # END-FOR (each parameter)

        return sub_run_vec, param_vec

    def get_scan_indexes(self):
        """
        get a vector of scan indexes and assume that the scan log indexes are from 0 and consecutive
        :return: vector of integer from 0 and up consecutively
        """
        data_workspace = self.retrieve_workspace(self._mantid_workspace_name, True)
        indexes_list = range(data_workspace.getNumberHistograms())

        return np.array(indexes_list)

    # def get_fitted_params_error(self, param_name):
    #     """ Get the value of a specified parameters's fitted error of whole scan
    #     Note: from FitPeaks's output workspace "OutputParameterFitErrorsWorkspace"
    #     :param param_name:
    #     :return: float vector of parameters...
    #     """
    #     # TODO - NOW - Continue (2) from here to sort out how do we demo result to front-end
    #     # check
    #     checkdatatypes.check_string_variable('Function parameter', param_name)
    #
    #     # init parameters
    #     error_vec = np.ndarray(shape=(self._fitted_function_error_table.rowCount()), dtype='float')
    #
    #     col_names = self._fitted_function_error_table.getColumnNames()
    #     if param_name in col_names:
    #         col_index = col_names.index(param_name)
    #         for row_index in range(self._fitted_function_error_table.rowCount()):
    #             error_vec[row_index] = self._fitted_function_error_table.cell(row_index, col_index)
    #     elif param_name == 'center_d':
    #         error_vec = self._peak_center_d_error_vec
    #     else:
    #         err_msg = 'Function parameter {0} does not exist. Supported parameters are {1}' \
    #                   ''.format(param_name, col_names)
    #         # raise RuntimeError()
    #         raise KeyError(err_msg)
    #
    #     return error_vec
    #
    # def get_good_fitted_params(self, param_name, max_chi2=1.E20):
    #     """
    #     get fitted parameter's value for good fit specified by maximum chi2
    #     :param param_name:
    #     :param max_chi2:
    #     :return: 2-vector of same size
    #     """
    #     # check
    #     checkdatatypes.check_string_variable('Function parameter', param_name)
    #     checkdatatypes.check_float_variable('Chi^2', max_chi2, (1., None))
    #
    #     # get all the column names
    #     col_names = self._fitted_function_param_table.getColumnNames()
    #     if not ('chi2' in col_names and (param_name in col_names or param_name == 'center_d')):
    #         err_msg = 'Function parameter {0} does not exist. Supported parameters are {1}' \
    #                   ''.format(param_name, col_names)
    #         # raise RuntimeError()
    #         raise KeyError(err_msg)
    #     elif param_name == 'chi2':
    #         is_chi2 = True
    #     else:
    #         is_chi2 = False
    #
    #     # get chi2 first
    #     chi2_col_index = col_names.index('chi2')
    #     if param_name == 'center_d':
    #         param_col_index = 'center_d'
    #     elif not is_chi2:
    #         param_col_index = col_names.index(param_name)
    #     else:
    #         param_col_index = chi2_col_index
    #
    #     param_list = list()
    #     selected_row_index = list()
    #     for row_index in range(self._fitted_function_param_table.rowCount()):
    #         chi2 = self._fitted_function_param_table.cell(row_index, chi2_col_index)
    #         if math.isnan(chi2) or chi2 > max_chi2:
    #             continue
    #
    #         if is_chi2:
    #             value_i = chi2
    #         elif param_col_index == 'center_d':
    #             value_i = self._peak_center_d_vec[row_index]
    #         else:
    #             value_i = self._fitted_function_param_table.cell(row_index, param_col_index)
    #
    #         param_list.append(value_i)
    #         selected_row_index.append(row_index)
    #     # END-IF
    #
    #     log_index_vec = np.array(selected_row_index) + 1
    #     param_vec = np.array(param_list)
    #
    #     return log_index_vec, param_vec

    def get_function_parameter_names(self):
        """
        get all the function parameters' names
        :return:
        """
        fitted_param_names = self._fitted_function_param_table.getColumnNames()
        fitted_param_names.append('center_d')

        return fitted_param_names

    def get_number_scans(self):
        """ Get number of scans in input data to fit
        :return:
        """
        data_workspace = self.retrieve_workspace(self._mantid_workspace_name, True)
        return data_workspace.getNumberHistograms()

    @staticmethod
    def retrieve_workspace(ws_name, throw_if_not_exist):
        """
        retrieve the workspace.
        optionally throw a runtime error if the workspace does not exist.
        :param ws_name:
        :param throw_if_not_exist:
        :return: workspace instance or None (if throw_if_not_exist is set to False)
        """
        # check inputs
        checkdatatypes.check_string_variable('Workspace name', ws_name)
        checkdatatypes.check_bool_variable('Throw exception if workspace does not exist', throw_if_not_exist)

        # get
        if AnalysisDataService.doesExist(ws_name):
            workspace = AnalysisDataService.retrieve(ws_name)
        elif throw_if_not_exist:
            raise RuntimeError('Workspace {0} does not exist in Mantid ADS'.format(throw_if_not_exist))
        else:
            workspace = None

        return workspace

    # TODO URGENT - #84 - Need to talk to IS about wave length!!!
    def set_wavelength(self, wavelengths):
        """ Set wave length to each spectrum
        :param wavelengths:
        :return:
        """
        # Call base class method
        super(MantidPeakFitEngine, self).set_wavelength(wavelengths)

        # Set to an ndarray for
        mtd_ws = mantid_helper.retrieve_workspace(self._mantid_workspace_name)
        self._wavelength_vec = np.ndarray(shape=(mtd_ws.getNumberHistograms(), ), dtype='float')

        sub_runs = sorted(wavelengths.keys())
        if len(sub_runs) == self._wavelength_vec.shape[0]:
            for i in range(len(sub_runs)):
                self._wavelength_vec[i] = wavelengths[sub_runs[i]]
        else:
            # FIXME TODO - #84 NOWNOW URGENT
            self._wavelength_vec = np.zeros_like(self._wavelength_vec) + 1.1723
            # raise RuntimeError('Input wavelength dictionary {} has different number than the sub runs {}.'
            #                    ''.format(wavelengths, self._wavelength_vec.shape[0]))

        return
