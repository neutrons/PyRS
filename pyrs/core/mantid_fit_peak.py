# Peak fitting engine by calling mantid
from pyrs.core import mantid_helper
from pyrs.utilities import checkdatatypes
from pyrs.core import peak_fit_engine
from pyrs.core.peak_collection import PeakCollection
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
        # the Mandid workspace and HiDRA workspace have consistent spectrum/sub run map.
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

    # FIXME TODO - This method can be abstracted to base class PeakFitEngine
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

        return peak_center_vec

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

    # TODO - #89 - Cleanup
    def fit_peaks(self, peak_tag, sub_run_range, peak_function_name, background_function_name,
                  peak_center, peak_range, cal_center_d):
        """Fit peaks

        Fit peaks given from sub run range and with option to calculate peak center in d-spacing

        Note: method _set_profile_parameters_values_from_fitting() must be called at the end of fit_peaks()

        Parameters
        ----------
        sub_run_range: 2-tuple
            first sub run, last sub run (included)
            start sub run (None as first 1) and end sub run (None as last 1) for
            range of sub runs (including both end) to refine
        peak_function_name
        background_function_name
        peak_center: str / Mantid Workspace2D / ndarray / float
            Center workspace name, Center workspace, peak centers for each spectrum, peak center (universal)
        peak_range:
            Peak range
        cal_center_d

        Returns
        -------

        """
        # Check inputs
        self._fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                               cal_center_d)

        # Get workspace and gather some information
        mantid_ws = mantid_helper.retrieve_workspace(self._mantid_workspace_name, True)
        num_spectra = mantid_ws.getNumberHistograms()
        sub_run_array = self._hd_workspace.get_sub_runs()

        # Get the sub run range
        start_sub_run, end_sub_run = sub_run_range
        if start_sub_run is None:
            # default start sub run as first sub run
            start_spectrum = 0
            start_sub_run = int(sub_run_array[0])
        else:
            # user specified
            start_spectrum = int(self._hd_workspace.get_spectrum_index(start_sub_run))
        if end_sub_run is None:
            # default end sub run as last sub run (included)
            end_spectrum = num_spectra - 1
            end_sub_run = int(sub_run_array[-1])
        else:
            end_spectrum = int(self._hd_workspace.get_spectrum_index(end_sub_run))

        # TODO FIXME - Next step: this section to create peak center and windows will be moved to a
        #              separate method and expanded to support multiple peaks

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

        # TODO FIXME - Using standard constants and separate this part to a separate method
        # Make the difference between peak profiles
        if peak_function_name == 'Gaussian':
            # Gaussian
            peak_param_names = "{}".format(width_dict[peak_function_name][0])
            peak_param_values = "{}".format(width_dict[peak_function_name][1])
        elif peak_function_name == 'PseudoVoigt':
            peak_param_names = "{}, {}".format(width_dict[peak_function_name][0], 'Mixing')
            peak_param_values = "{}, {}".format(width_dict[peak_function_name][1], '0.5')
        else:
            raise RuntimeError('Peak function {} is not supported for pre-set guessed starting value'
                               ''.format(peak_function_name))

        # END-IF

        # Fit peak by Mantid.FitPeaks
        r = FitPeaks(InputWorkspace=self._mantid_workspace_name,
                     PeakCentersWorkspace=peak_center_ws,
                     FitPeakWindowWorkspace=peak_window_ws_name,
                     PeakFunction=peak_function_name,
                     BackgroundType=background_function_name,
                     StartWorkspaceIndex=start_spectrum,
                     StopWorkspaceIndex=end_spectrum,
                     FindBackgroundSigma=1,
                     HighBackground=True,
                     ConstrainPeakPositions=False,
                     PeakParameterNames=peak_param_names,
                     PeakParameterValues=peak_param_values,
                     RawPeakParameters=True,
                     OutputWorkspace=r_positions_ws_name,
                     OutputPeakParametersWorkspace=r_param_table_name,
                     OutputParameterFitErrorsWorkspace=r_error_table_name,
                     FittedPeaksWorkspace=r_model_ws_name)
        # r is a class containing multiple outputs (workspaces)
        # print (r,  r.OutputParameterFitErrorsWorkspace.getColumnNames(), r.OutputPeakParametersWorkspace,
        #        r.FittedPeaksWorkspace)
        assert r, 'return from FitPeaks cannot be None'

        # Save all the workspaces automatically for further review
        # mantid_helper.study_mantid_peak_fitting()
        # END-IF-DEBUG (True)

        # process output
        self._fitted_peak_position_ws = AnalysisDataService.retrieve(r_positions_ws_name)
        self._fitted_function_param_table = AnalysisDataService.retrieve(r_param_table_name)
        self._fitted_function_error_table = AnalysisDataService.retrieve(r_error_table_name)
        self._model_matrix_ws = AnalysisDataService.retrieve(r_model_ws_name)

        # Calculate d-spacing with wave length given
        if cal_center_d:
            # optionally to use calibrated wave length as default
            self.calculate_peak_position_d(wave_length=self._wavelength_vec)

        # Set the fit result to private class structure numpy arrays
        # Get sub runs considering fitting only being applied to a sub set of sub runs
        sub_runs = sub_run_array[np.where((sub_run_array >= start_sub_run) & (sub_run_array <= end_sub_run))]
        fitted_peak = self._set_profile_parameters_values_from_fitting(peak_tag, sub_runs, peak_function_name,
                                                                       background_function_name)

        return fitted_peak

    def _set_profile_parameters_values_from_fitting(self, peak_tag, sub_runs, peak_profile_name,
                                                    background_type_name):
        """Set (fitted) profile names from TableWorkspaces

        Parameters
        ----------
        peak_tag : str
            peak tag
        sub_runs : numpy.ndarray
            sub run numbers for those which are fitted
        peak_profile_name : str
        background_type_name : str

        Returns
        -------
        ~pyrs.core.peak_collecton.PeakCollection
            Fitted peak's information

        """
        def convert_from_table_to_arrays(table_ws):
            # Table column names
            table_col_names = table_ws.getColumnNames()
            num_sub_runs = table_ws.rowCount()

            # Set the structured numpy array
            data_type_list = list()
            for param_name in table_col_names:
                data_type_list.append((param_name, np.float32))

            struct_array = np.zeros(num_sub_runs, dtype=data_type_list)

            # get fitted parameter value
            for col_index, param_name in enumerate(table_col_names):
                # get value from column in value table
                struct_array[param_name] = table_ws.column(col_index)

            return struct_array

        peak_params_value_array = convert_from_table_to_arrays(self._fitted_function_param_table)
        peak_params_error_array = convert_from_table_to_arrays(self._fitted_function_error_table)
        fit_cost_array = peak_params_value_array['chi2']

        # Create PeakCollection instance
        peak_object = PeakCollection(peak_tag)
        peak_object.set_peak_fitting_values(peak_profile_name, background_type_name, sub_runs,
                                            peak_params_value_array, peak_params_error_array, fit_cost_array)

        # Set to dictionary
        self._peak_collection_dict[peak_tag] = peak_object

        return peak_object

    # FIXME - Abstract to base class and re-implement
    @staticmethod
    def get_observed_peaks_centers():
        """
        get center of mass vector and X value vector corresponding to maximum Y value
        :return:
        """
        return np.zeros(shape=(1000,))

    def get_calculated_peak(self, sub_run):
        """Get the "model" peak, calculated from fitted parameters

        Parameters
        ----------
        sub_run: int
            sub run number

        Returns
        -------
        ndarray, ndarray
            vector X, vector Y
        """
        if self._model_matrix_ws is None:
            raise RuntimeError('There is no fitting result!')

        # Check sub run: just a positive integer
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))

        # Convert to workspace
        ws_index = int(self._hd_workspace.get_spectrum_index(sub_run))

        # Get data
        vec_x = self._model_matrix_ws.readX(ws_index)
        vec_y = self._model_matrix_ws.readY(ws_index)

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

    # def _get_fitted_parameters_value(self, spec_index_vec, param_name_list, param_value_array):
    #     """
    #     Get fitted peak parameters' value
    #     :param spec_index_vec:
    #     :param param_name_list:
    #     :param param_value_array: a (p, s, e) array: p = param_name_list.size, s = sub runs size, e = 1 or 2
    #     :return:
    #     """
    #     # table column names
    #     col_names = self._fitted_function_param_table.getColumnNames()
    #
    #     # get fitted parameter value
    #     for out_index, param_name in enumerate(param_name_list):
    #         # get value from column
    #         if param_name in col_names:
    #             param_col_index = col_names.index(param_name)
    #             param_vec = np.array(self._fitted_function_param_table.column(param_col_index))
    #         elif param_name == 'center_d':
    #             param_vec = self._peak_center_d_vec[:, 0]
    #         else:
    #             raise RuntimeError('Peak parameter {} does not exist. Available parameters are {} and center_d'
    #                                ''.format(param_name, col_names))
    #         # set value
    #         param_value_array[out_index, :, 0] = param_vec[spec_index_vec]
    #     # END-FOR
    #
    #     return

    # def get_fit_cost(self, max_chi2):
    #     """ Get the peak function cost
    #     :param max_chi2:
    #     :return:
    #     """
    #     # Get chi2 column
    #     col_names = self._fitted_function_param_table.getColumnNames()
    #     chi2_col_index = col_names.index('chi2')
    #
    #     # Get chi2 from table workspace (native return is List)
    #     chi2_vec = np.array(self._fitted_function_param_table.column(chi2_col_index))  # form to np.ndarray
    #
    #     # Filter out the sub runs/spectra with large chi^2
    #     if max_chi2 is not None and max_chi2 < 1.E20:
    #         # selected
    #         good_fit_indexes = np.where(chi2_vec < max_chi2)
    #         chi2_vec = chi2_vec[good_fit_indexes]
    #         spec_vec = good_fit_indexes[0]
    #     else:
    #         # all
    #         print(chi2_vec)
    #         spec_vec = np.arange(chi2_vec.shape[0])
    #
    #     return spec_vec, chi2_vec

    def get_scan_indexes(self):
        """
        get a vector of scan indexes and assume that the scan log indexes are from 0 and consecutive
        :return: vector of integer from 0 and up consecutively
        """
        data_workspace = self.retrieve_workspace(self._mantid_workspace_name, True)
        indexes_list = range(data_workspace.getNumberHistograms())

        return np.array(indexes_list)

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
