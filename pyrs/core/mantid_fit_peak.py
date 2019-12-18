# Peak fitting engine by calling mantid
from pyrs.core import mantid_helper
from pyrs.utilities import checkdatatypes
from pyrs.core import peak_fit_engine
from pyrs.core.peak_collection import PeakCollection
import numpy as np
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateWorkspace, FitPeaks


DEBUG = False   # Flag for debugging mode


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

        # Create Mantid workspace: generate a workspace with all sub runs and for all peaks
        mantid_workspace = mantid_helper.generate_mantid_workspace(workspace, workspace.name, mask_name)
        # the Mandid workspace and HiDRA workspace have consistent spectrum/sub run map.
        self._mantid_workspace_name = mantid_workspace.name()

        # fitting results in Mantid workspaces (Mantid specific)
        self._fitted_peak_position_ws = None  # fitted peak position workspace
        self._fitted_function_param_table = None  # fitted function parameters table workspace
        self._fitted_function_error_table = None  # fitted function parameters' fitting error table workspace
        self._model_matrix_ws = None  # MatrixWorkspace of the model from fitted function parameters

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

    def _create_peak_window_workspace(self, num_spectra, peaks_range_list):
        """Create Mantid Matrix workspace for peak fitting window used by FitPeaks

        Parameters
        ----------
        num_spectra : int
            number of spectrum
        peaks_range_list

        Returns
        -------
        str
            name of peak range MatrixWorkspace required by FitPeaks

        """
        # Check inputs
        checkdatatypes.check_list('Peaks ranges', peaks_range_list)

        # Create Peak range/window workspace
        peak_window_ws_name = 'fit_window_{0}'.format(self._mantid_workspace_name)

        # Create vector X and vector Y
        range_vector = np.tile(np.array(peaks_range_list).flatten(), num_spectra)

        # Create workspace
        CreateWorkspace(DataX=range_vector, DataY=range_vector,
                        NSpec=num_spectra, OutputWorkspace=peak_window_ws_name)

        return peak_window_ws_name

    def _set_default_peak_params_value(self, peak_function_name, peak_range):
        """Set up the starting peak parameters values for Mantid.FitPeaks

        Parameters
        ----------
        peak_function_name : str
            peak function name
        peak_range : float, float
            peak range

        Returns
        -------
        str, str
            parameter names (native), parameter values (as a list in str)

        """
        from peak_profile_utility import Gaussian, PseudoVoigt

        # Specify instrument resolution for both Gaussian and FWHM
        hidra_fwhm = 0.5

        # Estimate
        estimated_heights, flat_bkgds = self.estimate_peak_height(peak_range)
        max_estimated_height = estimated_heights.max()
        flat_bkgd = flat_bkgds[np.argmax(estimated_heights)]

        # Make the difference between peak profiles
        if peak_function_name == 'Gaussian':
            # Gaussian
            peak_param_names = '{}, {}'.format('Height', 'Sigma', 'A0')

            # sigma
            instrument_sigma = Gaussian.cal_sigma(hidra_fwhm)

            # set value
            peak_param_values = "{}, {}".format(max_estimated_height, instrument_sigma, flat_bkgd)

        elif peak_function_name == 'PseudoVoigt':
            # Pseudo-voig
            default_mixing = 0.6

            peak_param_names = '{}, {}, {}'.format('Mixing', 'Intensity', 'FWHM', 'A0')

            # intensity
            max_intensity = PseudoVoigt.cal_intensity(max_estimated_height, hidra_fwhm, default_mixing)

            # set values
            peak_param_values = "{}, {}, {}".format(default_mixing, max_intensity, hidra_fwhm, flat_bkgds)

        else:
            # Non-supported case
            raise RuntimeError('Peak function {} is not supported for pre-set guessed starting value'
                               ''.format(peak_function_name))

        return peak_param_names, peak_param_values

    def fit_peaks(self, peak_tag, sub_run_range, peak_function_name, background_function_name,
                  peak_center, peak_range, max_chi2=1E3, max_peak_shift=2, min_intensity=None):
        """Fit 1 peak on multiple sub runs

        Fit peaks given from sub run range and with option to calculate peak center in d-spacing

        Note: method _set_profile_parameters_values_from_fitting() must be called at the end of fit_peaks()

        Parameters
        ----------
        peak_tag: str
            peak tag
        sub_run_range: 2-tuple
            first sub run, last sub run (included)
            start sub run (None as first 1) and end sub run (None as last 1) for
            range of sub runs (including both end) to refine
        peak_function_name : str
            peak profile function name
        background_function_name
        peak_center: str / Mantid Workspace2D / ndarray / float
            Center workspace name, Center workspace, peak centers for each spectrum, peak center (universal)
        peak_range: (float, float)
            Peak range in 2-theta
        max_chi2 : float
            maximum allowed chi2
        max_peak_shift : float or None
            maximum allowed peak shift from specified peak position
        min_intensity : float or None
            minimum allowed peak intensity

        Returns
        -------

        """
        # Check inputs
        self._fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range)

        # Get workspace and gather some information
        mantid_ws = mantid_helper.retrieve_workspace(self._mantid_workspace_name, True)
        num_spectra = mantid_ws.getNumberHistograms()
        sub_run_array = self._hidra_wksp.get_sub_runs()

        # Get the sub run range and spectrum index range for Mantid
        start_sub_run, end_sub_run = sub_run_range
        if start_sub_run is None:
            # default start sub run as first sub run
            start_spectrum = 0
            start_sub_run = int(sub_run_array[0])
        else:
            # user specified
            start_spectrum = int(self._hidra_wksp.get_spectrum_index(start_sub_run))
        if end_sub_run is None:
            # default end sub run as last sub run (included)
            end_spectrum = num_spectra - 1
            end_sub_run = int(sub_run_array[-1])
        else:
            end_spectrum = int(self._hidra_wksp.get_spectrum_index(end_sub_run))

        # Create peak range/fitting window workspace
        peak_window_ws_name = self._create_peak_window_workspace(num_spectra, [peak_range])

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

        # Create output workspace names
        r_positions_ws_name = 'fitted_peak_positions_{0}'.format(self._mantid_workspace_name)
        r_param_table_name = 'param_m_{0}'.format(self._mantid_workspace_name)
        r_error_table_name = 'param_e_{0}'.format(self._mantid_workspace_name)
        r_model_ws_name = 'model_full_{0}'.format(self._mantid_workspace_name)

        # Set up the default parameter values according to peak profile and instrument property
        peak_param_names, peak_param_values = self._set_default_peak_params_value(peak_function_name, peak_range)

        if DEBUG:
            mantid_helper.export_workspaces([self._mantid_workspace_name, peak_center_ws, peak_window_ws_name])

        # Fit peak by Mantid.FitPeaks
        fit_return = FitPeaks(InputWorkspace=self._mantid_workspace_name,
                              PeakCentersWorkspace=peak_center_ws,
                              FitPeakWindowWorkspace=peak_window_ws_name,
                              PeakFunction=peak_function_name,
                              BackgroundType=background_function_name,
                              StartWorkspaceIndex=start_spectrum,
                              StopWorkspaceIndex=end_spectrum,
                              FindBackgroundSigma=1,
                              HighBackground=True,
                              ConstrainPeakPositions=True,
                              PeakParameterNames=peak_param_names,  # 'Mixing, FWHM, Intensity',
                              PeakParameterValues=peak_param_values,  # '0.8, 0.5, 0.1',
                              RawPeakParameters=True,
                              OutputWorkspace=r_positions_ws_name,
                              OutputPeakParametersWorkspace=r_param_table_name,
                              OutputParameterFitErrorsWorkspace=r_error_table_name,
                              FittedPeaksWorkspace=r_model_ws_name,
                              MaxFitIterations=500)
        # r is a class containing multiple outputs (workspaces)
        if fit_return is None:
            raise RuntimeError('return from FitPeaks cannot be None')

        # Save all the workspaces automatically for further review
        if DEBUG:
            mantid_helper.study_mantid_peak_fitting(self._mantid_workspace_name, r_param_table_name,
                                                    r_model_ws_name, r_positions_ws_name,
                                                    peak_function_name, info=peak_tag)
        # END-IF-DEBUG (True)

        # Process output
        # retrieve workspaces from Analysis cluster
        self._fitted_peak_position_ws = AnalysisDataService.retrieve(r_positions_ws_name)
        self._fitted_function_param_table = AnalysisDataService.retrieve(r_param_table_name)
        self._fitted_function_error_table = AnalysisDataService.retrieve(r_error_table_name)
        self._model_matrix_ws = AnalysisDataService.retrieve(r_model_ws_name)

        # Set the fit result to private class structure numpy arrays
        # Get sub runs considering fitting only being applied to a sub set of sub runs
        sub_runs = sub_run_array[np.where((sub_run_array >= start_sub_run) & (sub_run_array <= end_sub_run))]
        fitted_peak = self._set_profile_parameters_values_from_fitting(peak_tag, sub_runs, peak_function_name,
                                                                       background_function_name)

        # Apply criteria
        fitted_peak.apply_fitting_cost_criteria(max_chi2)
        if min_intensity is not None:
            fitted_peak.apply_intensity_criteria(min_intensity)
        if isinstance(peak_center, (float, int)) and max_peak_shift is not None:
            fitted_peak.apply_peak_position_criteria(peak_center, max_peak_shift)

        return fitted_peak

    # TODO FIXME - peak_tag, peak_center and peak_range will be replaced by a PeakObject class (namedtuple)
    def fit_multiple_peaks(self, sub_run_range, peak_function_name, background_function_name, peak_tag_list,
                           peak_center_list, peak_range_list, max_chi2=1E3, max_peak_shift=2, min_intensity=None):
        """Fit multiple peaks on multiple sub runs

        Parameters
        ----------
        sub_run_range : 2-tuple
            start sub run (None as first 1) and end sub run (None as last 1) for
            range of sub runs (including both end) to refine
        peak_function_name : str
            name of peak profile function
        background_function_name : str
            name of background function
        peak_tag_list : List
            list of str for peak tags
        peak_center_list : List
            list of float for peak centers
        peak_range_list : List
            list of 2-tuple for each peak's range in 2-theta
        max_chi2 : float
            maximum allowed chi2
        max_peak_shift : float or None
            maximum allowed peak shift from specified peak position
        min_intensity : float or None
            minimum allowed peak intensity

        Returns
        -------
        Dict
            dictionary of ~pyrs.core.peak_collection.PeakCollection with peak tag as key

        """
        peak_collection_dict = dict()

        error_message = ''
        for i_peak, peak_tag_i in enumerate(peak_tag_list):
            # get fit setup parameters
            peak_center = peak_center_list[i_peak]
            peak_range = peak_range_list[i_peak]

            # fit peak
            try:
                pl = self.fit_peaks(peak_tag_i, sub_run_range, peak_function_name, background_function_name,
                                    peak_center, peak_range, max_chi2, max_peak_shift, min_intensity)
                peak_collection_dict[peak_tag_i] = pl
            except RuntimeError as run_err:
                error_message += 'Failed to fit (tag) {} due to {}\n'.format(peak_tag_i, run_err)
        # END-FOR

        if len(error_message) > 0:
            raise RuntimeError(error_message)

        return peak_collection_dict

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
            data_type_list = [(name, np.float32) for name in table_col_names]
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
        peak_object = PeakCollection(peak_tag, peak_profile_name, background_type_name)
        peak_object.set_peak_fitting_values(sub_runs, peak_params_value_array, peak_params_error_array, fit_cost_array)

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

    def get_mantid_workspace_name(self):
        """
        get the data workspace name
        :return:
        """
        return self._mantid_workspace_name

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
