# This is the virtual base class as the fitting frame
import numpy
import math
from pyrs.core import workspaces
from pyrs.utilities import rs_project_file
from pyrs.core import peak_profile_utility
from pyrs.utilities import checkdatatypes


class PeakFitEngine(object):
    """
    Virtual peak fit engine
    """

    def __init__(self, workspace, mask_name):
        """
        initialization
        :param workspace: HidraWorksapce containing the diffraction data
        :param mask_name: name of mask ID (or main/None) for reduced diffraction data
        """
        # check
        checkdatatypes.check_type('Diffraction workspace', workspace, workspaces.HidraWorkspace)

        # for scipy: keep the numpy array will be good enough
        self._hd_workspace = workspace  # hd == HiDra
        self._mask_name = mask_name

        # wave length information
        self._wavelength_dict = None

        # for fitted result
        self._peak_center_vec = None  # 2D vector for observed center of mass and highest data point
        self._peak_center_d_vec = None  # 1D vector for calculated center in d-spacing

        # Peak function
        self._peak_function_name = None
        self._background_function_name = None
        self._fit_cost_array = None
        self._peak_params_value_array = None
        self._peak_params_error_array = None
        # shall be a structured numpy array
        # columns are peak and background parameters names, rows are index corresponding to sorted run numbers

        return

    def calculate_peak_position_d(self, wave_length):
        """ Calculate peak positions in d-spacing
        Output: result will be saved to self._peak_center_d_vec
        Parameters
        ----------
        wave_length: float or numpy.ndarray(dtype=float)
            uniform wave length or wave length for each sub run
        Returns
        -------
        None
        """
        # TODO/FIXME - #80+ - Must have a better way than try and guess
        try:
            r = self.get_fitted_params(param_name_list=['PeakCentre'], including_error=True)
        except KeyError:
            r = self.get_fitted_params(param_name_list=['centre'], including_error=True)
        sub_run_vec = r[0]
        params_vec = r[2]

        # Other parameters
        num_sub_runs = sub_run_vec.shape[0]

        # Process wave length
        if isinstance(wave_length, numpy.ndarray):
            assert wave_length.shape[0] == num_sub_runs
            various_wl = True
            wl = 0
        else:
            various_wl = False
            wl = wave_length

        # init vector for peak center in d-spacing with error
        self._peak_center_d_vec = numpy.ndarray((params_vec.shape[1], 2), params_vec.dtype)

        for sb_index in range(num_sub_runs):
            # convert to d-spacing: both fitted value and fitting error
            # set wave length if various to sub runs
            if various_wl:
                wl = wave_length[sb_index]

            # calculate peak position and propagating fitting error
            for sub_index in range(2):
                peak_i_2theta_j = params_vec[0][sb_index][sub_index]
                if wl:
                    peak_i_d_j = wl * 0.5 / math.sin(peak_i_2theta_j * 0.5 * math.pi / 180.)
                else:
                    # case for None or zero
                    peak_i_d_j = -1  # return a non-physical number
                self._peak_center_d_vec[sb_index][0] = peak_i_d_j
        # END-FOR

        return

    def export_to_hydra_project(self, hydra_project_file, peak_tag):
        """Export fit result from this fitting engine instance to Hydra project file

        Parameters
        ----------
        hydra_project_file
        peak_tag

        Returns
        -------

        """
        # Check input
        checkdatatypes.check_type('Hidra project file', hydra_project_file, rs_project_file.HydraProjectFile)

        # Get parameter values
        sub_run_vec = self._hd_workspace.get_sub_runs()

        hydra_project_file.set_peak_fit_result(peak_tag, self._peak_function_name, self._background_function_name,
                                               sub_run_vec, self._fit_cost_array, self._peak_params_value_array,
                                               self._peak_params_error_array)

        return

    def fit_peaks(self, sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                  cal_center_d):
        """ Fit peaks with option to calculate peak center in d-spacing
        :param sub_run_range: range of sub runs (including both end) to refine
        :param peak_function_name:
        :param background_function_name:
        :param peak_center:
        :param peak_range:
        :param cal_center_d:
        :return:
        """
        raise NotImplementedError('Virtual base class member method fit_peaks()')

    @staticmethod
    def _fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                          cal_center_d):
        """Check parameters used to fit peaks

        Parameters
        ----------
        sub_run_range : 2-tuple
            range of sub runs including the last run specified
        peak_function_name : str
            name of peak function
        background_function_name :
            name of background function
        peak_center: float or numpy ndarray
            peak centers
        peak_range:
        cal_center_d : boolean
            flag to calculate d-spacing of fitted peaks

        Returns
        -------
        None
        """
        checkdatatypes.check_tuple('Sub run numbers range', sub_run_range, 2)
        checkdatatypes.check_string_variable('Peak function name', peak_function_name,
                                             allowed_values=['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian'])
        checkdatatypes.check_string_variable('Background function name', background_function_name,
                                             allowed_values=['Linear', 'Flat', 'Quadratic'])
        checkdatatypes.check_bool_variable('Flag to calculate peak center in d-spacing', cal_center_d)
        if not isinstance(peak_center, float or numpy.ndarray):
            raise AssertionError('Peak center {} must be float or numpy array'.format(peak_center))
        checkdatatypes.check_tuple('Peak range', peak_range, 2)

        return

    def get_calculated_peak(self, sub_run_number):
        """
        get the calculated peak's value
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')

    def get_fit_cost(self, max_chi2):
        raise NotImplementedError('This is virtual')

    def _get_fitted_parameters_value(self, spectrum_index_vec, parameter_name_list, param_value_array):
        raise NotImplementedError('This is virtual')

    def get_fitted_params(self, param_name_list, including_error, max_chi2=1.E20):
        """ Get specified parameters' fitted value and optionally error with optionally filtered value
        :param param_name_list:
        :param including_error:
        :param max_chi2: Default is including all.
        :return: 3-tuple: (1) (n, ) vector for sub run number (2) costs
                          (3) (p, n, 1) or (p, n, 2) vector for parameter values
                            and
                            optionally fitting error: p = number of parameters , n = number of sub runs
        """
        # Deal with multiple default
        if max_chi2 is None:
            max_chi2 = 1.E20

        # Check inputs
        checkdatatypes.check_list('Function parameters', param_name_list)
        checkdatatypes.check_bool_variable('Flag to output fitting error', including_error)
        checkdatatypes.check_float_variable('Maximum cost chi^2', max_chi2, (1, None))

        # Get number of sub-runs meets the requirement
        spec_index_vec, fit_cost_vec = self.get_fit_cost(max_chi2)

        # init parameters
        num_sub_runs = fit_cost_vec.shape[0]
        num_params = len(param_name_list)
        if including_error:
            num_items = 2
        else:
            num_items = 1
        param_value_array = numpy.zeros(shape=(num_params, num_sub_runs, num_items), dtype='float')

        # Set values of parameters
        self._get_fitted_parameters_value(spec_index_vec, param_name_list, param_value_array)

        # Convert
        sub_runs_vec = self._hd_workspace.get_sub_runs_from_spectrum(spec_index_vec)

        return sub_runs_vec, fit_cost_vec, param_value_array

    def get_fitted_effective_params(self, including_error, max_chi2=1.E20):
        """
        Get the effective peak parameters including
        peak position, peak height, peak intensity, FWHM and Mixing

        Parameters
        ----------
        including_error: boolean
            returned will include fitting error
        max_chi2: float
            filtering with chi2

        Returns
        -------
        list, ndarray, ndarray, ndarray
            list of string: effective parameter names
            (n,) for sub run numbers
            (n,) for fitting cost
            (p, n, 1) or (p, n, 2) for fitted parameters value,
            p = number of parameters , n = number of sub runs, 2 containing fitting error
        """
        # Create native -> effective parameters converter
        print('[DB...BAT] Current peak function: {}'.format(self._peak_function_name))
        converter = peak_profile_utility.get_effective_parameters_converter(self._peak_function_name)

        # Get raw peak parameters
        param_name_list = converter.get_native_peak_param_names()
        param_name_list.extend(converter.get_native_background_names())
        sub_run_array, fit_cost_array, param_value_array = self.get_fitted_params(param_name_list,
                                                                                  including_error,
                                                                                  max_chi2)

        # Convert
        effective_params_list, effective_param_value_array =\
            converter.calculate_effective_parameters(param_name_list, param_value_array)

        return effective_params_list, sub_run_array, fit_cost_array, effective_param_value_array

    def get_number_scans(self):
        """ Get number of scans in input data to fit
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_number_scans()')

    def get_hidra_workspace(self):
        """
        Get the HidraWorkspace instance associated with this peak fit engine
        :return:
        """
        assert self._hd_workspace is not None, 'No HidraWorkspace has been set up.'

        return self._hd_workspace

    def get_peak_param_names(self, peak_function, is_effective):
        """ Get the peak parameter names
        :param peak_function: None for default/current peak function
        :param is_effective:
        :return:
        """
        # Default
        if peak_function is None:
            peak_function = self._peak_function_name

        if is_effective:
            # Effective parameters
            param_names = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS[:]
            if peak_function == 'PseudoVoigt':
                param_names.append('Mixing')

        else:
            # Native parameters
            try:
                param_names = peak_profile_utility.NATIVE_PEAK_PARAMETERS[peak_function][:]
            except KeyError as key_err:
                raise RuntimeError('Peak type {} not supported.  The supported peak functions are {}.  FYI: {}'
                                   ''.format(peak_function,
                                             peak_profile_utility.NATIVE_PEAK_PARAMETERS.keys(), key_err))

        return param_names

    def set_wavelength(self, wavelengths):
        """

        :param wavelengths:
        :return:
        """
        # TODO - #80 NOW - Implement
        self._wavelength_dict = wavelengths

        return
