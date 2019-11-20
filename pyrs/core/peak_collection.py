"""
Object to contain peak parameters (names and values) of a collection of peaks for sub runs
"""
import numpy as np
from pyrs.utilities import checkdatatypes
from pyrs.core.peak_profile_utility import get_effective_parameters_converter


class PeakCollection(object):
    """
    Class for a collection of peaks
    """
    def __init__(self, peak_tag):
        """Initialization

        Parameters
        ----------
        peak_tag : str
            tag for the peak such as 'Si111'

        """
        # Init variables from input
        self._tag = peak_tag

        # Init other parameters
        self._peak_profile = None
        self._background_type = None

        # sub run numbers: 1D array
        self._sub_run_array = None
        # parameter values: numpy structured array
        self._params_value_array = None
        # parameter fitting error: numpy structured array
        self._params_error_array = None
        # fitting cost (chi2): numpy 1D array
        self._fit_cost_array = None

        return

    @property
    def peak_tag(self):
        """Peak tag

        Returns
        -------
        str
            Peak tag

        """
        return self._tag

    @property
    def peak_profile(self):
        """Get peak profile name

        Returns
        -------
        str
            peak profile name such as Gaussian

        """
        return self._peak_profile

    @property
    def background_type(self):
        """Get background type

        Returns
        -------
        str
            background type of the profile such as Linear

        """
        return self._background_type

    @property
    def sub_runs(self):
        return self._sub_run_array

    @property
    def parameters_values(self):
        return self._params_value_array

    @property
    def parameters_errors(self):
        return self._params_error_array

    @property
    def fitting_costs(self):
        return self._fit_cost_array

    def set_peak_fitting_values(self, peak_profile, background_type, sub_runs, parameter_values,
                                parameter_errors, fit_costs):
        """Set peak fitting values

        Parameters
        ----------
        peak_profile : str
            Peak profile
        background_type : str
            Background type
        sub_runs : numpy.array
            1D numpy array for sub run numbers
        parameter_values : numpy.ndarray
            numpy structured array for peak/background parameter values
        parameter_errors : numpy.ndarray
            numpy structured array for peak/background parameter fitted error
        fit_costs : numpy.ndarray
            numpy 1D array for

        Returns
        -------

        """
        self._peak_profile = peak_profile
        self._background_type = background_type

        self._sub_run_array = np.copy(sub_runs)
        self._params_value_array = np.copy(parameter_values)
        self._params_error_array = np.copy(parameter_errors)
        self._fit_cost_array = np.copy(fit_costs)

        return

    def get_parameters_values(self, param_name_list, max_chi2=None):
        """Get specified parameters' fitted value and optionally error with optionally filtered value

        The outputs will NOT be numpy structured array but ordered with parameters given in the list

        Parameters
        ----------
        param_name_list : list
            list of parameter names
            If None, use the native parameters
        max_chi2 : None or float
            Default is including all
        Returns
        -------
        tuple
            4-tuple: (1) (n, ) vector for sub run number
                     (2) costs
                     (3) (p, n) array for parameter values
                     (4) (p, n) array for parameter fitting error
            p = number of parameters , n = number of sub runs

        """
        # Check inputs
        checkdatatypes.check_list('Function parameters', param_name_list)
        if max_chi2 is not None:
            checkdatatypes.check_float_variable('Maximum cost chi^2', max_chi2, (1, None))

        num_params = len(param_name_list)

        # Create unfiltered output values
        chi2_vec = np.copy(self._fit_cost_array)
        sub_runs_vec = np.copy(self._sub_run_array)

        # array size: (P, N)  P = number of parameters, N = number of sub runs
        param_value_array = np.zeros(shape=(num_params, sub_runs_vec.shape[0]), dtype='float')
        param_error_array = np.zeros(shape=(num_params, sub_runs_vec.shape[0]), dtype='float')
        # Set value (unfiltered)
        for iparam, param_name in enumerate(param_name_list):
            param_value_array[iparam] = self._params_value_array[param_name]
            param_error_array[iparam] = self._params_error_array[param_name]
        # END-FOR

        # Set filter and create chi2 vector and sub run nun vector
        if max_chi2 is not None and max_chi2 < np.max(self._fit_cost_array):
            # There are runs to be filtered out
            good_fit_indexes = np.where(chi2_vec < max_chi2)
            # Filter
            chi2_vec = chi2_vec[good_fit_indexes]
            sub_runs_vec = sub_runs_vec[good_fit_indexes]
            # parameter values: [P, N] -> [N, P] for filtering -> [P, N']
            param_value_array = param_value_array.transpose()[good_fit_indexes].transpose()
            param_error_array = param_error_array.transpose()[good_fit_indexes].transpose()
        # END-IF

        return sub_runs_vec, chi2_vec, param_value_array, param_error_array

    def get_effective_parameters_values(self, max_chi2=None):
        """
        Get the effective peak parameters including
        peak position, peak height, peak intensity, FWHM and Mixing

        Parameters
        ----------
        max_chi2: float or None
            filtering with chi2

        Returns
        -------
        5-tuple:
                 (0) List as effective peak and background function parameters
                 (1) (n, ) vector for sub run number
                 (2) costs
                 (3) (p, n) array for parameter values
                 (4) (p, n) array for parameter fitting error
            p = number of parameters , n = number of sub runs
        """
        # Create native -> effective parameters converter
        converter = get_effective_parameters_converter(self._peak_profile)

        # Get raw peak parameters
        param_name_list = converter.get_native_peak_param_names()
        param_name_list.extend(converter.get_native_background_names())
        sub_run_array, fit_cost_array, param_value_array, param_error_array = \
            self.get_parameters_values(param_name_list, max_chi2)

        # Convert
        eff_params_list, eff_param_value_array, eff_param_error_array =\
            converter.calculate_effective_parameters(param_name_list, param_value_array, param_error_array)

        return eff_params_list, sub_run_array, fit_cost_array, eff_param_value_array, eff_param_error_array
