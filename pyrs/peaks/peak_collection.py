"""
Object to contain peak parameters (names and values) of a collection of peaks for sub runs
"""
import numpy as np
from pyrs.core.peak_profile_utility import get_parameter_dtype, get_effective_parameters_converter, PeakShape, \
    BackgroundFunction
from pyrs.dataobjects import SubRuns

__all__ = ['PeakCollection']


class PeakCollection(object):
    """
    Class for a collection of peaks
    """
    def __init__(self, peak_tag, peak_profile, background_type):
        """Initialization

        Parameters
        ----------
        peak_tag : str
            tag for the peak such as 'Si111'
        peak_profile : str
            Peak profile
        background_type : str
            Background type

        """
        # Init variables from input
        self._tag = peak_tag

        # Init other parameters
        self._peak_profile = PeakShape.getShape(peak_profile)
        self._background_type = BackgroundFunction.getFunction(background_type)

        # sub run numbers: 1D array
        self._sub_run_array = SubRuns()
        # parameter values: numpy structured array
        self._params_value_array = None
        # parameter fitting error: numpy structured array
        self._params_error_array = None
        # fitting cost (chi2): numpy 1D array
        self._fit_cost_array = None

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
        return str(self._peak_profile)

    @property
    def background_type(self):
        """Get background type

        Returns
        -------
        str
            background type of the profile such as Linear

        """
        return str(self._background_type)

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

    def __convertParameters(self, parameters):
        '''Convert the supplied parameters into an appropriate ndarray'''
        expected_names = self._peak_profile.native_parameters  # background defaults to zero if not provided
        supplied_names = parameters.dtype.names
        for name in expected_names:
            if name not in supplied_names:
                msg = 'Did not find "{}" parameter in fitting results (expected={}, found={})'.format(name,
                                                                                                      expected_names,
                                                                                                      supplied_names)
                raise RuntimeError(msg)
        converted = np.zeros(parameters.size, get_parameter_dtype(self._peak_profile, self._background_type))
        for name in converted.dtype.names:
            converted[name] = parameters[name]

        return converted

    def set_peak_fitting_values(self, subruns, parameter_values, parameter_errors, fit_costs):
        """Set peak fitting values

        Parameters
        ----------
        subruns : numpy.array
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
        self._sub_run_array = SubRuns(subruns)
        self._params_value_array = self.__convertParameters(parameter_values)
        self._params_error_array = self.__convertParameters(parameter_errors)
        self._fit_cost_array = np.copy(fit_costs)

    def get_native_params(self):
        return self._params_value_array, self._params_error_array

    def get_effective_params(self):
        '''
        'Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'
        '''
        converter = get_effective_parameters_converter(self._peak_profile)

        # Convert
        eff_values, eff_errors =\
            converter.calculate_effective_parameters(self._params_value_array, self._params_error_array)

        return eff_values, eff_errors

    def get_integrated_intensity(self):
        pass

    def get_chisq(self):
        return np.copy(self._fit_cost_array)

    def get_subruns(self):
        return self._sub_run_array.raw_copy()  # TODO should this be the actual object

    def get_fit_status(self):
        '''list of messages with "success" being it's good'''
        return [''] * self._sub_run_array.size  # TODO other things should give this information
