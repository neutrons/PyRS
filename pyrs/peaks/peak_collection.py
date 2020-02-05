from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import numpy as np
from pyrs.core.peak_profile_utility import get_parameter_dtype, get_effective_parameters_converter, PeakShape, \
    BackgroundFunction
from pyrs.dataobjects import SubRuns

__all__ = ['PeakCollection']


class PeakCollection(object):
    """
    Object to contain peak parameters (names and values) of a collection of peaks for sub runs
    """
    def __init__(self, peak_tag, peak_profile, background_type, wavelength=np.nan, d_reference=np.nan):
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
        self._wavelength = wavelength
        self._d_reference = d_reference

        # sub run numbers: 1D array
        self._sub_run_array = SubRuns()
        # parameter values: numpy structured array
        self._params_value_array = None
        # parameter fitting error: numpy structured array
        self._params_error_array = None
        # fitting cost (chi2): numpy 1D array
        self._fit_cost_array = None
        # status messages: list of strings
        self._fit_status = None

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

    def __set_fit_status(self):
        '''This requires self._fit_cost_array and self._params_error_array to be set first'''
        # default value is that everything worked
        self._fit_status = ['success'] * self._sub_run_array.size

        # check individual parameter errors
        bad_params = np.zeros(self._sub_run_array.size, dtype=bool)  # nothing is bad
        for name in self._params_error_array.dtype.names:
            bad_params = np.logical_or(bad_params, self._params_error_array[name] == 0.)
            bad_params = np.logical_or(bad_params, np.logical_not(np.isfinite(self._params_error_array[name])))
        if np.sum(bad_params) > 0:
            for i in np.where(bad_params)[0]:
                self._fit_status[i] = 'did not refine all parameters'

        # chisq in general is bad
        bad_chisq = np.logical_not(np.isfinite(self._fit_cost_array))
        if np.sum(bad_chisq) > 0:
            for i in np.where(bad_chisq)[0]:
                self._fit_status[i] = 'failed'

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
        self.__set_fit_status()

    def get_d_reference(self):
        return self._d_reference

    def set_d_reference(self, values):
        """Set d reference values

        Parameters
        ----------
        values :
            1D numpy array or floats

        Returns
        -------

        """
        self._d_reference = [values] * self._sub_run_array.size

    def get_strain(self, values=np.nan):
        """get strain values and uncertainties

          Parameters
          ----------
          values :
              1D numpy array or floats

          Returns
          -------
            tuple
                A two-item tuple containing the strain and its uncertainty.
          """
        d_fitted, d_fitted_error = self.get_dspacing_center()
        self.set_d_reference(values)
        strain = (d_fitted - self.get_d_reference())/self.get_d_reference()
        strain_error = np.reciprocal(self.get_d_reference())
        strain_error = strain_error.tolist()
        return strain, strain_error

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

    def get_dspacing_center(self):
        r"""
        peak center in unit of d spacing.

        Returns
        -------
        tuple
            A two-item tuple containing the peak center and its uncertainty.
        """
        effective_values, effective_errors = self.get_effective_params()
        theta_center_value = np.array([value['Center'] / 2 for value in effective_values])
        theta_center_error = np.array([error['Center'] / 2 for error in effective_errors])
        dspacing_center = self._wavelength / (2 * np.sin(theta_center_value))
        dspacing_center_error = self._wavelength * abs(np.cos(theta_center_value)) * theta_center_error /\
            (2 * np.sin(theta_center_value)**2)
        return dspacing_center, dspacing_center_error

    def get_integrated_intensity(self):
        pass

    def get_chisq(self):
        return np.copy(self._fit_cost_array)

    def get_subruns(self):
        return self._sub_run_array.raw_copy()  # TODO should this be the actual object

    def get_fit_status(self):
        '''list of messages with "success" being it's good'''
        return self._fit_status
