"""
Object to contain peak parameters (names and values) of a collection of peaks for sub runs
"""
import numpy as np


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
