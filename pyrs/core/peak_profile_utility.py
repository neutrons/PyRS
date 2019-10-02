"""
Containing peak profiles with method to calculate effective peak parameters and error from native values
"""
import numpy as np


def get_effective_parameters_converter(peak_profile):
    """
    This is a factory method to create a proper native to effective peak parameters value converter
    Parameters
    ----------
    peak_profile: string
        peak profile name including Gaussian, PseudoVoigt and Voigt

    Returns
    -------
    PeakParametersConverter
        Gaussian, PseudoVoigt or Voigt
    """
    profile_name = peak_profile.lower()
    if profile_name == 'gaussian':
        converter = Gaussian()
    elif profile_name == 'pseudovoigt':
        converter = PseudoVoigt()
    elif profile_name == 'voigt':
        converter = Voigt()
    else:
        raise RuntimeError('Profile {} is not supported.'.format(profile_name))

    return converter


class PeakParametersConverter(object):
    """

    """
    def __init__(self, param_value_dict):
        """

        Parameters
        ----------
        param_value_dict
        """

    @staticmethod
    def get_native_peak_param_names():
        """
        Get the list of native peak parameters
        Returns
        -------

        """
        raise NotImplementedError('Virtual')

    def calculate_effective_parameters(self, effective_params_list, param_value_array):
        """

        Parameters
        ----------
        effective_params_list
        param_value_array : numpy.ndarray
            (p, n, 1) or (p, n, 2) vector for parameter values and  optionally fitting error
            p = number of native parameters , n = number of sub runs
        Returns
        -------
        np.ndarray
            (p', n, 1) or (p', n, 2) array for  parameter values and  optionally fitting error
            p' = number of effective parameters , n = number of sub runs
        """
        raise NotImplementedError('Virtual')


class Gaussian(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        """
        Initialization
        """
        super(PeakParametersConverter, self).__init__()

        return

    @staticmethod
    def get_native_peak_param_names():
        """
        Get the list of native peak parameters
        Returns
        -------
        List
            list of string for peak parameter name used in Mantid as the standard
        """
        return ['Height', 'PeakCentre', 'Sigma']

    def calculate_effective_parameters(self, effective_params_list, param_value_array):
        """

        Parameters
        ----------
        effective_params_list
        param_value_array : numpy.ndarray
            (p, n, 1) or (p, n, 2) vector for parameter values and  optionally fitting error
            p = number of native parameters , n = number of sub runs
        Returns
        -------
        np.ndarray
            (p', n, 1) or (p', n, 2) array for  parameter values and  optionally fitting error
            p' = number of effective parameters , n = number of sub runs
        """
        height_index = effective_params_list.index('Height')
        peak_center_index = effective_params_list.index('PeakCentre')
        sigma_index = effective_params_list.index('Sigma')

        heights = param_value_array[height_index, :, 0]
        peak_center_index = peak_center_index[peak_center_index, :, 0]





    def get_intensity(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak intensity and fitting error
        """

    def get_height(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak height and fitting error
        """

        return

    def get_fwhm(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak FWHM and fitting error
        """
        return

    def get_mixing(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            Gaussian/Lorentzian mixing for PseudoVoigt and Voigt and fitting error
        """

        return


class PseudoVoigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self, param1, param2):
        """
        Initialization
        """

        self._param1 = blabla
        self._param2 = blabla

        if isinstance(param1, float):
            self._dimension = None
        else:
            self._dimension = param1.shape

        return

    @staticmethod
    def get_native_peak_param_names():
        """
        Get the list of native peak parameters
        Returns
        -------
        List
            list of string for peak parameter name used in Mantid as the standard
        """
        return ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']

    def get_intensity(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak intensity and fitting error
        """

    def get_height(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak height and fitting error
        """

        return

    def get_fwhm(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak FWHM and fitting error
        """
        return

    def get_mixing(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            Gaussian/Lorentzian mixing for PseudoVoigt and Voigt and fitting error
        """

        return


class Voigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self, param1, param2):
        """
        Initialization
        """

        self._param1 = blabla
        self._param2 = blabla

        if isinstance(param1, float):
            self._dimension = None
        else:
            self._dimension = param1.shape

        return

    @staticmethod
    def get_native_peak_param_names():
        """
        Get the list of native peak parameters
        Returns
        -------
        List
            list of string for peak parameter name used in Mantid as the standard
        """
        return ['LorentzAmp', 'LorentzPos', 'LorentzFWHM', 'GaussianFWHM']

    def get_intensity(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak intensity and fitting error
        """


    def get_height(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak height and fitting error
        """

        return

    def get_fwhm(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak FWHM and fitting error
        """
        return

    def get_mixing(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            Gaussian/Lorentzian mixing for PseudoVoigt and Voigt and fitting error
        """

        return

