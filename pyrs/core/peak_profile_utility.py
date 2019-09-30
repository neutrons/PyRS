"""
Containing peak profiles with method to calculate effective peak parameters and error from native values
"""
import numpy as np


def peak_parameter_converter(profile_name):
    """

    Parameters
    ----------
    profile_name

    Returns
    -------
    Converter Instance
    """
    profile_name = profile_name.lower()
    if profile_name == 'gaussian':
        converter = Gaussian(param_dict)
    elif profile_name == 'pseudovoigt':
        converter = PseudoVoigt(param_dict)
    elif profile_name == 'voigt':
        converter = Voigt(param_dict)
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


class Gaussian(PeakParametersConverter):
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

