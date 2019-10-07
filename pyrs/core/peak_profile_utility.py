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
        # peak_center_index = effective_params_list.index('PeakCentre')
        sigma_index = effective_params_list.index('Sigma')

        heights = param_value_array[height_index, :, 0]
        sigmas = param_value_array[sigma_index, :, 0]
        # peak_center_index = peak_center_index[peak_center_index, :, 0]

        # calculate FWHM
        self.cal_fwhm(param_value_array[sigma_index])
        # calculate intensity
        self.cal_intensity(heights, sigmas)

        return

    @staticmethod
    def cal_intensity(height, sigma):
        """ Calculate peak intensity (intensities)
        I = H * Sigma * sqrt(2 * PI)

        getParameter("Height") * getParameter("Sigma") * sqrt(2.0 * M_PI);
        Returns
        -------
        Float/ndarray, Float/ndarray
            peak intensity and fitting error
        """
        intensity = np.sqrt(2. * np.pi) * height * sigma

        return intensity

    def cal_height(self):
        """

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak height and fitting error
        """
        raise RuntimeError('Peak height is native ')

    @staticmethod
    def cal_fwhm(sigma):
        """
        Calculate FWHM from sigma
        FWHM = 2.0 * sqrt(2.0 * ln(2)) * sigma

        Parameters
        ----------
        sigma: float or ndarray
            Sigma

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak FWHM and fitting error

        """
        fwhm = 2. * np.sqrt(2. * np.log(2.)) * sigma

        return fwhm

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

    def cal_height(self, intensity, fwhm, mixing):
        """
        intensity =  m_height / 2. / (1 + (sqrt(M_PI * M_LN2) - 1) * eta) * (M_PI * gamma)
        -->
        height = 2 *  (1 + (sqrt(M_PI * M_LN2) - 1) * eta) * intensity / (M_PI * gamma)

        Parameters
        ----------
        intensity
        fwhm
        mixing

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak height and fitting error
        """
        height = 2. * intensity * (1 + (np.sqrt(np.pi * np.log(2)) - 1) * mixing) / (np.pi * fwhm)

        return height

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
