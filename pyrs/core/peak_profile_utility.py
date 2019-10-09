"""
Containing peak profiles with method to calculate effective peak parameters and error from native values
"""
import numpy as np


NATIVE_PEAK_PARAMETERS = {'Gaussian': ['Height', 'PeakCentre', 'Sigma', 'A0', 'A1'],
                          'PseudoVoigt': ['Mixing', 'Intensity', 'PeakCentre', 'FWHM', 'A0', 'A1'],
                          'Voigt': ['LorentzAmp', 'LorentzPos', 'LorentzFWHM', 'GaussianFWHM',
                                    'A0', 'A1']}
EFFECTIVE_PEAK_PARAMETERS = ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']


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

    def calculate_effective_parameters(self, native_param_names, param_value_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Parameters
        ----------
        native_param_names: list
            native peak parameter names of specific type of peak profile corresponding to param_value_array
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

    def calculate_effective_parameters(self, native_param_names, param_value_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Parameters
        ----------
        native_param_names: list or None
        param_value_array : numpy.ndarray
            (p, n, 1) or (p, n, 2) vector for parameter values and  optionally fitting error
            p = number of native parameters , n = number of sub runs
        Returns
        -------
        np.ndarray
            (p', n, 1) or (p', n, 2) array for  parameter values and  optionally fitting error
            p' = number of effective parameters , n = number of sub runs
        """
        # Check whether error is included or not: flag to include error in the output
        include_error = param_value_array.shape[3] == 2

        # Output array
        eff_value_array = np.zeros(shape=(len(EFFECTIVE_PEAK_PARAMETERS), param_value_array.shape[1],
                                          param_value_array.shape[2]),
                                   dtype=float)

        # Put to value
        try:
            height_index = native_param_names.index('Height')
            sigma_index = native_param_names.index('Sigma')
            center_index = native_param_names.index('PeakCentre')
            bkgd_a0_index = native_param_names.index('A0')
            bkgd_a1_index = native_param_names.index('A1')
        except ValueError as value_err:
            raise RuntimeError('Input native parameters are not complete: {}'.format(value_err))

        # Calculate effective parameter value
        fwhm_array = self.cal_fwhm(param_value_array[sigma_index, :, 0])
        intensity_array = self.cal_intensity(param_value_array[height_index, :, 0],
                                             param_value_array[sigma_index, :, 0])

        # Set
        eff_value_array[0, :, 0] = native_param_names[center_index, :, 0]  # center
        eff_value_array[1, :, 0] = native_param_names[height_index, :, 0]  # height
        eff_value_array[2, :, 0] = intensity_array[:]  # intensity
        eff_value_array[3, :, 0] = fwhm_array[:]  # FWHM
        eff_value_array[4, :, 0] = 1   # no mixing for Gaussian
        eff_value_array[5, :, 0] = native_param_names[bkgd_a0_index, :, 0]  # A0
        eff_value_array[6, :, 0] = native_param_names[bkgd_a1_index, :, 0]  # A1

        # Error propagation
        if include_error:
            fwhm_error_array = self.cal_fwhm_error(param_value_array[sigma_index, :, 1])
            intensity_error_array = self.cal_intensity_error(intensity_array,
                                                             param_value_array[height_index, :, 0],
                                                             param_value_array[height_index, :, 1],
                                                             param_value_array[sigma_index, :, 0],
                                                             param_value_array[sigma_index, :, 1])

            # Set
            eff_value_array[0, :, 1] = native_param_names[center_index, :, 1]  # center
            eff_value_array[1, :, 1] = native_param_names[height_index, :, 1]  # height
            eff_value_array[2, :, 1] = intensity_error_array[:]  # intensity
            eff_value_array[3, :, 1] = fwhm_error_array[:]  # FWHM
            eff_value_array[4, :, 1] = 0  # no mixing for Gaussian
            eff_value_array[5, :, 1] = native_param_names[bkgd_a0_index, :, 1]  # A0
            eff_value_array[6, :, 1] = native_param_names[bkgd_a1_index, :, 1]  # A1
        # END-IF

        return EFFECTIVE_PEAK_PARAMETERS, eff_value_array

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

    @staticmethod
    def cal_intensity_error(intensity, height, height_error, sigma, sigma_error):
        """Propagate Sigma error and Height error to peak intensity (intensities) error

        I = H * Sigma * sqrt(2 * PI)

        dI = I * sqrt((d(Sigma)/Sigma)^2 + (d(Height)/Height)^2)

        getParameter("Height") * getParameter("Sigma") * sqrt(2.0 * M_PI);
        Returns
        -------
        Float/ndarray, Float/ndarray
            peak intensity and fitting error
        """
        intensity_error = intensity * np.sqrt((sigma_error / sigma)**2 + (height_error / height)**2)

        return intensity_error

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

    @staticmethod
    def cal_fwhm_error(sigma_error):
        """Propagate FWHM error from Sigma error

        FWHM = 2.0 * sqrt(2.0 * ln(2)) * sigma

        Parameters
        ----------
        sigma_error: float or ndarray
            Sigma

        Returns
        -------
        Float/ndarray, Float/ndarray
            peak FWHM and fitting error

        """
        fwhm_error = 2. * np.sqrt(2. * np.log(2.)) * sigma_error

        return fwhm_error

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
