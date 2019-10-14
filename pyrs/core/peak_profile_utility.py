"""
Containing peak profiles with method to calculate effective peak parameters and error from native values
"""
import numpy as np


# Native peak parameters in Mantid naming convention
NATIVE_PEAK_PARAMETERS = {'Gaussian': ['Height', 'PeakCentre', 'Sigma'],
                          'PseudoVoigt': ['Mixing', 'Intensity', 'PeakCentre', 'FWHM'],
                          'Voigt': ['LorentzAmp', 'LorentzPos', 'LorentzFWHM', 'GaussianFWHM']}
# Native background parameters in Mantid naming convention
NATIVE_BACKGROUND_PARAMETERS = {'Linear': ['A0', 'A1']}
# Effective peak and background parameters
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
    """Virtual base class to convert peak parameters from native to effective

    """
    def __init__(self):
        """Initialization
        """
        # Peak profile name
        self._peak_name = 'Virtual Peak'
        # Background name
        self._background_name = 'Linear'  # so far, one and only supported

        return

    def get_native_peak_param_names(self):
        """
        Get the list of native peak parameters
        Returns
        -------
        List
            list of string for peak parameter name used in Mantid as the standard
        """
        return NATIVE_PEAK_PARAMETERS[self._peak_name][:]

    def get_native_background_names(self):
        """
        Get the list of background parameters
        Returns
        -------

        """
        return NATIVE_BACKGROUND_PARAMETERS[self._background_name][:]

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
        super(Gaussian, self).__init__()

        self._peak_name = 'Gaussian'

        return

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
        include_error = param_value_array.shape[2] == 2

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
            raise RuntimeError('Input native parameters (\n\t{}\n)are not complete: {}.'
                               ''.format(native_param_names, value_err))

        # Calculate effective parameter value
        fwhm_array = self.cal_fwhm(param_value_array[sigma_index, :, 0])
        intensity_array = self.cal_intensity(param_value_array[height_index, :, 0],
                                             param_value_array[sigma_index, :, 0])

        # Set
        eff_value_array[0, :, 0] = param_value_array[center_index, :, 0]  # center
        eff_value_array[1, :, 0] = param_value_array[height_index, :, 0]  # height
        eff_value_array[2, :, 0] = intensity_array[:]  # intensity
        eff_value_array[3, :, 0] = fwhm_array[:]  # FWHM
        eff_value_array[4, :, 0] = 1   # no mixing for Gaussian
        eff_value_array[5, :, 0] = param_value_array[bkgd_a0_index, :, 0]  # A0
        eff_value_array[6, :, 0] = param_value_array[bkgd_a1_index, :, 0]  # A1

        # Error propagation
        if include_error:
            fwhm_error_array = self.cal_fwhm_error(param_value_array[sigma_index, :, 1])
            intensity_error_array = self.cal_intensity_error(intensity_array,
                                                             param_value_array[height_index, :, 0],
                                                             param_value_array[height_index, :, 1],
                                                             param_value_array[sigma_index, :, 0],
                                                             param_value_array[sigma_index, :, 1])

            # Set
            eff_value_array[0, :, 1] = param_value_array[center_index, :, 1]  # center
            eff_value_array[1, :, 1] = param_value_array[height_index, :, 1]  # height
            eff_value_array[2, :, 1] = intensity_error_array[:]  # intensity
            eff_value_array[3, :, 1] = fwhm_error_array[:]  # FWHM
            eff_value_array[4, :, 1] = 0  # no mixing for Gaussian
            eff_value_array[5, :, 1] = param_value_array[bkgd_a0_index, :, 1]  # A0
            eff_value_array[6, :, 1] = param_value_array[bkgd_a1_index, :, 1]  # A1
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


class PseudoVoigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        """Initialization

        """
        super(PseudoVoigt, self).__init__()

        self._peak_name = 'PseudoVoigt'

        return

    def calculate_effective_parameters(self, native_param_names, param_value_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Native PseudoVoigt: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']

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
            intensity_index = native_param_names.index('Intensity')
            fwhm_index = native_param_names.index('FWHM')
            center_index = native_param_names.index('PeakCentre')
            mix_index = native_param_names.index('Mixing')
            bkgd_a0_index = native_param_names.index('A0')
            bkgd_a1_index = native_param_names.index('A1')
        except ValueError as value_err:
            raise RuntimeError('Input native parameters are not complete: {}'.format(value_err))

        # Calculate effective parameter value
        heights = self.cal_height(intensity=param_value_array[intensity_index, :, 0],
                                  fwhm=param_value_array[fwhm_index, :, 0],
                                  mixing=param_value_array[mix_index, :, 0])

        # Set
        eff_value_array[0, :, 0] = param_value_array[center_index, :, 0]  # center
        eff_value_array[1, :, 0] = heights[:]  # height
        eff_value_array[2, :, 0] = param_value_array[intensity_index, :, 0]  # intensity
        eff_value_array[3, :, 0] = param_value_array[fwhm_index, :, 0]  # FWHM
        eff_value_array[4, :, 0] = param_value_array[mix_index, :, 0]  # no mixing for Gaussian
        eff_value_array[5, :, 0] = param_value_array[bkgd_a0_index, :, 0]  # A0
        eff_value_array[6, :, 0] = param_value_array[bkgd_a1_index, :, 0]  # A1

        # Error propagation
        if include_error:
            # Calculate effective parameter value
            heights_error = self.cal_height_error(heights,
                                                  param_value_array[intensity_index, :, 0],
                                                  param_value_array[intensity_index, :, 1],
                                                  param_value_array[fwhm_index, :, 0],
                                                  param_value_array[fwhm_index, :, 1],
                                                  param_value_array[mix_index, :, 0],
                                                  param_value_array[mix_index, :, 1])

            # Set
            eff_value_array[0, :, 1] = param_value_array[center_index, :, 0]  # center
            eff_value_array[1, :, 1] = heights_error[:]  # height
            eff_value_array[2, :, 1] = param_value_array[intensity_index, :, 0]  # intensity
            eff_value_array[3, :, 1] = param_value_array[fwhm_index, :, 0]  # FWHM
            eff_value_array[4, :, 1] = param_value_array[mix_index, :, 0]  # no mixing for Gaussian
            eff_value_array[5, :, 1] = param_value_array[bkgd_a0_index, :, 0]  # A0
            eff_value_array[6, :, 1] = param_value_array[bkgd_a1_index, :, 0]  # A1
        # END-IF

        return EFFECTIVE_PEAK_PARAMETERS, eff_value_array

    @staticmethod
    def cal_height(intensity, fwhm, mixing):
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
        Float/ndarray
            peak height
        """
        height = 2. * intensity * (1 + (np.sqrt(np.pi * np.log(2)) - 1) * mixing) / (np.pi * fwhm)

        return height

    @staticmethod
    def cal_height_error(height, intensity, intensity_error, fwhm, fwhm_error, mixing, mixing_error):
        """Calculate propagated error of peak height

        Parameters
        ----------
        height
        intensity
        intensity_error
        fwhm
        fwhm_error
        mixing
        mixing_error

        Returns
        -------
        Float/ndarray
            Peak height fitting error
        """
        # TODO - ASAP

        return None


class Voigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        """Initialization

        """
        super(Voigt, self).__init__()

        self._peak_name = 'Voigt'

        return

    def calculate_effective_parameters(self, native_param_names, param_value_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Native PseudoVoigt: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']

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

        # Calculate effective parameters
        # ... ...

        # Optionally propagate the uncertainties
        if include_error:
            pass

        # TODO - Implement the main body ASAP
        raise NotImplementedError('ASAP')

        return EFFECTIVE_PEAK_PARAMETERS, eff_value_array


"""
From here are a list of static method of peak profiles
"""


def gaussian(x, a, sigma, x0):
    """
    Gaussian with linear background
    :param x:
    :param a:
    :param sigma:
    :param x0:
    :return:
    """
    return a * np.exp(-((x - x0) / sigma) ** 2)


def pseudo_voigt(x, intensity, fwhm, mixing, x0):
    """PseudoVoigt function

    References: https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html

    Parameters
    ----------
    x
    intensity
    fwhm
    mixing
    x0

    Returns
    -------

    """
    # Calculate normalized Gaussian part
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2)))
    part_gauss = gaussian(x, a=1 / (2. * np.sqrt(2 * np.pi)),
                          sigma=sigma, x0=x0)

    # Calculate normalized Lorentzian
    part_lorenz = lorenzian(x, 1., fwhm, x0)

    # Together
    pv = intensity * (mixing * part_gauss + (1 - mixing) * part_lorenz)

    return pv


def lorenzian(x, a, fwhm, x0):
    """Normalized Lorentzian

    Parameters
    ----------
    x
    a
    fwhm
    x0

    Returns
    -------
    float or np.ndarray
    """
    return a * fwhm * 0.5 / (np.pi * ((x - x0)**2 + (fwhm * 0.5)**2))


def quadratic_background(x, b0, b1, b2, b3):
    """
    up to 3rd order
    :param x:
    :param b0:
    :param b1:
    :param b2:
    :param b3:
    :return:
    """
    return b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3


def fit_peak(peak_func, vec_x, obs_vec_y, p0, p_range):
    """

    :param peak_func:
    :param vec_x:
    :param obs_vec_y:
    :param p0:
    :param p_range: example  # bounds=([a, b, c, x0], [a, b, c, x0])
    :return:
    """
    import scipy.optimize

    def calculate_chi2(covariance_matrix):
        """

        :param covariance_matrix:
        :return:
        """
        # TODO
        return 1.

    # check inputs
    # fit
    fit_results = scipy.optimize.curve_fit(peak_func, vec_x, obs_vec_y, p0=p0, bounds=p_range)

    fit_params = fit_results[0]
    fit_covmatrix = fit_results[1]
    cost = calculate_chi2(fit_covmatrix)

    return cost, fit_params, fit_covmatrix
