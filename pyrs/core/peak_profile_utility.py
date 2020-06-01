"""
Containing peak profiles with method to calculate effective peak parameters and error from native values
"""
from enum import Enum
import numpy as np


# Effective peak and background parameters
EFFECTIVE_PEAK_PARAMETERS = ['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']


class PeakShape(Enum):
    GAUSSIAN = 'Gaussian'
    PSEUDOVOIGT = 'PseudoVoigt'
    VOIGT = 'Voigt'

    def __str__(self):
        return self.value

    @staticmethod
    def getShape(shape):
        try:  # for python 3
            shape = shape.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        if isinstance(shape, PeakShape):
            return shape
        else:
            try:
                return PeakShape[str(shape).upper()]
            except KeyError:
                raise KeyError('Cannot determine peak shape from "{}"'.format(shape))

    @property
    def native_parameters(self):
        # Native peak parameters in Mantid naming convention
        NATIVE_PEAK_PARAMETERS = {'Gaussian': ['Height', 'PeakCentre', 'Sigma'],
                                  'PseudoVoigt': ['Mixing', 'Intensity', 'PeakCentre', 'FWHM'],
                                  'Voigt': ['LorentzAmp', 'LorentzPos', 'LorentzFWHM', 'GaussianFWHM']}

        return NATIVE_PEAK_PARAMETERS[self.value][:]


class BackgroundFunction(Enum):
    LINEAR = 'Linear'  # so far, one and only supported

    def __str__(self):
        return self.value

    @staticmethod
    def getFunction(function):
        try:  # for python 3
            function = function.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        if isinstance(function, BackgroundFunction):
            return function
        else:
            try:
                return BackgroundFunction[str(function).upper()]
            except KeyError:
                raise KeyError('Cannot determine background function from "{}"'.format(function))

    @property
    def native_parameters(self):
        # Native background parameters in Mantid naming convention
        NATIVE_BACKGROUND_PARAMETERS = {'Linear': ['A0', 'A1']}

        return NATIVE_BACKGROUND_PARAMETERS[self.value][:]


def get_parameter_dtype(peak_shape=None, background_function=None, effective=False):
    '''Convert the peak parameters into a dtype to ge used in numpy constructors

    ``np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype('Gaussian'))``
    '''
    if effective:
        param_names = EFFECTIVE_PEAK_PARAMETERS
    else:
        param_names = PeakShape.getShape(peak_shape).native_parameters
        if background_function is not None:
            param_names.extend(BackgroundFunction.getFunction(background_function).native_parameters)

    return [(name, np.float32) for name in param_names]


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
    try:
        peak_profile = PeakShape.getShape(peak_profile)
    except KeyError:
        raise KeyError('Profile {} is not supported.'.format(peak_profile))

    if peak_profile == PeakShape.GAUSSIAN:
        converter = Gaussian()
    elif peak_profile == PeakShape.PSEUDOVOIGT:
        converter = PseudoVoigt()
    elif peak_profile == PeakShape.VOIGT:
        converter = Voigt()
    else:
        raise RuntimeError('if/else tree is incomplete')

    return converter


class PeakParametersConverter(object):
    """Virtual base class to convert peak parameters from native to effective

    """
    def __init__(self, peak_shape):
        """Initialization
        """
        # Peak profile name
        self._peak_shape = PeakShape.getShape(peak_shape)
        # Background name
        self._background = BackgroundFunction.LINEAR  # so far, one and only supported

    def calculate_effective_parameters(self, param_value_array, param_error_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Parameters
        ----------
        param_value_array : numpy.ndarray
            (p, n) array for parameter values
            p = number of native parameters , n = number of sub runs
        param_error_array : numpy.ndarray
            (p, n) array for parameter fitting error
            p = number of native parameters , n = number of sub runs
        Returns
        -------
        np.ndarray, np.ndarray
            (p', n) and (p', n) array for  parameter values and  fitting error respectively
            p' = number of effective parameters , n = number of sub runs

        """
        raise NotImplementedError('Virtual')


class Gaussian(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        super(Gaussian, self).__init__(PeakShape.GAUSSIAN)

    def calculate_effective_parameters(self, param_value_array, param_error_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Parameters
        ----------
        param_value_array : numpy.ndarray
            (p, n) array for parameter values
            p = number of native parameters , n = number of sub runs
        param_error_array : numpy.ndarray
            (p, n) array for parameter fitting error
            p = number of native parameters , n = number of sub runs
        Returns
        -------
        np.ndarray, np.ndarray
            (p', n) and (p', n) array for  parameter values and  fitting error respectively
            p' = number of effective parameters , n = number of sub runs
        """
        # error check inputs
        if param_value_array.dtype != param_error_array.dtype:
            raise RuntimeError('dtype of values and errors do not match: {} and {}'.format(param_value_array.dtype,
                                                                                           param_error_array.dtype))
        if param_value_array.size != param_error_array.size:
            raise RuntimeError('size of values and errors do not match: {} and {}'.format(param_value_array.size,
                                                                                          param_error_array.size))
        for name in PeakShape.GAUSSIAN.native_parameters:
            if name not in param_value_array.dtype.names:
                raise RuntimeError('Did not find "{}" in dtype={}'.format(name, param_value_array.dtype.names))

        # Output array
        eff_value_array = np.zeros(param_value_array.size, dtype=get_parameter_dtype(effective=True))
        eff_error_array = np.zeros(param_value_array.size, dtype=get_parameter_dtype(effective=True))

        # Calculate effective parameters' value: FWHM, Intensity
        fwhm_array = self.cal_fwhm(param_value_array['Sigma'])
        intensity_array = self.cal_intensity(param_value_array['Height'],
                                             param_value_array['Sigma'])

        # Set effective parameter values
        eff_value_array['Center'] = param_value_array['PeakCentre']  # center
        eff_value_array['Height'] = param_value_array['Height']  # height
        eff_value_array['FWHM'] = fwhm_array[:]  # FWHM
        eff_value_array['Mixing'] = 1.   # no mixing for Gaussian
        eff_value_array['A0'] = param_value_array['A0']  # A0
        eff_value_array['A1'] = param_value_array['A1']  # A1
        eff_value_array['Intensity'] = intensity_array[:]  # intensity

        # Calculate error propagation
        fwhm_error_array = self.cal_fwhm_error(param_error_array['Sigma'])
        intensity_error_array = self.cal_intensity_error(intensity_array,
                                                         param_value_array['Height'],
                                                         param_error_array['Height'],
                                                         param_value_array['Sigma'],
                                                         param_error_array['Sigma'])

        # Set Error
        eff_error_array['Center'] = param_error_array['PeakCentre']  # center
        eff_error_array['Height'] = param_error_array['Height']  # height
        eff_error_array['FWHM'] = fwhm_error_array[:]  # FWHM
        eff_error_array['Mixing'] = 0.  # no uncertainty in mixing for Gaussian
        eff_error_array['A0'] = param_error_array['A0']  # A0
        eff_error_array['A1'] = param_error_array['A1']  # A1
        eff_error_array['Intensity'] = intensity_error_array[:]  # intensity

        return eff_value_array, eff_error_array

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

    @staticmethod
    def cal_sigma(fwhm):
        """Calculate Sigma from FWHM

        Parameters
        ----------
        fwhm

        Returns
        -------
        float
            Sigma

        """
        sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))

        return sigma


class PseudoVoigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        super(PseudoVoigt, self).__init__(PeakShape.PSEUDOVOIGT)

    def calculate_effective_parameters(self, param_value_array, param_error_array):
        """Calculate effective peak parameter values

        If input parameter values include fitting error, then this method will calculate
        the propagation of error

        Native PseudoVoigt: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']

        Parameters
        ----------
        param_value_array : numpy.ndarray
            (p, n) array for parameter values
            p = number of native parameters , n = number of sub runs
        param_error_array : numpy.ndarray
            (p, n) array for parameter fitting error
            p = number of native parameters , n = number of sub runs

        Returns
        -------
        np.ndarray, np.ndarray
            (p', n) and (p', n) array for  parameter values and  fitting error respectively
            p' = number of effective parameters , n = number of sub runs

        """
        # error check inputs
        if param_value_array.dtype != param_error_array.dtype:
            raise RuntimeError('dtype of values and errors do not match: {} and {}'.format(param_value_array.dtype,
                                                                                           param_error_array.dtype))
        if param_value_array.size != param_error_array.size:
            raise RuntimeError('size of values and errors do not match: {} and {}'.format(param_value_array.size,
                                                                                          param_error_array.size))
        for name in PeakShape.PSEUDOVOIGT.native_parameters:
            if name not in param_value_array.dtype.names:
                raise RuntimeError('Did not find "{}" in dtype={}'.format(name, param_value_array.dtype.names))

        # Output array
        eff_value_array = np.zeros(param_value_array.size, dtype=get_parameter_dtype(effective=True))
        eff_error_array = np.zeros(param_value_array.size, dtype=get_parameter_dtype(effective=True))

        # Calculate effective parameter value
        heights = self.cal_height(intensity=param_value_array['Intensity'],
                                  fwhm=param_value_array['FWHM'],
                                  mixing=param_value_array['Mixing'])

        # Set effective parameters
        eff_value_array['Center'] = param_value_array['PeakCentre']  # center
        eff_value_array['Height'] = heights[:]  # height
        eff_value_array['FWHM'] = param_value_array['FWHM']  # FWHM
        eff_value_array['Mixing'] = param_value_array['Mixing']  # no mixing for Gaussian
        eff_value_array['A0'] = param_value_array['A0']  # A0
        eff_value_array['A1'] = param_value_array['A1']  # A1
        eff_value_array['Intensity'] = param_value_array['Intensity']  # intensity

        # Calculate error propagation: effective parameter value
        heights_error = self.cal_height_error(param_value_array['Intensity'],
                                              param_error_array['Intensity'],
                                              param_value_array['FWHM'],
                                              param_error_array['FWHM'],
                                              param_value_array['Mixing'],
                                              param_error_array['Mixing'])

        # Set
        eff_error_array['Center'] = param_error_array['PeakCentre']  # center
        eff_error_array['Height'] = heights_error[:]  # height
        eff_error_array['FWHM'] = param_error_array['FWHM']  # FWHM
        eff_error_array['Mixing'] = param_error_array['Mixing']  # no mixing for Gaussian
        eff_error_array['A0'] = param_error_array['A0']  # A0
        eff_error_array['A1'] = param_error_array['A1']  # A1
        eff_error_array['Intensity'] = param_error_array['Intensity']  # intensity

        return eff_value_array, eff_error_array

    @staticmethod
    def cal_height(intensity, fwhm, mixing):
        """Calculate peak height from I(intensity), Gamma (fwhm) and eta (mixing)

        According to Mantid doc:
        h = 2 * I * (1 + ((pi * ln 2)^(1/2) - 1) * eta) / (pi * Gamma)

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
    def cal_height_error(intensity, intensity_error, fwhm, fwhm_error, mixing, mixing_error):
        """Calculate propagated error of peak height

        Note: 's' is used for uncertainty/error
        - s_i : sigma_I
        - s_g : sigma_Gamma (sigma_FWHM)
        - s_e : sigma_eta (mixing)
        - s_h : sigma_height (output)

        s_h^2 = (partial h()/partial I)^2 s_i^2 + (partial h()/partial Gamma)^2 s_g^2
              + (partial h()/partial eta)^2 s_e^2

        Parameters
        ----------
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
        # Define a factor
        mixing_factor = np.sqrt(np.pi * np.log(2)) - 1
        two_inv_pi = 2. / np.pi

        # FIXME - all the terms shall get SQUARED!
        # Partial derivative to intensity
        # partial h()/partial I = 2. * (1 + (np.sqrt(np.pi * np.log(2)) - 1) * mixing) / (np.pi * fwhm)
        #                       = (2 / np.pi) * (1 + F1 * mixing) / fwhm
        part_h_part_i = two_inv_pi * (1 + mixing_factor * mixing) / fwhm

        # Partial derivative to FWHM
        # partial h()/partial G = -2. * intensity * (1 + (np.sqrt(np.pi * np.log(2)) - 1) * mixing) / (np.pi * fwhm^2)
        part_h_part_gamma = -two_inv_pi * intensity * (1 + mixing_factor * mixing) / fwhm**2

        # Partial derivative to Eta
        # partial h()/partial eta = 2 * I * ((pi * ln 2)^(1/2) - 1) / (pi * Gamma)
        part_h_part_eta = two_inv_pi * intensity * mixing_factor / fwhm

        # sum
        s_h2 = (part_h_part_i * intensity_error)**2 + (part_h_part_gamma * fwhm_error)**2 + \
               (part_h_part_eta * mixing_error)**2

        return np.sqrt(s_h2)

    @staticmethod
    def cal_intensity(height, fwhm, mixing):
        """Calculate peak intensity

        I = h * pi * fwhm * 0.5 / (1 + mixing * (sqrt(pi * log2) - 1)))

        Parameters
        ----------
        height : float
            peak height
        fwhm : float
            full width half maximum
        mixing : float
            mixing value

        Returns
        -------
        float

        """
        intensity = 0.5 * height * np.pi * fwhm / (1 + mixing * (np.sqrt(np.pi * np.log(2)) - 1.0))

        return intensity


class Voigt(PeakParametersConverter):
    """
    class for handling peak profile parameters' conversion
    """
    def __init__(self):
        super(Voigt, self).__init__(PeakShape.VOIGT)

    def calculate_effective_parameters(self, param_value_array, param_error_array):
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
        param_error_array : numpy.ndarray
        Returns
        -------
        np.ndarray
            (p', n, 1) or (p', n, 2) array for  parameter values and  optionally fitting error
            p' = number of effective parameters , n = number of sub runs
        """
        raise NotImplementedError('Somebody should write this')


"""
From here are a list of static method of peak profiles
"""


def calculate_profile(peak_type, background_type, vec_x, param_value_dict, peak_range):
    """Calculate peak and background profile of a given set of NATIVE peak parameters

    Parameters
    ----------
    peak_type
    background_type : BackgroundFunction or str
        Background function (type)
    vec_x
    param_value_dict : dict
        dictionary with key as peak and background function native parameters name
    peak_range : integer or float
        range (R) is equal N * FWHM
        Then calculated range will be peak center +/- R

    Returns
    -------
    numpy.ndarray
        1D array as calculated intensity

    """
    # Calculate peak range
    if peak_type == str(PeakShape.GAUSSIAN):
        # Gaussian: ['Height', 'PeakCentre', 'Sigma']
        peak_center = param_value_dict['PeakCentre']
        fwhm = Gaussian.cal_fwhm(param_value_dict['Sigma'])

    elif peak_type == str(PeakShape.PSEUDOVOIGT):
        # PseudoVoigt: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']
        peak_center = param_value_dict['PeakCentre']
        fwhm = param_value_dict['FWHM']

    else:
        # Not supported
        raise RuntimeError('Peak type {} is not supported'.format(peak_type))

    left_x = peak_center - peak_range * fwhm
    right_x = peak_center + peak_range * fwhm

    left_x_index = np.abs(vec_x - left_x).argmin()
    right_x_index = np.abs(vec_x - right_x).argmin()

    # Init Y
    vec_intensity = np.zeros_like(vec_x)

    # Calculate peak range
    if peak_type == str(PeakShape.GAUSSIAN):
        # Gaussian: ['Height', 'PeakCentre', 'Sigma']
        vec_intensity[left_x_index:right_x_index] = gaussian(vec_x[left_x_index:right_x_index],
                                                             param_value_dict['Height'],
                                                             param_value_dict['Sigma'],
                                                             param_value_dict['PeakCentre'])

    elif peak_type == str(PeakShape.PSEUDOVOIGT):
        # PseudoVoigt: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM']
        print('Calculating PV....')
        vec_intensity[left_x_index:right_x_index] = pseudo_voigt(vec_x[left_x_index:right_x_index],
                                                                 param_value_dict['Intensity'],
                                                                 param_value_dict['FWHM'],
                                                                 param_value_dict['Mixing'],
                                                                 param_value_dict['PeakCentre'])

    else:
        # Not supported
        raise RuntimeError('Peak type {} is not supported'.format(peak_type))

    # Calculate background
    if str(background_type) == str(BackgroundFunction.LINEAR):
        # Linear background
        bkgd_i = quadratic_background(vec_x[left_x_index:right_x_index],
                                      b0=param_value_dict['A0'],
                                      b1=param_value_dict['A1'],
                                      b2=0., b3=0.)
        vec_intensity[left_x_index:right_x_index] += bkgd_i

    else:
        raise RuntimeError('Background type {} is not supported'.format(background_type))

    return vec_intensity


def gaussian(x, a, sigma, x0):
    """
    Gaussian without normalization (Mantid compatible)
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
    part_gauss = 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Calculate normalized Lorentzian
    part_lorenz = 1. / np.pi * (fwhm / 2.) / ((x - x0)**2 + (fwhm / 2.)**2)

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
    print('Bug!')
    print(x)
    print(b0)
    print(b1)

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
