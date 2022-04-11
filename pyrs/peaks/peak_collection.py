import numpy as np
from pathlib import Path
from pyrs.core.peak_profile_utility import get_parameter_dtype, get_effective_parameters_converter, PeakShape, \
    BackgroundFunction
from pyrs.dataobjects import SubRuns  # type: ignore
from typing import Optional, Tuple, Union
from uncertainties import unumpy

__all__ = ['PeakCollection', 'PeakCollectionLite']


def get_strain_conversion_factor(units: str = 'strain') -> float:
    '''
    get factor to convert strain to correct units

    Parameters
    ----------
    units: str
        Can be ``strain`` or ``microstrain``

    Returns
    -------
    float'''
    # prepare to return the requested units
    conversion_factor = 1.
    if units == 'strain':
        pass  # data is calculated as strain
    elif units == 'microstrain':
        conversion_factor = 1.e6
    else:
        raise ValueError('Cannot return units of "{}". Must be "strain" or "microstrain"'.format(units))
    return conversion_factor


def to_microstrain(strains):
    r"""
    convert a list of strains from strain to microstrains units

    Parameters
    ----------
    strains: np.ndarray, tuple, list

    Returns
    -------
    np.ndarray, list
        numpy array if `strains` is also a numpy array, otherwise return a list
    """
    microstrain = get_strain_conversion_factor(units='microstrain')
    if isinstance(strains, np.ndarray):
        return microstrain * strains
    return [microstrain * x for x in strains]


def _create_d_reference_array(values: Union[float, np.ndarray],
                              errors: Union[float, np.ndarray], size: int) -> unumpy.uarray:
    '''Convert the d-reference values to a :py:obj:`unumpy.uarray`

    Parameters
    ----------
    values :
        1D numpy array or floats
    errors:
        1D numpy array or floats
    size:
        The number of elements the final object should contain

    Returns
    -------
    :py:obj:`unumpy.uarray`
    '''
    # d-reference should be, at minimum, length one
    num_values = size if size else 1

    if isinstance(values, np.ndarray) and values.size > 1:
        msg = 'Incompatible number of values for d-reference: {} should be 1 or {}'.format(values.size,
                                                                                           size)
        assert values.size == size, msg
        nd_values = values
    else:
        nd_values = np.array([values] * num_values)

    if isinstance(errors, np.ndarray):
        nd_errors = errors
    else:
        nd_errors = np.array([errors] * num_values)

    # store value and uncertainties together
    return unumpy.uarray(nd_values, nd_errors)


class PeakCollectionLite:
    r"""
    A variant of the :py:obj:PeakCollection which does not have the full peak profile information.

    The intent is to be a very lightweight version of a :py:obj:PeakCollection to be created for
    the in-plane strain and in-plane stress special cases.
    """
    def __init__(self, peak_tag: str,
                 strain: np.ndarray,
                 strain_error: np.ndarray,
                 strain_units: str = 'strain',
                 d_reference: Union[float, np.ndarray] = np.nan,
                 d_reference_error: Union[float, np.ndarray] = 0.) -> None:
        self._tag: str = peak_tag

        # We need to store strains in strain units, NOT in microstrains
        self._strain = unumpy.uarray(strain, strain_error) / get_strain_conversion_factor(strain_units)

        # must happen after the sub_run array is set
        self._d_reference = unumpy.uarray(np.nan, np.nan)  # set this correctly in next call
        self.set_d_reference(d_reference, d_reference_error)

    def __len__(self):
        return self._strain.size

    def __bool__(self):
        return True

    def __eq__(self, other) -> bool:
        if self._tag != other._tag:
            return False
        strain_self, error_self = self.get_strain()
        strain_other, error_other = self.get_strain()
        if not np.all(strain_self == strain_other):
            return False
        if not np.all(error_self == error_other):
            return False
        return True

    def set_d_reference(self, values: Union[float, np.ndarray] = np.nan,
                        errors: Union[float, np.ndarray] = 0.) -> None:
        """Set d reference values
        """
        # store value and uncertainties together
        self._d_reference = _create_d_reference_array(values, errors, self._strain.size)

    def get_d_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get d reference for all the sub runs

        Returns
        -------
        numpy.ndarray
            1D array for peak's reference position in dSpacing.  NaN for not being set.

        """
        return unumpy.nominal_values(self._d_reference), unumpy.std_devs(self._d_reference)

    def get_strain(self, units: str = 'strain') -> Tuple[np.ndarray, np.ndarray]:
        """get strain values and uncertainties in units of strain

          Parameters
          ----------
          units: str
              Can be ``strain`` or ``microstrain``

          Returns
          -------
            tuple
                A two-item tuple containing the strain and its uncertainty.
          """
        # prepare to return the requested units
        conversion_factor = get_strain_conversion_factor(units)

        # multiplying by 1e6 converts to micro
        strain = conversion_factor * self._strain

        # unpack the values to return
        return unumpy.nominal_values(strain), unumpy.std_devs(strain)

    @property
    def runnumber(self) -> int:
        '''Negative one means it was never set'''
        return -1

    @property
    def projectfilename(self) -> str:
        '''Empty string because these never came from a project file'''
        return ''


class PeakCollection:
    """
    Object to contain peak parameters (names and values) of a collection of peaks for sub runs
    """
    def __init__(self, peak_tag: str, peak_profile, background_type, wavelength: float = np.nan,
                 d_reference: Union[float, np.ndarray] = np.nan,
                 d_reference_error: Union[float, np.ndarray] = 0.,
                 projectfilename: str = '',
                 runnumber: int = -1) -> None:
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
        self._filename: str = ''
        self.projectfilename = projectfilename  # use the setter
        self._runnumber: int = runnumber

        # Init other parameters
        self._peak_profile = PeakShape.getShape(peak_profile)
        self._background_type = BackgroundFunction.getFunction(background_type)
        self._wavelength = wavelength

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

        # must happen after the sub_run array is set
        self._d_reference: Optional[unumpy.uarray]
        self.set_d_reference(d_reference, d_reference_error)

    def __len__(self):
        return len(self._sub_run_array)

    def __bool__(self):
        return True

    @property
    def peak_tag(self) -> str:
        """Peak tag

        Returns
        -------
        str
            Peak tag

        """
        return self._tag

    @property
    def peak_profile(self) -> str:
        """Get peak profile name

        Returns
        -------
        str
            peak profile name such as Gaussian

        """
        return str(self._peak_profile)

    @property
    def background_type(self) -> str:
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
    def fitting_costs(self):
        return self._fit_cost_array

    @property
    def runnumber(self) -> int:
        '''The run number. Negative one means it was never set'''
        return self._runnumber

    @property
    def projectfilename(self) -> str:
        return self._filename

    @projectfilename.setter
    def projectfilename(self, filename: str) -> None:
        if not filename or filename == '/':
            # convert all "False" things to empty string
            self._filename = ''
        else:
            # only the name of the file rather than full path
            self._filename = Path(filename).name

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

    def get_d_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get d reference for all the sub runs

        Returns
        -------
        numpy.ndarray
            1D array for peak's reference position in dSpacing.  NaN for not being set.

        """
        return unumpy.nominal_values(self._d_reference), unumpy.std_devs(self._d_reference)

    def set_d_reference(self, values: Union[float, np.ndarray] = np.nan,
                        errors: Union[float, np.ndarray] = 0.) -> None:
        '''Set d reference values'''
        # store value and uncertainties together
        self._d_reference = _create_d_reference_array(values, errors, self._sub_run_array.size)

    def get_strain(self, units: str = 'strain') -> Tuple[np.ndarray, np.ndarray]:
        """get strain values and uncertainties in units of strain

          Parameters
          ----------
          units: str
              Can be ``strain`` or ``microstrain``

          Returns
          -------
            tuple
                A two-item tuple containing the strain and its uncertainty.
          """
        # prepare to return the requested units
        conversion_factor = get_strain_conversion_factor(units)

        d_fitted = self._get_dspacing_center()

        # multiplying by 1e6 converts to micro
        strain = conversion_factor * (d_fitted - self._d_reference)/self._d_reference

        # unpack the values to return
        return unumpy.nominal_values(strain), unumpy.std_devs(strain)

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

    def _get_dspacing_center(self):
        '''Internal function for getting d-spacing position'''
        effective_values, effective_errors = self.get_effective_params()
        theta_center = unumpy.uarray(0.5 * np.deg2rad(effective_values['Center']),
                                     0.5 * np.deg2rad(effective_errors['Center']))
        sine_theta = unumpy.sin(theta_center)
        try:
            dspacing_center = 0.5 * self._wavelength / sine_theta
        except ZeroDivisionError:
            # replace zeros in the denominator with nan explicitly
            dspacing_center = np.where(unumpy.nominal_values(sine_theta) != 0.,
                                       unumpy.std_devs(0.5 * self._wavelength / sine_theta.clip(1e-9)), np.nan)

        return dspacing_center

    def get_dspacing_center(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        peak center in unit of d spacing.

        Returns
        -------
        tuple
            A two-item tuple containing the peak center and its uncertainty.
        """
        d_spacing = self._get_dspacing_center()
        return unumpy.nominal_values(d_spacing), unumpy.std_devs(d_spacing)

    def get_chisq(self):
        return np.copy(self._fit_cost_array)

    def get_subruns(self):
        return self._sub_run_array.raw_copy()  # TODO should this be the actual object

    def get_fit_status(self):
        '''list of messages with "success" being it's good'''
        return self._fit_status
