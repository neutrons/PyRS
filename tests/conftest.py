from copy import deepcopy
import numpy as np
import os
import pytest
import sys

from pyrs.dataobjects.fields import StrainField, StrainFieldSingle, StressField
from pyrs.dataobjects.sample_logs import _coerce_to_ndarray, PointList
from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.peaks.peak_collection import PeakCollection

# set to True when running on build servers
ON_GITHUB_ACTIONS = bool(os.environ.get('GITHUB_ACTIONS', False))


@pytest.fixture(scope='session')
def test_data_dir():
    this_module_path = sys.modules[__name__].__file__
    this_module_directory = os.path.dirname(this_module_path)
    return os.path.join(this_module_directory, 'data')


@pytest.fixture(scope='session')
def assert_almost_equal_with_sorting():
    r"""
    np.testing.assert_almost_equal with sorting incorporated, since different
    versions of scipy.cluster can yield different ordering.
    """
    def inner_function(left, right, *args, **kwargs):
        try:
            np.testing.assert_almost_equal(left, right, *args, **kwargs)
        except AssertionError:
            np.testing.assert_almost_equal(sorted(left), sorted(right), *args, **kwargs)
    return inner_function


def assert_allclose_with_sorting(left, right, *args, **kwargs) -> None:
    r"""
    np.allclose with sorting incorporated, since different versions of scipy.cluster
    can yield different ordering.
    """
    if np.allclose(left, right, *args, **kwargs):
        return

    equal_nan = kwargs.pop('equal_nan', False)
    if equal_nan is True:
        assert len(np.where(np.isnan(left))[0]) == len(np.where(np.isnan(right))[0])  # same number of nan
    # create copies of the arrays and sort them
    left_array, right_array = np.array(left), np.array(right)  # cast to numpy array
    left_array, right_array = sorted(left_array[np.isfinite(left)]), sorted(right_array[np.isfinite(right)])

    np.testing.assert_allclose(left_array, right_array, *args, **kwargs)


@pytest.fixture(scope='session')
def allclose_with_sorting():
    r"""
    np.allclose with sorting incorporated, since different versions of scipy.cluster
    can yield different ordering.
    """
    def inner_function(left, right, *args, **kwargs):
        if np.allclose(left, right, *args, **kwargs) is False:
            equal_nan = kwargs.pop('equal_nan', False)
            if equal_nan is True:
                assert len(np.where(np.isnan(left))[0]) == len(np.where(np.isnan(right))[0])  # same number of nan
            left_array, right_array = np.array(left), np.array(right)  # cast to numpy array
            left_array, right_array = sorted(left_array[np.isfinite(left)]), sorted(right_array[np.isfinite(right)])
            return np.allclose(left_array, right_array, *args, **kwargs)
        return True
    return inner_function


@pytest.fixture(scope='session')
def approx_with_sorting():
    r"""
    pytest.approx with sorting incorporated, since different versions of scipy.cluster
    can yield different ordering.
    """
    def inner_function(left, right, *args, **kwargs):
        try:
            assert left == pytest.approx(right, *args, **kwargs)
        except AssertionError:
            assert sorted(left) == pytest.approx(sorted(right), *args, **kwargs)
    return inner_function


@pytest.fixture(scope='session')
def strain_single_builder():
    def wrapped_function(peaks_data):
        r"""
        Constructor of `StrainField` objects

        Parameters
        ----------
        peaks_data: dict

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField

        Examples
        --------

        """

        # Required arguments
        for required in ('subruns', 'wavelength', 'peak_profile', 'background_type', 'd_spacing',
                         'error_fraction', 'vx', 'vy', 'vz'):
            assert required in peaks_data

        for key in ('subruns', 'd_reference', 'd_spacing', 'fit_costs', 'vx', 'vy', 'vz'):
            if isinstance(peaks_data[key], (list, tuple)):
                peaks_data[key] = np.array(peaks_data[key])
        for key in peaks_data['native']:
            peaks_data['native'][key] = _coerce_to_ndarray(peaks_data['native'][key])
        # Default values for optional arguments of the PeakCollection constructor
        runnumber = peaks_data.get('runnumber', -1)

        peak_collection = PeakCollection(peaks_data['peak_tag'],
                                         peaks_data['peak_profile'], peaks_data['background_type'],
                                         peaks_data['wavelength'],
                                         runnumber=runnumber)

        # Back-calculate the peak centers from supplied lattice spacings
        centers = 2 * np.rad2deg(np.arcsin(peaks_data['wavelength'] / (2 * peaks_data['d_spacing'])))

        # Enter the native parameters in the peak collection
        subruns_count = len(peaks_data['subruns'])
        peaks_data['native'].update({'PeakCentre': centers})
        dtype = get_parameter_dtype(peaks_data['peak_profile'], peaks_data['background_type'])
        parameters_value = np.zeros(subruns_count, dtype=dtype)
        parameters_error = np.zeros(subruns_count, dtype=dtype)
        for parameter_name in peaks_data['native']:
            values = peaks_data['native'][parameter_name]
            parameters_value[parameter_name] = values
            parameters_error[parameter_name] = peaks_data['error_fraction'] * values
        # set_peak_fitting_values ensures all required fitting parameters are being passed
        peak_collection.set_peak_fitting_values(peaks_data['subruns'],
                                                parameters_value, parameters_error,
                                                peaks_data['fit_costs'])

        # set the reference lattice spacing
        peak_collection.set_d_reference(1.0, 0.1)

        # Point list
        point_list = PointList([peaks_data['vx'], peaks_data['vy'], peaks_data['vz']])

        return StrainFieldSingle(point_list=point_list, peak_collection=peak_collection)

    return wrapped_function


@pytest.fixture(scope='function')
def strain_single_object_0(strain_single_builder):
    r"""Serves a StrainFieldSingle object"""
    scan = {
        'runnumber': '1234',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # parameters appropriate to the selected peak shape, except for parameter PeakCentre
        'native': {
            'Intensity': [100, 110, 120, 130, 140, 150, 160, 170],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            'Mixing': [1.0] * 8,
            'A0': [10., 11., 12., 13., 14., 15., 16., 17.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        'fit_costs': [1.0] * 8,
        # will back-calculate PeakCentre values in order to end up with these lattice spacings
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
        'vx': [0., 1., 2., 3., 4., 5., 6., 7.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    }

    # create a StrainField object
    return strain_single_builder(scan)


@pytest.fixture(scope='session')
def strain_builder(strain_single_builder):

    def wrapped_function(peaks_data):
        r"""
        Constructor of `StrainField` objects

        Parameters
        ----------
        peaks_data: dict

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField

        Examples
        --------

        """
        return StrainField(strain_single=strain_single_builder(peaks_data))

    return wrapped_function


@pytest.fixture(scope='function')
def strain_object_0(strain_builder):
    r"""Serves a StrainField object made up of one StrainFieldSingle object"""
    scan = {
        'runnumber': '1234',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # parameters appropriate to the selected peak shape, except for parameter PeakCentre
        'native': {
            'Intensity': [100, 110, 120, 130, 140, 150, 160, 170],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            'Mixing': [1.0] * 8,
            'A0': [10., 11., 12., 13., 14., 15., 16., 17.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        'fit_costs': [1.0] * 8,
        # will back-calculate PeakCentre values in order to end up with these lattice spacings
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
        'vx': [0., 1., 2., 3., 4., 5., 6., 7.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    }

    # create a StrainField object
    return strain_builder(scan)


@pytest.fixture(scope='function')
def strain_object_1(strain_builder):
    r"""Serves a StrainField object made up of two non-overlapping StrainFieldSingle objects"""
    common_data = {
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
    }

    strain_1235_data = deepcopy(common_data)
    strain_1235_data.update({
        'runnumber': '1235',
        'subruns': [1, 2, 3, 4],
        'native': {
            'Intensity': [110, 120, 130, 140],
            'FWHM': [1.1, 1.2, 1.3, 1.4],
            'Mixing': [1.0] * 4,
            'A0': [11., 12., 13., 14.],
            'A1': [0.01, 0.02, 0.03, 0.04],
        },
        'fit_costs': [1.0] * 4,
        'd_spacing': [1.01, 1.02, 1.03, 1.04],  # notice the last value
        'vx': [1., 2., 3., 4.],
        'vy': [0.] * 4,
        'vz': [0.] * 4
    })
    strain_1235 = strain_builder(strain_1235_data)

    strain_1236_data = deepcopy(common_data)
    strain_1236_data.update({
        'runnumber': '1236',
        'subruns': [1, 2, 3, 4],
        'native': {
            'Intensity': [150, 160, 170, 180],
            'FWHM': [1.5, 1.6, 1.7, 1.8],
            'Mixing': [1.0] * 4,
            'A0': [15., 16., 17., 18.],
            'A1': [0.05, 0.06, 0.07, 0.08],
        },
        'fit_costs': [1.0] * 4,
        'd_spacing': [1.05, 1.06, 1.07, 1.08],  # notice the first value
        'vx': [5., 6., 7., 8.],
        'vy': [0.] * 4,
        'vz': [0.] * 4
    })
    strain_1236 = strain_builder(strain_1236_data)

    return strain_1235 + strain_1236


@pytest.fixture(scope='function')
def strain_object_2(strain_builder):
    r"""Serves a StrainField object made up of two overlapping StrainFieldSingle objects"""
    common_data = {
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
    }
    strain_1235_data = deepcopy(common_data)
    strain_1235_data.update({
        'runnumber': '1235',
        'subruns': [1, 2, 3, 4],
        'native': {
            'Intensity': [110, 120, 130, 140],
            'FWHM': [1.1, 1.2, 1.3, 1.4],
            'Mixing': [1.0] * 4,
            'A0': [11., 12., 13., 14.],
            'A1': [0.01, 0.02, 0.03, 0.04],
        },
        'fit_costs': [1.0] * 4,
        'd_spacing': [1.01, 1.02, 1.03, 1.04],  # notice the last value
        'vx': [1., 2., 3., 4.],
        'vy': [0.] * 4,
        'vz': [0.] * 4
    })
    strain_1235 = strain_builder(strain_1235_data)

    strain_1236_data = deepcopy(common_data)
    strain_1236_data.update({
        'runnumber': '1236',
        'subruns': [1, 2, 3, 4, 5],
        'native': {
            'Intensity': [140, 150, 160, 170, 180],
            'FWHM': [1.4, 1.5, 1.6, 1.7, 1.8],
            'Mixing': [1.0] * 5,
            'A0': [14., 15., 16., 17., 18.],
            'A1': [0.04, 0.05, 0.06, 0.07, 0.08],
        },
        'fit_costs': [1.0] * 5,
        'd_spacing': [1.045, 1.05, 1.06, 1.07, 1.08],  # notice the first value
        'vx': [4., 5., 6., 7., 8.],
        'vy': [0.] * 5,
        'vz': [0.] * 5
    })
    strain_1236 = strain_builder(strain_1236_data)

    return strain_1235 + strain_1236


@pytest.fixture(scope='function')
def strain_stress_object_0(strain_builder):
    r"""
    We create three single strain instances, all sharing the same set of sample points

          RUN          vx-coordinate
        NUMBER  0.0  1.0  2.0  3.0  4.0  5.0
        1234    ****************************
        1235    ****************************
        1236    ****************************

    For simplicity, vy and vz values are all zero, so these are unidimensional scans

    From these four strains, we create strains in three dimensions:
        strain11 = strain_1234
        strain22 = strain_1235
        strain33 = strain_1237

    We create three stress objects with the previous strains:
        'diagonal' uses strain11, strain22, and strain33
        'in-plane-strain': uses strain11 and strain22
        'in-plane-stress': uses strain11 and strain22

    Returns
    -------
    dict
        {
            'strains': {
                '11': strain11, '22': strain22, '33': strain33
            },
            'stresses': {
                'diagonal': StressField(strain11, strain22, strain33, 1. / 3, 4. / 3, 'diagonal'),
                'in-plane-strain': StressField(strain11, strain22, None, 1. / 3, 4. / 3, 'in-plane-strain'),
                'in-plane-stress': StressField(strain11, strain22, None, 1. / 2, 3. / 2, 'in-plane-stress')
            }
        }
    """

    common_data = {
        'peak_tag': 'test',
        'subruns': [1, 2, 3, 4, 5],
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        'fit_costs': [1.0] * 5,
        'vx': [0., 1., 2., 3., 4.],
        'vy': [0.] * 5,
        'vz': [0.] * 5,
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
    }

    strain11_data = deepcopy(common_data)
    strain11_data.update({
        'runnumber': '1234',
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04],
        'native': {
            'Intensity': [100, 110, 120, 130, 140],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4],
            'Mixing': [1.0] * 5,
            'A0': [10., 11., 12., 13., 14.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04],
        },
    })
    strain11 = strain_builder(strain11_data)

    strain22_data = deepcopy(common_data)
    strain22_data.update({
        'runnumber': '1235',
        'd_spacing': [1.10, 1.11, 1.12, 1.13, 1.14],
        'native': {
            'Intensity': [101, 111, 121, 131, 141],
            'FWHM': [1.01, 1.11, 1.21, 1.31, 1.41],
            'Mixing': [1.01] * 5,
            'A0': [10.1, 11.1, 12.1, 13.1, 14.1],
            'A1': [0.001, 0.011, 0.021, 0.031, 0.041],
        },
    })
    strain22 = strain_builder(strain22_data)

    strain33_data = deepcopy(common_data)
    strain33_data.update({
        'runnumber': '1236',
        'd_spacing': [1.20, 1.21, 1.22, 1.23, 1.24],
        'native': {
            'Intensity': [102, 112, 122, 132, 142],
            'FWHM': [1.02, 1.12, 1.22, 1.32, 1.42],
            'Mixing': [1.02] * 5,
            'A0': [10.2, 11.2, 12.2, 13.2, 14.2],
            'A1': [0.002, 0.012, 0.022, 0.032, 0.042],
        },
    })
    strain33 = strain_builder(strain33_data)

    # values of Young's modulus and Poisson's ratio to render simpler strain-to-stress formulae
    return {
        'strains': {
            '11': strain11, '22': strain22, '33': strain33
        },
        'stresses': {
            'diagonal': StressField(strain11, strain22, strain33, 4. / 3, 1. / 3, 'diagonal'),
            'in-plane-strain': StressField(strain11, strain22, None, 4. / 3, 1. / 3, 'in-plane-strain'),
            'in-plane-stress': StressField(strain11, strain22, None, 3. / 2, 1. / 2, 'in-plane-stress')
        }
    }


@pytest.fixture(scope='function')
def strain_stress_object_1(strain_builder):
    r"""
    We create four single strain instances, below is how they overlap over the vx's extent

          RUN          vx-coordinate
        NUMBER  0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0
        1234    **************************************
        1235         ******************
        1236                        ***********************
        1237              **************************************

    Runs 1235 and 1236 overlap in one point. 1235 is the run will the smallest error in strain.

    For simplicity, vy and vz values are all zero, so these are unidimensional scans

    From these four strains, we create strains in three dimensions:
        strain11 = strain_1234
        strain22 = strain_1235 + strain_1236
        strain33 = strain_1237

    We create three stress objects with the previous strains:
        'diagonal' uses strain11, strain22, and strain33
        'in-plane-strain': uses strain11 and strain22
        'in-plane-stress': uses strain11 and strain22

    Returns
    -------
    dict
        {
            'strains': {
                '11': strain11, '22': strain22, '33': strain33
            },
            'stresses': {
                'diagonal': StressField(strain11, strain22, strain33, 1. / 3, 4. / 3, 'diagonal'),
                'in-plane-strain': StressField(strain11, strain22, None, 1. / 3, 4. / 3, 'in-plane-strain'),
                'in-plane-stress': StressField(strain11, strain22, None, 1. / 2, 3. / 2, 'in-plane-stress')
            }
        }
    """

    common_data = {
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
    }

    strain11_data = deepcopy(common_data)
    strain11_data.update({
        'runnumber': '1234',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'native': {
            'Intensity': [100, 110, 120, 130, 140, 150, 160, 170],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            'Mixing': [1.0] * 8,
            'A0': [10., 11., 12., 13., 14., 15., 16., 17.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        'fit_costs': [1.0] * 8,
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        'vx': [0., 1., 2., 3., 4., 5., 6., 7.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    })
    strain11 = strain_builder(strain11_data)

    strain_1235_data = deepcopy(common_data)
    strain_1235_data.update({
        'runnumber': '1235',
        'subruns': [1, 2, 3, 4],
        'native': {
            'Intensity': [110, 120, 130, 140],
            'FWHM': [1.1, 1.2, 1.3, 1.4],
            'Mixing': [1.0] * 4,
            'A0': [11., 12., 13., 14.],
            'A1': [0.01, 0.02, 0.03, 0.04],
        },
        'fit_costs': [1.0] * 4,
        'd_spacing': [1.01, 1.02, 1.03, 1.04],  # notice the last value
        'vx': [1., 2., 3., 4.],
        'vy': [0.] * 4,
        'vz': [0.] * 4
    })
    strain_1235 = strain_builder(strain_1235_data)

    strain_1236_data = deepcopy(common_data)
    strain_1236_data.update({
        'runnumber': '1236',
        'subruns': [1, 2, 3, 4, 5],
        'native': {
            'Intensity': [140, 150, 160, 170, 180],
            'FWHM': [1.4, 1.5, 1.6, 1.7, 1.8],
            'Mixing': [1.0] * 5,
            'A0': [14., 15., 16., 17., 18.],
            'A1': [0.04, 0.05, 0.06, 0.07, 0.08],
        },
        'fit_costs': [1.0] * 5,
        'd_spacing': [1.045, 1.05, 1.06, 1.07, 1.08],  # notice the first value
        'vx': [4., 5., 6., 7., 8.],
        'vy': [0.] * 5,
        'vz': [0.] * 5
    })
    strain_1236 = strain_builder(strain_1236_data)

    strain22 = strain_1235 + strain_1236

    strain33_data = deepcopy(common_data)
    strain33_data.update({
        'runnumber': '1237',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'native': {
            'Intensity': [120, 130, 140, 150, 160, 170, 180, 190],
            'FWHM': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'Mixing': [1.0] * 8,
            'A0': [12., 13., 14., 15., 16., 17., 18., 19.],
            'A1': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        },
        'fit_costs': [1.0] * 8,
        'd_spacing': [1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
        'vx': [2., 3., 4., 5., 6., 7., 8., 9.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    })
    strain33 = strain_builder(strain33_data)

    # values of Young's modulus and Poisson's ratio to render simpler strain-to-stress formulae
    return {
        'strains': {
            '11': strain11, '22': strain22, '33': strain33
        },
        'stresses': {
             'diagonal': StressField(strain11, strain22, strain33, 4. / 3, 1. / 3, 'diagonal'),
             'in-plane-strain': StressField(strain11, strain22, None, 4. / 3, 1. / 3, 'in-plane-strain'),
             'in-plane-stress': StressField(strain11, strain22, None, 3. / 2, 1. / 2, 'in-plane-stress')
        }
    }
