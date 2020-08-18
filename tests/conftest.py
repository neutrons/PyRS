import numpy as np
import os
import pytest
import sys

from pyrs.dataobjects.fields import StrainField
from pyrs.dataobjects.sample_logs import _coerce_to_ndarray, PointList
from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection


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
def strain_builder():
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

        return StrainField(point_list=point_list, peak_collection=peak_collection)

    return wrapped_function
