import numpy as np
import os
import pytest
import sys


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


@pytest.fixture(scope='session')
def allclose_with_sorting():
    r"""
    np.allclose with sorting incorporated, since different versions of scipy.cluster
    can yield different ordering.
    """
    def inner_function(left, right, *args, **kwargs):
        if np.allclose(left, right, *args, **kwargs) is False:
            return np.allclose(sorted(left), sorted(right), *args, **kwargs)
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
