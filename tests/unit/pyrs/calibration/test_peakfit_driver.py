"""The pyrs-calibration (FitCalibration) usage text names the default it actually uses (#1079)"""

import pytest

from pyrs.calibration import peakfit_driver


@pytest.fixture(scope="module")
def driver():
    return peakfit_driver


def test_usage_names_the_default_method(driver):
    # Act
    method_line = next(line for line in driver._options.splitlines() if line.startswith('"method"'))

    # Assert - the default the usage text gives is the one the tool refines with
    assert f'(default is "{driver.REFINE_METHOD}"' in method_line


def test_default_method_is_a_valid_method(driver):
    # Act / Assert - and it is one of the methods the tool accepts
    assert driver.REFINE_METHOD in driver.method_options
