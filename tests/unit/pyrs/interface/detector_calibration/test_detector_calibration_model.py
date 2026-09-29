"""Tests for `pyrs.interface.detector_calibration.detector_calibration_model`.

Unit tier: the model is a `QObject`, which needs no display or `QApplication`,
and a plain namespace stands in for the calibration object it delegates to.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel


@pytest.mark.xfail(
    reason="DD-2: `fitted_ws` is indexed by position, and looked up as `sub_run - 1`",
    raises=IndexError,
    strict=True,
)
def test_get_fitted_diffraction_data_sub_runs_not_starting_at_one() -> None:
    """`fit_peaks` fills `fitted_ws` by position over `sub_runs`. Recorded under
    DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    model = DetectorCalibrationModel(None)
    model._calibration_obj = SimpleNamespace(fitted_ws=["five", "six", "seven"], sub_runs=np.array([5, 6, 7]))

    # Act / Assert
    assert model.get_fitted_diffraction_data(6) == "six"
