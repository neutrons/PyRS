"""Tests for `pyrs.calibration.mantid_peakfit_calibration`.

Unit tier: `FitCalibration` is built with `object.__new__` and only the
collaborators `fit_peaks` touches are supplied -- no NeXus file is reduced and
no Mantid fit is run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pyrs.calibration.mantid_peakfit_calibration as calibration_module
from pyrs.calibration.mantid_peakfit_calibration import FitCalibration


@pytest.mark.xfail(
    reason="DD-2: the reference peaks are looked up as `sub_run - 1`", raises=AssertionError, strict=True
)
def test_fit_peaks_looks_up_reference_peaks_by_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_diff_peaks` indexes the per-position powder tables, so it must be given a
    position. Recorded under DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    calibration = object.__new__(FitCalibration)
    calibration._keep_subrun_list = [True, True, True]
    calibration._calib = np.zeros(8)
    calibration._hidra_ws = SimpleNamespace(reduction_masks=["mask"], get_sub_runs=lambda: np.array([5, 6, 7]))
    two_theta = np.linspace(30.0, 90.0, 61)
    calibration.reducer = SimpleNamespace(
        get_diffraction_data=lambda sub_run, mask_id: (two_theta, np.ones(61), np.ones(61))
    )
    calibration.reduce_data = lambda: None
    looked_up: list = []
    calibration.get_diff_peaks = lambda index: looked_up.append(index) or np.array([])
    monkeypatch.setattr(calibration_module.PeakFitEngineFactory, "getInstance", lambda *args, **kwargs: None)

    # Act
    calibration.fit_peaks()

    # Assert -- used to be [4, 5, 6]
    assert looked_up == [0, 1, 2]
