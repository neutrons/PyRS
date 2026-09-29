"""Tests for `pyrs.interface.detector_calibration.detector_calibration_crtl`.

Unit tier: the controller module imports headless, and a plain namespace stands
in for its model and for the plot it draws on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl


@pytest.mark.xfail(reason="DD-2: the keep list is indexed as `sub_run - 1`", raises=AssertionError, strict=True)
def test_update_diff_view_sub_runs_not_starting_at_one() -> None:
    """The keep list is positional, over the model's sub-runs. A sub-run number
    used as a position raised `IndexError`, which the view swallows, so nothing
    was drawn. Recorded under DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    drawn: list = []
    plot = SimpleNamespace(
        plot_diffraction=lambda *args, **kwargs: drawn.append(kwargs["line_label"]),
        set_no_null_plot=lambda: None,
    )
    two_theta = np.linspace(30.0, 90.0, 5)
    model = SimpleNamespace(
        sub_runs=np.array([5, 6, 7]),
        reduction_masks=["mask"],
        get_reduced_diffraction_data=lambda sub_run, mask: (two_theta, np.ones(5), np.ones(5)),
        get_fitted_diffraction_data=lambda sub_run: None,
    )

    # Act -- sub-run 6 is at position 1, which is kept
    DetectorCalibrationCrtl(model).update_diff_view(plot, 1, 6, [False, True, False])

    # Assert
    assert drawn == ["mask"]
