"""FitCalibration's calibration powders come from pyrs.calibration.powder_standards (#1022)

Each sub-run is matched to the powder standard at its ``sy`` position; the expected peak positions
are that standard's reflections at the run's wavelength, and sub-runs whose ``sy`` matches no standard
are not used (with a warning) instead of failing later.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.calibration import mantid_peakfit_calibration as mpc
from pyrs.calibration.powder_standards import HB2B_STANDARDS

NI, FE, MO = HB2B_STANDARDS
WAVELENGTH = 1.54


def _calibration(sy_values, wavelength=WAVELENGTH):
    """A FitCalibration over a stand-in workspace that has only the sy log, sub-runs and a wavelength."""
    sy = np.asarray(sy_values, dtype=np.float32)  # as HB2B logs it
    workspace = SimpleNamespace(
        get_sample_log_values=lambda name: sy,
        get_sub_runs=lambda: np.arange(1, sy.size + 1),
        get_wavelength=lambda calibrated, throw_if_not_set: wavelength,
    )
    fit = mpc.FitCalibration.__new__(mpc.FitCalibration)
    fit._hidra_ws = workspace
    fit._standards = HB2B_STANDARDS
    fit.get_powder_lines()
    return fit


def test_powders_matched_by_sy():
    # Act
    fit = _calibration([62.0019, 11.9999, -13.0004])

    # Assert
    assert fit.powders == ["Ni", "Fe", "Mo"]
    assert fit._keep_subrun_list.tolist() == [True, True, True]


def test_diff_peaks_are_the_standards_bragg_angles():
    # Arrange
    fit = _calibration([62.0, 12.0])

    # Act
    ni_peaks, fe_peaks = fit.get_diff_peaks(0), fit.get_diff_peaks(1)

    # Assert - full float64 lattice constants (they were rounded to float32 before), Ni 400 included
    np.testing.assert_array_equal(ni_peaks, NI.two_theta(WAVELENGTH))
    np.testing.assert_array_equal(fe_peaks, FE.two_theta(WAVELENGTH))
    ni_400 = 2 * np.degrees(np.arcsin(WAVELENGTH / (2 * NI.lattice_a / 4)))
    assert ni_400 == pytest.approx(ni_peaks[5])


def test_diff_peaks_skip_unreachable_reflections():
    # Arrange - at the Si331 setting, 1.731 A, Ni 331 (2d = 1.617 A) cannot diffract
    fit = _calibration([62.0], wavelength=1.731)

    # Act
    peaks = fit.get_diff_peaks(0)

    # Assert - ascending 2theta and no NaN, so fit_peaks' "first peak in range" is the lowest angle
    assert np.all(np.isfinite(peaks)) and np.all(np.diff(peaks) > 0)
    assert peaks.size == 6


def test_unknown_sy_sub_run_not_used_with_warning():
    # Act
    with pytest.warns(UserWarning, match=r"sub-runs \[2\]: sy matches no powder standard"):
        fit = _calibration([62.0, 40.0, 12.0])

    # Assert
    assert fit.powders == ["Ni", "", "Fe"]
    assert fit._keep_subrun_list.tolist() == [True, False, True]


def test_diff_peaks_of_unknown_sy_raises():
    # Arrange
    with pytest.warns(UserWarning):
        fit = _calibration([40.0])

    # Act / Assert - e.g. the GUI re-enabling that sub-run
    with pytest.raises(ValueError, match="matches no powder standard"):
        fit.get_diff_peaks(0)


def test_sy_tolerance_is_exclusive_at_2_mm():
    # Act
    with pytest.warns(UserWarning):
        fit = _calibration([63.999, 64.0])

    # Assert
    assert fit._keep_subrun_list.tolist() == [True, False]
