"""Tests for `pyrs.interface.texture_fitting.texture_fitting_crtl`.

Unit tier: the controller module imports headless, and `trim_data` reads only
the model's `sub_runs`, so a plain namespace stands in for the model.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl


def _crtl(sub_runs: list[int]) -> TextureFittingCrtl:
    return TextureFittingCrtl(SimpleNamespace(sub_runs=np.array(sub_runs)))


class TestTrimDataIncludeList:
    """The typed include list holds sub-run numbers, which are not positions.

    Recorded under DD-2 in `plans/PR_review/deferred-defects.md`.
    """

    @pytest.mark.xfail(reason="DD-2: included sub-runs are used as 0-based positions", raises=IndexError, strict=True)
    def test_sub_runs_not_starting_at_one(self):
        # Arrange
        data = np.array([1.0, 2.0, 3.0])

        # Act
        x, _ = _crtl([5, 6, 7]).trim_data(data, data, include_list=[5, 7])

        # Assert
        assert x.tolist() == [1.0, 3.0]

    @pytest.mark.xfail(
        reason="DD-2: a single included sub-run is ignored (`len(...) > 1`)", raises=AssertionError, strict=True
    )
    def test_a_single_sub_run(self):
        # Arrange
        data = np.array([1.0, 2.0, 3.0])

        # Act
        x, _ = _crtl([1, 2, 3]).trim_data(data, data, include_list=[2])

        # Assert
        assert x.tolist() == [2.0]


class _FitWorkspace:
    """Stands in for a Mantid fit workspace: one spectrum per sub-run, by position."""

    def __init__(self, n_spectra: int, n_bins: int = 4) -> None:
        self._n_spectra = n_spectra
        self._n_bins = n_bins

    def readX(self, index: int) -> np.ndarray:
        if not 0 <= index < self._n_spectra:
            raise IndexError(f"spectrum {index} out of range")
        return np.arange(self._n_bins, dtype=float)

    readY = readX


@pytest.mark.xfail(reason="DD-2: the first sub-run number is used as a spectrum index", raises=IndexError, strict=True)
def test_fit_peaks_sub_runs_not_starting_at_one() -> None:
    """Recorded under DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    fit = SimpleNamespace(fitted=_FitWorkspace(3), difference=_FitWorkspace(3))
    model = SimpleNamespace(
        sub_runs=np.array([5, 6, 7]),
        ws=SimpleNamespace(reduction_masks=["", "_var"]),
        fit_diff_peaks=lambda *args, **kwargs: fit,
    )
    crtl = TextureFittingCrtl(model)
    crtl.texture_run = lambda: False  # type: ignore[method-assign]

    # Act
    crtl.fit_peaks(SimpleNamespace(rowCount=lambda: 0), "PseudoVoigt", "Linear")

    # Assert -- one row per sub-run
    tth, _ = crtl._fitted_patterns[""][0]
    assert tth.shape == (3, 4)
