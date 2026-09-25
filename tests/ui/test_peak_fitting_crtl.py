"""GUI-tier tests for `PeakFittingCrtl`.

No `QWidget` is constructed -- the view is a duck-typed stub -- but this is still
GUI tier: `peak_fitting_crtl` imports matplotlib's `QtAgg` backend at module
scope, so the module cannot even be imported without a display (verified with
`DISPLAY`/`QT_QPA_PLATFORM` unset).

Both markers are required -- `test-gui` selects `-m gui` while
`test-integration` selects `-m 'integration and not gui'`, so `gui` alone would
drop these from the integration tier.
"""

from pathlib import Path
from typing import Any, Callable

import pytest

from pyrs.core.pyrscore import PyRsCore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.peak_fitting.peak_fitting_crtl import PeakFittingCrtl
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult

pytestmark = [pytest.mark.gui, pytest.mark.integration]


class _FakeFitSetupView:
    """Minimal stand-in for the Qt fit-setup view -- only the methods
    `plot_diff_and_fitted_data` actually calls, no real widget construction."""

    def __init__(self) -> None:
        self.experiment_calls: list = []
        self.fitted_calls: list = []

    def plot_experiment_data(self, diff_data_set: Any, data_reference: str) -> None:
        self.experiment_calls.append((diff_data_set, data_reference))

    def plot_fitted_data(self, x_array: Any, y_array: Any) -> None:
        self.fitted_calls.append((x_array, y_array))

    def plot_fitting_diff_data(self, x_axis: Any, y_axis: Any) -> None:
        pass


@pytest.fixture
def peak_model() -> PeakFittingModel:
    return PeakFittingModel(PyRsCore())


def test_plot_diff_and_fitted_data_fitted_none_does_not_raise(
    peak_model: PeakFittingModel,
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
    tmp_path: Path,
) -> None:
    """The plotting path tolerates the None fitted spectrum an NXstress load leaves behind."""
    # Arrange
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
    n_subrun = len(ws.get_sub_runs())
    peak = minimal_PeakCollection(N_subrun=n_subrun)

    peak_model.hidra_workspace = ws
    peak_model.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)
    peak_model._project_name = "test_project"

    out_path = tmp_path / "roundtrip.nxs"
    peak_model.save_fit_result(str(out_path))

    reloaded = PeakFittingModel(PyRsCore())
    reloaded.load_hidra_project(str(out_path))

    crtl = PeakFittingCrtl(reloaded)
    fake_view = _FakeFitSetupView()

    # Act -- previously crashed unconditionally on fit_result.fitted.readX(...).
    # _FakeFitSetupView is a duck-typed stub, not a PeakFitSetupView subclass.
    crtl.plot_diff_and_fitted_data(fake_view, sub_run_number=1)  # type: ignore[arg-type]

    # Assert
    assert len(fake_view.experiment_calls) == 1
    assert len(fake_view.fitted_calls) == 0
