"""Integration tests for `PeakFittingModel` -- suffix-dispatched NXstress/.h5
save-load and the `PyRsCore.register_hidra_workspace` session hookup.

Each test drives `PeakFittingModel` together with a second component -- NXstress
or `HidraProjectFile`, and `PyRsCore`'s session registry -- writing real HDF5
through `nexusformat`/`h5py`. That is a multi-component workflow, so these are
`integration` even though the data is synthetic and the files land in `tmp_path`
(see CLAUDE.md's marker definitions).

Nothing here needs a display; the display-requiring tests for the same model
live in `tests/ui/test_peak_fitting_model.py`.
"""

from pathlib import Path
from typing import Callable

import pytest

from pyrs.core.pyrscore import PyRsCore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult

pytestmark = pytest.mark.integration


@pytest.fixture
def peak_model() -> PeakFittingModel:
    return PeakFittingModel(PyRsCore())


class TestPeakFittingModelNXstressRoundtrip:
    def test_save_fit_result_nxstress_roundtrip_matches_workspace_and_peaks(
        self,
        peak_model: PeakFittingModel,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        n_subrun = len(ws.get_sub_runs())
        peak = minimal_PeakCollection(N_subrun=n_subrun)

        peak_model.hidra_workspace = ws
        peak_model._project_name = "test_project"

        peak_model.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        out_path = tmp_path / "roundtrip.nxs"
        peak_model.save_fit_result(str(out_path))
        assert out_path.exists()

        reloaded = PeakFittingModel(PyRsCore())
        reloaded.load_hidra_project(str(out_path))

        assert reloaded.hidra_workspace is not None
        assert len(reloaded.hidra_workspace.get_sub_runs()) == n_subrun
        assert reloaded._curr_file_name == str(out_path)

        # Phase-1 scope: fitted/difference stay None (documented limitation) -- see
        # the guard in PeakFittingCrtl.plot_diff_and_fitted_data, covered by
        # tests/ui/test_peak_fitting_model.py.
        assert reloaded.fit_result is not None
        assert reloaded.fit_result.fitted is None
        assert reloaded.fit_result.difference is None
        assert len(reloaded.fit_result.peakcollections) == 1

    def test_load_hidra_project_multiple_nxstress_files_raises_value_error(
        self, peak_model: PeakFittingModel, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            peak_model.load_hidra_project([str(tmp_path / "a.nxs"), str(tmp_path / "b.nxs")])

    def test_load_hidra_project_nxstress_get_diffraction_data_succeeds(
        self,
        peak_model: PeakFittingModel,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        # Verifies the fix for the gap where an NXstress-loaded session left
        # PyRsCore's session registry empty -- get_diffraction_data (and therefore
        # any sub-run plotting) would otherwise raise immediately after a load.
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

        # Would raise (session not registered) before the register_hidra_workspace fix.
        diff_data_set = reloaded.get_diffraction_data(sub_run=1, mask=None)
        assert diff_data_set is not None


class TestPeakFittingModelSuffixRouting:
    def test_save_fit_result_h5_suffix_routes_through_hidraprojectfile(
        self,
        peak_model: PeakFittingModel,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        project_path = write_minimal_h5_project(ws, filename="source.h5")

        peak_model.load_hidra_project([str(project_path)])
        assert peak_model._curr_file_name == str(project_path)
        # The .h5 load path never populates fit_result -- unaffected by the new
        # .nxs branch above.
        assert peak_model.fit_result is None

        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()))

        peak_model.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        out_path = tmp_path / "saved.h5"
        peak_model.save_fit_result(str(out_path))
        assert out_path.exists()

    def test_load_hidra_project_h5_bare_string_loads(
        self,
        peak_model: PeakFittingModel,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ) -> None:
        """A bare-string .h5 path behaves like a one-element list (it used to be
        indexed character-by-character, loading "/" as the project file)."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        project_path = str(write_minimal_h5_project(ws, filename="source.h5"))

        # Act
        loaded = peak_model.load_hidra_project(project_path)

        # Assert
        assert len(loaded.get_sub_runs()) == len(ws.get_sub_runs())
        assert peak_model._curr_file_name == project_path
