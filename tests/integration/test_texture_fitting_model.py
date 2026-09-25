"""Integration tests for `TextureFittingModel` -- suffix-dispatched NXstress/.h5
save-load, and a regression test for the pre-existing `self.parent` crash fixed
alongside it.

Each test drives `TextureFittingModel` together with a second component --
NXstress or `HidraProjectFile` -- writing real HDF5 through `nexusformat`/`h5py`.
That is a multi-component workflow, so these are `integration` even though the
data is synthetic and the files land in `tmp_path` (see CLAUDE.md's marker
definitions). No widget is constructed and nothing here needs a display, so
`gui` does not apply.
"""

from pathlib import Path
from typing import Callable

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult

pytestmark = pytest.mark.integration


@pytest.fixture
def texture_model() -> TextureFittingModel:
    # `peak_fit_core` is unused by save_fit_result/load_hidra_project_file -- see
    # texture_fitting_model.py, `self._peak_fit` is stored but never read.
    return TextureFittingModel(None)


class TestTextureFittingModelH5SaveRegression:
    """Regression coverage for the pre-existing `self.parent` bug: `save_fit_result`
    referenced `self.parent._curr_file_name`, but `TextureFittingModel` never sets a
    `self.parent` attribute anywhere -- calling this method raised `AttributeError`
    unconditionally. Fixed by tracking `self._curr_file_name` directly (set in
    `load_hidra_project_file`), mirroring `PeakFittingModel`'s equivalent.
    """

    def test_save_fit_result_h5_does_not_raise(
        self,
        texture_model: TextureFittingModel,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True)
        project_path = write_minimal_h5_project(ws, filename="source.h5")

        texture_model.load_hidra_project_file(str(project_path))
        assert texture_model._curr_file_name == str(project_path)

        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()))
        fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        out_path = tmp_path / "saved.h5"
        # This is the call that previously raised AttributeError unconditionally.
        texture_model.save_fit_result(str(out_path), fit_result=fit_result)

        assert out_path.exists()

    def test_save_fit_result_h5_updates_in_place(
        self,
        texture_model: TextureFittingModel,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
    ) -> None:
        # Suffix-routing regression: an .h5 out_file_name must still go through
        # HidraProjectFile, not NXstress -- exercised implicitly here since no
        # NXstress import/call would succeed against this pre-existing file layout.
        ws = minimal_HidraWorkspace(with_instrument=True)
        project_path = write_minimal_h5_project(ws, filename="source.h5")

        texture_model.load_hidra_project_file(str(project_path))
        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()))
        fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        # out_file_name == currently loaded file -> update in place, no copy.
        texture_model.save_fit_result(str(project_path), fit_result=fit_result)
        assert project_path.exists()


class TestTextureFittingModelNXstressRoundtrip:
    def test_save_fit_result_nxstress_roundtrip_matches_workspace_and_peaks(
        self,
        texture_model: TextureFittingModel,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        n_subrun = len(ws.get_sub_runs())
        peak = minimal_PeakCollection(N_subrun=n_subrun)
        fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        texture_model.ws = ws
        out_path = tmp_path / "roundtrip.nxs"

        texture_model.save_fit_result(str(out_path), fit_result=fit_result)
        assert out_path.exists()

        reloaded = TextureFittingModel(None)
        reloaded.load_hidra_project_file(str(out_path))

        assert reloaded.ws is not None
        assert len(reloaded.ws.get_sub_runs()) == n_subrun
        assert reloaded._curr_file_name == str(out_path)

        # Phase-1 scope: peaks round-trip and are exposed via `fit_result` for this
        # assertion, but deliberately are NOT wired into `fit_table_operator`/the
        # plot overlay -- see texture_fitting_model.py's load_hidra_project_file.
        assert reloaded.fit_result is not None
        assert reloaded.fit_result.fitted is None
        assert reloaded.fit_result.difference is None
        assert len(reloaded.fit_result.peakcollections) == 1
        # NXstress decomposes a peak tag into (phase name, (h, k, l)) and
        # reconstructs it in canonical no-space form on read -- "Fe 110" round
        # trips as "Fe110", not verbatim. See _Peaks._parse_peak_tag.
        assert reloaded.fit_result.peakcollections[0].peak_tag == peak.peak_tag.replace(" ", "")
