"""Integration round-trip tests for spec 02 (NXstress I/O for PeakFitting & Texture
viewers): write via NXstress, read back, assert HidraWorkspace/PeakCollection
equality; and confirm the .h5 suffix still routes through HidraProjectFile
(no regression to the pre-existing path).

Model-level only (no QWidget construction) -- see tests/ui/test_peak_fitting.py /
test_texture_fitting.py for the GUI-level enablement-wiring coverage.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pyrs.core.pyrscore import PyRsCore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult
from pyrs.utilities.NXstress.NXstress import NXstress

pytestmark = pytest.mark.integration


class TestPeakFittingViewerRoundtrip:
    def test_save_fit_result_nxstress_roundtrip_matches_workspace_and_peaks(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        n_subrun = len(ws.get_sub_runs())
        peaks = [
            minimal_PeakCollection(N_subrun=n_subrun, peak_tag="Fe110", runnumber=1),
            minimal_PeakCollection(N_subrun=n_subrun, peak_tag="Ni200", runnumber=1),
        ]

        model = PeakFittingModel(PyRsCore())
        model.hidra_workspace = ws
        model.fit_result = FitResult(peakcollections=peaks, fitted=None, difference=None)
        model._project_name = "test_project"

        out_path = tmp_path / "roundtrip.nxs"
        model.save_fit_result(str(out_path))
        assert out_path.exists()

        reloaded = PeakFittingModel(PyRsCore())
        reloaded.load_hidra_project(str(out_path))

        assert len(reloaded.hidra_workspace.get_sub_runs()) == n_subrun
        assert len(reloaded.fit_result.peakcollections) == len(peaks)
        reloaded_tags = sorted(p.peak_tag for p in reloaded.fit_result.peakcollections)
        assert reloaded_tags == sorted(p.peak_tag for p in peaks)

    def test_save_fit_result_h5_suffix_routes_through_hidraprojectfile(
        self,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True)
        project_path = write_minimal_h5_project(ws, filename="source.h5", with_instrument=True)

        model = PeakFittingModel(PyRsCore())
        model.load_hidra_project([str(project_path)])
        assert model._curr_file_name == str(project_path)
        # The .h5 load path never populates fit_result -- confirms the new .nxs
        # branch above didn't change this pre-existing behavior.
        assert model.fit_result is None

        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()))
        model.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        out_path = tmp_path / "saved.h5"
        model.save_fit_result(str(out_path))
        assert out_path.exists()


class TestTextureFittingViewerRoundtrip:
    def test_save_fit_result_nxstress_roundtrip_matches_workspace_and_peaks(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        n_subrun = len(ws.get_sub_runs())
        peaks = [
            minimal_PeakCollection(N_subrun=n_subrun, peak_tag="Fe110", runnumber=1),
            minimal_PeakCollection(N_subrun=n_subrun, peak_tag="Ni200", runnumber=1),
        ]

        model = TextureFittingModel(None)
        model.ws = ws
        fit_result = FitResult(peakcollections=peaks, fitted=None, difference=None)

        out_path = tmp_path / "roundtrip.nxs"
        model.save_fit_result(str(out_path), fit_result=fit_result)
        assert out_path.exists()

        reloaded = TextureFittingModel(None)
        reloaded.load_hidra_project_file(str(out_path))

        assert reloaded.ws is not None
        assert reloaded.fit_result is not None
        assert len(reloaded.ws.get_sub_runs()) == n_subrun
        assert len(reloaded.fit_result.peakcollections) == len(peaks)
        reloaded_tags = sorted(p.peak_tag for p in reloaded.fit_result.peakcollections)
        assert reloaded_tags == sorted(p.peak_tag for p in peaks)

    def test_save_fit_result_h5_suffix_routes_through_hidraprojectfile(
        self,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        ws = minimal_HidraWorkspace(with_instrument=True)
        project_path = write_minimal_h5_project(ws, filename="source.h5", with_instrument=True)

        model = TextureFittingModel(None)
        model.load_hidra_project_file(str(project_path))
        assert model._curr_file_name == str(project_path)

        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()))
        fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

        out_path = tmp_path / "saved.h5"
        model.save_fit_result(str(out_path), fit_result=fit_result)
        assert out_path.exists()


class TestCombineRunsViewerRoundtrip:
    def test_export_project_files_nxstress_roundtrip_matches_merged_workspace(
        self,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        tmp_path: Path,
    ) -> None:
        # combine_project_files merges with load_raw_counts=False, so the merged
        # workspace never carries raw counts -- the .nxs export never populates
        # the optional input_data group. This test's scope is deliberately
        # narrowed to sub-run counts and sample-log arrays as a direct
        # consequence, not an arbitrary choice -- see
        # open-questions/03-combine-runs-nxstress.md Q1.
        ws1 = minimal_HidraWorkspace(name="run1", with_masks=True)
        ws2 = minimal_HidraWorkspace(name="run2", with_masks=True)
        project_path_1 = write_minimal_h5_project(ws1, filename="run1.h5", with_instrument=True, with_masks=True)
        project_path_2 = write_minimal_h5_project(ws2, filename="run2.h5", with_instrument=True, with_masks=True)

        model = CombineRunsModel()
        model.combine_project_files([str(project_path_1), str(project_path_2)])

        merged_n_subrun = len(model._hidra_ws.get_sub_runs())
        merged_vx = model._hidra_ws.get_sample_log_values("vx")

        out_path = tmp_path / "combined.nxs"
        model.export_project_files(str(out_path))
        assert out_path.exists()

        with NXstress(out_path, "r") as nx:
            ws_read, peaks_read = nx.read()

        assert peaks_read == []
        assert len(ws_read.get_sub_runs()) == merged_n_subrun
        np.testing.assert_allclose(ws_read.get_sample_log_values("vx"), merged_vx)

    def test_export_project_files_h5_suffix_routes_through_hidraprojectfile(
        self,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        tmp_path: Path,
    ) -> None:
        ws1 = minimal_HidraWorkspace(name="run1")
        project_path_1 = write_minimal_h5_project(ws1, filename="run1.h5", with_instrument=True)

        model = CombineRunsModel()
        model.combine_project_files([str(project_path_1)])

        out_path = tmp_path / "combined.h5"
        model.export_project_files(str(out_path))
        assert out_path.exists()
