from pathlib import Path
from typing import List

from qtpy.QtCore import Signal, QObject  # type:ignore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile  # type: ignore
from pyrs.utilities.NXstress.NXstress import NXstress


class CombineRunsModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self):
        super().__init__()
        self._hidra_ws = None

    def combine_project_files(self, project_files: List[str]) -> None:
        self._hidra_ws = HidraWorkspace("Combined Project Files")
        _project = HidraProjectFile(project_files[0])
        self._hidra_ws.load_hidra_project(_project, load_raw_counts=False, load_reduced_diffraction=True)
        _project.close()

        for project in project_files[1:]:
            _project = HidraProjectFile(project)
            self._hidra_ws.append_hidra_project(_project)
            _project.close()

    def export_project_files(self, fileout: str) -> None:
        if Path(fileout).suffix == ".nxs":
            # peakss is always [] for this viewer -- CombineRunsModel has no
            # PeakCollection concept, only merges/exports raw/reduced diffraction
            # data. Already a valid, no-op-safe input: _Peaks.validateNoDuplicatePeaks([])
            # / _Fit.validateWorkspaceAndPeaksData(ws, []) both no-op, and
            # _Peaks.init_group([], ...) / _Fit.init_group(ws, [], ...) both produce
            # a valid, empty PEAKS/FIT group rather than raising.
            with NXstress(Path(fileout), "w") as nx:
                nx.write(self._hidra_ws, [])
            return

        export_project = HidraProjectFile(fileout, "w")
        self._hidra_ws.save_experimental_data(
            export_project, sub_runs=self._hidra_ws._sample_logs.subruns, ignore_raw_counts=True
        )
        self._hidra_ws.save_reduced_diffraction_data(export_project)
        export_project.save()
