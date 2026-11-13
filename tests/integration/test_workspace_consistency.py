"""A workspace's logs survive a file round trip as given.

The tests run a `HidraWorkspace` through a file format and back -- the legacy project
file and NXstress -- so they exercise the workspace and a reader together.
"""

import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.NXstress.NXstress import NXstress
from tests.util.workspace_helpers import make_minimal_HidraWorkspace

pytestmark = pytest.mark.integration


def _write_project(ws: HidraWorkspace, path) -> None:
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    ws.save_experimental_data(project, sub_runs=None, ignore_raw_counts=True)
    ws.save_reduced_diffraction_data(project, sub_runs=None)
    project.save()
    project.close()


def _load_project(path) -> HidraWorkspace:
    ws = HidraWorkspace("readback")
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        ws.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)
    return ws


class TestBlankLogRoundTrip:
    """A blank text log is read back as "", not as a placeholder.

    "None" was once substituted for a blank log on save. A log set to the empty
    string -- left unfilled by the user, or by ADARA -- would then be
    indistinguishable from one that was never set.
    """

    VALUES = ["first", "", "third"]

    @pytest.fixture
    def ws(self) -> HidraWorkspace:
        ws = make_minimal_HidraWorkspace(sub_runs=[1, 2, 3])
        ws.set_sample_log("note", np.array([1, 2, 3]), np.array(self.VALUES))
        return ws

    def test_project_file(self, ws, tmp_path):
        path = tmp_path / "blank.h5"

        _write_project(ws, path)

        assert _load_project(path).get_sample_log_values("note").tolist() == self.VALUES

    def test_nxstress(self, ws, tmp_path):
        path = tmp_path / "blank.nxs"

        with NXstress(path, "w") as nx:
            nx.write([ws], [[]])
        with NXstress(path, "r") as nx:
            (read_ws,), _ = nx.read()

        assert read_ws.get_sample_log_values("note").tolist() == self.VALUES
