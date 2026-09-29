"""
A single-workspace viewer refuses an NXstress file holding more, by name.

Integration rather than unit by `CLAUDE.md`'s dividing line: `read_single_workspace`
is interface code driving the NXstress library, so this exercises two components
together even though every fixture is synthetic and lives under `tmp_path`.

What these pin is a *silent* failure, not a crash. `PeakFittingModel` and
`TextureFittingModel` used to take `wss[0]` under a comment asserting that an
entry written by the viewer holds one workspace -- true of files it wrote, and
not of files a user may open. A three-direction entry loaded with no error and
two thirds of its content missing, which the next save would have made permanent.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.nxstress_load import UnsupportedNXstressFile, read_single_workspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.config import Config
from pyrs.utilities.NXstress.NXstress import NXstress

pytestmark = pytest.mark.integration


@pytest.fixture
def discriminated(tmp_path):
    """`direction` as the discriminator, so a multi-workspace entry can be written."""
    override = tmp_path / "override.yml"
    override.write_text("nxstress:\n  discriminator_fields: ['direction']\n")
    Config.loadEnv(str(override))
    yield
    Config.loadEnv("application.yml")


def _workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction, sub_runs):
    points = np.array(sub_runs)
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
    ws.set_sample_log("direction", points, np.array([direction] * len(points)))
    return ws, [minimal_PeakCollection(N_subrun=len(points), sub_runs=points)]


class TestSingleWorkspaceFileLoads:
    def test_one_workspace_in_one_entry_reads_back(
        self,
        discriminated,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        # Arrange
        path = tmp_path / "one.nxs"
        ws, peaks = _workspace(minimal_HidraWorkspace, minimal_PeakCollection, "11", (1, 2, 3))
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])

        # Act
        recovered, recovered_peaks = read_single_workspace(path)

        # Assert -- the shape `NXstress.read` returned before spec 04b
        assert isinstance(recovered, HidraWorkspace)
        assert recovered.get_sub_runs().raw_copy().tolist() == [1, 2, 3]
        assert len(recovered_peaks) == len(peaks)


class TestFilesAViewerCannotShowAreRefused:
    def test_a_multi_workspace_entry_is_refused(
        self,
        discriminated,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        """The case that used to load with two thirds of its content dropped."""
        # Arrange
        path = tmp_path / "three.nxs"
        built = [
            _workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction, points)
            for direction, points in (("11", (1, 2, 3)), ("22", (4, 5, 6)), ("33", (7, 8, 9)))
        ]
        with NXstress(path, "w") as nx:
            nx.write([ws for ws, _ in built], [peaks for _, peaks in built])

        # Act / Assert -- the count is in the message, so the user learns what is there
        with pytest.raises(UnsupportedNXstressFile, match=r"holds 3 workspaces"):
            read_single_workspace(path)

    def test_a_multi_entry_file_is_refused(
        self,
        discriminated,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        """`read()` defaults to entry 1, so this used to open the first and say nothing."""
        # Arrange
        path = tmp_path / "two_entries.nxs"
        first = _workspace(minimal_HidraWorkspace, minimal_PeakCollection, "11", (1, 2, 3))
        second = _workspace(minimal_HidraWorkspace, minimal_PeakCollection, "22", (4, 5, 6))
        with NXstress(path, "w") as nx:
            nx.write([first[0]], [first[1]])
        with NXstress(path, "a", entry_number=2) as nx:
            nx.write([second[0]], [second[1]])

        # Act / Assert
        with pytest.raises(UnsupportedNXstressFile, match=r"holds 2 NXentry groups"):
            read_single_workspace(path)

    def test_the_refusal_is_a_runtime_error(self) -> None:
        """So it lands in the viewers' existing "unable to load" handling.

        `peak_fitting_viewer.load_and_plot` catches `(RuntimeError, TypeError)`
        and pops a dialog; the refusal has to be caught there, or it escapes into
        Qt and the user sees nothing.
        """
        assert issubclass(UnsupportedNXstressFile, RuntimeError)


class TestReadReturnsObjectsDetachedFromTheFile:
    """`read()` must return objects that outlive the context manager.

    `read_single_workspace` counts the workspaces and indexes `[0]` after
    `NXstress.read` returns, and both viewers then use the result long after the
    file is closed. That works because `read` materialises everything through
    `.nxdata` -- but nothing in `NXstress` *states* it, and the read path does
    index `NXfield`s lazily before materialising (`_InputData.readSubruns`
    selects rows in h5py). A future change that returned a field instead of its
    data would fail here, far from its cause, rather than in this test.
    """

    def test_every_part_of_a_workspace_survives_the_close(
        self,
        discriminated,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection: Callable[..., PeakCollection],
        tmp_path: Path,
    ) -> None:
        # Arrange
        path = tmp_path / "detached.nxs"
        points = np.array([1, 2, 3])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points, with_raw_counts=True)
        ws.set_sample_log("direction", points, np.array(["11"] * 3))
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points)]

        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])

        # Act -- everything below happens with the file closed
        with NXstress(path, "r") as nx:
            workspaces, peak_collections = nx.read()
        recovered = workspaces[0]

        # Assert -- one per family, since each is read by a different module
        assert recovered.get_sub_runs().raw_copy().tolist() == [1, 2, 3]
        assert recovered.get_detector_counts(1).shape == ws.get_detector_counts(1).shape
        assert list(recovered.reduction_masks) == list(ws.reduction_masks)
        assert recovered.get_sample_log_value("direction") == "11"
        assert peak_collections[0][0].get_effective_params() is not None
