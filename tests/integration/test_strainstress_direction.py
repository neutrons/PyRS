"""
The strain/stress viewer refuses a project file whose recorded direction disagrees.

`Model.load_hidra_project_file(filename, direction)` is told which direction a
file represents by the user, in the file-selection UI, and since this PR it also
records that on the workspace's `direction` property for NXstress to carry. If
the project file itself records a direction and the two disagree, one of them is
wrong and neither the model nor the viewer can tell which.

Reported rather than fatal: the model's existing `failureMsg` signal puts it in
front of the user and returns `(None, {})`, so the viewer stays usable and the
file can be re-selected under the right direction.

Integration: a real `HidraProjectFile` written to `tmp_path` and read back
through a GUI model class.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
import pyrs.interface.strainstressviewer.model as model_module
from pyrs.interface.strainstressviewer.model import Model
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode

pytestmark = pytest.mark.integration


def _project_with_direction(tmp_path: Path, minimal_HidraWorkspace, direction: str | None) -> Path:
    """A Hidra project file whose `direction` log holds `direction`, or has none."""
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
    if direction is not None:
        points = ws.get_sub_runs().raw_copy()
        ws.set_sample_log("direction", points, np.array([direction] * len(points)))
    path = tmp_path / f"direction_{direction}.h5"
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    ws.save_experimental_data(project, sub_runs=ws.get_sub_runs(), ignore_raw_counts=True)
    ws.save_reduced_diffraction_data(project)
    project.save(False)
    return path


@pytest.fixture
def model_and_failures():
    """A `Model` with its failure signal collected rather than shown."""
    model = Model()
    failures: list = []
    model.failureMsg.connect(lambda *args: failures.append(args))
    return model, failures


class TestDirectionMismatchIsReported:
    def test_a_disagreeing_direction_is_refused(
        self, model_and_failures, minimal_HidraWorkspace: Callable[..., HidraWorkspace], tmp_path: Path
    ) -> None:
        # Arrange
        model, failures = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, "22")

        # Act -- selected by the user as the 11 direction
        ws, peaks = model.load_hidra_project_file(str(path), "11")

        # Assert -- reported, and nothing loaded
        assert ws is None
        assert peaks == {}
        assert failures, "a mismatch must reach the user"
        assert "22" in str(failures[0]) and "11" in str(failures[0])

    def test_an_agreeing_direction_loads(
        self, model_and_failures, minimal_HidraWorkspace: Callable[..., HidraWorkspace], tmp_path: Path
    ) -> None:
        # Arrange
        model, failures = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, "11")

        # Act
        ws, _ = model.load_hidra_project_file(str(path), "11")

        # Assert
        assert not failures
        assert ws is not None
        assert ws.direction == "11"

    def test_a_file_recording_no_direction_loads(
        self, model_and_failures, minimal_HidraWorkspace: Callable[..., HidraWorkspace], tmp_path: Path
    ) -> None:
        """The ordinary case: most project files carry no `direction` log at all."""
        # Arrange
        model, failures = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, None)

        # Act
        ws, _ = model.load_hidra_project_file(str(path), "33")

        # Assert -- the user's choice stands, and is recorded for NXstress
        assert not failures
        assert ws is not None
        assert ws.direction == "33"


class TestProjectFileIsClosed:
    """The loader closes the project file it opens, whoever else holds a reference.

    `HidraProjectFile.__del__` also closes it, so with no other reference CPython's
    reference counting hides a missing `close()`: the file is closed as the
    loader's frame is torn down. Any surviving reference -- a stored traceback, a
    reference cycle, a debugger -- would keep it open. The recording subclass
    below is such a reference.
    """

    @pytest.fixture
    def opened(self, monkeypatch) -> list:
        """Every project file the model opens, kept alive by this list."""
        opened: list = []

        class Recording(HidraProjectFile):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                opened.append(self)

        monkeypatch.setattr(model_module, "HidraProjectFile", Recording)
        return opened

    @pytest.mark.parametrize("recorded, selected", [("11", "11"), ("22", "11")], ids=["loaded", "refused"])
    def test_the_file_is_closed_after_loading(
        self,
        model_and_failures,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        tmp_path: Path,
        opened: list,
        recorded: str,
        selected: str,
    ) -> None:
        # Arrange
        model, _ = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, recorded)

        # Act
        model.load_hidra_project_file(str(path), selected)

        # Assert
        assert len(opened) == 1
        assert opened[0]._project_h5 is None, "the project file was left open"


class TestControllerReportsARefusedLoad:
    """`Controller.filesSelected` answers whether the selection loaded.

    The viewer's file field shows the selected names only when it returns True.
    A refused load leaves the model holding an empty list, not None, so a check
    for None reported every refusal as a success and the field showed a file
    that had not loaded.
    """

    def test_a_refused_load_is_not_reported_as_loaded(
        self, model_and_failures, minimal_HidraWorkspace: Callable[..., HidraWorkspace], tmp_path: Path
    ) -> None:
        # Arrange
        from pyrs.interface.strainstressviewer.controller import Controller

        model, failures = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, "22")

        # Act
        loaded = Controller(model).filesSelected("e11", [str(path)])

        # Assert
        assert failures, "the load was expected to be refused"
        assert loaded is False

    def test_an_accepted_load_is_reported_as_loaded(
        self, model_and_failures, minimal_HidraWorkspace: Callable[..., HidraWorkspace], tmp_path: Path
    ) -> None:
        # Arrange
        from pyrs.interface.strainstressviewer.controller import Controller

        model, failures = model_and_failures
        path = _project_with_direction(tmp_path, minimal_HidraWorkspace, "11")

        # Act
        loaded = Controller(model).filesSelected("e11", [str(path)])

        # Assert
        assert not failures
        assert loaded is True
