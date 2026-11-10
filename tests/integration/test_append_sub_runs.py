"""Appended project files get sub-run numbers after the largest existing one

Appended sub-runs used to be numbered from ``count + 1``; a workspace whose sub-runs do not
start at 1 then got duplicate numbers.
"""

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration


def _loaded(path):
    workspace = HidraWorkspace("appended")
    with HidraProjectFile(str(path)) as project:
        workspace.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)
    return workspace


def _without_first_sub_run(tmp_path):
    """A project file holding HB2B_1331's sub-runs 2..24 (so it does not start at 1)."""
    full = _loaded("tests/data/HB2B_1331.h5")
    sub_runs = list(full.get_sub_runs())[1:]
    path = tmp_path / "HB2B_1331_from_2.h5"
    with HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE) as project:
        full.save_experimental_data(project, sub_runs=sub_runs, ignore_raw_counts=True)
        full.save_reduced_diffraction_data(project, sub_runs=sub_runs)
        project.save(False)
    return path


def test_append_after_sub_runs_not_starting_at_one(tmp_path):
    """Appending to sub-runs 2..24 numbers the new ones 25.., with no duplicates."""
    # Arrange
    workspace = _loaded(_without_first_sub_run(tmp_path))

    # Act
    with HidraProjectFile("tests/data/HB2B_1332.h5") as project:
        workspace.append_hidra_project(project)

    # Assert
    sub_runs = list(workspace.get_sub_runs())
    assert len(sub_runs) == 23 + 26
    assert len(set(sub_runs)) == len(sub_runs)
    assert sub_runs[23:] == list(range(25, 25 + 26))


def test_append_after_sub_runs_from_one_is_unchanged():
    """The usual case, sub-runs 1..N, still numbers the appended ones N+1.."""
    # Arrange
    workspace = _loaded("tests/data/HB2B_1331.h5")

    # Act
    with HidraProjectFile("tests/data/HB2B_1332.h5") as project:
        workspace.append_hidra_project(project)

    # Assert
    assert list(workspace.get_sub_runs()) == list(range(1, 24 + 26 + 1))
