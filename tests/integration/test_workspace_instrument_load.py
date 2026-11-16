"""Integration tests for how `HidraWorkspace.load_hidra_project` loads instrument geometry.

Each test writes a real `.h5` project file with `HidraProjectFile` and loads it
back into a workspace -- a multi-component workflow with real file I/O, so these
are `integration` (see CLAUDE.md's marker definitions).
"""

from pathlib import Path
from typing import Callable

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration


def _load(ws: HidraWorkspace, path: Path) -> None:
    project = HidraProjectFile(str(path), mode=HidraProjectFileMode.READONLY)
    try:
        ws.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)
    finally:
        project.close()


def test_geometry_loaded_without_raw_counts(
    write_minimal_h5_project: Callable[..., Path], minimal_HidraWorkspace: Callable[..., HidraWorkspace]
) -> None:
    """Geometry no longer depends on `load_raw_counts`."""
    # Arrange
    path = write_minimal_h5_project(
        minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True), "geometry.h5", with_instrument=True
    )
    ws = HidraWorkspace()

    # Act
    _load(ws, path)

    # Assert
    assert ws.get_instrument_setup() is not None


def test_reload_without_geometry_clears_previous_geometry(
    write_minimal_h5_project: Callable[..., Path], minimal_HidraWorkspace: Callable[..., HidraWorkspace]
) -> None:
    """A workspace loaded again from a project with no geometry must not keep the
    previous project's geometry, which NXstress would otherwise write as this
    project's (`HB2BReductionManager` reloads its session workspace this way)."""
    # Arrange
    with_geometry = write_minimal_h5_project(
        minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True), "geometry.h5", with_instrument=True
    )
    without_geometry = write_minimal_h5_project(minimal_HidraWorkspace(sub_runs=[1, 2, 3]), "no_geometry.h5")
    ws = HidraWorkspace()
    _load(ws, with_geometry)
    assert ws.get_instrument_setup() is not None

    # Act
    _load(ws, without_geometry)

    # Assert
    assert ws.get_instrument_setup() is None
