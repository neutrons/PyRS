"""
Shared fixtures for the NXstress test suite.

Fixture conventions
--------------------
- `minimal_HidraWorkspace` — build a small, entirely synthetic (no disk I/O)
  `HidraWorkspace`, via its public setters. This is the default choice for new
  NXstress tests: it lets a test's assertions be judged purely on the code under
  test, not on incidental facts about whichever real HB2B run happens to be on
  disk. Toggle `with_instrument`/`with_masks`/`with_raw_counts` for the specific
  structure your test needs.
- `minimal_PeakCollection` — a thin convenience wrapper around `createPeakCollection`
  (see tests/util/peak_collection_helpers.py) that fills in sensible defaults;
  still fully overridable.
- `.nxs` files written by a test must land under pytest's built-in `tmp_path`
  fixture and need no separate cleanup — this is already the convention every
  test in this directory follows.
- `load_HidraWorkspace` — legacy: reads a real HidraProject file from disk. Kept
  for a test that genuinely needs real project-file content (e.g. testing
  `HidraProjectFile`'s own I/O); prefer `minimal_HidraWorkspace` for anything else.
- `default_config` — see `tests/unit/pyrs/utilities/conftest.py` (one directory up):
  it lives there, not here, since `pyrs/utilities/config.py` isn't itself
  NXstress-specific code, and a fixture defined there is visible down into this
  directory too.
"""

from collections.abc import Callable, Generator
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile.file_object import HidraProjectFile, HidraProjectFileMode

import pytest
from tests.util.peak_collection_helpers import createPeakCollection  # noqa: F401
from tests.util.workspace_helpers import make_minimal_HidraWorkspace


@pytest.fixture
def minimal_HidraWorkspace() -> Generator[Callable[..., HidraWorkspace]]:
    # Factory fixture: builds a small but valid `HidraWorkspace` entirely from
    # synthetic in-memory data -- no project file is read from disk. The builder
    # itself is a plain function, so by-hand scripts can call it too.

    yield make_minimal_HidraWorkspace

    # teardown follows
    pass


@pytest.fixture
def minimal_PeakCollection(createPeakCollection):
    # Convenience wrapper around `createPeakCollection` with defaults suited to
    # `minimal_HidraWorkspace` -- still fully overridable.

    def _init(
        *,
        N_subrun: int,
        peak_tag: str = "Fe 110",
        peak_profile: str = "Gaussian",
        background_type: str = "Linear",
        wavelength: float = 1.486,
        projectfilename: str = "/does/not/exist.h5",
        runnumber: int = 1,
        **kwargs,
    ):
        return createPeakCollection(
            peak_tag=peak_tag,
            peak_profile=peak_profile,
            background_type=background_type,
            wavelength=wavelength,
            projectfilename=projectfilename,
            runnumber=runnumber,
            N_subrun=N_subrun,
            **kwargs,
        )

    return _init


@pytest.fixture
def load_HidraWorkspace(test_data_dir) -> Generator[Callable[..., HidraWorkspace]]:
    # Legacy: loads a `HidraWorkspace` instance from a real `HidraProject`-format
    # file on disk. Prefer `minimal_HidraWorkspace` for new tests -- this fixture
    # is kept for a test that genuinely needs real project-file content (e.g.
    # testing `HidraProjectFile`'s own I/O).

    def _init(*, file_name: str, name: str, load_raw_counts=True, load_reduced_diffraction=True) -> HidraWorkspace:
        file_path = Path(test_data_dir) / file_name
        ws = HidraWorkspace(name)
        with HidraProjectFile(file_path, mode=HidraProjectFileMode.READONLY) as project_file:
            ws.load_hidra_project(
                project_file, load_raw_counts=load_raw_counts, load_reduced_diffraction=load_reduced_diffraction
            )
        return ws

    yield _init

    # teardown follows
    pass
