"""
Shared fixture for persisting a synthetic `HidraWorkspace` to a real `.h5` file.

Consumers live in more than one tier (`tests/integration/`, `tests/ui/`), so the
fixture lives here rather than in any one directory's `conftest.py`; each tier's
`conftest.py` re-exports it.

Typical use:

1. Build a workspace with the `minimal_HidraWorkspace` factory fixture.
2. Persist it with `write_minimal_h5_project(ws)`, opting into instrument
   geometry and/or masks as the component under test requires.
3. Point the code under test at the returned path.
"""

from pathlib import Path
from typing import Callable

import pytest

from pyrs.core.instrument_geometry import HidraSetup
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.projectfile.file_object import HidraProjectFile, HidraProjectFileMode


@pytest.fixture
def write_minimal_h5_project(tmp_path: Path) -> Callable[..., Path]:
    """Persist a synthetic in-memory `HidraWorkspace` to a real `.h5`
    `HidraProjectFile` on disk under `tmp_path`.

    `PeakFittingModel.save_fit_result` and `TextureFittingModel.save_fit_result`
    both copy an existing project file and patch peak parameters into the copy for
    their `.h5` branch, so exercising that branch (as opposed to the NXstress
    branch, which writes fresh with no existing file) requires a real file on disk
    to start from.

    Returns:
        A factory `(ws, filename=..., with_instrument=..., with_masks=...) -> Path`.

    Example:
        >>> path = write_minimal_h5_project(ws, with_instrument=True, with_masks=True)
    """

    def _init(
        ws: HidraWorkspace,
        filename: str = "project.h5",
        *,
        with_instrument: bool = False,
        with_masks: bool = False,
    ) -> Path:
        """Write `ws` to `tmp_path / filename`.

        Args:
            ws: Workspace to persist, e.g. from `minimal_HidraWorkspace`.
            filename: Name of the file to create under `tmp_path`.
            with_instrument: Also write instrument geometry and wavelength.
                `CombineRunsModel.combine_project_files()` reads these back from
                disk, and NXstress requires a real geometry to write at all -- a
                bare `save_experimental_data()`/`save_reduced_diffraction_data()`
                round trip leaves geometry as None.
            with_masks: Also write the workspace's default mask. NXstress's own
                default-mask fallback, used when no default mask is present, has
                an unrelated pre-existing shape bug (found during spec 02);
                writing a real default mask avoids hitting it.

        Returns:
            Path to the file written.

        Raises:
            ValueError: If a flag is set but `ws` carries no such data.
        """
        if with_instrument and ws.get_instrument_setup() is None:
            raise ValueError(
                "with_instrument=True requires a workspace carrying instrument geometry; "
                "build it with minimal_HidraWorkspace(with_instrument=True)"
            )
        if with_masks and ws.get_detector_mask(is_default=True) is None:
            raise ValueError(
                "with_masks=True requires a workspace carrying a default mask; "
                "build it with minimal_HidraWorkspace(with_masks=True)"
            )

        file_path = tmp_path / filename
        project = HidraProjectFile(str(file_path), mode=HidraProjectFileMode.OVERWRITE)
        # `sub_runs=None` exports all sub-runs -- passing `ws.get_sub_runs()` (a
        # `SubRuns` instance) directly breaks `save_reduced_diffraction_data`'s
        # internal `sub_runs_array - 1` indexing, which assumes a plain ndarray.
        ws.save_experimental_data(project, sub_runs=None, ignore_raw_counts=True)
        ws.save_reduced_diffraction_data(project, sub_runs=None)

        if with_instrument:
            project.write_instrument_geometry(HidraSetup(ws.get_instrument_setup()))
            project.write_wavelength(ws.get_wavelength(True, False))
        if with_masks:
            project.write_mask_detector_array(HidraConstants.DEFAULT_MASK, ws.get_detector_mask(is_default=True))

        project.save()
        project.close()
        return file_path

    return _init
