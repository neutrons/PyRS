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

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile.file_object import HidraProjectFile, HidraProjectFileMode

import pytest
from tests.util.peak_collection_helpers import createPeakCollection  # noqa: F401


@pytest.fixture
def minimal_HidraWorkspace() -> Generator[Callable[..., HidraWorkspace]]:
    # Factory fixture: builds a small but valid `HidraWorkspace` entirely from
    # synthetic in-memory data -- no project file is read from disk.

    def _init(
        *,
        name: str = "test_workspace",
        n_subruns: int = 3,
        with_instrument: bool = True,
        with_masks: bool = False,
        mask_names: tuple = (),
        with_raw_counts: bool = False,
        with_reduced_diffraction: bool = True,
        n_two_theta: int = 20,
    ) -> HidraWorkspace:
        ws = HidraWorkspace(name)

        subruns = np.arange(1, n_subruns + 1, dtype=int)
        ws.set_sub_runs(subruns)

        # Minimal sample logs: coordinates, timestamps, sample rotation.
        ws.set_sample_log("vx", subruns, np.arange(n_subruns, dtype=float))
        ws.set_sample_log("vy", subruns, np.zeros(n_subruns, dtype=float))
        ws.set_sample_log("vz", subruns, np.zeros(n_subruns, dtype=float))
        ws.set_sample_log(
            "start_time", subruns, np.array([f"2024-01-15T10:{n:02d}:00".encode("utf-8") for n in range(n_subruns)])
        )
        ws.set_sample_log(
            "end_time", subruns, np.array([f"2024-01-15T10:{n:02d}:30".encode("utf-8") for n in range(n_subruns)])
        )
        ws.set_sample_log("mrot", subruns, np.zeros(n_subruns, dtype=float))
        # `NXstress._init`/`_Fit._init` read `start_time`/`end_time`/`Filename` sample-log
        # values as bytes (`t.decode("utf-8")`) -- matching what comes back from a real
        # h5py-backed string dataset. A synthetic workspace needs to match that, not a
        # plain numpy unicode string.
        ws.set_sample_log("Filename", subruns, np.array([f"{name}.h5".encode("utf-8")] * n_subruns))

        ws.set_wavelength(1.486, calibrated=True)

        n_pixels = 16  # small synthetic detector -- no test asserts a specific pixel count
        if with_instrument:
            geometry = DENEXDetectorGeometry(
                num_rows=4,
                num_columns=4,
                pixel_size_x=0.001,
                pixel_size_y=0.001,
                arm_length=2.0,
                calibrated=True,
            )
            ws.set_instrument_geometry(geometry)
            shift = DENEXDetectorShift(
                shift_x=0.01, shift_y=0.02, shift_z=0.03, rotation_x=1.0, rotation_y=2.0, rotation_z=3.0, tth_0=0.5
            )
            ws.set_detector_shift(shift)

        if with_masks:
            default_mask = np.ones(n_pixels, dtype=np.int64)
            ws.set_detector_mask(default_mask, True)
            for mask_name in mask_names:
                ws.set_detector_mask(np.ones(n_pixels, dtype=np.int64), False, mask_name)

        if with_raw_counts:
            for subrun in subruns:
                ws.set_raw_counts(int(subrun), np.arange(n_pixels, dtype=np.int64))

        if with_reduced_diffraction:
            two_theta_matrix = np.tile(np.linspace(60.0, 120.0, n_two_theta), (n_subruns, 1))
            intensities = np.ones((n_subruns, n_two_theta), dtype=float)
            variances = np.ones((n_subruns, n_two_theta), dtype=float)
            ws.set_reduced_diffraction_data_set(two_theta_matrix, {None: intensities}, {None: variances})

        return ws

    yield _init

    # teardown follows
    pass


@pytest.fixture
def minimal_PeakCollection(createPeakCollection):
    # Convenience wrapper around `createPeakCollection` with defaults suited to
    # `minimal_HidraWorkspace` -- still fully overridable.

    def _init(*, N_subrun: int, peak_tag: str = "Fe 110", peak_profile: str = "Gaussian",
              background_type: str = "Linear", wavelength: float = 1.486,
              projectfilename: str = "/does/not/exist.h5", runnumber: int = 1, **kwargs):
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
