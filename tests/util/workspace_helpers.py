"""
Synthetic `HidraWorkspace` builder shared by the NXstress tests and by-hand scripts.

`make_minimal_HidraWorkspace` builds a small but valid workspace entirely from
in-memory data via its public setters -- no project file is read from disk. The
`minimal_HidraWorkspace` fixture (tests/unit/pyrs/utilities/NXstress/conftest.py)
returns it; scripts under `tests/scripts/` call it directly.
"""

from collections.abc import Sequence

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from tests.util.peak_collection_helpers import SEED


def make_minimal_HidraWorkspace(
    *,
    sub_runs: Sequence[int] | np.ndarray,
    name: str = "test_workspace",
    seed: int = SEED,
    with_instrument: bool = True,
    with_masks: bool = False,
    mask_names: tuple = (),
    with_raw_counts: bool = False,
    with_reduced_diffraction: bool = True,
    n_two_theta: int = 20,
) -> HidraWorkspace:
    """A small, valid `HidraWorkspace` over `sub_runs`, built in memory.

    Every per-scan-point and per-mask array holds *distinct* values -- seeded random
    counts, masks and intensities; distinct coordinates per axis -- so that a round
    trip that attaches a row to the wrong sub-run, or one mask's data to another
    mask, is caught rather than compared equal.

    Args:
        sub_runs: Its scan points, strictly increasing. Required: two workspaces
            sharing one NXstress entry must cover disjoint scan points, so no
            single default is right for every test.
        name: The workspace name; also used for the `Filename` log.
        seed: Seeds a fresh generator per call, so a workspace's values never
            depend on what was built before it.
        with_instrument: Set a synthetic detector geometry and shift.
        with_masks: Set a default mask, plus one named mask per `mask_names`.
        mask_names: The named masks to set, when `with_masks`.
        with_raw_counts: Set raw detector counts for every sub-run.
        with_reduced_diffraction: Set reduced data, one data set per mask key.
        n_two_theta: The number of two-theta bins.
    """
    rng = np.random.default_rng(seed)
    ws = HidraWorkspace(name)

    subruns = np.asarray(sub_runs, dtype=int)
    n_subruns = len(subruns)
    ws.set_sub_runs(subruns)

    # Minimal sample logs: coordinates, timestamps, sample rotation.
    # Distinct per axis as well as per scan point, so that a swapped axis or a
    # positional (rather than by-sub-run) lookup does not compare equal.
    ws.set_sample_log("vx", subruns, np.arange(n_subruns, dtype=float))
    ws.set_sample_log("vy", subruns, 10.0 + np.arange(n_subruns, dtype=float))
    ws.set_sample_log("vz", subruns, 20.0 + np.arange(n_subruns, dtype=float))
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
        ws.set_detector_mask(rng.integers(0, 2, n_pixels, dtype=np.int64), True)
        for mask_name in mask_names:
            ws.set_detector_mask(rng.integers(0, 2, n_pixels, dtype=np.int64), False, mask_name)

    if with_raw_counts:
        for subrun in subruns:
            ws.set_raw_counts(int(subrun), rng.integers(0, 1000, n_pixels, dtype=np.int64))

    if with_reduced_diffraction:
        two_theta_matrix = np.tile(np.linspace(60.0, 120.0, n_two_theta), (n_subruns, 1))
        mask_keys = [None, *mask_names] if with_masks else [None]
        intensities = {key: rng.uniform(1.0, 1000.0, (n_subruns, n_two_theta)) for key in mask_keys}
        # Poisson-like: a variance equal to its intensity, so each is as distinct.
        variances = {key: values.copy() for key, values in intensities.items()}
        ws.set_reduced_diffraction_data_set(two_theta_matrix, intensities, variances)

    return ws
