"""
Synthetic `HidraWorkspace` builder shared by the NXstress tests and by-hand scripts.

`make_minimal_HidraWorkspace` builds a small but valid workspace entirely from
in-memory data via its public setters -- no project file is read from disk. The
`minimal_HidraWorkspace` fixture (tests/unit/pyrs/utilities/NXstress/conftest.py)
returns it; scripts under `tests/scripts/` call it directly.
"""

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace

#: `n_subruns`' default, named so that `make_minimal_HidraWorkspace` can tell a
#: caller who passed it explicitly from one who did not, and refuse the first in
#: combination with `sub_runs` rather than silently discarding it.
_DEFAULT_N_SUBRUNS = 3


def make_minimal_HidraWorkspace(
    *,
    name: str = "test_workspace",
    n_subruns: int = _DEFAULT_N_SUBRUNS,
    sub_runs: np.ndarray | None = None,
    with_instrument: bool = True,
    with_masks: bool = False,
    mask_names: tuple = (),
    with_raw_counts: bool = False,
    with_reduced_diffraction: bool = True,
    n_two_theta: int = 20,
) -> HidraWorkspace:
    ws = HidraWorkspace(name)

    # `sub_runs` replaces the default 1..N. Two workspaces sharing one
    # NXstress entry must cover disjoint scan points, and the interesting
    # case -- values that interleave between workspaces -- cannot be built
    # from 1..N at all.
    #
    # It also replaces an explicitly passed `n_subruns`, so the two are
    # refused together rather than one silently winning: a test reading
    # `n_subruns=6, sub_runs=(1, 2, 3)` asks for six and would get three.
    if sub_runs is None:
        subruns = np.arange(1, n_subruns + 1, dtype=int)
    else:
        subruns = np.asarray(sub_runs, dtype=int)
        if n_subruns != len(subruns) and n_subruns != _DEFAULT_N_SUBRUNS:
            raise ValueError(
                f"make_minimal_HidraWorkspace: n_subruns={n_subruns} disagrees with "
                f"sub_runs={subruns.tolist()} ({len(subruns)} values). Pass one or the other."
            )
        n_subruns = len(subruns)
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
