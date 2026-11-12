# Deferred defects

Defects in the PyRS codebase that were found and reproduced, but deliberately not fixed in the change
that found them. Each PR's commit message lists the ones it found, under "Defects recorded, not fixed".

One line per defect: its number, the date it was found, and enough of a description to identify it.
Numbers are never reused. When a defect is fixed, mark its line fixed rather than deleting it.

Numbering starts at DD-10 here: DD-1 to DD-9 are recorded by later, still-pending PRs of the same
series, and keeping the numbers distinct lets those PRs merge without renumbering.

- **DD-10**, 2026-11-09: a legacy save truncates its destination on open, so a failed write destroys an
  existing file of that name. `HidraProjectFile(path, "w")` opens with `h5py.File(path, "w")`, which empties
  the file before anything is written; any error after that -- a refused write, a full disk -- leaves a
  truncated `.h5` where the user's file was. NXstress saves avoid this by writing a temporary sibling and
  replacing the destination only on success (`gui_helper.write_nxstress_or_report`); legacy saves do not.

  **Found during:** the review of the NXstress hookup, phase 1 (CodeRabbit comment D3).
  **Why it is deferred.** Giving legacy saves the same protection changes how every legacy write behaves,
  outside the NXstress scope of that PR; the save helper would also have to become format-neutral.

- ~~**DD-15**~~ **Fixed** (pole-figure defects PR, after #1102): the ASCII export (`export_arrays_to_ascii` and the `ascii` branch of `export_pole_figure`) was removed; asking for any format but MTEX raises `ValueError`. Originally: 2026-11-12: the ASCII pole-figure export scrambles its rows. `export_arrays_to_ascii`
  (`pyrs/core/polefigurecalculator.py`) sorts with `np.sort(combined_array, axis=0)`, which sorts each
  column on its own, so a written row's α, β and intensity no longer belong to the same measured point.
  The same ASCII branch of `export_pole_figure` also passes the whole peak list on every pass, so each
  per-peak `.txt` file holds every peak's points. Both are latent: the texture window exports MTEX
  (`.jul`), so ASCII is reached only by calling `export_pole_figure(file_type="ascii")` directly.
  `test_export_ascii_keeps_each_point_together` pins the correct row behaviour as a strict `xfail`.

  **Found during:** #411 (pole-figure tests).
  **Why it is deferred.** A production change outside #411, which adds tests only.

- ~~**DD-16**~~ **Fixed** (pole-figure defects PR, after #1102): `rotate_project_q` clips both `arccos` arguments into range, so a rim pole is α = 90°. Originally: 2026-11-12: a pole on the rim of the pole figure (α = 90°) can come out as NaN.
  `rotate_project_q` takes `arccos(√(h₁²+h₂²))`, and the root can round to just above 1. Example:
  θ = 41.1°, ω = θ, χ = 90° gives NaN, while θ = 41.3° gives 90°. Precision also drops near the rim
  (about 1e-6° at χ = 89.9999°). Whether HB2B measures at χ = 90° decides how much this matters.
  `test_tilt_to_rim_gives_alpha_ninety` pins it as a strict `xfail`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411, which adds tests only.

- ~~**DD-17**~~ **Fixed** (pole-figure defects PR, after #1102): `add_input_data_set` refuses data missing a required log name, naming it (`RuntimeError`). Originally: 2026-11-12: `PoleFigureCalculator.add_input_data_set` checks only for **unknown** log
  names (`checkdatatypes.check_list`), not missing ones. Data without e.g. `"phi"` is accepted, and
  fails later in `calculate_pole_figure` (or when more data is appended) with a bare `KeyError`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.

- ~~**DD-18**~~ **Fixed** (pole-figure defects PR, after #1102): `add_input_data_set` stores a copy of the first data for a peak, and drops the peak's calculated pole figure whenever data is added, so it is recalculated. Originally: 2026-11-12: `PoleFigureCalculator` bookkeeping:
  - **The caller's data is changed in place.** The first `add_input_data_set` for a peak stores the
    caller's dict itself, so adding more data to that peak later rewrites the caller's arrays.
  - **A calculated pole figure goes stale.** `get_polefigure_array` keeps returning it after more
    points are appended, until `calculate_pole_figure` runs again. The export recalculates, but
    `texture_fitting_crtl.extract_polar_projection` does not.

  Both are latent with today's callers (`load_pole_data` builds a new dict each time).

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.
