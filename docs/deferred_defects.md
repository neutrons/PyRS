# Deferred defects

Defects in the PyRS codebase that were found and reproduced, but deliberately not fixed in the change
that found them. Each PR's commit message lists the ones it found, under "Defects recorded, not fixed".

One line per defect: its number, the date it was found, and enough of a description to identify it.
Numbers are never reused. When a defect is fixed, mark its line fixed rather than deleting it.

DD-6 to DD-9 are recorded by a later, still-pending PR of the same series (NXstress hookup, phase 3), and
DD-11 to DD-14 by another pending PR (`fix/931-missing-run-file`); keeping the numbers distinct lets each merge
without renumbering.

- **DD-1**, 2026-10-11: `validate_config()` runs only at import, so a configuration installed later with
  `Config.loadEnv` is never validated. The gap is in `neutrons_standard.Config`, which has no validation hook.
- **DD-2**, 2026-10-20: a workspace's sub-runs must be in increasing order, which nothing needs, and about ten
  sites use a sub-run number as an array position, so they break on any workspace whose sub-runs do not run
  1..N. Each site has a strict `xfail` test.
- **DD-3**, 2026-10-21: strain/stress viewer: loading a project file with no peak collections as e22 or e33
  raises `IndexError`.
- **DD-4**, 2026-10-21: strain/stress viewer: `Controller.fileSelected` is never called, and calls
  `Model.set_workspace`, which does not exist.
- **DD-5**, 2026-11-05: detector calibration viewer: a recipe saved before any data were loaded cannot be
  loaded back. `PeakLinesSetupView.save_json` writes `nexus_file` only when a file is loaded, but `load_json`
  reads it unconditionally and raises `KeyError`.

- **DD-10**, 2026-11-09: a legacy save truncates its destination on open, so a failed write destroys an
  existing file of that name. `HidraProjectFile(path, "w")` opens with `h5py.File(path, "w")`, which empties
  the file before anything is written; any error after that -- a refused write, a full disk -- leaves a
  truncated `.h5` where the user's file was. NXstress saves avoid this by writing a temporary sibling and
  replacing the destination only on success (`gui_helper.write_nxstress_or_report`); legacy saves do not.

  **Found during:** the review of the NXstress hookup, phase 1 (CodeRabbit comment D3).
  **Why it is deferred.** Giving legacy saves the same protection changes how every legacy write behaves,
  outside the NXstress scope of that PR; the save helper would also have to become format-neutral.

- **DD-15**, 2026-11-12: `HidraWorkspace.append_hidra_project` grows the sub-run axis with `append_subruns` and the
  reduced data with `np.append`, and never checks that the two grew by the same number of rows. The axis
  setters and both project-file readers now refuse a workspace whose data disagree with its sub-runs; this
  path does not.

  **Found during:** the review of the NXstress hookup, phase 2 (CodeRabbit comment D2).
  **Why it is deferred.** The legacy append path is outside that PR's changes, and checking it needs the
  append's own row bookkeeping rather than the whole-axis check the other paths share.
- **DD-16**, 2026-11-12: the legacy project-file extension is hard-coded as `.h5` in about ten GUI file
  filters and file-name constructions, although it is configurable as `legacy_io.extension`. With another
  extension configured, those dialogs neither offer nor recognise legacy files of that extension.

  **Found during:** the review of the NXstress hookup, phase 2 (CodeRabbit comment D4), which made every
  NXstress extension check read `nxstress.extension`.
  **Why it is deferred.** Out of that comment's scope, which was the NXstress extension.
- **DD-17**, 2026-11-12: a project file's reduced data carries no sub-run labels. A reduction of a subset of
  sub-runs saved to the file (`ReductionApp.save_diffraction_data` after `reduce_data(sub_runs=[...])`) writes
  one row per reduced sub-run, and `HidraWorkspace.load_hidra_project` pairs those rows with the file's sub-runs
  by position: the rows of any subset that does not start at the first sub-run are attributed to the wrong
  sub-runs, and nothing on disk can show it. Related to DD-2.

  **Found during:** the review of the NXstress hookup, phase 2 (CodeRabbit comment D2). A load-time check that
  the rows match the sub-runs refused such files, which PyRS itself writes, and was withdrawn.
  **Why it is deferred.** The legacy project-file format and its subset save predate that PR and are outside
  its NXstress scope; fixing it means recording the rows' sub-runs on save.
- **DD-18**, 2026-11-12: the GUI suite still segfaults at interpreter shutdown without `os._exit()`, after
  the strain/stress SliceViewer observer leak was fixed (see the shutdown-segfault section of
  `docs/ground_truths.md`). Some other object outlives the Qt or mantid teardown it depends on. The first
  thing to check is defects of the same kind as the SliceViewer one: a widget or helper that holds a C++
  resource or registers a callback -- a mantid ADS or algorithm observer, an embedded mantid or matplotlib
  canvas, a `QThread`, `QTimer` or signal connection to a long-lived object -- and is removed from its layout
  or dropped, but never closed, stopped or disconnected. Closing a window sends no `closeEvent` to the
  widgets inside it, so each such widget's cleanup must be called explicitly.

  **Found during:** the review of the NXstress hookup, phase 2, when the GUI tier was run without `os._exit()`.
  **Why it is deferred.** Finding the source means auditing every viewer, and confirming the CI crash needs
  `xvfb-run`; both are outside that PR's scope.

- **DD-22**, 2026-11-12: the ASCII pole-figure export scrambles its rows. `export_arrays_to_ascii`
  (`pyrs/core/polefigurecalculator.py`) sorts with `np.sort(combined_array, axis=0)`, which sorts each
  column on its own, so a written row's α, β and intensity no longer belong to the same measured point.
  The same ASCII branch of `export_pole_figure` also passes the whole peak list on every pass, so each
  per-peak `.txt` file holds every peak's points. Both are latent: the texture window exports MTEX
  (`.jul`), so ASCII is reached only by calling `export_pole_figure(file_type="ascii")` directly.
  `test_export_ascii_keeps_each_point_together` pins the correct row behaviour as a strict `xfail`.

  **Found during:** #411 (pole-figure tests).
  **Why it is deferred.** A production change outside #411, which adds tests only.

- **DD-23**, 2026-11-12: a pole on the rim of the pole figure (α = 90°) can come out as NaN.
  `rotate_project_q` takes `arccos(√(h₁²+h₂²))`, and the root can round to just above 1. Example:
  θ = 41.1°, ω = θ, χ = 90° gives NaN, while θ = 41.3° gives 90°. Precision also drops near the rim
  (about 1e-6° at χ = 89.9999°). Whether HB2B measures at χ = 90° decides how much this matters.
  `test_tilt_to_rim_gives_alpha_ninety` pins it as a strict `xfail`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411, which adds tests only.

- **DD-24**, 2026-11-12: `PoleFigureCalculator.add_input_data_set` checks only for **unknown** log
  names (`checkdatatypes.check_list`), not missing ones. Data without e.g. `"phi"` is accepted, and
  fails later in `calculate_pole_figure` (or when more data is appended) with a bare `KeyError`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.

- **DD-25**, 2026-11-12: `PoleFigureCalculator` bookkeeping:
  - **The caller's data is changed in place.** The first `add_input_data_set` for a peak stores the
    caller's dict itself, so adding more data to that peak later rewrites the caller's arrays.
  - **A calculated pole figure goes stale.** `get_polefigure_array` keeps returning it after more
    points are appended, until `calculate_pole_figure` runs again. The export recalculates, but
    `texture_fitting_crtl.extract_polar_projection` does not.

  Both are latent with today's callers (`load_pole_data` builds a new dict each time).

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.
