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

- ~~**DD-22**~~ **Fixed** (pole-figure defects PR, after #1102): the ASCII export (`export_arrays_to_ascii` and the `ascii` branch of `export_pole_figure`) was removed; asking for any format but MTEX raises `ValueError`. Originally: 2026-11-12: the ASCII pole-figure export scrambles its rows. `export_arrays_to_ascii`
  (`pyrs/core/polefigurecalculator.py`) sorts with `np.sort(combined_array, axis=0)`, which sorts each
  column on its own, so a written row's α, β and intensity no longer belong to the same measured point.
  The same ASCII branch of `export_pole_figure` also passes the whole peak list on every pass, so each
  per-peak `.txt` file holds every peak's points. Both are latent: the texture window exports MTEX
  (`.jul`), so ASCII is reached only by calling `export_pole_figure(file_type="ascii")` directly.
  (Its strict-`xfail` test was replaced by `test_export_other_than_mtex_raises` when the export was removed.)

  **Found during:** #411 (pole-figure tests).
  **Why it is deferred.** A production change outside #411, which adds tests only.

- ~~**DD-23**~~ **Fixed** (pole-figure defects PR, after #1102): `rotate_project_q` clips both `arccos` arguments into range, so a rim pole is α = 90°. Originally: 2026-11-12: a pole on the rim of the pole figure (α = 90°) can come out as NaN.
  `rotate_project_q` takes `arccos(√(h₁²+h₂²))`, and the root can round to just above 1. Example:
  θ = 41.1°, ω = θ, χ = 90° gives NaN, while θ = 41.3° gives 90°. Precision also drops near the rim
  (about 1e-6° at χ = 89.9999°). Whether HB2B measures at χ = 90° decides how much this matters.
  `test_tilt_to_rim_gives_alpha_ninety` pins it as a strict `xfail`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411, which adds tests only.

- ~~**DD-24**~~ **Fixed** (pole-figure defects PR, after #1102): `add_input_data_set` refuses data missing a required log name, naming it (`RuntimeError`). Originally: 2026-11-12: `PoleFigureCalculator.add_input_data_set` checks only for **unknown** log
  names (`checkdatatypes.check_list`), not missing ones. Data without e.g. `"phi"` is accepted, and
  fails later in `calculate_pole_figure` (or when more data is appended) with a bare `KeyError`.

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.

- ~~**DD-25**~~ **Fixed** (pole-figure defects PR, after #1102): `add_input_data_set` stores a copy of the first data for a peak, and drops the peak's calculated pole figure whenever data is added, so it is recalculated. Originally: 2026-11-12: `PoleFigureCalculator` bookkeeping:
  - **The caller's data is changed in place.** The first `add_input_data_set` for a peak stores the
    caller's dict itself, so adding more data to that peak later rewrites the caller's arrays.
  - **A calculated pole figure goes stale.** `get_polefigure_array` keeps returning it after more
    points are appended, until `calculate_pole_figure` runs again. The export recalculates, but
    `texture_fitting_crtl.extract_polar_projection` does not.

  Both are latent with today's callers (`load_pole_data` builds a new dict each time).

  **Found during:** the review of #411.
  **Why it is deferred.** A production change outside #411.

- **DD-19**, 2026-11-12: `PoleFigureCalculator.export_pole_figure` and `get_pole_figure_1_pt` read the
  stored pole figures directly. Since the DD-25 fix, adding data to a peak drops that peak's stored pole
  figure, so calling either one afterwards without `calculate_pole_figure` raises a bare `KeyError`. The
  export may already have written the earlier peaks' files by then. Before, both silently used the
  stale pole figure. No caller does this: the texture window calls `calculate_pole_figure` right before
  exporting, and `get_pole_figure_1_pt` has no callers. A fix would read through
  `get_polefigure_array`, which recalculates.

  **Found during:** the review of the DD-22 to DD-25 fixes.
  **Why it is deferred.** A further production change, beyond the four approved fixes.
- **DD-20**, 2026-11-12: integration tests write fixed file names into the working folder instead of
  `tmp_path`, and several leave them behind. `test_reduction.py` (several `HB2B_*.h5`, some never
  removed), `test_manual_reduction_ui.py` (`output_dir = os.getcwd()`; `test_reduction_with_vanadium`
  never removes its output), `test_project_file_rw.py` (both outputs left behind), and `test_file_object.py`,
  `test_write_stress_csv.py`, `test_peak_fitting.py` and `test_calibration_file_io.py` (removed only on
  success). No two modules share a name today, so parallel runs don't collide.

  **Found during:** the parallel-test review for #1100.
  **Why it is deferred.** No collision exists under `--dist loadscope`. Moving every writer to `tmp_path`
  touches seven modules outside the CI change. CLAUDE.md now asks new and changed tests to use `tmp_path`.

- **DD-21**, 2026-11-12: `test_peakfit_calibration.py`'s gold-file comparison passes its arguments in the
  wrong order. `are_equivalent_jsons(test_json_name, gold_json_name, atol)` loops over the keys of its
  second argument, but the three tests call it as `are_equivalent_jsons(goldfile, file_name, ...)`. So it
  loops over the written file's keys, and a key in the gold file that is missing from the written
  calibration is never detected.

  **Found during:** the parallel-test review for #1100.
  **Why it is deferred.** It changes what three integration tests check, which needs its own run against
  the gold files, outside the CI change.

- **DD-27**, 2026-11-16: `TextureFittingViewer.save` and `saveas` (`pyrs/interface/texture_fitting/texture_fitting_viewer.py`)
  read `self._parent.fit_summary`, but `TextureFittingViewer` never sets `_parent`; only its child widgets do. Either
  would raise `AttributeError`. Both are unreachable: their legacy Save actions are disabled, and
  `tests/ui/test_texture_fitting.py` checks they stay disabled. A TODO beside them already notes it. The fix is
  `self.fit_summary.fit_table_operator.fit_result`, as `save_as_nxstress` uses.

  **Found during:** the mypy failure after #1112: mypy saw it through PySide6's new type stubs, and the PySide6
  override hides it again.
  **Why it is deferred.** Unreachable today, and outside a lint fix. To be fixed with the scoped-enum change,
  when the override is removed.

- **DD-28**, 2026-11-18: `pyrs-calibration` (`pyrs/calibration/peakfit_driver.py:main`, until #1101
  `scripts/pyrs_calibration.py`) raises `UnboundLocalError` unless the input JSON sets every optional key.
  - **Cause:** `main()` assigns `REFINE_METHOD`, `nexus_file`, `CalibName`, `KEEP_SUB_RUNS`,
    `INSTRUMENT_CALIBRATION`, `DATA_MASK`, `SAVE_CALIB`, `WRITE_LATEST` and `HFIR_CYCLE` inside its loop over the
    JSON keys. Python therefore treats them as locals of `main()`, and the module-level defaults are never read.
  - **Reproduced:** `{"powder scan": 3510, "method": "full"}` stops at `if nexus_file is None:` before reading any
    data. The template the tool writes (`template.json`) fails the same way, and so does any input without a
    `"method"` key, at `if "+" in REFINE_METHOD`.
  - **Since:** the code dates from 7a6068b5 (2023-07-18).
  - **Untested:** `test_peakfit_driver.py` checks only the usage text, so no test catches this.
  - **Fix:** start `main()` from a settings dict made from the defaults and update it from the JSON. Add a unit test
    that runs `main()` on a minimal JSON with `FitCalibration` replaced by a stand-in.

  **Found during:** the code review of #1101, which moved the file unchanged.
  **Why it is deferred.** A production change, outside a pure move. It needs its own approval.
- **DD-31**, 2026-11-19: `check_sweeping_motor` (`pyrs/core/nexus_conversion.py`) removes motors from
  `motor_search` while iterating over it (`for motor in motor_search: ... motor_search.pop(...)`).
  - **Effect:** the motor after each removed one is never tested, so a sweeping motor can stay in the list used to
    correct the first sub-run's start time. On HB2B_5344, `sx` and `2theta` are removed and `sy` and `omega` are
    skipped, so `sy` stays in; HB2B_3489 and HB2B_1017 are similar.
  - **Impact so far:** none found. The start time was unchanged on 5340, 5344, 1017 and 3489.
  - **Fix:** build the list with a comprehension, `[m for m in motor_search if runObj[m].size() <= num_points]`.
  - **Found during:** the pixelation study (#1106), which reproduced the start-time correction outside PyRS.
  - **Why it is deferred:** a production change outside the approved #1109 change.
- **DD-30**, 2026-11-19: `FitCalibration` (`pyrs/calibration/mantid_peakfit_calibration.py`) determines the detector
  rotations poorly on HB2B_3510.
  - **Reproduced:** with 2θ₀ held at 0 and all sub-runs, two σ weightings give solutions of equal RMS (0.0244° and
    0.0240°) whose Rot_x differs by 4.4° (+3.215° against −1.230°, quoted errors 0.05–0.08°). Rot_z also differs
    (0.162° against −0.001°).
  - **Comparison:** the ring calibration of the same run gives Rot_x 0.29 ± 0.50°, Rot_z 0.185 ± 0.029°.
  - **Effect:** `FitCalibration`'s quoted rotation errors understate how far its result can move. Its η-wedge
    residuals constrain the rotations weakly.
  - **Found during:** #1109 (Poisson bin variance in reduction), which changed `FitCalibration`'s peak weights.
  - **Why it is deferred:** a change to the established calibration, outside #1109, needing its own approval.

- **DD-29**, 2026-11-19: `tests/integration/test_nexus_calibration_reduction.py::test_nexus_and_json_reduce_bit_identically[eta-slices]`
  depends on the test order.
  - **Cause:** it compares `get_diffraction_data(sub_run)`, whose default is the whole-detector pattern (mask
    `None`). An `eta_step=3.0` reduction makes only `eta_*` patterns.
  - **Why it passes today:** the module-scoped `workspace` still holds the `[whole-detector]` case's patterns,
    and that case runs first.
  - **Failure:** run alone, or on another xdist worker (`-n` without `--dist loadscope`), it raises
    `RuntimeError: Mask ID None does not exist`.
  - **Same on `next`:** checked on a scratch checkout. The pixi tasks use `--dist loadscope`, so CI passes.
  - **Fix:** compare every mask the reduction made (`eta_*` here), not only the default one, so the case checks
    what it reduced.

  **Found during:** #1098 step 4, running the project-file and NeXus reduction tests together.
  **Why it is deferred.** A test defect outside #1098, and CI is not affected.
- **DD-34**, 2026-11-25: `calculate_sub_run_time_average` (`pyrs/core/nexus_conversion.py`) returns NaN for
  `vx/vy/vz` (about 47% of autoreduced files, usually sub-run 0) because the raw logs start with a NaN placeholder
  that overlaps the sub-run window by milliseconds. The strain/stress viewer then aborts on `PointList`'s finite-value
  assertion when a peak is selected (#1147, [ground_truths.md](ground_truths.md)).
- **DD-35**, 2026-11-25: strain/stress viewer: the `selectedPeak` setter calls `create_strain` outside any
  handler, so any `StrainField` error escapes a Qt slot and the viewer process aborts instead of reporting it.
