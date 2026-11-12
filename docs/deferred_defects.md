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

- **DD-11**, 2026-11-12: integration tests write fixed file names into the working folder instead of
  `tmp_path`, and several leave them behind. `test_reduction.py` (several `HB2B_*.h5`, some never
  removed), `test_manual_reduction_ui.py` (`output_dir = os.getcwd()`; `test_reduction_with_vanadium`
  never removes its output), `test_project_file_rw.py` (both outputs left behind), and `test_file_object.py`,
  `test_write_stress_csv.py`, `test_peak_fitting.py` and `test_calibration_file_io.py` (removed only on
  success). No two modules share a name today, so parallel runs don't collide.

  **Found during:** the parallel-test review for #1100.
  **Why it is deferred.** No collision exists under `--dist loadscope`. Moving every writer to `tmp_path`
  touches seven modules outside the CI change. CLAUDE.md now asks new and changed tests to use `tmp_path`.

- **DD-12**, 2026-11-12: `test_peakfit_calibration.py`'s gold-file comparison passes its arguments in the
  wrong order. `are_equivalent_jsons(test_json_name, gold_json_name, atol)` loops over the keys of its
  second argument, but the three tests call it as `are_equivalent_jsons(goldfile, file_name, ...)`. So it
  loops over the written file's keys, and a key in the gold file that is missing from the written
  calibration is never detected.

  **Found during:** the parallel-test review for #1100.
  **Why it is deferred.** It changes what three integration tests check, which needs its own run against
  the gold files, outside the CI change.
