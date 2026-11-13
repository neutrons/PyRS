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

- **DD-11**, 2026-11-10: the texture-fitting window carries on after a project file fails to load.
  `FileLoad.load_project_plot` goes on to read `model.sub_runs` and `model.ws` after
  `load_hidra_project_file` has reported its failure through `failureMsg` (it returns nothing either way),
  so the window continues with mixed state:
  - **a file h5py cannot open** (not HDF5): the previous workspace stays, shown under the failed file's
    name; with no earlier load, a `'NoneType' object has no attribute …` error;
  - **an HDF5 file that is not a project file:** `model.ws` is already replaced by an empty workspace while
    `model.sub_runs` still belongs to the previous file, and the failed file's handle is never closed.

  Loading by run number no longer reaches this for a missing file: the run lookup now
  refuses it first. (The peak-fitting window had the same flaw; #1026 fixed it.)

  **Found during:** #931 (PR #1089).
  **Why it is deferred.** Making the window stop after a failed load was proposed and declined.

- **DD-12**, 2026-11-11: peak fitting's sub-run boxes don't agree on what a number means.
  - **Sub-runs for the 2D plot** (`PeakFittingCrtl.parse_sub_run_list`): entries are 0-based array
    positions, kept only from 1 to N−1, so the first sub-run can never be plotted, and `1:5` is sub-runs
    2–6 for a file numbered 1..N.
  - **Sub-runs to plot** (`plot_diff_data`): a single entry is a 1-based position, and two or more are
    sub-run numbers.
  - **Run numbers** (`load_run_number`): run numbers, as everywhere else.

  **Found during:** the second review of #931 (PR #1089), after ranges were made inclusive in
  `gui_helper.parse_integers`.
  **Why it is deferred.** Choosing one meaning changes what users select in the peak-fitting window; it
  needs its own decision.

- **DD-13**, 2026-11-11: a sub-run that doesn't exist escapes the Qt slot in peak fitting's
  `plot_diff_data`. With two or more entries, the loop catches only `RuntimeError`, but a sub-run number
  missing from the file raises `IndexError` from `SubRuns.get_indices` (e.g. `1:20` for a 10-sub-run
  file). A single out-of-range entry fails the same way through `sub_run_list[scan_value - 1]`.

  **Found during:** the second review of #931 (PR #1089).
  **Why it is deferred.** Catching it is a production change outside #931's approved scope.

- **DD-14**, 2026-11-11: `CombineRunsModel.combine_project_files` doesn't close a project file when
  `file_record` or `read_reduction_provenance` raises. And if `append_hidra_project` fails partway,
  `_hidra_ws` keeps the partly combined workspace. The second is latent: after a failure the window
  clears `_project_files`, which both save dialogs check, so it can't be exported.

  **Found during:** the second review of #931 (PR #1089).
  **Why it is deferred.** A production change outside #931's approved scope (the same kind of leak as
  DD-11).
