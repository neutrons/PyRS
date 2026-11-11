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

- **DD-11**, 2026-11-10: the peak-fitting and texture windows carry on after a project file fails to load.
  `PeakFittingViewer.load_and_plot` shows its error pop-up and then reads the sub-runs of a workspace that
  was never loaded; texture's `FileLoad.load_project_plot` reads `model.sub_runs` after
  `load_hidra_project_file` has reported its failure (it returns nothing either way). Both end in a
  `'NoneType' object has no attribute …` error, e.g. for a non-project `.h5` picked in the file dialog.
  Loading by run number no longer reaches this for a missing file: the run lookup now refuses it first.

  **Found during:** #931 (PR #1089).
  **Why it is deferred.** Making both windows stop after a failed load was proposed and declined.
