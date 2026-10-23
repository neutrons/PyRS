# NXstress hookup: phase 1

**Changes to the PyRS codebase:**

- `HidraWorkspace.load_hidra_project` loads instrument geometry independently of `load_raw_counts`, logging a warning when the project file has none.

- Added `PyRsCore.register_hidra_workspace()` to register an in-memory `HidraWorkspace` as a session data source.

- `PeakFittingViewer` takes peak tags from the peak-range table's *Label* column; previously every tag was `peak<N>`, which NXstress cannot parse. Blank or auto-generated `Peak<N>` labels still give `peak<N>`, and duplicate tags are reported as an error before fitting (`pyrs/interface/peak_fitting/peak_tags.py`).

- `PeakFittingModel.load_hidra_project` accepts a single `.h5` path as well as a list; a bare string previously loaded `"/"`.

- `TextureFittingModel.save_fit_result` no longer references the undefined `self.parent` (previously always raised `AttributeError`); `load_hidra_project_file` now closes its project-file handle and returns `None`.

- `TextureFittingViewer`: the *Save* and *Save as* actions no longer share one attribute.

- `CombineRunsViewer`: fixed a `TypeError` in the "Run Numbers" input path, ignored a cancelled save dialog, and removed the unused `set_text_values`.

- Added `gui_helper.impose_extension()`; every save/export dialog, `.h5` included, now forces its format's configured extension onto the chosen filename (see config settings in [`application.yml`](../../pyrs/resources/application.yml)).

- Added type hints to the peak-fitting, texture-fitting, and combine-runs model/viewer classes.

- Tests: `tests/conftest.py` performs the first `pyrs.utilities.config` import under a throwaway `HOME`; added `tests/util/project_file_helpers.py` and conftest re-exports of the shared NXstress fixtures for `tests/integration` and `tests/ui`.

**Phase 1 implementation:**

- PeakFitting and TextureFitting viewers: added *File → Save as NXstress…* and an `NXstress (*.nxs)` load filter; the models dispatch on the `.nxs` suffix to `NXstress.write`/`read`.

- CombineRuns viewer: after loading, an NXstress export prompt follows the `.h5` one and writes the merged workspace with no peaks (`NXstress.write(ws, [])`).

- NXstress output is offered only when `Config["nxstress.enable"]` is set **and** the loaded workspace has instrument geometry: otherwise *Save as NXstress…* is grayed out, and CombineRuns skips its NXstress prompt. PeakFitting and TextureFitting previously checked only the config flag, so a workspace without geometry failed inside the writer. Legacy `.h5` actions follow `Config["legacy_io.enable"]`.

- *Save as NXstress…* writes to a temporary file that replaces the target only on success, and reports a failed save (e.g. a peak tag that is not `<phase> <hkl>`) in an error message. Previously PeakFitting silently ignored the error, and in TextureFitting the uncaught exception aborted the application; both left a truncated `.nxs` behind, destroying any existing file of that name. Shared helpers `gui_helper.write_nxstress_or_report()` and `gui_helper.nxstress_save_available()` serve both viewers.

- TextureFitting's *Save as NXstress…* reads its fit result from `self.fit_summary`; it referenced the undefined `self._parent`, so every save raised `AttributeError`.

- Cancelling PeakFitting's *Save as NXstress…* dialog no longer raises `TypeError`.

- The NXstress writer's generated default detector mask is 1-D (the same change as phase 2); it was 2-D, so a file written from a workspace without its own mask could not be read back (`Mask array with shape … is not acceptable`).

- The NXstress reader returns no peaks for files written with an empty peak list (`UNDEFINED_PEAK_TAG`).

- `PeakFittingCrtl.plot_diff_and_fitted_data` tolerates an NXstress load with no fitted spectrum; an `.nxs` load also registers its workspace with `PyRsCore`.

- Tests: model-level `.nxs` round-trip and `.h5` routing tests for all three viewers (integration), GUI tests for the new save/export actions, their geometry gating and error handling, and the `fitted=None` guard, and unit tests for the peak-tag rules and the NXstress save helpers.

## How to test

Start PyRS with `pixi run pyrs`; all files below are in `tests/data/`. NXstress output is on by default (`nxstress.enable` in `application.yml`), but is offered only for a workspace with instrument geometry: `HB2B_1017.h5` and `HB2B_1017_w_mask.h5` have it; `HB2B_1423.h5`, `HB2B_1599.h5`, `HB2B_1331.h5` and `HB2B_1332.h5` do not.

### Peak Fitting

1. Main window → **Peak Fitting** → **Browse Exp. Data File ...** → `HB2B_1423.h5`. **File → Save as NXstress...** is grayed out.
2. **Browse Exp. Data File ...** → `HB2B_1017.h5`. **File → Save as NXstress...** is now enabled.
3. Drag-select the peak near 81° (≈ 79–84), set its **Label** to `Fe 211` (NXstress needs `<phase> <hkl>`), then **Fit Peak(s)**.
4. **File → Save as NXstress...** → any name; it is saved with a `.nxs` extension.
5. **Browse Exp. Data File ...** → the saved `.nxs`: the data and sub-runs reload (the fitted curve is not reconstructed yet).
6. Optional: reload `HB2B_1017.h5`, fit with the default label `Peak0`, and save again: an error is shown and no file is written.

### Texture Fitting

1. Main window → **Texture Fitting Window** → **Browse Exp Data** → `HB2B_1599.h5`. **File → Save as NXstress...** is grayed out.
2. **Browse Exp Data** → `HB2B_1017.h5`; drag-select ≈ 79–84 on the fit plot, set **Peak Label** to `Fe 211`, then **Fit**.
3. **File → Save as NXstress...** → any name; a `.nxs` file is written and can be reloaded with **Browse Exp Data**.

### Combine Runs

1. Main window → **Advanced → Combine Runs** → **Browse Exp Data** → select `HB2B_1331.h5` and `HB2B_1332.h5`. Only the `.h5` save dialog appears.
2. **Browse Exp Data** → select `HB2B_1017.h5` and `HB2B_1017_w_mask.h5`. The `.h5` save dialog appears, then the NXstress one; saving writes a `.nxs` file. Cancelling either dialog skips that format.
