# Deferred defects

Defects in the PyRS codebase that were found and reproduced, but deliberately not fixed in the change
that found them. Each PR's commit message lists the ones it found, under "Defects recorded, not fixed".

Each defect starts with one line: its number, the date it was found, and enough of a description to
identify it. Its details follow, indented beneath that line: where it was found, what happens, with the
measurement, why it was deferred, and what to do if it is picked up. Numbers are never reused. When a
defect is fixed, mark its line fixed and say how in its details, rather than deleting it.

Code is cited by symbol name, not line number. Plan documents cited below (`plans/...`) are developer
metadata kept outside this repository.

DD-11 to DD-14 are recorded by a still-pending PR (`fix/931-missing-run-file`); keeping the numbers distinct
lets it merge without renumbering.

- **DD-1**, 2026-10-11: `validate_config()` runs only at import, so a configuration installed later with
  `Config.loadEnv` is never validated. The gap is in `neutrons_standard.Config`, which has no validation hook.

  **Found during:** the PR review of NXstress subspecs 04b and 04c, part 7.
  **Owner:** `neutrons_standard.Config`, not PyRS.

  **What happens.** `pyrs/utilities/config.py` calls `validate_config()` once, at import, so a malformed
  configuration fails at the first import rather than at the first I/O call site. That works for the
  configuration that exists at import. `Config.loadEnv(<path>)` installs a new configuration at any later
  point, and nothing re-validates it:

  ```console
  >>> Config.loadEnv("bad.yml")            # nxstress: {use_production_names: "yes"}
  >>> Config["nxstress.use_production_names"]
  'yes'                                     # a truthy string, never checked
  ```

  `validate_config()` is public and can be called again by hand, but nothing in `Config` invites it: there
  is no validation hook, no post-refresh callback, and `_Config.validate` is a documented no-op. So the
  guarantee that a malformed configuration is rejected before it is acted on holds only at import.

  **Why it is deferred.** Closing the gap means `neutrons_standard.Config` growing a validation hook that
  `refresh`/`loadEnv` invoke. PyRS cannot add that from outside, and it wants its own review with the
  package's owners. The PyRS-side workarounds are all worse than the gap:
  - calling `validate_config()` from every consumer, the scattering `config.py` exists to prevent;
  - wrapping `Config` in a PyRS proxy that re-validates on access, a second `Config`, where the module
    docstring says there must be exactly one way to reach it;
  - re-validating inside `_discriminator.field_names()` and the like, which validates some keys at some
    call sites and reads as coverage when it is not.

  It is tolerable for now because the runtime-override path is test-only by decision (subspec 04b,
  part 2): the `env` OS variable is the documented production mechanism, and it is read at import.
  Production never calls `loadEnv`, but nothing says so and nothing stops a future caller.

  **Partially mitigated in tests (2026-10-16).** The reproduction still held when re-verified:

  ```console
  before override: False
  after loadEnv  : 'yes'   (type str, truthy=True)
  validate_config(): RuntimeError -- so it WOULD catch it, if called
  ```

  The `config_override` fixture (now in `tests/conftest.py`) calls `validate_config()` after each merge,
  so a test cannot install a configuration production would reject. A first version of this claimed the
  guarantee was measured, which was true of a standalone script but not of the suite: an independent test
  review deleted the hook and every test stayed green. The hook had also made `test_config.py`'s own
  `validate_config()` call unreachable, so the tests in `TestValidateConfigRejectsMalformedValues` were
  asserting against the fixture's exception, not the subject's. Both were closed:
  `test_config.py::TestConfigOverrideRefusesMalformedConfig` pins the hook, and deleting it now fails 11
  tests. Measuring a guarantee once is not the same as constraining it.

  **What the mitigation uncovered: three consumer checks production cannot reach.** Adding validation
  broke three uses of `config_override`. Each installed a malformed configuration to test that a consumer
  rejects it: `_discriminator.field_names()` on a blank and on a repeated discriminator name, and
  `_Instrument._instrument_names()` on a blank instrument name. Measured against `ConfigSchema` directly:

  ```console
  blank discriminator   : rejected by schema -> consumer check NOT reachable in production
  repeated discriminator: rejected by schema -> consumer check NOT reachable in production
  blank instrument_name : rejected by schema -> consumer check NOT reachable in production
  ```

  `NonBlankText` and `_distinct` refuse all three at import, and the `env` override is merged before
  `validate_config()` runs, so no production path delivers such a configuration to a consumer. The three
  tests passed only because nothing validated the override. They now call `override(..., validate=False)`,
  each with a comment saying why. Whether the consumers' own re-checks should stay is a layering question
  that means touching `_discriminator.py` and `_instrument.py`, and is left open.

  **If picked up.** Ask `neutrons_standard` for a validation hook that `refresh`/`loadEnv` invoke. Not yet
  raised with the package's owners.

- **DD-2**, 2026-10-20: a workspace's sub-runs must be in increasing order, which nothing needs, and about ten
  sites use a sub-run number as an array position, so they break on any workspace whose sub-runs do not run
  1..N. Each site has a strict `xfail` test.

  **Found during:** phase-2 PR assembly (post-review probe D7).
  **Owner:** `pyrs.dataobjects.sample_logs.SubRuns`, and the sites below that treat a sub-run number as an
  array position.

  **Recommendation: drop the requirement that a workspace's sub-runs be in increasing order.** What PyRS
  needs is that sub-run numbers are unique within a workspace, and disjoint between workspaces that share an
  NXstress entry. Both are already checked, and neither depends on order. The increasing order is enforced
  in one place, protects one lookup that can be rewritten without it, and is otherwise never used. It makes
  a correct NXstress round trip impossible for workspaces that share a discriminator value and are not
  already in increasing order, and it hides a separate class of real bugs (item 4 below) behind the
  appearance of a sorted, contiguous axis.

  **What happens.** An entry whose workspaces share one discriminator value is read back as one workspace,
  with the rows in on-disk order, which is the inputs' order. So was a merged write, until the
  `nxstress.merge_workspaces` key was removed (Decisions Log row 45, `plans/NXstress-prod/README.md`).
  Writing `[4, 5, 6]` then `[1, 2, 3]` produced a file that `read` refused, from `SubRuns.set`. Measured
  with the merged write:

  ```console
  D7 write merged [4,5,6]+[1,2,3] -> [4, 5, 6, 1, 2, 3]
  D7 read RAISES RuntimeError subruns are not sorted in increasing order
  D7 increasing order (control) -> (1, [1, 2, 3, 4, 5, 6])
  ```

  The same happened on an append of lower scan points under a discriminator value the entry already held.

  **What was done instead, in phase 2: refuse, don't re-sort.** Phase 2 first refused, with
  `NotImplementedError`, any write or append that would read back a workspace whose scan points were not
  strictly increasing, and kept a strict xfail for the shared-value case. The rows were deliberately not
  re-sorted: the on-disk order is the inputs' order, and sorting it would leave no way to re-assign the rows
  to their workspaces on read.

  **Now: one workspace per key.** Phase 2's review replaced that refusal. Two inputs sharing a workspace key
  -- the case above, which read back as one workspace -- are refused outright, because NXstress never merges
  workspaces: at write by `NXstress/_validation.py`'s `validate_multi_workspace`, and at append by
  `_append.py`'s `reject_existing_workspace_key`. Each workspace is therefore read back from its own rows,
  which are increasing because its `SubRuns` is, and **NXstress round trips no longer depend on this
  defect**. `test_multi_workspace.py::TestOneWorkspacePerKey` pins the refusal, in increasing, decreasing
  and interleaved order, and the round trip of distinct keys in decreasing order. The order requirement
  itself, and the sites below, are unchanged.

  **What actually depends on the order** (a survey of the whole codebase, 2026-10-20; the key sites were
  re-read against the code, and each defect was reproduced on a synthetic workspace):

  1. **Where the order is used: one place.** `SubRuns.get_indices` finds the positions of several sub-runs
     with `np.searchsorted`, a binary search, which is correct only on a sorted axis.
     - On an unsorted axis it returns wrong positions without raising. `SubRuns.append` skips the ordering
       check, so this is reachable: `[4,5,6]` appended with `[1,2,3]`, then `get_indices([5, 2])` gives
       `[6, 4]`.
     - Even when sorted, it checks only that a query's first and last values exist:
       `SubRuns([1,2,4]).get_indices([1,3,4])` gives `[0, 2, 2]`, so the missing 3 maps to the position of 4.
     - Single-value lookups use `np.nonzero(value == x)` and need only uniqueness.
     - It is reached through `SampleLogs.__getitem__((name, subruns))`, `get_pointlist(subruns)` (including
       NXstress `_Peaks._sample_positions`), `HidraWorkspace.get_sample_log_values`,
       `save_experimental_data(sub_runs=subset)` and the peak-fitting plot.
  2. **Where the order is checked, but not used.**
     - `SubRuns.set` checks `value[:-1] < value[1:]`. Every `SubRuns` is built through it: a project-file
       read, `HidraWorkspace._set_subruns`, `PeakCollection.set_peak_fitting_values` and NXstress
       `_Sample.sampleLogsFromNexus`, which is where the D7 read failed. What it guarantees downstream is
       that values do not repeat.
     - `_Peaks.peakCollectionRanges` checks that scan points increase within each peak-collection block.
       It exists only because the `PeakCollection` built from the block would fail `SubRuns.set` anyway.
     - `HidraWorkspace._set_subruns` sorts (with `sort=True`) only so that its input passes the check.
     - The checks that enforce the real invariant are already order-free: `_validation.py`'s
       `validate_multi_workspace` (a dict of seen scan points); `_append.py`'s
       `reject_scan_point_collisions` and `classify_compound_keys` (set intersections);
       `_Peaks.validateNoDuplicatePeaks`; and the reader's partition check in `_read.py`'s
       `workspace_selections` (`np.isin`, plus the overlap and unclaimed checks).
  3. **Places that need only one consistent order.** These pair a sub-run array with data rows by
     position, and work in any order all the arrays share:
     - reduced-data rows and sample-log columns;
     - `PeakCollection` parameter rows and its exclude list;
     - the Mantid FitPeaks spectrum index;
     - NXstress's `input_data` axis check, diffractogram rows, wavelength-dict order, and the block order
       shared by PEAKS, `peak_parameters` and `background_parameters` (`IndexedPeaks.sort_key` never
       sorts by scan point);
     - `SubRuns.__eq__` and `matching_subruns` compare element-wise, which needs the same order, not
       sorted order.

     One exception would surface once the requirement is dropped: `HidraWorkspace.set_wavelength`
     compares `sorted(dict_keys)` to the sub-runs element-wise, so it would refuse a valid dictionary for
     unsorted sub-runs. It fails loudly, and comparing as sets fixes it. There is no test, because it is
     unreachable while `SubRuns` is sorted.
  4. **Places that assume sub-runs are exactly 1..N.** These treat the sub-run number as a position. They
     break on a gap, or a first sub-run other than 1, even when sorted, so they are defects whether or not
     the requirement is dropped. Sortedness never protected them; a 1..N test workspace hides them.

     | Location | What goes wrong | Test |
     |---|---|---|
     | `HidraWorkspace.save_reduced_diffraction_data` (`pyrs/core/workspaces.py`) | Selects rows as `sub_runs_array - 1`. With no raw counts loaded it always takes this branch, so saving any workspace whose sub-runs do not start at 1 raises `IndexError`. | `test_workspaces.py::TestSubRunNumbersAreNotPositions` |
     | `HidraWorkspace.set_sample_log` | Discards the result of `matching_subruns`, so values given for other sub-runs, or in another order, are stored by position | same class, two tests |
     | `HB2BReductionManager.setup_reduction_engine` (`pyrs/core/reduction_manager.py`) | Decides whether to rebuild the 2θ map by comparing with sub-run `n - 1`, not the one reduced last. A gap or a start above 1 raises; iterating out of order would reuse the wrong engine silently. | `test_reduction_manager.py` |
     | `fit_peaks` in `pyrs/calibration/mantid_peakfit_calibration.py` | `get_diff_peaks(sub_run - 1)` against position-indexed powder tables | `tests/unit/pyrs/calibration/test_mantid_peakfit_calibration.py` |
     | `get_fitted_diffraction_data` in `detector_calibration_model.py` | `fitted_ws[sub_run - 1]`, while `fit_peaks` fills `fitted_ws` by position | `test_detector_calibration_model.py` |
     | `update_diff_view` in `detector_calibration_crtl.py` | `exclude_list[sub_run - 1]`, the same pattern | `test_detector_calibration_crtl.py` |
     | `plot_scan` in `peak_fitting_crtl.py` | One typed sub-run is used as a 1-based position in the sub-run list | `tests/ui/test_peak_fitting_crtl.py::TestSubRunNumbersAreNotPositions` |
     | `parse_sub_run_list` in `peak_fitting_crtl.py` | Typed sub-runs are used as 0-based positions, and the range `[1, N)` drops the first position and sub-run N | same class |
     | `trim_data` in `texture_fitting_crtl.py` | Included sub-runs are used as 0-based positions, and a single included sub-run is ignored (`len(...) > 1`) | `test_texture_fitting_crtl.py` |
     | `fit_peaks` in `texture_fitting_crtl.py` | The first sub-run number is used as a spectrum index | `test_texture_fitting_crtl.py::test_fit_peaks_sub_runs_not_starting_at_one` |

     `parse_sub_run_list`'s `(x >= 1) == (x < N)` reads like a typo for `&`, but it is equivalent for
     `N >= 1`, since both conditions cannot be false together. The defect is the number-as-position use.
  5. **Display order only.** `rstables.py` and `manual_reduction_viewer.py` sort for display. The browse
     order of sliders and tables is also display-only.

  Also noted, outside the ordering question: `nexus_conversion` would silently overwrite a repeated
  sub-run's counts, which is a uniqueness issue.

  **The xfail tests.** Each is `strict=True`, so a fix turns it into a failing XPASS until its marker is
  removed. Each names DD-2 in its `reason` and pins `raises=` to the exception the defect produces
  (`AssertionError`, or `pytest.fail.Exception` for "did not raise"), so a broken arrangement errors rather
  than passing as an xfail. None needs Mantid or a widget: the collaborators are plain stand-ins, each
  checked to pass with 1..N sub-runs.
  - `tests/unit/pyrs/dataobjects/test_sample_logs.py::TestSubRunOrdering`: unique unsorted values are
    accepted (the recommendation itself); `get_indices` with a missing interior value; `get_indices` on an
    unsorted axis; plus a passing test pinning that repeated values are still refused.
  - `tests/unit/pyrs/core/test_workspaces.py::TestSubRunNumbersAreNotPositions`
  - `tests/unit/pyrs/core/test_reduction_manager.py`
  - `tests/unit/pyrs/calibration/test_mantid_peakfit_calibration.py`
  - `tests/unit/pyrs/interface/texture_fitting/test_texture_fitting_crtl.py`
  - `tests/unit/pyrs/interface/detector_calibration/test_detector_calibration_model.py` and
    `test_detector_calibration_crtl.py`
  - `tests/ui/test_peak_fitting_crtl.py::TestSubRunNumbersAreNotPositions`

  Phase 2's NXstress xfail for a shared value in decreasing order is gone: that write is now refused by
  design (`test_multi_workspace.py::TestOneWorkspacePerKey`), not deferred.

  **Why it is deferred.** `SubRuns` is used across the whole data model, so relaxing it touches every loader
  and every view, and it deserves its own review. NXstress no longer depends on it (see "Now: one workspace per key" above).

  **If picked up.**
  1. In `SubRuns.set`, check uniqueness only (`np.unique(value).size == value.size`).
  2. Replace the `searchsorted` in `get_indices` with an order-free lookup, such as a value-to-position
     dictionary, that raises `IndexError` for any missing value, not only the first and last.
  3. Compare as sets in `set_wavelength`. Make `set_sample_log` act on `matching_subruns`: either refuse a
     mismatch, or reorder the values to the workspace's axis.
  4. Remove `_Peaks.peakCollectionRanges`'s within-block ordering check and `_set_subruns`'s `sort`, then
     remove the xfail markers above. NXstress's one-workspace-per-key refusal stays: it is a design rule,
     not a workaround for this defect.
  5. Separately, fix the item-4 sites to look sub-runs up through `get_indices` rather than by
     `sub_run - 1`. These are independent of steps 1-4, and each has its own xfail test.

  See also [ground_truths.md](ground_truths.md), "Two data-model limits that bite any multi-record NeXus
  write", which first recorded that `SubRuns` cannot hold a non-monotonic array.

- **DD-3**, 2026-10-21: strain/stress viewer: loading a project file with no peak collections as e22 or e33
  raises `IndexError`. **Fixed** in NXstress hookup phase 3.

  **Found during:** the phase-2 PR, checking its strain/stress GUI steps.
  **Owner:** `pyrs/interface/strainstressviewer/model.py`.

  **Fixed** in phase 3 (subspec 05's PR review, Follow-up 6 F6.2). A file's empty peak dictionary is now
  kept, so `check_peak_for_direction` answers False and the setter never builds the strain; the viewer
  reports "Peak … is not in e22" instead. Pinned by
  `tests/integration/test_strain_stress_model_json.py::test_a_file_with_no_peaks_loads_as_e22`.

  **What happened.** `Model.check_peak_for_direction` answered "does every file carry the selected peak?"
  by looping over the files' peak dictionaries, so with none it returned True. The `e22`/`e33` setters then
  called `create_strain`, which built an empty strain list and indexed its first element. The
  `selectedPeak` setter took the same path for every loaded direction. e11 was not affected, because its
  setter does not build a strain. Measured on `tests/data/HB2B_1320.h5` with its `peaks/peak0` group
  deleted, through `Controller.filesSelected("e22", [path])`:

  ```
  e22 -> IndexError: list index out of range
    File "/home/ux0/workspaces/PyRS/pyrs/interface/strainstressviewer/model.py", line 134, in create_strain
      setattr(self, f"_e{direction}_strain", sum(strain_list[1:], strain_list[0]))
                                                                  ~~~~~~~~~~~^^^
  IndexError: list index out of range
  ```

  In the viewer the exception left the **Browse** button's slot uncaught. It was deferred from phase 2
  because it predated phase 2, which did not change what it depends on.

- **DD-4**, 2026-10-21: strain/stress viewer: `Controller.fileSelected` is never called, and calls
  `Model.set_workspace`, which does not exist.

  **Found during:** the phase-2 PR, fixing `Controller.filesSelected`'s success check.
  **Owner:** `pyrs/interface/strainstressviewer/controller.py`.

  **What happens.** `Controller.fileSelected(name, filename)` calls `self._model.set_workspace(...)`, which
  `Model` does not define (`hasattr(Model(), "set_workspace")` is False), so any call raises
  `AttributeError`. A `grep` over `pyrs/` and `tests/` finds only its definition; the plural
  `filesSelected` is the one the viewer uses. The method dates from af206107 (2020-06-25).

  **Why it is deferred.** Dead code has no symptom, and removing it was unrelated to phase 2.

  **If picked up.** Delete `fileSelected`.

- **DD-5**, 2026-11-05: detector calibration viewer: a recipe saved before any data were loaded cannot be
  loaded back. `PeakLinesSetupView.save_json` writes `nexus_file` only when a file is loaded, but `load_json`
  reads it unconditionally and raises `KeyError`.

  **Found during:** the rebase of NXstress hookup phase 2 onto `next`.
  **Owner:** `pyrs/interface/detector_calibration/detector_calibration_viewer.py`, `PeakLinesSetupView`.

  **What happens.** `load_json` compares the loaded file with the recipe's before anything else:

  ```
  >       if self._parent._nexus_file != input_dict["nexus_file"]:
  E       KeyError: 'nexus_file'
  pyrs/interface/detector_calibration/detector_calibration_viewer.py:732: KeyError
  ```

  It was hidden by the GUI tests' shared session-scoped window: an earlier test had always loaded data.
  Phase 2 builds each window per test, so `tests/ui/test_calibration_ui.py::test_recipe_keeps_method_and_strips`
  (the #1045 recipe test) failed, and now loads its data explicitly first.

  **Why it is deferred.** It predates phase 2, and the fix is a change to the calibration viewer.

  **If picked up.** In `load_json`, treat a missing `nexus_file` as "no data to load" (`input_dict.get`),
  and add a regression test that saves and loads a recipe with no data loaded.

- **DD-6**, 2026-10-29: a workspace whose file carries no instrument geometry cannot be saved as NXstress
  (*Save as NXstress…* is disabled). Reading the geometry from Mantid's instrument definition is deferred.

  **Found during:** subspec 05's implementation pass. Planned as F-1
  (`plans/NXstress-prod/F-1-mantid-instrument-geometry.md`), not scheduled.

  **What happens.** `NXstress.write` requires instrument geometry: the `instrument` group's detector size,
  pixel size and arm length come from `HidraWorkspace.get_instrument_setup()`, with no fallback. A
  workspace loaded from a project file with no usable geometry has `get_instrument_setup() is None`, and
  every viewer's *Save as NXstress…* is disabled for it (`gui_helper.nxstress_save_available`). Every
  strain/stress project file in `tests/data` is in that state: its `instrument/geometry setup/detector`
  group exists and is empty, `HidraProjectFile.read_instrument_geometry` raises
  `KeyError: object 'detector size' doesn't exist`, and `HidraWorkspace.load_hidra_project` logs a warning
  and continues without geometry. `HB2B_1327_with_instrument.h5`, the same run re-saved with geometry, holds
  `L2 = 0.985`, `detector size = [1024, 1024]` and `pixel dimension = [0.0003, 0.0003]`. So the strain/stress
  workflow cannot save NXstress from most real files.

  **If picked up.** When a project file has no usable geometry, build a `DENEXDetectorGeometry` from the
  installed Mantid's `HB2B_Definition.xml` (cached per instrument), name the default instrument in `Config`,
  and log a warning naming the file and the definition. The result is the nominal geometry, not a
  calibrated one, and the NXstress file must not present it as calibrated.

- **DD-7**, 2026-10-29: the model tests in `tests/ui/test_stress_strain_viewer.py` carry only the
  `integration` marker, but the module cannot be imported without a display.

  **Found during:** subspec 05's implementation pass.

  **What happens.** The module imports the viewer, which pulls in mantidqt and matplotlib's `QtAgg` backend
  at module scope, so by `CLAUDE.md`'s tier rules it is a `gui` module. Its model tests are marked
  `integration` only, so `test-integration` selects them though they need a display. Since phase 3,
  `tests/ui/conftest.py` creates an offscreen `QApplication` when there is no display, so collection no
  longer fails, but a bare `python -c 'import tests.ui.test_stress_strain_viewer'` still does. Phase 3's new
  model tests are in `tests/integration/test_strain_stress_model_json.py` instead.

  **Why it is deferred.** It predates phase 3.

  **If picked up.** Move the model tests to `tests/integration/`, against the model alone, and mark what
  remains `gui` and `integration`.

- **DD-8**, 2026-10-29: the developer documentation does not build (`pixi run build-dev-docs`): autodoc fails
  with `AttributeError: __name__` on a `unittest.mock` stand-in that `docs/developer/source/conf.py` installs.

  **Found during:** subspec 05's vendored NeXus definitions work.

  **What happens.** autodoc reaches one of the stand-ins in `conf.py`'s `MOCK_MODULES` and fails with
  `AttributeError: __name__`. The same failure occurs at the commit before phase 3's changes.

  **Why it is deferred.** It predates phase 3.

  **If picked up.** Find which mocked module autodoc reaches, and replace the hand-rolled stand-ins with
  Sphinx's `autodoc_mock_imports`, whose mocks carry `__name__`.

- **DD-9**, 2026-11-05: `StressField.__init__` and `StressField._stack_strains` annotate `strain33` as
  `StrainField`, but `None` is how an in-plane case is built, so a fully type-checked caller fails mypy.

  **Found during:** the rebase of NXstress hookup phase 3 onto `next`, when `pixi run mypy pyrs scripts
  tests` checked the new demo scripts.
  **Owner:** `pyrs/dataobjects/fields.py`, `StressField`.

  **What happens.** `StressField(strain11, strain22, None, ..., stress_type="in-plane-strain")` is how the
  in-plane cases are built, and the code handles `None`, but mypy reports `arg-type` against the annotation.
  `tests/scripts/cis_tests/NXstress/viewer_defects.py` carries a `type: ignore[arg-type]` for it.

  **Why it is deferred.** `StressField` is scientific production code, and the annotation predates phase 3.

  **If picked up.** Annotate `strain33` as `Optional[StrainField]` in both places, then remove the
  `type: ignore` from `viewer_defects.py`.

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

- **DD-33**, 2026-11-17: PyRS reads Mantid workspaces through `MatrixWorkspace.readX`/`readY`, which Mantid 7.0
  deprecates in favour of `x()`/`y()` (`pyrs/peaks/peak_fit_engine.py:133,140`, the peak-fitting, texture-fitting
  and detector-calibration controllers, and `tests/unit/pyrs/peaks/test_peak_fit_engine.py:706`). They still work
  and emit `DeprecationWarning`; a future Mantid release will remove them.

  **Found during:** the Mantid 7 upgrade (see [ground_truths.md](ground_truths.md)).
  **Why it is deferred.** `pyrs/peaks/` and the UI computations are scientific production code; the swap needs
  its own approval under the constitution.

- **DD-32**, 2026-11-18: strain/stress viewer: a field cannot be plotted when the sample points lie on one
  straight line that is not parallel to a sample axis; the viewer opens a modal "Failed to generate field"
  error instead.

  **Found during:** the phase-3 rebase onto `next`, when phase 2's `make_minimal_HidraWorkspace` began to
  vary `vy` and `vz` along with `vx`, and the GUI tier hung on the dialog.
  **Owner:** `pyrs/dataobjects/fields.py`, `ScalarFieldSample.interpolated_sample`.

  **What happens.** `interpolated_sample` reduces the grid only along axes on which the points do not vary
  (`coordinates_irreducible`). Points on a diagonal vary along all three, so `griddata(..., method="linear")`
  is asked for a 3-D Delaunay triangulation of collinear points, which Qhull refuses. Measured with three
  points `(0, 0, 0)`, `(1, 1, 1)`, `(2, 2, 2)`:

  ```
  diagonal -> QhullError: QH6214 qhull input error: not enough points(3) to construct initial simplex (need 5)
  line -> ok, 3 grid points
  ```

  (`line` is the same three values along `vx` only.) `StrainStressViewer` catches the exception and shows
  it with `show_failure_msg`, which is modal, so a headless test blocks on it until its timeout.

  **Why it is deferred.** It is in field interpolation, which is scientific production code and outside the
  rebase. A real scan along a diagonal is unusual; the test fixture was made a line scan along `vx`
  (`tests/ui/test_nxstress_roundtrip.py`).

  **If picked up.** Detect a degenerate (collinear or coplanar) point set before `griddata`, for example from
  the rank of the centred coordinates, and interpolate along the line or plane it spans, or fall back to
  `method="nearest"` with a warning. Either changes plotted values for such scans, so it needs approval.
