# PR 1.1 — responses to review comments

Base for every diff below: `6368e3fe`.

Verification after all changes: `pixi run test-unit` 298 passed / 120 deselected;
`pixi run test-integration` 81 passed, 28 skipped, 2 xfailed; `pixi run test-gui`
9 passed; full `pixi run test` **388 passed, 28 skipped, 2 xfailed** (5m44s, 82%
coverage). `ruff check`, `ruff format` and `taplo-format` clean. `test-gui` now
forces offscreen itself; the full `pixi run test` still wants
`QT_QPA_PLATFORM=offscreen` exported on a machine with a real desktop session —
see the last section.

---

## 1. `pyrs/core/workspaces.py` — why `print` and not `log`?

**Fixed.** No good reason — `HidraWorkspace` simply never had a logger, so the
module had grown five `print()` calls with hand-written `[INFO]` / `[WARNING]`
prefixes. This PR touched two of them (rewording the "no reduced data" message)
and should have converted them at the same time.

`HidraWorkspace.__init__` now holds `self._log = Logger(__name__)` and all five
sites use it. That is the existing codebase convention — `mantid.kernel.Logger`
stored as `self._log`:

| existing example | |
|---|---|
| `pyrs/projectfile/file_object.py:5,83` | `HidraProjectFile` |
| `pyrs/core/nexus_conversion.py:7,171,354` | `NeXusConvertingApp`, `Hidra…` |
| `pyrs/peaks/peak_fit_engine.py:33` | `PeakFitEngine` |
| `pyrs/peaks/mantid_fit_peak.py:34` | `MantidPeakFitEngine` |

The hand-written level prefixes are dropped, since the logger supplies the level:

- `[INFO] No reduced-diffraction data recorded…` → `self._log.information(…)` (×2)
- `[INFO] Loaded diffraction data from …` → `self._log.information(…)` (×2)
- `[WARNING] sub run {} is not exported to {}` → `self._log.warning(…)` (×1)

---

## 2. `pyrs/dataobjects/fields.py` — what is `from None` for?

`from None` sets `__suppress_context__` on the raised exception, which stops
Python printing the *implicitly chained* original exception above it.

In `Direction.get`, the caught exception is whatever `Enum.__call__` raises for
an unknown value:

```
ValueError: 'FOO' is not a valid Direction
```

That says nothing the replacement message does not already say, and better. But
because the `raise` happens *inside* an `except` block, Python would otherwise
print it first, followed by `During handling of the above exception, another
exception occurred:`, and only then the message we actually want the user to
read. `from None` suppresses that, leaving a single, actionable traceback.

Convention being applied: **`raise … from <exc>` when the caught exception adds
information; `from None` when it does not.** The chaining form is used in ten
places in `pyrs/` where the original genuinely carries context —
`pyrs/utilities/convertdatatypes.py:59,61,84,86`,
`pyrs/peaks/peak_collection.py:26`, `pyrs/core/nexus_conversion.py:178`. This is
the only `from None`.

The inline comment has been expanded to state all of this, since it was clearly
too terse.

Separately, on the `except ValueError` half of that same line: the original code
caught `KeyError`, which `Enum.__call__` never raises — so the "clearer error
message" was dead code and a bad direction string propagated the raw enum error.
**Note for a follow-up (deliberately not fixed here, to keep this PR scoped):**
the same latent bug is still present in `StressType.get`
(`pyrs/dataobjects/fields.py:1858`) and in `pyrs/core/peak_profile_utility.py:32,64`.

---

## 3. `tests/plot_sample_points.py` — why deleted?

**Restored**, as `tests/scripts/cis_tests/plot_sample_points.py`.

It should have been moved, not deleted. It is not a test — it is a by-hand
plotting script, and `tests/scripts/cis_tests/` is exactly the directory this
repo keeps such things in ("scripts used for by hand smoke-tests to verify the
implementation of PyRS features… automatically ignored by the pytest collection
system" — `tests/scripts/cis_tests/README.rst`).

Two defects were repaired on the way, both of which made it unrunnable by
anyone but its original author:

1. `test_data_dir = "/home/jbq/repositories/pyrs/pyrs1/tests/data"` — an absolute
   path into another developer's home directory. Now resolved from the repo:
   `Path(__file__).resolve().parents[2] / "data"`.
2. The bottom-of-file call `plot_sample_points("HB2B_1327.h5", …)` ran at
   **import** time, so merely importing the module opened a blocking
   `plt.show()` window. Now guarded by `if __name__ == "__main__":`.

A module docstring was added in the style of the neighbouring
`NXstress_demo_script.py`.

Related: `norecursedirs` was widened from `["tests/scripts/cis_tests"]` to
`["tests/scripts"]`, so the "nothing under `tests/scripts/` is ever collected"
rule is true for the directory as a whole, not just that one subdirectory.

---

## 4. Summary lists for the changed set

> This section is the one intended to be pasted into the PR description.

### 4.1 Files moved or renamed

| From | To | Why |
|---|---|---|
| `tests/unit/pyrs/core/test_d0_grid.py` | `tests/integration/test_d0_grid.py` | reads real project files from `tests/data` |
| `tests/unit/pyrs/projectfile/test_file_object.py` | `tests/integration/test_file_object.py` | reads and writes real HDF5 project files |
| `tests/unit/pyrs/calibration/test_peakfit_calibration.py` | `tests/integration/test_peakfit_calibration.py` | reads real calibration + diffraction data |
| `tests/unit/pyrs/core/test_pyrscore.py` | `tests/integration/test_pyrscore.py` | reads `tests/data/Hidra_16-1_cor_log.h5` |
| `tests/ui/test_plot_data_preparer.py` | `tests/unit/pyrs/interface/test_plot_data_preparer.py` | pure functions, constructs no Qt widget — was misfiled in the GUI tier |
| `tests/plot_sample_points.py` | `tests/scripts/cis_tests/plot_sample_points.py` | not a test; by-hand script (see comment 3) |

The first four are all the same move: a test that reads real data does not
belong in the unit tier now that the tiers are separable.

Empty `__init__.py` housekeeping — note that `git diff` renders these as
renames/copies of one another purely because all three files are zero bytes;
they are unrelated:

- **deleted** `tests/integration/pyrs/__init__.py` — an empty, unused package directory
- **added** `tests/unit/pyrs/interface/__init__.py` — new unit-test package
- **added** `pyrs/resources/__init__.py` — required by `neutrons_standard` to find `application.yml`

### 4.2 File splits (new files, not moves)

| Source | New file | Split |
|---|---|---|
| `tests/unit/pyrs/dataobjects/test_fields.py` | `tests/integration/test_fields_from_files.py` | 70 → 56 tests; the real-file half became 17 integration tests |
| `tests/integration/test_write_stress_csv.py` | `tests/unit/pyrs/core/test_summary_generator_stress.py` | 11 → 9 tests; 2 tests that read no project file became unit tests |
| `tests/ui/test_manual_reduction_runspec.py` (deleted, 8 tests) **and** `tests/integration/test_batch_reduction.py` (11 → 7 tests) | `tests/unit/pyrs/interface/test_manual_reduction_runspec.py` | the two partly-overlapping sets of tests for `parse_run_numbers` / `is_run_specification` merged into 9 unit tests |

The third row is worth a note: the *same two functions* had one set of tests
behind an HFIR-archive skip and another set in the GUI tier, though neither
needs a file or a widget. Both sets were effectively never running.

### 4.3 Added TODO comments

**Exactly one TODO is new in this PR:**

| File | Comment |
|---|---|
| `tests/unit/pyrs/dataobjects/test_fields.py` | `# TODO: please verify that this is the expected behavior! …` |

Context: `test_peak_collection` asserted that a composite (multi-scan)
`StrainField` raises `RuntimeError` from a singular `.peak_collection` property.
That property was removed in commit `d340cccb` (2020) in favour of the plural
`.peak_collections`, which returns all of them with no restriction. No current
operation on `StrainField` raises `RuntimeError` merely because a strain is
composite, so the test had nothing left to assert and was dropped — the TODO
asks for confirmation that losing that restriction was intentional.

**Three TODO lines that `git diff` shows as `+` are *not* new.** They are
pre-existing comments carried verbatim from `tests/unit/pyrs/dataobjects/test_fields.py`
into `tests/integration/test_fields_from_files.py` by the split above:

- `# TODO: substitute/fix HB2B_1628.h5 with other data, because reported vx, vy, and vz are all 'nan'`
- `# TODO HB2B_1320_peak0 and HB2B_1320_ are the same scan. We need two different scans` (×2)

### 4.4 Tests moved or split, by file and name

**(a) Relocated unchanged** — 30 tests, same name, new file:

| Old file | New file | Tests |
|---|---|---|
| `tests/unit/pyrs/core/test_d0_grid.py` | `tests/integration/test_d0_grid.py` | `test_correction_full_match`, `test_correction_no_match`, `test_correction_partial_match`, `test_validation_full_match`, `test_validation_no_match`, `test_validation_partial_match` |
| `tests/unit/pyrs/calibration/test_peakfit_calibration.py` | `tests/integration/test_peakfit_calibration.py` | `test_all_refinements`, `test_load_print_calibration`, `test_wavelength` |
| `tests/unit/pyrs/core/test_pyrscore.py` | `tests/integration/test_pyrscore.py` | `test_main` |
| `tests/ui/test_plot_data_preparer.py` | `tests/unit/pyrs/interface/test_plot_data_preparer.py` | `test_prepare_3d_plot_data_accepts_python_lists`, `…_contour_auto_promotes_to_scatter_when_ungriddable`, `…_contour_grids_without_colors`, `…_empty_z_returns_no_colors`, `…_lines_grids_with_colors`, `…_scatter_returns_input_copies`, `…_unknown_mode_defaults_to_scatter` |
| `tests/unit/pyrs/projectfile/test_file_object.py` | `tests/integration/test_file_object.py` | `test_read_sample_logs`, `test_append_experiment_log`, `test_read_log_units`, `test_mask`, `test_detector_efficiency`, `test_reduced_diffraction_masks_excludes_variance_datasets`, `test_wave_length_rw`, `test_peak_fitting_result_io`, `test_strain_io`, `test_write_reduction_provenance_records_calibration_version_and_vanadium`, `test_write_reduction_provenance_no_calibration_or_vanadium_only_records_version`, `test_write_reduction_provenance_missing_calibration_file_skips_hash`, `test_write_reduction_provenance_second_call_does_not_clear_omitted_field` |

**(b) `tests/unit/pyrs/dataobjects/test_fields.py` → `tests/integration/test_fields_from_files.py`.**
14 tests removed from the unit file; 13 of them reappear as 17 tests (four were
split in two), one was dropped:

| Old name | New name(s) |
|---|---|
| `test_get_peak_params` | `test_get_peak_param_supported_name` **+** `test_get_peak_param_invalid_name_raises` |
| `test_eq` (with its `StrainFieldMock` helper class) | `test_eq_matching_and_differing_strains_returns_expected_bool` |
| `test_peak_collections` | `test_peak_collections_property_returns_single_element_list` |
| `test_coordinates` | `test_coordinates_property_matches_sample_log_positions` |
| `test_fuse_with` | `test_fuse_with_two_strains_returns_merged_strain` **+** `test_fuse_with_invalid_criterion_raises_value_error` |
| `test_add` | `test_add_operator_two_strains_returns_merged_strain` |
| `test_create_strain_field_from_file_no_peaks` | `test_strain_field_init_file_without_peaks_raises_io_error` |
| `test_from_file` | `test_strain_field_init_from_file_returns_populated_field` |
| `test_fuse_strains` | `test_fuse_strains_classmethod_matches_sequential_addition` |
| `test_stack_strains` | `test_stack_overlapping_and_disjoint_strains` **+** `test_stack_strains_unimplemented_mode_raises` |
| `test_fuse_and_stack_strains` | `test_fuse_then_stack_strains` |
| `test_to_md_histo_workspace` | `test_to_md_histo_workspace_returns_expected_bin_geometry` |
| `test_stress_field_from_files` | `test_stress_field_identical_strains` **+** `test_stress_field_select_invalid_direction` |
| `test_peak_collection` | *dropped* — see §4.3 |

Note `test_coordinates`, `test_fuse_with`, `test_add` and
`test_to_md_histo_workspace` still exist in `test_fields.py` under a different
class; only the file-backed variants moved.

**(c) `tests/integration/test_write_stress_csv.py` → `tests/unit/pyrs/core/test_summary_generator_stress.py`:**

| Old name | New name |
|---|---|
| `test_write_csv_empty_strain_filenames` | `test_init_strain_without_filenames_raises` |
| `test_write_csv_none_stress` | `test_init_none_stress_raises` |

Both build strains from in-memory `PeakCollectionLite` objects and assert only
that `SummaryGeneratorStress.__init__` rejects bad input — no project file, so
not integration tests. The local `strain_instantiator` helper moved with them.

**(d) Run-spec merge → `tests/unit/pyrs/interface/test_manual_reduction_runspec.py`:**

| Old file / name | New name |
|---|---|
| `tests/ui/…/test_parse_run_numbers_single` | `test_parse_run_numbers_single_run_returns_one_element_list` |
| `tests/ui/…/test_parse_run_numbers_dash_range_is_inclusive` | `test_parse_run_numbers_dash_range_returns_inclusive_list` |
| `tests/ui/…/test_parse_run_numbers_comma_list` | `test_parse_run_numbers_comma_separated_returns_ordered_list` |
| `tests/ui/…/test_parse_run_numbers_mixed_range_and_list` | `test_parse_run_numbers_mixed_range_and_list` (unchanged) |
| `tests/ui/…/test_parse_run_numbers_trailing_comma_ignored` | `test_parse_run_numbers_stray_comma_skipped` |
| `tests/ui/…/test_parse_run_numbers_invalid_raises` | `test_parse_run_numbers_non_integer_token_raises_value_error` |
| `tests/ui/…/test_is_run_specification_accepts_run_specs` | `test_is_run_specification_accepts_run_specs` (unchanged) |
| `tests/ui/…/test_is_run_specification_rejects_paths_and_empty` | `test_is_run_specification_path_or_blank_returns_false` |
| `tests/integration/…/test_parse_run_numbers_range` | *merged into* `test_parse_run_numbers_dash_range_returns_inclusive_list` |
| `tests/integration/…/test_parse_run_numbers_comma_and_range` | *merged into* `test_parse_run_numbers_mixed_range_and_list` |
| `tests/integration/…/test_is_run_specification_run_numbers` | *merged into* `test_is_run_specification_accepts_run_specs` |
| `tests/integration/…/test_is_run_specification_rejects_path` | *merged into* `test_is_run_specification_path_or_blank_returns_false` |
| — | `test_parse_run_numbers_blank_returns_empty` (new) |

**(e) Renamed in place:** `tests/integration/test_reduction.py` —
`test_reduce_method_data` → `test_reduce_method_data_valid_nexus_writes_diffraction_files`.

**(f)** Plus the 19 shortening renames listed under comment 6 below.

### 4.5 Added `xfail` markers

Two, both `strict=True` with an explicit `raises=`, so they fail the build if
the underlying problem is fixed without removing the marker. **Neither is a new
failure caused by this PR's logic** — both are pre-existing broken *data files*
that this PR stopped silently hiding.

Background: `HidraProjectFile.read_diffraction_2theta_array` used to fall back to
a legacy capitalized `"2Theta"` key (marked `# FIXME - This is a patch for
'legacy' data. It will be removed after codes are stable`), and its caller in
`workspaces.py` wrapped the whole read in a blanket `except KeyError: return`.
Between them, *any* failure to read the 2θ coordinate — corruption included —
was swallowed and the file silently loaded with no reduced-diffraction data.
This PR removes the fallback and narrows the catch to the one legitimate case
(the `REDUCED_DATA` group exists but is empty), raising `RuntimeError` otherwise.
Two data files turned out to be relying on the old silence.

**1. `tests/integration/test_pyrscore.py::test_main`** — `raises=RuntimeError`

`tests/data/Hidra_16-1_cor_log.h5` is written with the legacy capitalized
`"2Theta"` coordinate key. It now raises a clear "format is not up-to-date and
must be re-reduced" `RuntimeError` instead of loading with silently missing
data. Migrating the file is deferred to a separate follow-up.

**2. `tests/unit/pyrs/peaks/test_peak_fit_engine.py::test_pseudovoigt_HB2B_1060`** — `raises=RuntimeError`
(also newly marked `@pytest.mark.integration`)

`tests/data/HB2B_1060_first3_subruns.h5`'s `2theta` dataset raises at the HDF5
level: `h5py.KeyError: Unable to synchronously open object (invalid dataset
size, likely file corruption)`. This reproduces with bare `h5py` and no PyRS
code at all, so it is a defect in the file (or an hdf5/h5py version
incompatibility), not a schema issue. Root cause is a separate follow-up.

---

## 5. `tests/unit/pyrs/test_trigger.py` — why deleted?

**Restored, unchanged.** It was removed as apparent dead weight — the file
contains nothing but the comment

```
# This file is present only to allow PyCharm to run pytest on this directory and all subdirectories
```

— and `tests/unit/pyrs/__init__.py` already exists, which is what modern PyCharm
uses to resolve the package. But that reasoning doesn't justify breaking
somebody's IDE workflow for zero benefit, so it is back.

---

## 6. Test names are too long

**Fixed — 19 renames.** Repo baseline for comparison: 402 test names, median 27
characters, 90th percentile 57. The names added by this PR occupied most of the
repo's twenty-five longest. Every new name over 60 characters has been
shortened; the longest new name introduced by this PR is now 60.

Convention applied (now written down in `CLAUDE.md`, see below): keep
`test_<function>_<scenario>_<expected_outcome>` **under 60 characters**, omitting
any part the module or class name already carries and dropping `_returns_*` tails
when the scenario implies the outcome. `_raises` is kept for error cases, but the
exception type is not repeated — `pytest.raises(...)` in the body already states it.

| File | Old (len) | New |
|---|---|---|
| `tests/unit/pyrs/core/test_summary_generator_stress.py` | `test_summary_generator_stress_init_strain_without_filenames_raises_runtime_error` (80) | `test_init_strain_without_filenames_raises` |
| | `test_summary_generator_stress_init_none_stress_raises_runtime_error` (67) | `test_init_none_stress_raises` |
| `tests/integration/test_file_object.py` | `test_read_diffraction_2theta_array_missing_reduced_data_group_raises_key_error` (78) | `test_read_2theta_empty_group_raises_key_error` |
| | `test_read_diffraction_2theta_array_unrecognized_schema_raises_runtime_error` (75) | `test_read_2theta_bad_schema_raises_runtime_error` |
| `tests/unit/pyrs/interface/test_manual_reduction_runspec.py` | `test_parse_run_numbers_mixed_range_and_list_with_spaces_returns_combined_list` (77) | `test_parse_run_numbers_mixed_range_and_list` |
| | `test_parse_run_numbers_stray_comma_returns_list_with_empty_tokens_skipped` (73) | `test_parse_run_numbers_stray_comma_skipped` |
| | `test_is_run_specification_digit_dash_comma_strings_returns_true` (63) | `test_is_run_specification_accepts_run_specs` |
| | `test_parse_run_numbers_blank_or_whitespace_returns_empty_list` (61) | `test_parse_run_numbers_blank_returns_empty` |
| `tests/integration/test_fields_from_files.py` | `test_stack_operator_overlapping_and_disjoint_strains_returns_expected_fields` (76) | `test_stack_overlapping_and_disjoint_strains` |
| | `test_get_effective_peak_parameter_supported_name_returns_scalar_field` (69) | `test_get_peak_param_supported_name` |
| | `test_fuse_then_stack_strains_matches_expected_finite_and_nan_counts` (67) | `test_fuse_then_stack_strains` |
| | `test_stack_strains_unimplemented_mode_raises_not_implemented_error` (66) | `test_stack_strains_unimplemented_mode_raises` |
| | `test_stress_field_from_identical_strains_computes_expected_values` (65) | `test_stress_field_identical_strains` |
| | `test_get_effective_peak_parameter_invalid_name_raises_value_error` (65) | `test_get_peak_param_invalid_name_raises` |
| | `test_stress_field_select_invalid_direction_raises_value_error` (61) | `test_stress_field_select_invalid_direction` |
| `tests/util/test_peak_collection_helpers.py` | `test_createPeakCollection_inverted_error_fraction_bounds_raises_value_error` (75) | `test_inverted_error_fraction_bounds_raises` |
| | `test_createPeakCollection_zero_error_fraction_min_raises_value_error` (68) | `test_zero_error_fraction_min_raises` |
| `tests/unit/pyrs/utilities/test_config.py` | `test_validate_config_raises_when_legacy_io_enable_is_not_bool` (61) | `test_validate_legacy_io_enable_not_bool` |
| | `test_validate_config_raises_when_nxstress_enable_is_not_bool` (60) | `test_validate_nxstress_enable_not_bool` |

The last entry is at the 60-character threshold rather than over it; renamed
anyway so it stays symmetric with its `legacy_io` sibling.

Two of the run-spec renames restore the names those tests had *before* this PR
moved them (`test_parse_run_numbers_mixed_range_and_list`,
`test_is_run_specification_accepts_run_specs`) — the move had made them longer,
which is a fair part of what prompted this comment.

**Not renamed**, because this PR only relocated them and did not create them:
`test_write_reduction_provenance_*` (four, in `test_file_object.py`) and
`test_prepare_3d_plot_data_contour_auto_promotes_to_scatter_when_ungriddable`
(in `test_plot_data_preparer.py`).

### `CLAUDE.md` now documents this

Two additions under **🧪 Testing Guidelines**, so the convention is written down
rather than re-litigated per PR:

1. The 60-character naming limit above, with a worked before/after example.
2. A new **Test tiers, markers, and locations** subsection: the
   unit / integration / GUI / by-hand-script table (location, required marker,
   `pixi run` task), verbatim definitions of the `integration` and `gui` markers,
   and the non-obvious rules — `--strict-markers` is on, GUI tests need **both**
   `gui` and `integration` (because `test-integration` selects
   `-m 'integration and not gui'`), and `pytestmark` is preferred over
   per-function decoration.

---

## Additional changes made while addressing the above

- **Three dangling comment references removed.** `test_batch_reduction.py`,
  `test_write_stress_csv.py` and `test_manual_reduction_runspec.py` each pointed
  a reader at `plans/test-framework.md`, which is not part of this PR (the
  `plans/` tree is deliberately excluded). Each comment now states its reason
  inline instead.
- **`norecursedirs`** widened from `tests/scripts/cis_tests` to `tests/scripts`
  (see comment 3).
- **Every test now has a 300-second timeout.** `pytest-timeout` was added to the
  test feature and configured globally in `pyproject.toml` (`timeout = 300`,
  `timeout_method = "thread"`), so a hung test fails with a stack dump instead of
  blocking the run forever. 300s is roughly seven times the slowest test in the
  suite (41.8s, `test_detector_calibration`), leaving ample room on a slower CI
  runner; a genuinely long-running test can opt out with `@pytest.mark.timeout(N)`.
  The `thread` method is required rather than preferred — measured against a
  `QEventLoop().exec()` that never returns to the interpreter, the plugin's default
  `signal` method let the timeout pass unnoticed and an external kill was needed,
  while `thread` caught it and named the exact blocking line. SIGALRM is only
  delivered when the interpreter next runs bytecode, which a blocked C++ event
  loop never does.

- **`pixi run test-gui` now sets `QT_QPA_PLATFORM=offscreen` itself**, via
  `env = { … }` on the task, so it opens no windows on a developer's desktop.
  It is deliberately *not* set on the `test` task: a pixi task `env` overrides the
  ambient environment unconditionally — verified that a caller's
  `QT_QPA_PLATFORM=xcb` is ignored and that neither `${VAR:-default}` in `env` nor
  in `cmd` expands — so setting it there would silently defeat CI's
  `xvfb-run --server-args=… -a pixi run test` and move CI off the `xcb` platform
  that `run_tests.py`'s shutdown workaround exists for. `test-unit` and
  `test-integration` need nothing: the only `tests/ui/` tests the integration tier
  selects are `test_model`, `test_model_multiple_files` and `test_model_from_json`,
  which construct `Model()` and touch no widget.

  Both changes came out of a hang found while verifying this PR: on a workstation
  with a real desktop session and no `QT_QPA_PLATFORM` override, `pixi run test`
  blocks indefinitely at `tests/ui/test_calibration_ui.py`. That is pre-existing
  and unrelated to this PR's logic, but it is a concrete argument for the tier
  split — neither `test-unit` nor `test-integration` can reach it. Written up in
  `docs/ground_truths.md`; `CLAUDE.md` covers both rules in the tier section.
- **`pyproject.toml` reindented by `taplo-format`.** The `[tool.pytest.ini_options]
  markers` array this PR added used 4-space indentation; taplo (a pre-commit hook
  in this repo) normalizes it to 2. Worth knowing that it would otherwise have
  failed the hook on the next run.

## Two things noticed, not changed — flagging for a follow-up

1. `tests/integration/test_fields_from_files.py` carries three near-identical
   class names inherited from the original file: `TestStrainFieldSingle`,
   `Test_StrainField`, and `TestStrainField`. The middle one holds the mock-based
   equality tests. Confusing, but renaming them is churn better done separately.
2. `pyproject.toml` adds `F811` to the `**/conftest.py` ruff per-file-ignores.
   This is needed because fixtures are re-exported between `conftest.py` levels,
   which ruff reads as redefinition. Worth a look if a cleaner re-export pattern
   exists.
