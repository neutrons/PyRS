# Test tree cleanup (follow-up to marker classification)

## Context

The `integration`/`gui` pytest markers (see `pyproject.toml`'s `[tool.pytest.ini_options]`)
are now the authoritative classification for every test — `-m "not integration and not gui"`
selects the fast tier regardless of where a file lives. That raised the question of whether
the directory tree should be restructured to match.

Investigation found the tree is **not** uniformly type-first or uniformly module-mirrored:

| Branch | Form |
|---|---|
| `tests/unit/pyrs/<module>/` | type, then mirrors the `pyrs/` package (`core/`, `dataobjects/`, `peaks/`, `utilities/NXstress/`, ...) |
| `tests/integration/` | type only, flat — no module subdivision |
| `tests/ui/` | type only, flat — no module subdivision |

`tests/integration/pyrs/` exists containing nothing but an empty `__init__.py` — an abandoned
start at extending the `unit/` module-mirror pattern into `integration/`.

Decision (discussed with the user): **do not** extend the module-mirror into `integration/`
or `ui/` — there's no natural module split for cross-subsystem workflow tests or GUI windows
that each wrap a distinct `pyrs.interface` subpackage 1:1 anyway. The existing three-branch
form "largely makes sense" and should be kept. What needs fixing is **consistency**: the dead
stub, files whose entire content contradicts the branch they're filed under, and files that
mix unit and integration tests so thoroughly that one `pytestmark` line couldn't classify them
(which is why those got per-function decorators in the marker pass — see git history on
`tests/unit/pyrs/dataobjects/test_fields.py` etc.).

This document is a plan, not a completed change — none of the moves below have been made.
Implement in a separate pass so the diff stays reviewable independent of the marker work.

## 1. Delete the dead stub

`tests/integration/pyrs/__init__.py` (and the now-empty `tests/integration/pyrs/` directory)
— zero tests, zero non-init files. Delete both.

## 2. Whole-file moves — 100% classification mismatch

These files are **entirely** one marker class, and that class contradicts their current
directory. Each is a straight move (`git mv`), no content split needed. Absolute
`from tests.conftest import ...` / `from tests.util... import ...` imports are unaffected by
location — verified during the marker pass that no test uses `__file__`-relative or relative
imports.

| Current path | New path | Why |
|---|---|---|
| `tests/unit/pyrs/calibration/test_peakfit_calibration.py` | `tests/integration/test_peakfit_calibration.py` | 100% integration (loads real `HB2B_3510.nxs.h5`, runs Mantid `FitPeaks` calibration, writes to cwd) |
| `tests/unit/pyrs/core/test_d0_grid.py` | `tests/integration/test_d0_grid.py` | 100% integration (loads 3 real `.h5` + CSV grids via `test_data_dir`) |
| `tests/unit/pyrs/core/test_pyrscore.py` | `tests/integration/test_pyrscore.py` | 100% integration (loads `Hidra_16-1_cor_log.h5`); note most of its content is dead `broken_test_*` code — worth a follow-up issue, out of scope here |
| `tests/unit/pyrs/projectfile/test_file_object.py` | `tests/integration/test_file_object.py` | 100% integration (reads `HB2B_938_v2.h5`, 7 `tmpdir` round-trips) |
| `tests/unit/pyrs/utilities/NXstress/test_helper_util.py` | `tests/integration/test_NXstress_helper_util.py` | 100% integration (all 3 tests read `HB2B_1628.h5`) |
| `tests/unit/pyrs/utilities/NXstress/test_input_data.py` | `tests/integration/test_NXstress_input_data.py` | 100% integration (all tests use `load_HidraWorkspace` + `tmp_path`) |
| `tests/unit/pyrs/utilities/NXstress/test_peaks.py` | `tests/integration/test_NXstress_peaks.py` | 100% integration (all 6 tests use `load_HidraWorkspace`) |
| `tests/unit/pyrs/utilities/NXstress/test_sample.py` | `tests/integration/test_NXstress_sample.py` | 100% integration (all 9 tests use `load_HidraWorkspace`) |
| `tests/ui/test_plot_data_preparer.py` | `tests/unit/pyrs/interface/test_plot_data_preparer.py` | 0% gui — imports only `pyrs.interface.utilities.plot_data_preparer` (numpy/matplotlib, no Qt). Misfiled under `ui/`. |
| `tests/ui/test_manual_reduction_runspec.py` | `tests/unit/pyrs/interface/test_manual_reduction_runspec.py` | 0% gui — pure string parsing (`parse_run_numbers`, `is_run_specification`); 7 of 8 tests are unit, 1 is integration but none touch Qt, so it doesn't belong under `ui/` regardless |

Note the `test_NXstress_*` prefix for the four NXstress files moving into flat
`tests/integration/`: without it they'd lose the grouping the `NXstress/` subdirectory
currently provides, and `test_helper_util.py`/`test_input_data.py`/`test_peaks.py`/
`test_sample.py` are generic enough names to risk confusion once flattened.

`tests/unit/pyrs/interface/` is new — create it (with `__init__.py`, see §5) as the unit-side
home for `pyrs.interface` code that has no Qt dependency, mirroring how `unit/pyrs/<module>/`
already mirrors every other `pyrs/` subpackage.

## 3. Split candidates — files mixing unit and integration substantially

Unlike §2, these files are **not** uniformly one class — the marker pass applied per-function
decorators because both slices are real and substantial. Splitting turns "read the whole file
to know what you're running" into "the filename tells you." Only propose a split where the
minority slice is large enough to justify a new file (rough bar: 3+ tests); see §4 for cases
below that bar.

| File | Split | Rationale |
|---|---|---|
| `tests/unit/pyrs/dataobjects/test_fields.py` | Extract the 14 `@pytest.mark.integration` tests (see git blame from the marker commit) into `tests/integration/test_fields_from_files.py`. Leave the remaining ~56 in-memory tests in place. | At 1876 lines / 70 tests this is the single largest mixed file. The integration slice all funnels through the `strain_field_samples` fixture or `test_data_dir` directly — self-contained enough to extract cleanly. `tests/integration/` already has a `test_fields.py`, hence the different name. |
| `tests/integration/test_batch_reduction.py` | Move the 4 unit tests (`test_parse_run_numbers_range`, `test_parse_run_numbers_comma_and_range`, `test_is_run_specification_run_numbers`, `test_is_run_specification_rejects_path`) into `tests/unit/pyrs/interface/test_manual_reduction_runspec.py` (see §2 — same functions being tested, see §6 duplication note). Leave the 7 `reduce_runs_*` integration tests in `test_batch_reduction.py`. | These 4 tests are currently *skipped entirely* whenever `/HFIR` isn't mounted, via the module-level `pytestmark = pytest.mark.skipif(...)` — see §6. Moving them out fixes that as a side effect. |
| `tests/integration/test_write_stress_csv.py` | Extract `test_write_csv_empty_strain_filenames` and `test_write_csv_none_stress` (the only 2 unit tests, pure `SummaryGeneratorStress` error-path checks) into `tests/unit/pyrs/core/test_summary_generator_stress.py` (new file — no existing unit coverage for this module). Leave the other 9 in place. | Small slice (2 of 11), but they're testing pure validation logic unrelated to the CSV-comparison machinery the rest of the file depends on (`compare_csv`, gold files). |

## 4. Mixed files to leave as-is

These also carry both markers, but the minority slice is 1-3 tests embedded in a cohesive
test class — splitting would cost more (new file, new imports, broken class grouping) than it
buys. The per-function markers already make them fully selectable; no action needed:

- `tests/unit/pyrs/utilities/NXstress/test_NXstress.py` (1 unit / 20 integration)
- `tests/unit/pyrs/utilities/NXstress/test_fit.py` (3 unit / 15 integration)
- `tests/unit/pyrs/utilities/NXstress/test_workspace_read.py` (3 unit / 9 integration — already
  separated into `TestStandaloneMethods` vs `TestWorkspaceRoundtrip`/`TestReadErrors` classes,
  which is arguably enough)
- `tests/unit/pyrs/utilities/NXstress/test_peaks_read.py` (mixed across 5 classes, none lopsided)
- `tests/unit/pyrs/utilities/NXstress/test_instrument.py` (2 unit / 5 integration)
- `tests/unit/pyrs/core/test_live_conversion.py` / `test_nexus_conversion.py` (3 unit / 1
  integration each — see §6, these two files are near-duplicates of each other)
- `tests/unit/pyrs/core/test_workspaces.py` (1 unit / 1 integration)
- `tests/unit/pyrs/utilities/test_calibration_file_io.py` (2 unit / 1 integration)
- `tests/unit/pyrs/utilities/test_file_util.py` (1 unit / 4 integration, all 4 already gated by
  `skipif(not os.path.exists("/HFIR/HB2B/shared/"))`)
- `tests/integration/test_fields.py`, `test_manual_reduction_ui.py`, `test_peak_fitting.py`,
  `test_load_split.py`, `test_reduction.py` — small unit minorities (0-1 tests) inside files
  that are otherwise cohesive integration suites
- `tests/ui/test_stress_strain_viewer.py` (3 integration / 1 gui+integration — the 3
  `test_model*` tests exercise the non-widget `Model` class directly; leaving them alongside
  the one real GUI test keeps all "strain/stress viewer" coverage in one place)

## 5. Housekeeping surfaced during the marker pass

- **`tests/unit/pyrs/test_trigger.py`** — zero tests (a comment-only placeholder). Delete.
- **`tests/plot_sample_points.py`** — hard-codes `/home/jbq/repositories/...` at line 6; dead
  script, not collected as a test (no `test_` prefix). Delete or move to `tests/scripts/`.
- **`__init__.py` coverage is inconsistent**: present in `tests/unit/pyrs/core/`,
  `tests/unit/pyrs/dataobjects/`; absent in `tests/unit/pyrs/peaks/`,
  `tests/unit/pyrs/projectfile/`, `tests/unit/pyrs/utilities/`,
  `tests/unit/pyrs/utilities/NXstress/`, `tests/ui/`. Since none of these are currently
  imported as packages from outside `tests/`, this is cosmetic — but normalize one way or the
  other while touching these directories for the moves above. Add `__init__.py` to the new
  `tests/unit/pyrs/interface/`.

## 6. Related findings (not directory issues, flag separately)

- **Duplication**: `tests/unit/pyrs/core/test_live_conversion.py` and `test_nexus_conversion.py`
  are near-identical — same `TestSplitter` class (`test_empty_constructor`, `test_nominal`,
  `test_no_end`) duplicated verbatim; the only difference is whether `NeXusConvertingApp` is
  built from a live Mantid workspace or a filename. Consider parametrizing one file over both
  construction paths instead of maintaining two copies.
- **Duplication**: the 4 pure-logic tests in `tests/integration/test_batch_reduction.py`
  (`test_parse_run_numbers_*`, `test_is_run_specification_*`) test the same functions as
  `tests/ui/test_manual_reduction_runspec.py`. The §3 move consolidates these into one file
  instead of two.
- **Pre-existing bug**: `test_batch_reduction.py`'s module-level
  `pytestmark = pytest.mark.skipif(not hfir_available(), reason="HFIR archive not accessible")`
  currently skips *all 11* tests — including the 4 pure-logic ones its own docstring calls
  "pure-logic tests, no HFIR needed" — whenever `/HFIR` isn't mounted. The §3 move fixes this
  as a side effect (the 4 tests move to a file with no such skip); if the move is deferred,
  narrowing the skip to only the 7 `reduce_runs_*` tests is a one-line fix worth doing on its
  own.

## Verification

Before/after each move, the full-suite collection count must be unchanged (moves are not
supposed to add, drop, or duplicate tests):

```bash
pixi run python -m pytest ./tests --collect-only -q | tail -1   # expect 408 both times
```

And the three tiers must still partition the suite with no overlap and no gap (matches the
counts established by the marker pass: 219 unit / 180 integration / 9 gui):

```bash
pixi run python -m pytest ./tests -m "not integration and not gui" --collect-only -q | tail -1
pixi run python -m pytest ./tests -m "integration and not gui"     --collect-only -q | tail -1
pixi run python -m pytest ./tests -m "gui"                          --collect-only -q | tail -1
```

After each split (§3), diff the extracted file's test names against the original to confirm
nothing was dropped, and run both halves once each (`pixi run test-unit` /
`pixi run test-integration`) to confirm imports resolve from the new location.
