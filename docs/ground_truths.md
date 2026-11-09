# Ground Truths

Key findings recorded here for future reference. Link to this file from related
documentation and code comments when relevant.

## Mantid ecosystem moved to Python 3.12 across all channels (2026-07)

As of early July 2026, every Mantid conda channel used by this project —
`mantid/label/main`, `mantid/label/nightly`, `mantid-ornl/label/main`,
`mantid-ornl/label/rc`, and `mantid-ornl/label/nightly` — only publishes
Mantid 6.16.x builds linked against Python 3.12 (`py312`). Older 6.15.x
builds (Python 3.11) have been pruned from these channels' repodata; they
no longer resolve even though their version numbers still satisfy a
`>=6.15.0` constraint.

**Symptom:** `pixi update` (or `pixi install`) fails to solve any
environment that depends on `mantidworkbench`/`mantid`/`mantidqt`, with
errors like:
- `mantidworkbench >=6.15.0,<6.16.0 cannot be installed ... is excluded
  because candidate not in requested channel` — the upper-bound pin
  excludes the only builds left in the channel.
- A large `pyqt` vs `mantidqt`/`mantidworkbench` conflict tree ending in
  `python 3.12.13, which conflicts with the versions reported above` —
  caused by the base (unconstrained) `pyqt = "*"` dependency in
  `[tool.pixi.dependencies]` / `[tool.pixi.package.run-dependencies]`
  pulling in PyQt5, while `mantidqt` now requires PyQt6. This wasn't a
  true "no py312 build" problem (PyQt5 5.15.9 has a `py312` build); it's
  redundant PyQt5 that isn't needed once Mantid supplies PyQt6 via
  `mantidqt`.

**Fix applied** (see [pyproject.toml](../pyproject.toml)):
1. Bump `requires-python` to `>=3.12` (project-wide).
2. Drop the `<6.16.0` upper bound on `mantidworkbench` in the base
   dependencies and in the `qa`/`prod` feature dependencies — only
   `>=6.15.0` is needed now.
3. Remove the bare `pyqt = "*"` dependency from
   `[tool.pixi.dependencies]` and `[tool.pixi.package.run-dependencies]`.
   `qtpy` is kept (binding-agnostic); Mantid's own `mantidqt` dependency
   now supplies PyQt6.

After these changes, `pixi update` solves cleanly for all four
environments (`default`, `dev`, `prod`, `qa`), and
`pixi run -e <env> test-import-framework` (imports `pyrs`, `qtpy`,
`mantidqt`) passes in each.

**Why this matters going forward:** if `pixi update`/`pixi install`
starts failing again with a `pyqt`/`mantidqt`/`python_abi` conflict tree,
check whether Mantid has moved to a newer Python ABI first — the fix is
usually to relax an upper-bound version pin, not to add new pins.

## PyQt5→PyQt6 fallout after the Mantid/Python 3.12 update (2026-07)

Dropping the unconstrained `pyqt = "*"` dependency (above) moved the
environment from PyQt5 to PyQt6. Two latent bugs surfaced that had been
masked by PyQt5 being installed incidentally:

1. `pyrs/icons/icons_rc5.py` was a `pyrcc5`-compiled Qt resource module
   hard-coded to `from PyQt5 import QtCore`. With PyQt5 no longer
   installed, importing it raised `ModuleNotFoundError`, which aborted
   the entire pytest collection phase in CI (pytest treats a collection
   error as fatal by default, so *no* tests ran — not just the two
   files that imported it). PyQt6 dropped `pyrcc6` entirely, so
   recompiling isn't an option. **Fix:** stopped using the compiled Qt
   resource system for these 5 static PNGs; `peak_fitting_viewer.py` now
   builds file:// style paths directly from `pyrs.icons.__file__` at
   import time. `icons_rc5.py` and its source `icons.qrc` were deleted.

2. `QTableWidgetItem.setCheckState()` is strictly typed in PyQt6 (via
   sip) and rejects a plain `bool`/`int` — PyQt5 silently accepted both.
   Call sites in `fit_table.py` and
   `detector_calibration_viewer.py` passed raw `0`/`2`/`bool` values.
   Because the collection error in (1) aborted the whole test session,
   these paths were never actually exercised by CI and the bug went
   undetected until the import bug above was fixed and tests began
   running for real. **Fix:** pass `Qt.CheckState.Checked` /
   `Qt.CheckState.Unchecked` (from `qtpy.QtCore`) instead of raw values.

**Why this matters going forward:** a pytest collection error anywhere
in the suite silently skips *all* tests in that run, not just the
broken file — a green-looking "0 failed" or a short run time can hide
that nothing actually executed. When migrating a Qt binding, grep for
`PyQt5`/`pyrcc`/raw `setCheckState(` calls specifically; qtpy alone
doesn't catch either class of bug because both are binding-specific
(and PyQt6 doesn't provide a resource-compiler migration path — the
`.qrc`-compiled-module pattern needs to be replaced, not ported).

## `pixi-build-python` exact-version pin breaks whenever the pixi CLI advances (2026-07)

The `Build conda package` / `build-package` CI jobs failed with:

```
Error:   × could not initialize the build-backend
  ├─▶   × failed to solve the environment
  ╰─▶ Cannot solve the request because of: pixi-build-python ==0.5.2 cannot be installed because there are no viable options:
      └─ pixi-build-python 0.5.2 would require
         └─ pixi-build-api-version >=4,<5, for which no candidates were found.
      The following packages are incompatible
      └─ pixi-build-python ==0.5.2 cannot be installed because there are no viable options:
         └─ pixi-build-python 0.5.2 is excluded because due to strict channel priority not using this option from: 'https://prefix.dev/conda-forge/'
```

**First hypothesis (wrong, kept here as a warning against a plausible dead
end):** `[tool.pixi.package.build.backend].channels` listed
`https://prefix.dev/pixi-build-backends` — a rolling/nightly feed whose
`pixi-build-python` builds are versioned like
`0.5.2.20260612.1208.0827e23`, never a bare `0.5.2` — ahead of
`conda-forge`, which does publish a clean-semver `0.5.2`. It looked like
strict channel priority was locking onto the nightly channel's
non-matching versions and never falling through. Removing that channel
entry solved cleanly locally (even from a fully cleared rattler cache,
`pixi clean cache --yes`) — but pushing that fix alone did **not** fix
CI; the exact same error recurred on the next run.

**Actual root cause:** `prefix-dev/setup-pixi@v0.9.6` (used with no
`pixi-version` input) installs whatever the *latest* pixi CLI release is
at run time — it drifts forward on every CI run, unpinned. The pixi CLI
and `pixi-build-python` communicate over a versioned protocol
(`pixi-build-api-version`), and `pixi-build-python 0.5.2` only supports
protocol `>=4,<5` — `0.6.0` and later require `>=5,<6`
(`pixi search -c conda-forge "pixi-build-python==0.6.0"` shows the
dependency bump). My local pixi was 0.70.2; CI's was 0.72.2 by the time
this was diagnosed. Once the installed pixi CLI needs protocol 5, an
exact pin to a protocol-4-only backend build is unsolvable in *any*
channel, regardless of channel order — the "channel priority" wording in
the error is a red herring surfaced by the solver's generic reporting,
not the real constraint.

Confirmed by running `pixi self-update` locally to match CI's 0.72.2
exactly: the channel-only fix reproduced CI's exact failure locally for
the first time (my earlier pixi 0.70.2 had solved fine either way,
masking the real problem).

**Fix applied** (see [pyproject.toml](../pyproject.toml)): changed
`[tool.pixi.package.build.backend].version` from `"==0.5.2"` to `"*"`.
Verified by clearing the rattler cache and rebuilding with pixi 0.72.2 —
solves and builds successfully.

**Why this matters going forward:** an exact pin on a build-backend tool
that speaks a versioned protocol to its host CLI is only as stable as
the host CLI's own version — and `setup-pixi` here has no `pixi-version`
pin, so the host advances on every run. If this breaks again with a
`pixi-build-api-version` mismatch, check whether the currently-installed
pixi CLI (`pixi --version` in the failing job's log) needs a newer
protocol than the pinned backend supports before touching channel
config at all — `pixi search -c conda-forge "pixi-build-python==<pin>"`
shows a build's exact `pixi-build-api-version` dependency, and comparing
that across versions shows where the protocol bumped.

## PyQt6 `Qt.CheckState` enum doesn't compare equal to a plain `int` (2026-07)

A further fallout of the PyQt5→PyQt6 move (see above): PyQt6's
`Qt.CheckState` is a strict enum that does not compare equal to a plain
Python `int`, in either direction — `0 == Qt.CheckState.Unchecked` and
`Qt.CheckState.Unchecked == 0` are both `False` (`int(Qt.CheckState.Unchecked)`
also raises `TypeError`). PyQt5 allowed both directions transparently.

This broke `_mask_state`/`_calibration_state`/`_output_state` in
`manual_reduction_viewer.py`, which compare a `state` parameter that is
polymorphic in origin: `Qt.CheckState` when called directly with
`checkBox.checkState()` (in `__init__`), but a plain `int` when invoked
via the `stateChanged` signal (`QCheckBox.stateChanged` still emits
`int`, unlike the newer `checkStateChanged` signal). Comparing this
mixed-type value against the bare `Qt.Unchecked` enum member meant the
"is this checkbox checked" test was always wrong for whichever call path
didn't match by luck — in practice the default-mask/calibration/output
line edits stayed permanently disabled after the user toggled the
checkbox, silently keeping the stale default file path. This surfaced as
`tests/ui/test_manual_reduction.py` failures with a
`FileNotFoundError: /HFIR/HB2B/shared/CALIBRATION` (an ORNL-only network
path that doesn't exist in CI) instead of using the test's intended
`tests/data/...` file, because the qtbot-driven text entry silently
no-opped on the disabled, still-defaulted line edit.

The same bare-int-vs-enum bug also existed in
`fit_table.py`'s `_update_exclude_list` (`checkState() == 0`), separate
from the `setCheckState()` strict-typing bug fixed earlier in the same
file.

**Fix applied**: added a small `_is_checked(state)` normalizer in
`manual_reduction_viewer.py` that unwraps `.value` when `state` is a
`Qt.CheckState` and returns `bool(...)`, so the same comparison works
regardless of which call path supplied it. Changed the `fit_table.py`
comparison to `Qt.CheckState.Unchecked` (both sides now the enum type,
comparable). Also fixed `QFormLayout.setFieldGrowthPolicy(0)` in
`strain_stress_view.py` — same strict-enum-typing class of bug as
`setCheckState`, just a different enum
(`QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow`).

**Why this matters going forward:** when a value can arrive as *either*
a raw Qt signal payload (often a plain `int`/`bool` for
backward-compatible signals like `stateChanged`) or a direct getter call
(often the strict PyQt6 enum type, e.g. `checkState()`), don't compare
it directly against an enum member — normalize first. `grep -rn
"== Qt\.\|!= Qt\.\|checkState() ==" pyrs/` is a reasonable sweep for this
pattern after any further Qt-binding changes.

## Segfault at interpreter shutdown after all tests pass (2026-07)

Once the bugs above were fixed, CI's `tests` job still failed: pytest
itself reported `354 passed, 33 skipped` with zero failures, but
~40 seconds *after* that summary line the process crashed with
`Segmentation fault (core dumped)` (exit code 139), so GitHub Actions
still marked the job red.

This lines up with a `RuntimeError: wrapped C/C++ object of type
SliceViewerView has been deleted` seen locally in `mantidqt`'s own
`AnalysisDataService` (ADS) observer cleanup code
(`mantidqt/widgets/sliceviewer/models/adsobsever.py`), which fires
whenever a workspace tied to an open SliceViewer gets deleted. Locally
(under `QT_QPA_PLATFORM=offscreen`) this stays a caught, non-fatal
`RuntimeError` printed to stderr; under CI's real X server
(`xvfb-run` + the `xcb` platform), the same use-after-free apparently
escalates into an actual native segfault — I could not reproduce the
crash locally even after running the full suite twice, which means this
fix could not be directly verified against the real failure, only
against "no regressions in the passing 354 tests."

**First fix attempt (insufficient on its own):** added a session-scoped,
`autouse=True` fixture (`_clear_ads_at_session_end` in
[tests/conftest.py](../tests/conftest.py)) that calls
`AnalysisDataService.clear()` once, after the last test finishes, on the
theory that lingering SliceViewer-owned workspaces/observers were racing
with native object destruction at interpreter shutdown. Pushed alone,
the exact same segfault recurred on the next CI run — so ADS state
wasn't the (or wasn't the only) cause. Kept anyway as reasonable hygiene
since it's harmless, but it is not the fix that made CI green.

**Actual fix:** don't try to identify the exact native cleanup path that
crashes — instead, stop the interpreter from reaching it at all. Added
[scripts/development/run_tests.py](../scripts/development/run_tests.py),
a tiny wrapper that calls `pytest.main()` directly (rather than going
through the `pytest` CLI entry point), then explicitly flushes
stdout/stderr and calls `os._exit(code)`. `pytest.main()` only returns
once every hook — including pytest-cov's report writing and the
terminal reporter's final "N passed" summary — has already run, so
nothing is lost; `os._exit()` then skips the crashing native teardown
that would otherwise follow. The `[tool.pixi.tasks.test]` command in
`pyproject.toml` now runs this wrapper instead of `pytest` directly.

Dead ends worth recording so they aren't retried:
- A `pytest_sessionfinish` hook with `os._exit()` (even `trylast=True`)
  fires *before* the terminal reporter's own "N passed" summary and
  pytest-cov's coverage table are printed — those apparently happen in
  an even-later stage of pytest's shutdown sequence, not inside
  `pytest_sessionfinish` itself. Using this hook silently truncated all
  of pytest's own reporting from the log, even though the exit code was
  still correct. Only calling `pytest.main()` directly and exiting
  *after it returns* is late enough.
- `os._exit()` does not flush Python's buffered stdout/stderr (unlike
  `sys.exit()`/normal interpreter shutdown) — an explicit
  `sys.stdout.flush(); sys.stderr.flush()` immediately before it is
  required or the whole report vanishes from a non-tty (i.e. CI log)
  destination.
- `mantid.utils.logging.log_to_python()` (routing Mantid's C++ logging
  through Python's `logging` module instead of its native Poco console
  channel, on the theory that a native-channel log call during
  interpreter shutdown was the crash trigger) does **not** prevent the
  crash. Tried two ways: as a session-wide `autouse` fixture, and scoped
  narrowly to just `test_stress_strain_viewer` (the only test that
  exercises `SliceViewer`, matching the crash's traceback) with a
  proper backup/restore of the channel afterward, mirroring
  `mantid.utils.logging.capture_logs()`'s pattern. Neither prevented the
  crash — in fact this is the only way I ever reproduced the segfault
  locally at all (2 of 3 local full-suite runs crashed with some form of
  `log_to_python()` active, versus 0 of 4 without it). Small sample, but
  the direction is consistently unhelpful; abandoned in favor of the
  `os._exit()` wrapper, which is the only thing confirmed to pass real
  CI (twice).

**Why this matters going forward:** if the `tests` job fails again with
a `Segmentation fault` (exit code 139) *after* a passing pytest summary
line, that's a process-teardown crash, not a test regression. This
class of bug is hard to verify locally under `offscreen` — most local
runs never reproduce it at all; matching CI's real display server
(`xvfb-run` with the `xcb` platform, not `QT_QPA_PLATFORM=offscreen`)
is likely required to reproduce it reliably. When a fix like this can't
be verified against the actual failure, verify what you *can*: no
regressions in the full local suite, and (critically, since this is the
actual defect the first two dead ends above introduced) that the
summary output and coverage report are still present in the log after
the change.

## `tests/ui/test_calibration_ui.py` wrote/deleted files in the repo root, not a tmpdir (2026-07)

While iterating on the segfault fix above, running the full suite
locally (required, since the crash could only be reproduced/investigated
by running everything pytest runs in CI) revealed a separate, pre-existing
bug: `test_detector_calibration` in `test_calibration_ui.py` typed the
bare relative filenames `"HB2B_test_export.json"` and `"HB2B_CAL.json"`
into a save/load `QFileDialog`, then did `os.remove("HB2B_test_export.json")`
/ `os.remove("HB2B_CAL.json")` at the end — all relative to the current
working directory, i.e. the repo root when pytest is invoked from there
(exactly how both CI and a local dev shell run it).

This test had been silently doing this on every run; it just never
mattered in CI, which always starts from a clean checkout with no
same-named files to collide with. Locally, though, a developer working
in this exact repo root had files genuinely named `HB2B_CAL.json` and
`HB2B_test_export.json` sitting there (visible as untracked files in
`git status`, presumably outputs from manually running
`pixel_calibration.py`/`reduce_HB2B.py` from the repo root). Running the
test suite locally overwrote those files with the test's own output and
then deleted them via the test's cleanup step — untracked files have no
git history, so they were not recoverable.

**Fix applied**: the test now takes pytest's built-in `tmp_path` fixture
and builds the export/calibration paths as `tmp_path / "HB2B_..."`
instead of bare relative filenames, so it writes only inside a
pytest-managed temp directory that pytest itself cleans up — it can no
longer collide with anything in a developer's working directory
regardless of what they happen to have sitting there. The explicit
`os.remove(...)` calls were removed (no longer needed).

**Why this matters going forward:** a test that writes to a literal
relative path is writing to whatever the current working directory
happens to be when it's invoked — which is the repo root for both CI
and a typical local `pixi run test` — not a private, disposable
location. Prefer pytest's `tmp_path`/`tmpdir` fixtures for anything a
test writes to disk, even "cleaned up" output, since "clean up after
yourself" only works if you're the only thing that could have been
using that name. `grep -rn '"\.\./\|\.\(json\|csv\|h5\|xml\)"' tests/`
(and manually check for bare relative filenames passed to file dialogs,
`open()`, `os.remove()`, etc.) is a reasonable sweep for this pattern in
other UI tests.

## `pixi run test` hangs on the GUI tier under an interactive display (2026-09)

Running the full suite on a workstation with a real desktop session
(`DISPLAY=:0` plus a live Wayland compositor) and **no** `QT_QPA_PLATFORM`
override hangs indefinitely at the first GUI test,
`tests/ui/test_calibration_ui.py`. It is not slow — it is blocked. The
process sits at ~6% CPU with `wchan = poll_schedule_timeout`, holding the
compositor's cursor-shm, `mime.cache` and `icon-theme.cache` file
descriptors open: the signature of a real, mapped Qt window spinning its
event loop waiting for an interaction that never arrives. pytest's stdout
is block-buffered when redirected, so the log shows nothing after the
`tests/ui/test_calibration_ui.py` line and the run looks merely slow.

Set `QT_QPA_PLATFORM=offscreen` and the same suite completes in under six
minutes. This matches what `scripts/development/run_tests.py`'s own
docstring assumes ("the offscreen Qt platform used for local runs"), but
nothing in the repo actually *sets* it, so whether a local run works
depends on the developer's desktop environment.

Note the asymmetry with the segfault entry above: `offscreen` is what
makes a local run finish, while CI's real display server (`xvfb-run` +
`xcb`) is what makes the shutdown segfault reproducible. The two failure
modes want opposite platforms, which is why neither is reliably visible
from the other's environment.

**Resolved (2026-09-18), two ways:**

1. `test-gui` now carries `env = { QT_QPA_PLATFORM = "offscreen" }` in its
   pixi task definition, so it can no longer open a window regardless of
   the developer's desktop. It is *not* set on the `test` task: a pixi task
   `env` overrides the ambient environment unconditionally (verified — a
   caller's `QT_QPA_PLATFORM=xcb` is ignored, and neither `${VAR:-default}`
   in `env` nor in `cmd` expands), so setting it there would silently
   defeat CI's `xvfb-run` wrapper and switch CI off the `xcb` platform
   this very section is about.
2. Every test now has a 300s cap (`pytest-timeout`, configured in
   `pyproject.toml`). A hang is therefore a failure with a stack dump
   rather than an unbounded wait, including on the full `pixi run test`
   where offscreen is not forced.

`timeout_method = "thread"` is required, not a preference. Measured
directly against a `QEventLoop().exec()` that never returns to the
interpreter: with `--timeout-method=signal` (the plugin's Unix default)
the 5s timeout passed unnoticed and an external `timeout 40` had to kill
the process; with `--timeout-method=thread` it was caught at 5s and the
dump named the exact blocking line. SIGALRM is only delivered when the
interpreter next executes bytecode, which a blocked C++ event loop never
does.

**Practical consequence:** prefer `pixi run test-unit` /
`pixi run test-integration` for day-to-day work — they deselect the `gui`
marker entirely and never open a window. (Confirmed: the only `tests/ui/`
tests the integration tier selects are `test_model`,
`test_model_multiple_files` and `test_model_from_json`, which construct
`Model()` and touch no widget.) Before the full `pixi run test`, still
export `QT_QPA_PLATFORM=offscreen` yourself.

## Uncalibrated (`Status: -1`) calibration JSON silently applied during reduction (2026-07)

`read_calibration_json_file()` in
[pyrs/utilities/calibration_file_io.py](../pyrs/utilities/calibration_file_io.py)
returns a 5-tuple `(shift, shift_error, wave_length, wave_length_error,
status)`, where `status` is the exit code of the
`scipy.optimize.least_squares` call performed during calibration
refinement (`self._calibstatus = out[2]` in
`pyrs/calibration/mantid_peakfit_calibration.py`, `out.status` from
`FitDetector`). That field is initialized to `-1` in `__init__` before
any refinement has run, and negative values in general mean the
optimizer never converged. `ReductionApp.reduce_data()` in
[pyrs/core/powder_pattern.py](../pyrs/core/powder_pattern.py) unpacked
this 5-tuple but only ever used `calib_values[0]` (shift) and
`calib_values[2]` (wavelength) — it ignored `status` entirely, so a JSON
calibration file that was never successfully refined (e.g. a stale
template still carrying the `-1` sentinel, or one saved after a
non-converging fit) was applied to reduction exactly like a good one,
producing plausible-looking but systematically shifted diffraction
patterns with no diagnostic indication.

**Fix applied**: added `check_calibration_status(status)` to
`calibration_file_io.py`, which raises `RuntimeError` when `status < 0`.
`reduce_data()` now calls it immediately after
`read_calibration_json_file()` returns and before any of the returned
values are used, so an unconverged/never-refined calibration aborts
reduction instead of being silently applied. Regression coverage:
`tests/unit/pyrs/utilities/test_calibration_file_io.py::test_check_calibration_status_negative_status_raises`
(the guard in isolation) and
`tests/integration/test_reduction.py::test_reduce_data_with_unconverged_calibration_status_raises`
(exercises the real `ReductionApp.reduce_data()` call site against
`tests/data/HB2B_1017.h5` with a corrupted-status copy of
`tests/data/HB2B_CAL_Si333.json`).

**Why this matters going forward:** `read_calibration_json_file()` has a
single production caller (`powder_pattern.py`);
`import_calibration_ascii_file()` (the other calibration loader, used
for the non-JSON branch of the same `if`) has no `Status` field/concept
at all, so it needed no change. *(Removed 2026-11, #1051: a non-`.json` calibration
file now raises `ValueError`. `PyRsCore.reduce_diffraction_data` is a second caller of
`read_calibration_json_file()` and also calls `check_calibration_status()`.)* If another caller of
`read_calibration_json_file()` is added later, it must also call
`check_calibration_status()` on the returned status before applying the
shift/wavelength — `grep -rn "read_calibration_json_file" pyrs/` is the
sweep to check this hasn't been bypassed.

**Known gap (not yet fixed):** `check_calibration_status()` only rejects
negative status codes. `status == 0` is scipy's "maximum number of
function evaluations exceeded" code — the optimizer ran but did **not**
converge — which is the same class of "never successfully refined"
problem this fix targets, just not the specific `-1` sentinel that
motivated it. A calibration JSON saved after a non-converging fit with
`Status: 0` is currently still applied silently. If this needs closing,
change the guard's condition and confirm with whoever owns the
calibration refinement code whether `status == 0` should always be
rejected outright or only warned on.

## Combined-axis calibration rotation used element-wise multiply instead of matrix multiply (2026-07)

`generate_rotation_matrix()` in
[pyrs/core/reduce_hb2b_pyrs.py](../pyrs/core/reduce_hb2b_pyrs.py) combined
the three per-axis rotation matrices (`rot_x_matrix`, `rot_y_matrix`,
`rot_z_matrix`, each a plain `np.array`, not `np.matrix`) with NumPy's
`*` operator, which is element-wise (Hadamard) multiplication for plain
ndarrays — not matrix multiplication.

**Correction to the initial bug report:** the original report (and this
entry's first draft) described the bug as "quiescent whenever at most
one axis is non-zero" — i.e., only combined multi-axis rotations were
assumed broken, with single-axis calibrations assumed fine. That is
**not** correct, and was disproven by running the actual single-axis
regression test against the unfixed code (it failed) and independently
verifying the arithmetic:
```python
Rx = rotation matrix for rot_x=2deg, Ry0 = Rz0 = identity (angle=0)
Rx * Ry0 * Rz0   # elementwise: zeroes every off-diagonal term of Rx too,
                 # because elementwise-multiplying by an identity matrix
                 # kills any entry that lands on a zero of the identity
```
Elementwise-multiplying any matrix by the identity (which is what a
zero-angle axis matrix reduces to) zeroes all of that matrix's
off-diagonal terms, not just the identity's. So the **only** case the
old code ever got right was all three angles being exactly zero (no
rotation at all, both sides trivially the identity). Any calibration
with even one non-zero rotation angle — single-axis or combined — was
silently wrong.

This is not dead code: `generate_rotation_matrix()` is called from
`ResidualStressInstrument.build_instrument()`
(`reduce_hb2b_pyrs.py:133`), and the resulting matrix is applied to
every detector pixel's position via `_rotate_detector()`
(`reduce_hb2b_pyrs.py:38-52`) on the real reduction path
(`reduction_manager.py` → `PyHB2BReduction.build_instrument()` →
`ResidualStressInstrument.build_instrument()`), whenever a calibration
with any non-zero rotation is applied. Every pixel's 2theta, and
therefore every downstream peak position/d-spacing/strain, would be
silently wrong for such a calibration.

**Fix applied**: changed line 270 from
`rot_x_matrix * rot_y_matrix * rot_z_matrix` to
`rot_x_matrix @ rot_y_matrix @ rot_z_matrix` (matrix multiplication).
Regression coverage added in
`tests/unit/pyrs/core/test_reduce_hb2b_pyrs.py`:
`test_generate_rotation_matrix_combined_rotation_is_orthogonal` asserts
`R @ R.T ≈ I` and `det(R) ≈ 1` for simultaneously non-zero rotations on
all three axes; `test_generate_rotation_matrix_single_axis_matches_axis_matrix`
asserts a single-axis rotation reduces exactly to that axis's own
matrix. Both were confirmed to fail against the unfixed code (verified
by temporarily reverting the fix) and pass with it.

**Why this matters going forward:** the actual gap in test coverage
wasn't "single-axis tests don't catch multi-axis bugs" (the initial,
incorrect hypothesis) — it's that **no test exercised
`build_instrument()`/`generate_rotation_matrix()` with any non-`None`,
non-zero calibration at all** (the one test that calls
`build_instrument()`, in `tests/integration/test_powder_pattern.py`,
passes `instrument_calibration=None`). A function is only as tested as
the inputs actually exercised — a plausible-sounding severity
assessment ("this only breaks when N things are true") still needs to
be verified against the real unfixed code before it's written down;
don't propagate an unverified claim about a bug's scope, even from a
detailed-looking report, without checking it against a failing test.
`grep -rn '\* rot_\|rot_.*_matrix \*' pyrs/` (and generally
`grep -rn ' \* .*_matrix\b'`) is a reasonable sweep for the same
`*`-instead-of-`@` mistake elsewhere in this file or in similar
geometry code.

**Action needed on existing calibration files:** any previously-saved
calibration JSON with a non-zero rotation was *fit* using the old,
broken forward model (`least_squares` in
`pyrs/calibration/mantid_peakfit_calibration.py` calls the same
`generate_rotation_matrix()` during refinement, so calibration and
reduction shared the identical bug, and the fit self-consistently
absorbed it). Applying such a stale calibration through the now-fixed
model will **not** reproduce its old (also wrong) geometry — it needs
to be re-refined from raw data, not just re-applied. Check any
calibration JSON with 2+ non-zero rotation values against its
refinement date relative to this fix.

*Since 2026-11 (#1011):* the rotation helpers are `rotation_matrix`, `rotation_matrix_x/y/z` and
`rotate_positions` (formerly `_rotate_detector`) in `pyrs/core/detector_model.py`;
`generate_rotation_matrix`, `_cal_rotation_matrix_x/y/z` and `_rotate_detector` were removed from
`reduce_hb2b_pyrs.py`, so the `reduce_hb2b_pyrs.py` line numbers in this entry are historical. The two regression tests above now test
`rotation_matrix` in `tests/unit/pyrs/core/test_detector_model.py`
(`test_rotation_matrix_combined_rotation_is_orthogonal`,
`test_rotation_matrix_single_axis_matches_axis_matrix`).

## Vanadium normalization is a flat-field correction only — run-duration scaling is intentionally NOT applied (2026-07)

**Intent.** The vanadium normalization in
[pyrs/core/reduce_hb2b_pyrs.py](../pyrs/core/reduce_hb2b_pyrs.py)
(`PyHB2BReduction.histogram_by_numpy()`) is a **flat-field correction**:
it divides out per-pixel detector-efficiency and solid-angle variation
so that a uniform scatterer would produce a uniform response. Vanadium
is a near-isotropic incoherent scatterer, so its measured pattern
reflects instrument response rather than sample structure. The
normalized signal is:

```
normalized signal = Sample / (V_signal / V_signal.max())
                  = Sample * (V_signal.max() / V_signal)
```

which is exactly what the code computes:

```python
normalized_data = data_array * (van_array_max / van_array)
```

This ratio is deliberately **self-referencing** — `van_array_max` and
`van_array` are drawn from the same array, so it is algebraically
invariant to any uniform rescaling of the vanadium counts
(`(k·V_max)/(k·V) = V_max/V`). It corrects the *shape* of the response,
not its absolute scale. That scale-invariance is a feature of a
flat-field correction, not a defect: the correction should not depend on
how long either run counted.

**Why run-duration scaling was reverted.** A prior change (commit
`8a979f67`) threaded `sub_run_duration`/`van_duration` through
`convert_counts_to_diffraction` → `reduce_to_2theta_histogram` →
`histogram_by_numpy` and, after the flat-field ratio, applied an extra
multiplicative factor `van_duration / sub_run_duration`. That turns the
correction into an absolute/counting-time normalization, which is **out
of scope** for a flat-field correction and changed reduced intensities
by the ratio of the two runs' counting times (e.g. the
`test_manual_reduction` UI baseline shifted by ~0.575×). This was
reverted: `reduce_to_2theta_histogram()`/`histogram_by_numpy()` no
longer take duration parameters, and the reduction returns to
flat-field-only output.

**Current state of the duration parameters.**
`reduce_sub_run_diffraction()`/`reduce_sub_run_texture()`/
`reduce_diffraction_data()` in
[pyrs/core/reduction_manager.py](../pyrs/core/reduction_manager.py)
still *accept* `sub_run_duration` and `van_duration` (and their
docstrings still describe the "raw / vanadium * vanadium duration / sub
run duration" formula), but these values are **not** forwarded into the
histogram arithmetic and have no effect on the output. Treat that
docstring formula as describing an intent that is intentionally not
implemented here. If run-duration (absolute counting-time) normalization
is ever wanted, it belongs in a separate, explicitly-named step — not
folded into the vanadium flat-field correction — and needs its own
review of every reduction call site, since it changes output scale for
every reduced pattern.

## Output project file never recorded which calibration/vanadium/PyRS version produced it (2026-07)

The output HiDRA project file (HDF5) recorded the calibrated wavelength
and raw instrument geometry, but never recorded which calibration file
was applied, which PyRS version wrote the file, or which vanadium run
was used for normalization — a scientist could not verify, from the
file alone, how it was produced. The schema already had an empty
`CALIBRATION` subgroup under `INSTRUMENT`, created at file-creation
time by `_init_project`
([pyrs/projectfile/file_object.py](../pyrs/projectfile/file_object.py)),
apparently intended for exactly this — but nothing ever wrote into it
(`grep -rn "HidraConstants.CALIBRATION" pyrs/` found only the
group-creation call site).

**Important structural detail**: `write_instrument_geometry` — the
only function that touches the `INSTRUMENT` group — is called from
exactly one place in the whole codebase: `NeXusConvertingApp.save()`,
which runs during the *earlier* NeXus→HiDRA-project conversion stage,
before any calibration file has even been chosen. The calibration file
path is a local variable inside `ReductionApp.reduce_data()`
([pyrs/core/powder_pattern.py](../pyrs/core/powder_pattern.py)), a
later, separate stage — it was never reachable from
`write_instrument_geometry` no matter what was added to that function.
Any fix had to add a new write path at the point where `reduce_data()`
actually knows the calibration/vanadium inputs, not extend the existing
one.

**Fix applied**:
1. Added `write_reduction_provenance(calibration_file=None,
   vanadium_run=None)` to `HidraProjectFile`
   (`pyrs/projectfile/file_object.py`): writes `calibration_file` (path)
   and `calibration_file_sha256` (via a new `_compute_file_sha256`
   helper, 64KB-chunked, silently skipped if the file can't be read) as
   attributes on the existing `CALIBRATION` group when a calibration
   file is given; writes `vanadium_run` similarly when given; always
   (re-)writes a root-level `pyrs_version` attribute from
   `pyrs.__version__`.
2. `write_instrument_geometry` also now writes `pyrs_version` at the
   root, since it already runs for essentially every produced file
   during conversion — this means the version is recorded even for
   files that are never touched by `reduce_data()`/`save_diffraction_data()`.
3. `ReductionApp` (`powder_pattern.py`) now stores
   `self._calibration_file`/`self._van_file` from `reduce_data()`'s
   parameters (previously discarded once the function returned), and
   `save_diffraction_data()` calls `write_reduction_provenance()` with
   them before writing reduced diffraction data.

Verified end-to-end (NeXus conversion → calibrated `reduce_data()` →
`save_diffraction_data()` → reopened with `h5py`): `calibration_file`,
its real SHA-256, and `pyrs_version` are all present and correct;
`vanadium_run` is correctly absent when no vanadium was used. Unit
tests in `tests/unit/pyrs/projectfile/test_file_object.py` cover the
round-trip, the "neither given" case, and a calibration path that no
longer exists on disk (path still recorded, hash silently skipped, no
exception).

**Known gap (not fixed here) — verified by direct repro, not just
inspection:** `write_reduction_provenance` writes each of
`calibration_file`/`vanadium_run` only when that specific argument is
given; it never clears a previously-written attribute for the other
one. So calling it twice on the same file with different combinations
of arguments leaves a stale mix:
```python
f.write_reduction_provenance(calibration_file="calA.json", vanadium_run="van1")
f.write_reduction_provenance(calibration_file="calB.json")  # vanadium_run omitted
# CALIBRATION group now reads: calibration_file="calB.json", vanadium_run="van1" (stale!)
```
(Note: calling it with *neither* argument is safe — the method returns
immediately without touching either attribute, confirmed by
`test_write_reduction_provenance_no_calibration_or_vanadium_only_records_version`
— so `save_diffraction_data()` called without a prior `reduce_data()`
does not clobber good provenance with `None`; the risk above is
specifically a second call with a *different* partial combination of
real values.) Audited every current call site
(`pyrs/interface/manual_reduction/pyrs_api.py`,
`pyrs/calibration/mantid_peakfit_calibration.py`, all integration
tests): every one constructs a fresh `ReductionApp()` immediately
before its `reduce_data()` + `save_diffraction_data()` pair, so this is
a latent trap for future reuse, not a live bug today. If it needs
closing, `write_reduction_provenance` should delete the
`vanadium_run`/`calibration_file`/`calibration_file_sha256` attrs
explicitly for whichever of the two is `None`, not just skip writing
them.

**Second gap, resolved by removal (2026-07-22):** `PyRsCore.save_diffraction_data()`
(`pyrs/core/pyrscore.py`) called `HB2BReductionManager.save_reduced_diffraction()`
directly, bypassing `ReductionApp.save_diffraction_data()` (and therefore
`write_reduction_provenance`) entirely. `grep -rn
"\.save_diffraction_data("` across the repo confirmed zero callers of
`PyRsCore.save_diffraction_data` — it was dead code. Since nothing used
it, it was deleted outright rather than wired into provenance-writing;
if reduced-diffraction saving from `PyRsCore` is needed again in the
future, add it back calling `ReductionApp.save_diffraction_data()` (or
thread `write_reduction_provenance()` through directly) so it doesn't
silently reintroduce this same bypass.

**Why this matters going forward:** before adding a write to an
existing function because "that's where this kind of data gets
written," check whether the function is actually called at a stage of
the pipeline where the new data is available at all — extending
`write_instrument_geometry` here would have been a no-op fix, since the
calibration path doesn't exist yet at that call site. The right fix
needed a new write path anchored to where the data is actually known
(`save_diffraction_data`), not a bigger version of the existing one.

## `~/.config/QtProject.conf`'s lock file: "Could not remove our own lock file" on export (2026-08)

Exporting peak-fit results from PyRS intermittently produced
`Could not remove our own lock file "~/.config/QtProject.conf.lock.rmlock"`
(Permission denied). Reported reproduction: within a *single* PyRS GUI
session — remove any stale lock file, fit peaks, export (succeeds), load
a new data file, fit again, export a second time (fails). No test suite
or second PyRS instance needs to be running.

**Correction (2026-08-31): the first fix attempt below (commit `a1f7ba3d`,
calling `setOrganizationName()`/`setApplicationName()`) was based on a wrong
diagnosis and does not work.** It was believed that Qt falls back to a
generic `"QtProject"` identity only because PyRS never sets its own
org/app name. That's false: `QFileDialog`'s own C++ implementation
(`QFileDialogPrivate::saveSettings()`/`restoreSettings()`) **hardcodes**
`QSettings(QSettings::UserScope, "QtProject")` for its history/sidebar
state, completely independent of `QCoreApplication::organizationName()`/
`applicationName()`. This was verified empirically: a probe script that
called `setOrganizationName("PyRS")`/`setApplicationName("PyRS")` *before*
constructing `QApplication` and a `QFileDialog` still wrote
`[FileDialog]` state to the real `~/.config/QtProject.conf` (confirmed by
content and by a live, reproduced infinite `.rmlock`-chaining loop against
that exact file). So setting an org/app name does nothing to isolate
`QFileDialog`'s settings — it only affects code that explicitly
constructs its own `QSettings()`/`QSettings(org, app)`, which nothing in
`pyrs/` does.

**Actual root cause, in two parts:**
1. Every Qt/PyQt process on the machine that ever opens a `QFileDialog`
   — PyRS, Mantid Workbench, Qt Designer, another PyRS window, `pytest-qt`
   — reads/writes/locks the *same* `~/.config/QtProject.conf`, by Qt
   design (it's meant to share "recent places" across apps). Two of them
   racing on its `QLockFile` at the same moment is what produces "Could
   not remove our own lock file."
2. That race is normally recoverable (Qt detects and removes a stale
   lock), but `~/.config` here lives on an **NFS4-mounted home directory**
   (`mount` shows `nfs4 ... local_lock=none`). `QLockFile`'s stale-lock
   recovery depends on atomic `rename()`/`link()`, which NFS does not
   guarantee. On this filesystem the recovery attempt can itself fail and
   retry against its own leftover output, producing exactly the observed
   `.lock.rmlock.rmlock.rmlock...` chains (reproduced live during this
   investigation — a single hung Python process appended a new `.rmlock`
   suffix to `~/.config/QtProject.conf.lock` roughly once a minute,
   indefinitely, until killed).

**Could not reproduce deterministically** in a scripted, single-process,
offscreen repro at the time of the first fix attempt — consistent with
this being a race that depends on real concurrent Qt processes and NFS
timing, neither of which a quiet local repro exercises.

**Fix attempt #2 (superseded, same day):** redirect `XDG_CONFIG_HOME` to
a PyRS-private, persistent subdirectory *still under the NFS-mounted
home* (`~/.config/PyRS`) before constructing `QApplication`. This
correctly isolated PyRS's settings file from every other Qt process
(`~/.config/QtProject.conf` itself went untouched in verification), so
it does fix cross-app lock *contention*. It does **not** fix the
underlying problem: `~/.config/PyRS` is still on NFS, so `QLockFile`'s
stale-lock recovery is still non-atomic there. This was confirmed for
real: a PyRS session was killed (or crashed) leaving
`~/.config/PyRS/QtProject.conf.lock` referencing a dead PID, and the
*next* launch failed with `Could not remove our own lock file
.../PyRS/QtProject.conf.lock: Device or resource busy` — the identical
symptom, just against the now-private file instead of the shared one.
Moving the file off the contention path was necessary but not
sufficient; it needed to also move off NFS.

**Actual fix** ([scripts/pyrsplot.py](../scripts/pyrsplot.py)): redirect
`XDG_CONFIG_HOME` to `<tempfile.gettempdir()>/pyrs-qt-config-<user>` —
i.e. **local disk** (verified here to be `xfs` on `/`, not NFS),
namespaced per OS user (`getpass.getuser()`) so it can't collide with
other users on a shared multi-user analysis/login node, created with
`0o700` permissions so other users can't read one user's dialog
history. Because local filesystems support the atomic `rename()`/
`link()` operations `QLockFile`'s stale-lock recovery depends on, a
crashed/killed PyRS process's lock is now cleanly detected and reclaimed
by Qt's own recovery logic on the next launch — no manual intervention
needed. Verified empirically: two consecutive isolated launches (dialog
open → close → quit) both completed cleanly with zero lock file left
behind afterward, on both the first and second run.

Trade-off accepted: this directory is local to whichever compute/login
node the session happens to run on (typical at a shared, multi-node
facility), so "recent places" history does not follow the user between
nodes and may be cleared by OS temp-directory housekeeping. That's a
purely cosmetic loss — nothing in `pyrs/` reads this state — traded for
never hitting an NFS-unrecoverable lock again.
`setOrganizationName()`/`setApplicationName()` calls were kept (correct
practice for any future code that constructs its own `QSettings`), but
were never what fixed this.

**Why this matters going forward:** if a similar "permission denied /
device or resource busy removing our own lock file" resurfaces
elsewhere, check, in order: (a) whether the affected settings file is
one Qt itself hardcodes an identity for (`QFileDialog` does via
`QSettings::UserScope, "QtProject"`; your own explicit `QSettings()`
calls don't — `setOrganizationName()`/`setApplicationName()` only helps
the latter); (b) whether the file's directory is shared with other,
unrelated Qt processes; and (c), the one that actually matters for
*recovery* from a crash, whether that directory is on NFS — if so,
isolating the file (b) is not enough, it must also move off NFS
entirely, since `QLockFile` recovery is fundamentally unreliable there.
A quiet, single-process, offscreen repro will not reproduce (a)/(b), and
won't reproduce (c) either unless it specifically kills the process
mid-lock and relaunches — a clean repro passing is not evidence the fix
works under real crash/network conditions.

## Mantid `FitPeaks` `HighBackground` is per-engine, default `True` (2026-10)

*Superseded 2026-10-29:* ring extraction now uses the default too, so the per-engine option was
removed again and `pyrs/peaks` matches `next` (see "Ring extraction does not need
`HighBackground=False`" below). What follows explains the two settings.

`MantidPeakFitEngine` (and `FitEngineFactory.getInstance`) take
`high_background: bool = True`, forwarded to Mantid `FitPeaks` as
`HighBackground`. (`FindBackgroundSigma=1` was passed alongside it until #1029; Mantid ignores it,
see the `FindBackgroundSigma` entry below.)

- `True`: three fits per spectrum (Mantid `FitPeaks.cpp`, `fitFunctionHighBackground`):
  1. a linear background alone, on the window minus ±5 bins around the expected centre (skipped if
     either side has ≤ 10 bins);
  2. the peak alone, on the data minus that background;
  3. **peak and background together** on the original data, starting from steps 1–2, with the
     position tolerance applied.

  The reported parameters and errors come from step 3. Upstream default since `ab3dd7d5`
  (2019-10-15, "Peak fitting pytests are passed"); every normal PyRS fit (GUI, reduction,
  strain/stress) relies on it.
- `False`: step 3 only, from Mantid's own starting estimates. Used only by pixel-position
  calibration ([ring_extraction.py](../pyrs/calibration/ring_extraction.py)).

Both end in the same joint peak + background fit; they differ only in its starting point (checked
in the Mantid source, `main`, 2026-10-29).

**Why this matters:** the calibration branch originally hard-coded `False`
in `mantid_fit_peak.py`, which silently changed fitting for all of PyRS.
Change the behaviour for a caller by passing the flag, never by editing the
`FitPeaks` call.

## Detector rotation matrices are float64; float32 made angle derivatives zero (2026-10)

`ResidualStressInstrument._cal_rotation_matrix_x/y/z` (`pyrs/core/reduce_hb2b_pyrs.py`) built
their matrices as **float32**. Their entries carry ~7 significant digits, so any rotation angle
— the arm 2θ, 2θ₀, Rot_x/y/z — only resolved to ~3e-6°. For reduction that is irrelevant, but
calibration differentiates the residual numerically, and scipy's default finite-difference steps
sit at or below that resolution:

- `jac="2-point"` (used with `method="lm"`) steps by `sqrt(eps) ≈ 1.5e-8`: the 2θ₀ and rotation
  derivatives came out **exactly zero**, so `lm` never moved 2θ₀.
- `jac="3-point"` (trf/dogbox) steps by `eps**(1/3) ≈ 6e-6`: per-point angle derivatives were off
  by up to ~10–50 %, corrupting convergence and the error bars (they come from the same Jacobian).
  `FitCalibration` uses the same default step, so it was affected too.

The matrices are now float64 (user-approved change to reduction code). Effect on reduction: pixel
2θ/η move by ≤ 4.5e-7°; the unit and integration suites are unchanged. With float64 every step
from 1.5e-8 to 1e-2 gives the same derivative (Rot_y, the weakest, to ~1.5e-4 at the smallest
step), so `PixelCalibration` uses scipy's default step (`diff_step=None`).
`test_scipy_default_step_gives_accurate_angle_derivative` and (from #1029)
`test_rotation_matrix_resolves_tiny_angle_change` fail if the matrices go back to float32.
*Since 2026-11 (#1011)* the matrices are built by `rotation_matrix_x/y/z` in
`pyrs/core/detector_model.py`, and that test is in `tests/unit/pyrs/core/test_detector_model.py`.

## Ring calibration: why it has its own forward model, and the draft's defects (2026-10, #1012–#1014)

The ring-based calibration started from a draft, `pyrs/calibration/pixel_pos_calibration.py` as
first committed in 5a635584 (2024-12). The rewrite on `detector_calib_v3` keeps the idea: fit each
ring once in strips of the raw image, then refine the geometry from the ring pixels. It replaced
the parts that would have made its result wrong, or mean something different in reduction:

**Forward model (#1012).**
- **The rotations did not reproduce in reduction.** The draft's `get_pixel_tth` added
  `ARM_LENGTH + shift_z` *before* applying the calibration rotation. Reduction
  (`ResidualStressInstrument.build_instrument`) rotates about the detector centre first, then
  pushes the detector out. A Rot_x/Rot_y/Rot_z fitted by the draft therefore described a different
  detector when reduction applied it.
- **It hard-coded the instrument:** arm length 0.985 m and detector size 0.3 m, instead of reading
  `DENEXDetectorGeometry`.

**Refinement (#1014).**
- **λ was unconstrained.** Expected d-spacings were derived from the *nominal* wavelength, so the
  residual scaled with λ itself. λ ran to its bound and 2θ₀ absorbed the ring offset. Now the
  residual is 2θ_pixel − 2·asin(λ / 2d), with d from the powder standard.
- **The residual changed length between calls**, because an outlier cut ran inside it.
  `least_squares` needs a fixed-length residual with no side effects.
- **Single-parameter stages reported errors for the wrong parameter.** The final residual was
  evaluated without the stage's parameter index.
- **State accumulated across calls**, so iteration counts were cumulative. The error formula was
  non-standard; it is now `inv(JᵀJ)·Σr²/(n−p)`.
- **Lattice constants differed from `FitCalibration`'s** (Ni 3.523799 vs 3.526314). Both now read
  one table, `pyrs/calibration/powder_standards.py` (#1013), with the 2020 values (#1037 aligns
  `FitCalibration`).

**Why not call `build_instrument` directly.** It builds and rotates the full 1024×1024 pixel grid,
about 250 ms per sub-run, and only at integer pixels. The ring points sit at fractional pixels, and
the optimiser evaluates the model hundreds of times.

`DetectorModel` evaluates only the requested (fractional) pixels, about 0.4 ms per sub-run. It reuses
`ResidualStressInstrument`'s own rotation helpers and order of operations:
1. shift in X/Y;
2. rotate `Rx @ Ry @ Rz` about the detector centre;
3. push out by `l2 + shift_z`;
4. rotate about the sample by 2θ + 2θ₀.

Two consequences:
- **No new convention is introduced.** A unit test pins 2θ and η to be bit-identical to
  `build_instrument` on the full grid, nominal and calibrated, at two arm angles.
- **Reduction is unchanged.** Making `build_instrument` delegate to `DetectorModel`, so that one
  piece of code defines the detector, is the follow-up #1011 (done 2026-11, see the #1011 entry).

*Since 2026-10-30 (#1020):* `DetectorModel` lives in `pyrs/core/detector_model.py`, next to the
reduction geometry it reproduces. The calibration imports it from there, and
`pyrs.calibration.pixel_pos_calibration` still re-exports it. Its tests, including the bit-identity
test against `build_instrument`, are in `tests/unit/pyrs/core/test_detector_model.py`. It was a pure
move: unit and integration suites pass unchanged, and the driver output is bit-identical on
HB2B_3489/3510/4971. `reduce_hb2b_pyrs.py` was not touched.

**TTH_0.** The calibration writes `TTH_0`, and reduction now reads it (#1021, fixed in #1031). What
it represents, and why the ring fit cannot determine it at one L2, are the two 2θ₀ entries below.

## Ring calibration: Shift_x/Rot_y/TTH_0 are exactly degenerate; Shift_y is weak (2026-10)

Found with synthetic ring points (`tests/util/ring_calibration_helpers.py`) on the real HB2B_4971
scan layout. The SVD of the residual Jacobian (units: mm, deg, mÅ) shows:

- **Exact degeneracy** (singular value ~5e-10): 1 mm of `Shift_x` ≙ 0.057° of `Rot_y` and −0.057°
  of `TTH_0`. Turning the arm about the sample by −α, turning the detector about its own vertical
  axis by +α and shifting it sideways by L·sin α (L = 0.985 m, so 1 mm ↔ 0.058°) describe the same
  detector. Every geometry along this line gives the **same 2θ and η on every pixel** (checked to
  ~1e-6° at arms 60–120°), so **reduction is unaffected**, but the three values are individually
  meaningless when refined together: `CalibrateGeometry` and `FullCalibration` (here and in
  `FitCalibration`) do exactly that. Fixing any one of them (e.g. Rot_y = 0) makes all the others
  recoverable. Which one to fix is an open scientific convention.
- **Shift_y is weak** (0.0075° residual per mm, summed over 130 points): rings are symmetric about
  the horizontal plane, so a vertical shift moves 2θ only to second order. With 0.01° ring scatter
  it is uncertain to about ±1 mm.
- The evenly spaced arm layout behaves almost identically; walking each ring across the detector
  (as 4971 does) does not remove either effect.

Errors for an undetermined parameter are reported as **-2** (not NaN): along the degeneracy the
covariance inverse can return negative variances.

*Update (2026-10-29, #1019 test review):* that rule only fired when `inv(JᵀJ)` happened to give a
negative or NaN variance. On synthetic 4971-layout points it usually did not. `FullCalibration` then
reported finite but meaningless errors:
- **Degenerate set:** Shift_x 578 m and Rot_y/TTH_0 3.3e4° with 0.01° noise; 3e-7 m and 2e-5° with
  no noise.
- **Others inflated:** the near-singular inverse also pushed up the errors of other parameters,
  e.g. Shift_y 12 m and Rot_z 173°.

`_parameter_errors` now detects the rank deficiency directly. In the column-scaled Jacobian, the
degenerate direction's singular value is about 4e-10 of the largest, and the next-weakest about
5e-2. The cut-off is 1e-6.
- **Flagged -2:** every parameter whose removal restores full rank. That is Shift_x, Rot_y and
  TTH_0, and also Shift_y, Shift_z and Rot_z, which have a small, systematic share (about 3e-4
  scaled) in the exact null move: it also adjusts Shift_z and re-splits the rotations.
- **Real errors:** Rot_x and λ, which are orthogonal to the degenerate direction. They come from the
  pseudo-inverse.
- **At full rank:** e.g. `estimate_errors` with 2θ₀ held, the result is the same `inv(JᵀJ)` formula.
  The driver output on HB2B_4971 is unchanged to 3e-14 relative.

HB2B_4971 layout (from `Calib_Testing/HB2B_4971.h5` logs): arms 76, 80 (Ni), 78, 80, 82, 86 (Fe),
89, 93, 97 (Ni), 99, 103 (Fe); it sees Ni 220/311/222 and Fe 211/220, each at 2–4 arm positions.

## What 2θ₀ (TTH_0) represents on HB2B (2026-10)

`TTH_0` is not only an arm-encoder zero error: it is the combination of the encoder offset and a
**monochromator misalignment** (the incident beam leaving the monochromator slightly off its nominal
direction). Turning the incident beam about the sample's vertical axis is physically equivalent to
turning the detector arm the other way, so both appear as the same constant 2θ offset, and at a fixed
L2 both are also degenerate with Rot_y + Shift_x (see the degeneracy entry above).

**Decision (user, 2026-10-21):** `TTH_0` and `Lambda` are refined as **independent** parameters.
2θ₀ is *not* inferred from, or constrained by, the monochromator setting or take-off geometry, even
though a beam misalignment would in principle shift λ as well.

## 2θ₀ is a virtual swing: the ring fit cannot measure it, so it is set, not fitted (2026-10)

**Physics (user, 2026-10-21).** The detector arm does not move. 2θ₀ (beam misalignment from the
monochromator plus any encoder error) turns the *incident beam* relative to the fixed detector.
Seen from the beam this is a **virtual** swing of the detector about the sample, with four effects
on the detector image, all set by the one angle α:

| Effect | Exact size (L = sample–detector distance) | α = 0.656°, L = 0.985 m |
|---|---|---|
| uniform ring offset | α | 0.656° |
| sideways move of the beam centre | L·tan α | 11.28 mm (38.5 px) |
| distance change along the nominal ray | L·(1/cos α − 1) | +65 µm |
| rotation of the detector face | α | 0.656° |

`build_instrument` already models exactly this (it rotates the detector about the sample by
2θ + 2θ₀ in the beam frame), so **no change to the TTH_0 math** was made in reduction or calibration.

**Exact hand-back is a null move.** Those effects are, at one L2, exactly a change of the detector's
rotations and shifts: R → Ry(−Δ)·R (re-split into Rot_x/y/z), Shift_x by L·tan Δ, Shift_z by
L·(1/cos Δ − 1). Changing 2θ₀ and handing its virtual swing back this way changes no pixel's 2θ or η
(5.7e-14° at arms 60–120°). So **at a single L2 the ring data cannot determine 2θ₀** — every value,
with its handed-back shifts and rotations, fits identically. Its value must come from outside the
fit (beam/encoder alignment, or a convention), or from a scan at two distances (where L·tan α no
longer matches a real shift). `DetectorModel.with_two_theta_0` / `PixelCalibration.set_two_theta_0`
implement the exact swap (tests pin the L·tan α, L·(1/cos α − 1), α forms and map invariance).

Partial hand-backs are *not* null and are not the same thing: handing back only the position
(re-solving shifts to keep the detector centre fixed) leaves a pure face turn, which is close to but
**not** equal to Rot_y — they differ by 7e-4° (GUI-like geometry) to 2.3e-3° (Rot_x −1°, Rot_z 2°)
per 0.5° step, because Rot_y turns about the detector's own (tilted) axis and pivots about the
unshifted centre.

**Iterative workflow.** The production recipe is staged — CalibrateShift → CalibrateRotation →
calibrate_wave_shift, repeated — now `PixelCalibration.iterate()` (fixed passes or until no change
> 1e-3 mm/deg/mÅ, then an optional joint polish = `FullCalibration`). Study
(`tests/scripts/ring_calibration_degeneracy_study.py`, 4971 layout, 0.01° scatter, 20 trials):

- λ's residual pattern is 0.990 similar to TTH_0's and Shift_x's (a near-uniform ring offset over
  76–98°). **λ must be refined in the same stage as the parameter that carries the ring offset** —
  the recipe's λ+2θ₀ stage does that.
- Recipe with everything free (A): noise-free exact after ~11–13 passes by tolerance for small and
  GUI-like misalignments, but for a large one (Rot_x −1°, Rot_z 2°, 2θ₀ −1.5°) it had **not
  converged after 50 passes** (2e-3° map error). **3 passes + polish reaches the optimum for all
  three** (map error ≤ 2e-10°, 43–73 evaluations, ≤ 1.2 s), which is why `iterate()` polishes by
  default.
- Fixing a parameter in a staged recipe makes it crawl: Rot_y = 0 stalls at a 0.014° map error;
  TTH_0 = 0 is still 0.3° off after 50 passes; Shift_x = 0 needs ~30 passes. The coupled pair then
  sits in different stages. A final polish rescues all of them.
- Compensating TTH_0 steps to keep the detector centre fixed ("G") is as bad as TTH_0 = 0 (0.74°
  after one pass): it strips TTH_0 of the ring offset, leaving λ to fight Shift_x across stages.
  Rejected.
- One-shot fits give the same 2θ map for every convention (median error 0.018° at 0.01° scatter).

**Decisions (user, 2026-10-21):** Rot_y is not fixed; TTH_0 keeps its virtual-swing meaning and is
refined independently of λ (not inferred from the monochromator); `iterate()` leaves the
2θ₀/pose split where the recipe puts it, and `set_two_theta_0` re-expresses it only when called.

## Ring extraction: normalise strips per pixel, map back with the same geometry (2026-10)

`extract_ring_points` (`pyrs/calibration/pixel_pos_calibration.py`) turns raw detector images into
ring points. Two design rules, both verified on synthetic images rendered from a known geometry
(`tests/util/ring_calibration_helpers.py::synthetic_calibration_workspace`, HB2B_4971 layout,
pseudo-Voigt rings with FWHM 0.3°, Poisson counts):

1. **Each strip histogram is divided by the number of pixels (or vanadium counts) per 2θ bin.**
   Across a strip the pixel count per bin varies (curved rings, varying pixel density), so a plain
   sum of counts skews the peak and biases its centre. With a pixel efficiency falling 4× across the
   detector, passing the vanadium removes the resulting bias
   (`test_vanadium_normalisation_removes_efficiency_bias`).
2. **The fitted 2θ is converted back to a pixel column with the same geometry used to histogram.**
   Then a wrong starting geometry largely cancels: points extracted with the *nominal* geometry and
   wavelength agree with those from the true geometry to < 0.05 px. So extraction can start
   uncalibrated; re-extracting after refinement gives only a small gain.

Results on the synthetic 4971 run (15 strips, 11 sub-runs, ~6 s): all 13 ring observations found
(both Ni 311 and 222 at arms 93/97 — the draft and `FitCalibration` fit only the first peak),
points within 0.06 px RMS of the true rings with no bias; extract → `iterate(passes=3)` from the
nominal start reproduces the true 2θ map to 0.0008° and λ exactly (0.0005° after one re-extraction).

Other notes:

- **Rings are chosen per strip by a pixel margin from the side edges** (`side_margin`, default
  40 px), not per sub-run by a 2θ range: a curved ring can be near the edge in the corner strips
  only. The ring's crossing of each strip's centre row is *predicted* with the starting geometry
  and wavelength to decide whether to fit it, and the *fitted* point is checked again, because a
  wrong starting λ/2θ₀ moves rings by ~20 px (seen: predicted columns 80–86, fitted 86–104).
- **Fit windows are clipped to each strip's coverage** and need `min_window_side` (0.75°) of data
  on each side of the peak. A ring ~1° from the edge (one-sided window) is still extracted to
  < 0.07 px of the true ring, so clipping causes no measurable bias. Strips with identical windows
  (all of them, for rings well inside the detector) share one Mantid `FitPeaks` call, strips as
  spectra; for the 4971 layout nothing changed (same 195 points, 0.06 px RMS, ~6 s).
- *(Corrected 2026-10-29: these errors were wrong because strip fits passed σ² as σ, see
  "Strip fits handed Mantid σ² where it expects σ".)* **Mantid's centre uncertainty is unreliable at low signal**: fits to pure noise reported centre
  errors of 0.02–0.03° while the "rings" they found were 41–65 px off. A significance check (peak
  intensity ≥ `min_significance` = 5 × its error) removes them. Rings at 4 % contrast (height 2 on a
  background of 50) fail it — their points scattered by 0.5–5 px, against 0.06 px for normal rings.
- Outlier rejection per ring uses a **robust** (soft-L1) quadratic `index_0(index_1)`: a plain
  least-squares quadratic is dragged towards a single bad point, shrinking its residual so it
  escaped rejection (`test_reject_outliers_flags_far_point` failed before the change).
- Fully masked strips are not sent to the fitter (they produced divide-by-zero warnings) and are
  reported in the diagnostics as "band has no unmasked pixels".
- **Masks come from reduction's own path, not a second loader.** The instrument mask XML is given
  to `NeXusConvertingApp(nexus, mask_file)`, which stores it as the workspace's default mask
  (1 = keep); `extract_ring_points` uses that default mask (`apply_default_mask=True`), or an
  explicit `mask=` array instead. A calibration-side copy of the XML loading was written and then
  removed (user, 2026-10-21): `NeXusConvertingApp.__load_mask` is private and needs the converter's
  event workspace, so if a mask must ever be applied to an already-converted workspace, the fix is a
  public loader in core used by both, proposed separately.
- The ~290 `readX()/readY()` deprecation warnings in these tests come from the existing
  `PeakFitEngine._guess_center`, not from the extraction code.

## Ring extraction is two-pass: locate on all strips, then fit each strip (2026-10)

`PeakFitEngine._guess_center` starts every fit at (almost exactly) the **middle of the fit window**:
it takes the first moment of `y + |max(y)|`, and adding the maximum swamps the peak. Measured on
synthetic strips the guess was within 0.03° of the window centre whatever the data, and 0.27–0.34°
(about one FWHM) from the real peak. In extraction the window is centred on the ring *expected* from
the starting λ and geometry, so a wrong start put every fit's start off the ring. (This is existing
peak-engine code, also used by production fitting; not changed here — investigation tracked in #1023.)

`extract_ring_points` therefore runs two passes per sub-run and reflection, all inside the
calibration module: (1) **locate** — one fit on all unmasked strips combined (high statistics); its
centre (if significant) becomes the target; (2) **measure** — the side-edge prediction and every
strip's window are centred on that target, so each fit starts on the ring. `StripFit.start_center`
records the located value (NaN when pass 1 found nothing and the expectation was used).

Synthetic HB2B_4971 run (true λ 1.5308), one-pass (commit 41fba32f) vs two-pass:

| start λ | one-pass | two-pass |
|---|---|---|
| 1.5308, 1.540 | 195 pts, 13/13 rings, ≤ 0.17 px | 195 pts, 13/13, ≤ 0.16 px |
| 1.555 | 193 pts, one point 2.3 px off | 195 pts, 13/13, ≤ 0.16 px |
| 1.565 | crash (ZeroDivisionError) | 90 pts, 6/13, ≤ 0.14 px |
| 1.520 | crash | 47 pts, 7/13, one point 13 px off |
| ≥ 1.575, 1.505 | crash | 1–2 pts, 200–300 px off |

- Two-pass is clean up to an apparent ring offset of ~1° — real starts (mono setting 1.540 vs
  fitted 1.5308, 0.6 %) are well inside that.
- The one-pass crash: its significance used plain Python floats (`np.errstate` does not guard them)
  and a fit reported a zero intensity error. The shared helper divides NumPy arrays (inf/NaN, no
  crash).
- **Limitation, not yet guarded:** with a larger start error the expected ring leaves its ±2° window
  and the window captures a *different* reflection or a noise bump, accepted as a confident, wrongly
  labelled point. Per-ring outlier rejection cannot see it (the wrong ring is self-consistent). Next:
  a coarse global alignment of all expected reflections before pass 1, and a mis-assignment guard.


## A global ring alignment was tried and removed; first real-data observations (2026-10)

**What was removed and why.** To place fit windows when the starting λ/geometry is far off, an
alignment step grid-searched an effective wavelength plus an empirical 2θ offset (and then a
"scale") *added to the predicted ring position*. It worked on synthetic data, but it violates the
constitution in `CLAUDE.md`: a predicted ring position is the Bragg angle `2·asin(λ/2d)`, and
detector geometry (2θ₀, distance, …) is determined by the refinement, not by an empirical stand-in.
It was reverted (9d18115c). With a large starting error one window per expected ring can capture
a different reflection (documented above); this is handled by an assumption, not by code:

**Design assumption (user, 2026-10-22): the starting geometry is never badly wrong.** It is the
nominal geometry or the previous calibration, close enough that every ring lies inside its ±2° fit
window (HB2B_3510 from the nominal geometry: every ring located −0.24 to −1.09° from its Bragg
angle). The workflow is therefore **one extraction, then the refinement** — no alignment and no
extract → refine → re-extract loop. Ring points are meant to be geometry independent (the same
geometry bins the pixels and maps the fitted 2θ back), so re-extracting with a refined geometry
should not change them; when it does, the strip fits are unreliable (next entry), which is a fit
problem, not a geometry one.

**HB2B_3510 (real, `tests/data/calibration_tests`, mask `HB2B_Mask_12-18-19.xml`)** — same scan
layout as 4971 without the 82° sub-run (10 sub-runs, Ni/Fe, Si422 at 1.540 Å, no L2 log), and very
sparse: 5–9×10⁴ counts per sub-run, ~0.07 counts/pixel, at most 3–10 per pixel.

- With the *nominal* geometry the same reflection appears at a 2θ that depends on where it falls on
  the detector: Ni 220 at 75.73° (arm 76) / 76.03° (arm 80); Fe 211 at 81.36° / 81.51° / 81.91°
  (arms 78 / 80 / 86); Fe 220 at 98.06° / 98.38° (arms 99 / 103) — about 0.07° per degree of
  distance from the detector centre. That pattern is what a sample–detector distance error does
  (about 7 % short of 0.985 m); the GUI calibration of 4971 has Shift_z = −44 ± 31 mm and
  λ = 1.53078 Å.
- A refinement of those ring points (all 12 reflections on the detector, 179 points) left a
  systematic split by powder: every Ni ring +0.02 to +0.07° (mean +0.039°), every Fe ring −0.03 to
  −0.08° (mean −0.055°), at all arm angles. Open question: the Ni/Fe lattice-constant *ratio* (the
  draft's 3.523799 / 2.866398 differ from the production 3.526314 / 2.865579 by about the needed
  −0.1 %) or sample displacement (the powders sit at different `sy`). A lattice error scales with
  tan θ, displacement roughly with cos θ, so a refit can tell them apart.
- After `FullCalibration` every error except λ is flagged −2: the exact Shift_x/Rot_y/2θ₀
  degeneracy makes the covariance singular. Meaningful uncertainties need a final refinement with
  one member of the set held.
- `NeXusConvertingApp.convert()` of this 17 MB file peaks at ~3.3 GB of memory.


## Strip fits on sparse real data are unstable and their uncertainties are meaningless (2026-10)

On HB2B_3510 (~0.07 counts/pixel, 15 strips) re-extracting with a slightly different geometry —
which only re-bins the same counts — moved some ring points by 2–36 px, while the fitted centres
reported uncertainties of 0.0004–0.001°. Example, sub-run 10 Fe 220 (arm 103°, the faintest ring),
strip 7 over three geometries: 98.397 / 98.521 / 97.885° (±0.0007 / 0.0009 / 0.0005°) → columns
240.8 / 264.3 / 228.3. The 2θ→column conversion was verified exact (fitted − mapped = 0.0000°,
2θ monotonic along every row), so the jumps are the fits landing on different noise features.

- The reported Mantid centre uncertainty is ~1000× too small here, so `max_center_error`,
  `min_significance` and `RingPoints.sigma` do not detect these fits.
  *Correction (2026-10-29):* the cause was ours, not Mantid's. Strip fits were handed σ² as
  Mantid's σ; see "Strip fits handed Mantid σ² where it expects σ" below.
- Affected rings: Fe 220 (sub-run 10), Ni 222 (sub-runs 7, 8), Ni 311 (sub-run 8) — the weakest.
- The refinement results from such points are suspect; the per-strip counts are too low for 15
  strips. Open: measure point reliability from the data (geometry-perturbation test, scatter about
  each ring) and choose the strip count from it.

## Calibration lattice constants: the 2020 set, not the 2023 fitted values (2026-10)

`git log -S` on the values shows three sets, none citing a source:

| date / commit | Ni | Fe | Mo | note |
|---|---|---|---|---|
| 2020-04-29 `d6cd5e04` | 3.523799438 | 2.8663982 | 3.14719963 | `scripts/pyrs_calibration.py` materials table (with Ge 5.657568977, Si 5.4307); match common room-temperature reference values (Ni 3.5238, α-Fe 2.8664, Mo 3.1472 Å) |
| 2023-07-18 `7a6068b5` | 3.523421 | 2.866355 | 3.147664 | "refine detector calibration" |
| 2023-08-13 `47d1e558` | 3.526314 | 2.865579 | 3.147664 | "modified lattice based on newly calibrated data" — current `FitCalibration` |

The 2023 sets appear to have been fitted from HB2B data, which is circular for a calibration (only
λ/a is measured, and a fitted lattice constant also absorbs per-powder displacement). Their Fe/Ni
ratio (0.81263) is ~0.1 % off the 2020 ratio (0.81344) and off what HB2B_3510 shows (≈ 0.81335).

**Decision (user, 2026-10-22): `powder_standards.HB2B_STANDARDS` uses the 2020 set.** On
HB2B_3510 (5 strips, nominal start, `iterate(3)`): RMS 0.050° → **0.0175°**; per-powder mean
residual Ni/Fe +0.040/−0.056° → **−0.006/+0.009°**; λ 1.53122 → 1.53058 Å. With the 2020 values
the per-powder models (offset, lattice error, 2-direction sample displacement) no longer improve the
fit (ring-mean RMS 0.0155 → 0.014°), so there is no evidence for a sample displacement and it is not
added. The absolute λ rests on these lattice constants (only λ/a is measured); a certificate for the
actual Ni/Fe/Mo standards would be the final check. `FitCalibration` (production) still uses the
2023-08 set — changing it needs separate approval (#1022).

Remaining: sub-run-level ring-mean residuals of ~0.015° (≈ 4× their statistical error), largest
sub-run 4 Fe 211 (+0.035°) and sub-run 10 Fe 220 (+0.026°, the weakest ring); not explained by powder
or 2θ — possibly arm-positioning repeatability (untested).

## Ring-calibration driver, and why its files are not yet safe for reduction (2026-10)

`pixi run pyrs-pixel-calibration RUN.nxs.h5 -o CAL.json [--mask XML | --no-mask]`
(`scripts/pyrs_pixel_calibration.py`): NeXusConvertingApp (mask → default mask; default
`/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml`) → `extract_ring_points` (5 strips) →
`iterate(passes=3)` + polish → `estimate_errors()` with **2θ₀ held** (its error written as −1;
user decision 2026-10-22) → `write_calibration` (standard JSON via `write_calibration_to_json`).
On HB2B_3510 (19 s): 60 points / 12 rings, RMS 0.0175°, λ = 1.53058 ± 0.00023 Å,
Shift_z = −74.7 ± 1.4 mm, Shift_y = −2.0 ± 3.0 mm (weak), 2θ₀ = 1.455° (held).

**TTH_0 hazard — resolved (#1021, merged in #1031):** the file carries a non-zero TTH_0, which
`read_calibration_json_file` used to drop, placing the detector ~1.45° off in 2θ (Shift_x/Rot_y
applied without the 2θ₀ that balances them). Reduction now reads `TTH_0`, so these files load as
written.

**Still not for production use:** the refined geometry itself is in doubt — on HB2B_3510/3489 it needs
a ~7–8% larger pixel pitch or a ~75 mm shorter distance than surveyed, and a ~6° rotation of the
detection plane that the face survey cannot confirm or exclude (entries below). Do not reduce production data with these files until
the detector position response is measured (slit scan, `calibration-diagnostics` branch).

*Update (2026-10-28):* on the 2024 run HB2B_4971 the ring geometry is unremarkable. The detector
is nearly square-on (Rot_y +0.34°), and the geometry beats `FitCalibration` on every metric,
including `FitCalibration`'s own η-wedge residual (see the HB2B_4971 entries below). The 2023
anomalies above are unexplained. Whether the files are cleared for production is a user decision
and has not been made.

*Decision (user, 2026-10-30, #1045):* ring-based calibrations may become the production calibration.
The calibration GUI's Export writes them, like FitCalibration's, to `HB2B_Latest.json` when the
shared-calibration flag is ticked, and that flag is ticked by default for both methods (see the
#1045 entry below). The 2023 anomalies above remain unexplained.

## HB2B_3510: the ring spacing itself says the detector is ~7% "closer" than nominal (2026-10)

*The by-hand diagnostic scripts cited in this and the following entries (`tests/scripts/ring_overlay_figures.py`,
`calibration_head_to_head.py`, `two_theta_0_profile.py`, `pixel_pitch_at_nominal_distance.py`,
`constrained_pitch_fit.py`, `fixed_pitch_fit.py`, `joint_calibration_fit.py`, `geometry_1d_patterns.py`,
`srm640_detector_positions.py`) live on the `calibration-diagnostics` branch, with the pyFAI cross-checks and
the slit-scan simulation; they are not part of the calibration itself.*

The refined Shift_z ≈ −75 mm (and ~3–4.4° face tilt) is not physical (user, 2026-10-22), so it was
checked against the data without the refinement:

- **2θ₀ does not cause it.** Re-expressing the refined detector with `with_two_theta_0` (identical
  pixel 2θ/η) moves Shift_x by ≈ 16 mm and Rot_y by −1° per degree of 2θ₀, but Shift_z by only
  ≈ 1 mm: Shift_z stays −72…−75 mm for any 2θ₀ in [−0.5°, 1.5°]. Shift_x = −24 mm is entirely 2θ₀
  traded for a shift; Rot_y = 0 would need 2θ₀ ≈ 4.4° and Shift_x ≈ +70 mm.
- **The ring gap is short.** Sub-runs 7 and 8 each show Ni 311 and 222 (Δ2θ = 5.43° at λ 1.5306 Å).
  The extracted points are 299.0 / 295.6 px apart; the nominal geometry (L2 0.985 m, pixel
  0.29297 mm = 300 mm / 1024) puts them 319 px apart — ratio **0.937 / 0.926**. That is
  distance ÷ pixel pitch ≈ 7% below nominal: L2 ≈ 0.915 m (Shift_z ≈ −70 mm) *or* a pixel pitch
  ≈ 0.315 mm (a ~323 mm active width). Rings alone cannot separate the two. λ cannot explain it
  (1.5306 → 1.54 Å changes the ratio by 1.3%; a 7% change would need λ ≈ 1.43 Å).
- Overlay figures: `tests/scripts/ring_overlay_figures.py RUN CAL.json OUTDIR --mask XML` (nominal vs
  refined Bragg-ring contours on the masked detector image, plus the extracted points).

- **`FitCalibration` agrees on the distance.** Six runs on HB2B_3510 with the mask (fix/fitcalibration-mask
  worktree; default vs per-parameter `diff_step`; 2023 vs 2020 lattice; nominal vs perturbed start;
  `tests/scripts/fitcalibration_stepsize_study.py`) all end at **Shift_z = −69…−71 mm**, with
  Shift_x −5…−12 mm, 2θ₀ 0.46–0.79° and λ 1.5285–1.5295 Å. It does *not* reproduce the tilt:
  Rot_y −0.8…+0.2° (ring calibration at the same 2θ₀: ≈ 3.7°), so the tilt is unresolved.
  The 2020 lattice lowers its RMS 0.055 → 0.026°. The per-parameter step changed nothing material:
  run-to-run spread is the same in both modes, and its error estimates are unusable (Shift_z ± 7–44 mm,
  one all zero).

- **HB2B_3489 (mono 1.731 Å, 12 sub-runs) says the same.** Ring calibration (mask, 5 strips): 61 points,
  13 rings, RMS 0.036°, λ = 1.72077 ± 0.0003 Å. Head to head with HB2B_3510, both re-expressed at
  2θ₀ = 0 (`tests/scripts/calibration_head_to_head.py`):

  | at 2θ₀ = 0 | Shift_x | Shift_y | Shift_z | Rot_x | Rot_y | Rot_z | λ / mono |
  |---|---|---|---|---|---|---|---|
  | 3510 | −1.0 ± 0.3 | −2.1 ± 3.0 | −73.3 ± 1.4 | −0.02 ± 0.49 | 4.4 ± 0.9 | 0.18 ± 0.03 | 0.9939 |
  | 3489 | −5.9 ± 0.35 | 0.1 ± 3.7 | −78.8 ± 5.9 | −0.44 ± 1.03 | 7.9 ± 2.2 | 0.06 ± 0.07 | 0.9941 |

  (mm, deg; errors from each run's estimate, 2θ₀ held.) Shift_z, Shift_y, Rot_x, Rot_z agree within
  ~1.5σ. The 4.9 mm Shift_x difference is ≈ 0.3° of 2θ₀ — expected, since 2θ₀ includes the monochromator
  misalignment and the mono setting changed. Rot_y (the tilt) is large in both but poorly determined.
  Both refined wavelengths sit 0.6% below the mono setting. Two-ring gap in 3489 sub-run 12 (Ni 311/222,
  Δ2θ 7.37°): 398.8 px vs 432.9 px nominal, ratio **0.921** — the same ~7–8% short distance/pitch at a
  12% different wavelength, as a geometric (wavelength-independent) cause would give. (Sub-run 11's
  222 points from a re-extraction at the refined λ scatter 23 px and are spurious; the driver rejected
  them.)

- **2θ₀ cannot be refined from either run** (`tests/scripts/two_theta_0_profile.py`): holding 2θ₀ at
  −1…4° and refining the rest gives the same RMS to 5 digits (3510: 0.01746°, 3489: 0.03627°); only
  Shift_x (≈ −16 mm/deg) and Rot_y (−1°/deg) move. A joint fit of both runs with one shared pose and a
  2θ₀ and λ per run does measure the **difference: 2θ₀(3489) − 2θ₀(3510) = −0.299 ± 0.033°** (RMS
  0.0188 / 0.0370°, barely above the separate fits), i.e. changing the mono 1.54 → 1.731 Å turned the
  beam by 0.3°. The absolute value needs a second detector distance or an external reference.
- **Pixel pitch if the detector is at its nominal distance** (`tests/scripts/pixel_pitch_at_nominal_distance.py`,
  Shift_z = 0 held, common pitch scale k scanned): RMS minimum at k = 1.0804 (3510; pitch 0.3165 mm,
  RMS 0.01747° = the Shift_z-free fit) and k = 1.0869 (3489; 0.3184 mm) — **8.0–8.7% larger** than
  0.29297 mm (i.e. a 324–326 mm wide detector). The exact L/p scaling predicts 0.985 / 0.9117 = 1.0804.
  Rot_y is strongly coupled to k (3510: 1.8° at k = 1.07, 4.2° at 1.08, 8.9° at 1.09), so the tilt
  and the pitch/distance question are one problem. The vertical pitch (along the nearly straight rings
  near 90°) is barely constrained (ky 0.98–1.18 changes RMS ≤ 8%).
- **The illuminated area spans ~960 channels, not 1024.** Summed over all sub-runs, unmasked, the
  row profile (index_1, free of rings) crosses 50% of its centre level at channels 38–39 and 997 in
  both runs (959–960 channels, edges ~10 channels wide). If those edges are the 300 mm active area,
  the pitch is 300 / 960 = 0.3125 mm (k ≈ 1.067), which would leave Shift_z ≈ −13 mm (3510) /
  −18 mm (3489) instead of −73/−79 mm. Unconfirmed: the edges could be a frame or window shadow, or
  electronic edge compression; the vendor active area and the readout's channel mapping decide it.

- **Laser-scanner survey (user, 2026-10-23; corrected 2026-10-27):** the detector **housing face** is at
  the nominal distance and shows no rotation. **It does not measure the detection (wire) planes inside**
  (user, 2026-10-27): Shift_z refers to the face, with the wires an unknown depth behind it, and a rotation
  of the wire planes relative to the face is *not* excluded. Earlier statements below that "the survey
  excludes the tilt" apply to the face only.
- **The mask is exactly the dead border.** `HB2B_Mask_12-18-19.xml` keeps one rectangle, index_0 34–991
  (958 ch) × index_1 40–992 (953 ch); every masked pixel is within 40 channels of an edge and none lies
  inside that rectangle. Masked pixels get 0.16 counts/pixel against 1.58 kept (1.5% of all counts, the
  soft edge). The rectangle matches the illuminated area (50% edges 38–997 in index_1). If that
  rectangle is the 300 mm active area, the pitch is 300/955 ≈ 0.314 mm (k ≈ 1.07), not 300/1024.
- **Survey pose + one uniform pitch does not fully describe the rings**
  (`tests/scripts/constrained_pitch_fit.py`; Shift_z = 0, Rot = 0 held; Shift_x, Shift_y, 2θ₀, λ free;
  pitch scale optimised): nominal pitch RMS 0.24° (residuals +0.35° → −0.31° across the columns);
  best k = 1.0835 (3510) / 1.0874 (3489) / 1.0877 joint, i.e. pitch 0.317–0.319 mm, RMS 0.0231 / 0.0371°
  (free fit 0.0175 / 0.0363°) — but only with **Shift_x −77 to −101 mm and 2θ₀ 4.4–5.8°**: the old 4°
  "tilt" has moved into the Shift_x/2θ₀ swing, and those values are as unphysical as the tilt. The
  vertical pitch is unconstrained (ky 1.02–1.04, no RMS gain). So the data carry a left–right asymmetry
  in pixels per degree that a uniform pitch does not explain; a tilt of the face is excluded by the
  survey, a tilt of the wire planes is not (see the correction above). The
  two-ring ratios hint at it (3510: 0.926 at columns 246–541, 0.937 at 462–761), consistent with a
  pitch that varies across the detector (position non-linearity of the readout) — untested.
- **`FitCalibration` on HB2B_3489** (mask, 2020 lattice, nominal start; worktree code): per-parameter
  step → Shift_z −71.0 mm, Shift_x −17.5, Shift_y −7.3 mm, Rot −0.90/0.22/−0.09°, 2θ₀ 0.83°,
  λ 1.7199 Å, RMS 0.038° — the same −70 mm distance as on 3510. Default step went astray: CalibrateShift
  stopped at Shift_z +0.4 mm, CalibrateRotation drove Rot_z to −4.5° and the polish ended at Rot_z −5.0°
  (on its ±5° bound), RMS 0.059°, error estimate all zero. So the default step is not just noisy: on
  3489 it changed the answer. A per-parameter `diff_step` for `FitCalibration` is a production change
  and still needs approval.
- **Fixed pitch 0.314 mm** (mask-implied; `tests/scripts/fixed_pitch_fit.py`, outputs in
  `calib_diagnostics/fixed_pitch_*.txt`). Free fit: same RMS as at 0.293 mm (the L/p scaling) but
  **Shift_z −7.8 mm (3510) / −13.7 mm (3489)** instead of −73 / −79 mm — close to the survey. The tilt is
  unchanged (Rot_y 4.38° / 7.91° at 2θ₀ = 0): the pitch fixes the distance, not the asymmetry. With the
  survey pose (Shift_z = 0, Rot = 0) RMS 0.0323° (3510, free 0.0175°) and 0.0461° (3489, free 0.0363°),
  2θ₀ 1.07° / 1.37°, Shift_x −19 / −29 mm; the residuals keep a left–right gradient, +51 → −24 mdeg
  (3510) and +43 → −17 mdeg (3489) across the columns — what Rot_y absorbs in the free fit. Separate
  mask-edge pitches (0.3132 × 0.3148 mm) fit slightly worse. With Rot_y fixed at 0 (face survey), 2θ₀ is
  refinable, but its value carries whatever causes that gradient.
- **Rot_y = 2θ₀ + Rot_y_0 (user, 2026-10-23)** — the degeneracy in the 2θ₀ = 0 frame, with its shift
  part Shift_x = Shift_x_0 − L·sin 2θ₀ (verified: 3510 at 0.314 mm, Rot_y 4.375° ↔ ΔShift_x −74.5 mm =
  0.977 m · sin 4.375°). With the survey's Rot_y_0 = 0, the free fits would read 2θ₀ = 4.4° (3510) / 7.9°
  (3489) — but then Shift_x_0 = −76 / −140 mm, and the two runs' 2θ₀ would differ by 3.5°, against the
  joint fit's −0.30 ± 0.03°. Holding the whole surveyed pose (Shift_x = Shift_z = 0, Rot = 0; pitch
  0.314 mm; Shift_y, 2θ₀, λ free): **2θ₀ = −0.04 ± 0.03° (3510), −0.34 ± 0.03° (3489)**, difference
  −0.30° — matching the independent joint-fit difference. RMS 0.033 / 0.048°, same left–right residual
  slope (+53 → −21 mdeg). So the free fit's Rot_y is not 2θ₀: ~4–8° of it is the unexplained asymmetry.
  (User, 2026-10-23: the detector is not shifted sideways by ~75 mm, so Shift_x_0 ≈ 0 and the
  surveyed-pose fit is the physical one. Also user: |2θ₀| < 0.5° on HB2B; the surveyed-pose values
  −0.04° / −0.34° satisfy it, the free-fit readings 4.4° / 7.9° do not.)
- **Joint fit, shared physical geometry (user's model, 2026-10-23):** one pose for 3510 and 3489
  (Shift_x/y/z, Rot_x, Rot_y_0, Rot_z), a 2θ₀ and λ per run, each run seeing Rot_y = 2θ₀ + Rot_y_0;
  2θ₀ bounded to ±0.5°, Rot_y_0 free (`tests/scripts/joint_calibration_fit.py`). Pitch 0.314 mm:
  Shift_x −3.5, Shift_y 0.4, **Shift_z −9.9 ± 2.4 mm**, Rot_x −0.27, **Rot_y_0 5.82°**, Rot_z 0.12°;
  2θ₀ +0.14 (3510) / −0.16° (3489), λ 1.53052 / 1.72080 Å; RMS 0.0188 / 0.0370° (separate free fits
  0.0175 / 0.0363°). (Pitch 0.293 mm: identical except Shift_z −75.2 mm.) The Jacobian is singular
  (condition 2e8): Rot_y_0 + δ with both 2θ₀ − δ fits identically, so only the difference
  2θ₀(3489) − 2θ₀(3510) = −0.30° is measured and the reported ±0.38° are not meaningful. Within the
  ±0.5° bound, δ ∈ [−0.36, +0.34]°, so **Rot_y_0 ∈ [5.46, 6.16]°**: with |2θ₀| < 0.5° the rings demand a
  ~5.5–6.2° rotation of the detection plane (the face survey does not see the wire planes). One shared geometry does describe both runs
  (RMS barely above the separate fits).
- **1D patterns under three geometries** (`tests/scripts/geometry_1d_patterns.py`, figures in
  `calib_diagnostics/patterns_1d/`): unrefined (nominal pose, 2θ₀ = 0, 0.293 mm, mono λ) vs the joint
  refinement at 0.293 mm and at 0.314 mm. The two refined geometries give **identical** patterns
  (peak centres equal to 1e-4°): distance and pitch trade exactly, so 1D data cannot choose between them.
  Refined peaks sit within ±0.05° of Bragg on 3510 (−0.047…+0.014°) and within −0.04…+0.07° on 3489
  (one weak 222 ring at −0.13°). Unrefined peaks are −0.3…−1.0° from the mono-λ Bragg angle, mostly
  because of λ (mono 1.54 vs refined 1.5305 Å moves Bragg by ~0.6–0.9°); the peaks themselves move by
  only ±0.3° between geometries, with the sign depending on where the ring falls on the detector (the
  pixel-scale error). Refined FWHM in degrees is ~8–12% larger than unrefined, mostly because the
  refined geometry has ~8% more degrees per pixel; widths in degrees are not comparable across geometries
  with different scales.
- **Same joint fit and 1D patterns with the 2023 lattice** (`--lattice 2023`; `calib_diagnostics/patterns_1d_2023/`):
  Rot_y_0 7.02° (2020: 5.79°), Shift_z −76.1 mm at 0.293 mm / −10.8 mm at 0.314 mm, 2θ₀ +0.26 / −0.19°,
  λ 1.53113 / 1.72364 Å. The 1D peaks split by powder: peak − Bragg averages Ni +0.022° / Fe −0.081°
  (3510) and Ni +0.057° / Fe −0.025° (3489), a ~0.08–0.10° Ni–Fe split; with the 2020 lattice the
  same averages are Ni −0.025° / Fe −0.016° and Ni +0.029° / Fe +0.015° (split ≤ 0.015°). Confirms the
  2020 constants; the geometry conclusions (pitch/distance, ~6–7° rotation-like asymmetry) do not
  depend on the lattice set.
- **The HB2B wavelengths are one Si monochromator at a fixed take-off.** Nominal settings (user,
  2026-10-23): Si 333 / 511 1.452, 422 1.540, 331 1.731, 400 1.886, 311 2.275, 220 2.667 Å all give
  2θ_M = 2·asin(λ / 2d_Si) = 87.97–88.00°, so λ ratios between settings equal Si d-spacing ratios.
  Refined λ(3489)/λ(3510) = 1.12432 with the 2020 lattice (2023: 1.12573) against d331/d422 = 1.12390:
  0.04% apart (2023: 0.16%) — independent support for the 2020 constants. The refined λ correspond to
  2θ_M ≈ 87.32° (3510) / 87.36° (3489), i.e. the 0.6% λ deficit would be a ~0.66° take-off offset
  (or a lattice-constant offset; only λ/a is measured). Per the user, 2θ₀ is *not* inferred from this.
  Open tension: a fixed take-off suggests one beam direction for all settings, yet the joint fit gives
  2θ₀(331) − 2θ₀(422) = −0.30 ± 0.03°.
- **Lattice set Ni 3.52942 / Fe 2.8674 tried (user, 2026-10-23) — rejected by the data.** Joint fit at
  0.314 mm: RMS 0.077 / 0.138° (3510 / 3489) against 0.052 / 0.061° (2023) and **0.019 / 0.037° (2020)**;
  1D peaks split Ni − Fe by +0.155° in both runs (2020: −0.009 / +0.014°). The geometry runs away to absorb
  the mismatch (Rot_y_0 −7.7°, Shift_y +5.5, Shift_z −22 mm) and 3489 gains spurious points (one −0.43°).
  The rings test only a_Ni/a_Fe (one λ per run); the data want ≈ 1.2294 (the 2020 ratio, ±~1e-4), the
  proposed set has 1.2309. Consistent alternatives: Fe 2.8674 → Ni ≈ 3.5251; Ni 3.52942 → Fe ≈ 2.8709.

Open: the detector's position response (mm per channel across the face, and where the 300 mm active area
falls in channel space) — the masked-grid flood measurement decides it.
## FitCalibration ignored its mask file and then crashed (2026-10, #1024)

`FitCalibration(nexus_file=..., mask_file=...)` stored `mask_file` but called
`_reduce_diffraction_data(nexus_file)` without it, so `NeXusConvertingApp` ran unmasked; it then
passed the mask **path** to `ReductionApp.reduce_data(mask=...)`, which treats a string as the ID of
an already-loaded mask, so `HB2BReductionManager.get_mask_vector` raised `ValueError`. Any masked
production calibration (`scripts/pyrs_calibration.py` with `"mask"` in its JSON) crashed; the UI
never passes a mask, so UI calibrations are unmasked.

Fix (user-approved production change): pass `self.mask_file` to the conversion, which stores it as
the workspace's default mask, and call `reduce_data(mask=None)` — with `mask=None` the reduction
manager uses the default mask, and the per-evaluation `reduce_data()` only clears the η-wedge masks,
not the default mask. Unmasked calibrations take the identical code path (the existing
`test_peakfit_calibration` gold-file tests are unchanged); `test_mask_file_is_applied` failed with
the original `ValueError` before the fix. Open: whether the UI should offer the instrument mask.

## `FindBackgroundSigma` is ignored by Mantid `FitPeaks` (2026-10)

The Mantid in this environment logs `FitPeaks property "FindBackgroundSigma" is deprecated and will
be ignored` for every call. `MantidPeakFitEngine.fit_peaks` passed `FindBackgroundSigma=1`; it was
removed (user-approved). Verified: `FitPeaks` on the same synthetic spectra gives bit-identical
fitted parameters with and without the property, for `HighBackground=True` and `False`; peak-fitting
and FitCalibration tests unchanged.

## The calibration GUI applies the production detector mask by default (2026-10)

`DetectorCalibrationModel` never passed a mask, so GUI calibrations ran unmasked even after #1024.
It now uses `/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml` when that file exists and no mask
otherwise (user decision, 2026-10-26), resolved when a run is loaded, and passes it to *both* places
it builds `FitCalibration` — changing the 2θ/η bins rebuilds the object and used to drop any mask.
`DetectorCalibrationModel.mask_file` reports which was used. On machines without `/HFIR` (CI) the
behaviour is unchanged. Unit tests: `tests/unit/pyrs/interface/detector_calibration/`.

## `HidraWorkspace` stores 2θ and intensities as float64, never the input's dtype (2026-10)

`set_reduced_diffraction_data` used to create the 2θ matrix and the intensity matrix with
`intensity_array.dtype`. Integer intensities (e.g. an unweighted `np.histogram`) therefore truncated
**2θ** to whole degrees (75.79° → 75°) without warning, and float32 rounded it (~1e-5° at 100°); a
mask added later took its first sub-run's dtype, so later float intensities into an integer array
were truncated too. Both matrices are now always float64 (both branches). Every caller at the time
(reduction, `FitCalibration`, the ring calibration, tests) already passed float64, so results are
bit-identical; regression tests in `tests/unit/pyrs/core/test_workspaces.py`.

## Calibration files' 2θ offset (`TTH_0`) is now read by reduction (2026-10, #1021)

`DENEXDetectorShift.convert_to_dict` (and so `FitCalibration` and every calibration file) writes the
2θ offset as **`TTH_0`**, but `read_calibration_json_file` — the path reduction uses, via
`PowderPattern` — read **`two_theta_0`**, and `check_calibration_dictionary` filled the missing key with
0. Reduction therefore applied **2θ₀ = 0 for every calibration file**, whatever the calibration found.
(NXstress files and `FitCalibration.get_archived_calibration` already used the right key.)

Fix (user-approved): read `TTH_0` / `error_TTH_0`; accept the legacy `two_theta_0` /
`error_two_theta_0`; raise `ValueError` if a file gives both with different values; no key → 0 with
error −1, as before; `TTH_0` without `error_TTH_0` still loads (error −1).

**Effect on results:** files with `TTH_0 = 0` — every file in this repository — are bit-identical. A file
with `TTH_0 = δ` now shifts every reduced pattern by δ in 2θ (δ on the detector's centre row, 0.982–1.000 δ
over the face at arms 60–120°), an absolute d-spacing / strain change of (δ/2)·cot θ: for δ = 0.1°,
1511 / 873 / 504 µε at 2θ = 60 / 90 / 120°. `FitCalibration` refines TTH_0 jointly with the shifts and
rotations (`wavelength_tth0`, `full` stages), so such a file describes a geometry that reduction applied
only partly; the fix applies what the calibration found. Strains relative to a d₀ reduced with the same
file largely cancel; absolute d-spacings and lattice parameters do not.

**Audit the archive before relying on old reductions:** `tests/scripts/audit_calibration_tth0.py`
(default `/HFIR/HB2B/shared/CALIBRATION`) lists every calibration file's 2θ offset and the strain change.
Regression tests: `tests/unit/pyrs/utilities/test_calibration_file_io.py`.
## `FitCalibration` uses per-parameter finite-difference steps (2026-10, #1030)

`FitCalibration`'s residual is not smooth on small scales: every evaluation re-bins the data and re-fits
the peaks, leaving a noise floor of ~1e-3° per residual. Its stages call `least_squares(jac="3-point")`,
whose default relative step is ε^(1/3) ≈ 6e-6 (6 µm, 6e-6°, ~9e-6 Å) — deep in that noise. A step scan on
HB2B_3510 (mask applied) found derivatives stable only from ~1 mm (Shift_z), ≥ 3e-4 (TTH_0) and above
~3e-3° (rotations); Shift_y was noise at every step.

Every stage now passes `diff_step = DIFF_STEP[free]` (`mantid_peakfit_calibration.DIFF_STEP`, user-approved):
5e-4 for the shifts, 1e-2 for the angles, 1e-4 for λ. scipy's absolute step is `diff_step · max(1, |x|)`,
so 0.5 mm for the shifts (always < 1 m), 0.01° for angles below 1° (growing with the angle above it, e.g.
0.05° at the ±5° bound), ~1.5e-4 Å for λ; it follows the current value during a refinement (user decision:
keep scipy's relative `diff_step`, not fixed absolute steps).

**Effect** (full recipe, 2020 lattice, mask; `tests/scripts/fitcalibration_stepsize_study.py`):
HB2B_3510 RMS 0.0261 → 0.0259°, pixel 2θ change ≤ 0.042° with TTH_0 applied; HB2B_3489 changes from a
failed fit (Rot_z on its −5° bound, RMS 0.059°, error estimate all zero) to RMS 0.038° with rotations
−0.90 / 0.22 / −0.09° (pixel 2θ up to 1.2° different). Run time similar or shorter. The gold-file tests
(1–3 evaluations) still pass (max difference 6.8e-4 against a 5e-3 tolerance). The error estimates still
come from the Jacobian and remain unreliable. Tests: `tests/unit/pyrs/calibration/test_fitcalibration_diff_step.py`.

## Detection-plane tilt and the horizontal/vertical pixel pitch (2026-10-27)

The laser survey covers the housing face, not the wire planes (correction above), so the ~6° asymmetry
may be a real rotation of the detection plane. Ring points of HB2B_3510 + HB2B_3489 (mask, 5 strips, 2020
lattice; `calib_diagnostics/ring_points_3510_3489.npz`); scripts on `calibration-diagnostics`
(`tests/scripts/vertical_pitch_profile.py`, `horizontal_pitch_profile.py`, `joint_two_theta_0_locked.py`).

* **Vertical pitch is weakly constrained** (horizontal 0.314 mm, vertical scanned): best 0.301 ± 0.014 mm
  (3510) / 0.295 ± 0.018 mm (3489) with the free pose (survey pose 0.302 ± 0.027 / 0.293 ± 0.023 mm); RMS
  changes ≤ 5% over 0.290–0.338 mm. Rings near 90° are nearly straight, so the vertical scale only enters
  through their curvature. A ~1σ hint of py < px. The mask rectangle (958 × 953 ch) would give the
  opposite (0.3132 × 0.3148 mm for a square 300 mm area).
* **Horizontal pitch with the vertical locked at 0.29297 mm** (direct least squares):
  - surveyed face pose (Shift_z = 0, Rot = 0; Shift_x, Shift_y, 2θ₀, λ free): px 0.3176 ± 0.0005 / 0.3178
    ± 0.0007 mm, RMS 0.0231 / 0.0372°, but only with 2θ₀ ≈ +4.4° and Shift_x ≈ −80 mm — against
    |2θ₀| < 0.5° and no sideways shift, so an untilted plane with a uniform pitch is excluded;
  - 2θ₀ = 0 (removes the 2θ₀ / Shift_x / Rot_y degeneracy), Shift_z = 0: px 0.3170 ± 0.0004 / 0.3173 ±
    0.0007 mm, Rot_y 6.14 / 6.50°, RMS 0.0182 / 0.0361° (free pose 0.0173 / 0.0359°); with Shift_z free px
    is undetermined (0.306 ± 0.015 / 0.313 ± 0.021 mm: pitch and distance still trade);
  - both runs at 2θ₀ = 0 with one detector: 3510 RMS doubles (0.018 → 0.036°) — rejected, the two mono
    settings differ in beam direction.
* **Joint fit, converged** (one detector; 3510's 2θ₀ = 0 locked, 3489's offset free; λ per run; vertical
  pitch 0.29297 mm; horizontal pitch fitted as a diagnostic; Shift_z held, not refined with the pitch —
  user, 2026-10-27): the same solution from Rot_y starts 0–9°:

  | | Shift_z = 0 (face) | Shift_z = +10 mm (wires behind the face) |
  |---|---|---|
  | horizontal pitch | **0.3171 ± 0.0006 mm** | 0.3203 ± 0.0006 mm |
  | Rot_y (detection plane) | **+5.95 ± 1.12°** | +5.94 ± 1.12° |
  | Rot_x / Rot_z | −0.29 / +0.13° | −0.29 / +0.13° |
  | Shift_x / Shift_y | −1.7 / +0.4 mm | −1.8 / +0.4 mm |
  | 2θ₀(3489) − 2θ₀(3510) | −0.300 ± 0.033° | −0.300 ± 0.033° |
  | λ 3510 / 3489 | 1.53095 / 1.72129 Å | 1.53100 / 1.72134 Å |
  | RMS 3510 / 3489 | 0.0187 / 0.0366° | 0.0188 / 0.0366° |

  It fits as well as the fully free geometry. The horizontal pitch scales with the window-to-wire depth
  (+0.32% per mm). **Earlier fixed-pitch scans started at Rot_y = 0 were not converged** (they stopped at
  Rot_y 0.3–6.8° depending on the start: the cost is flat along a Rot_y–pitch valley), and the staged
  "Shift_z = 0, rotations free" fits (px 0.3178 / 0.3183 mm, Rot_y 6.95 / 7.26°) are superseded.
* **Reading:** a coherent geometry — face at the surveyed distance, beam aligned (−0.30° between the mono
  settings), no sideways offset — with the wire plane rotated ~6 ± 1° and a horizontal pitch ~8% above
  0.293 mm. A 6° plane over 300 mm means ~31 mm of depth variation: check against the DENEX gas-volume
  depth and wire mounting. A horizontal-only pitch gradient would mimic part of the tilt; a horizontal-slit
  scan (vertical scale against column; constant for a pitch gradient, ±1.6% across the face for a 6° tilt)
  separates the two.

## HB2B_4971 (2024): ring calibration against FitCalibration and the GUI calibration (2026-10-28, #1018)

HB2B_4971 (2024-10-30, cycle C510, "with RC" — radial collimator; Si 422, 1.54 Å, 11 sub-runs of Ni and Fe,
600–700k counts each, ~10× HB2B_3510). Mask `HB2B_Mask_12-18-19.xml`, 2020 lattice for the ring calibration.
Scripts on `calibration-diagnostics`: `score_calibrations.py`, `geometry_1d_patterns.py --calibration`,
`ring_overlay_figures.py`; outputs in `calib_diagnostics/hb2b_4971/`. The planned
`scripts/compare_calibrations.py` was replaced by these (user decision).

* **Ring calibration:** 26 s, 64 points / 13 rings, RMS 0.0065°, λ 1.53129 ± 0.00008 Å (not at a bound),
  Shift_z −44.25 ± 0.26 mm. **FitCalibration** (full recipe, mask, per-parameter steps, `max_nfev` 40):
  ~30 min; 2020 lattice η-wedge RMS 0.0166°, 2023 lattice 0.0577° with Rot_x −3.2° and Shift_z −51 mm (the
  2023 constants distort it again). GUI file: Shift_z −44.4 mm, rotations not refined.
* **Scores** (pixel 0.293 mm; geometries compared as objects, re-expressed at 2θ₀ = 0):

  | geometry | ring-point RMS | strip drift mean \|d\| | η-wedge RMS (2020 / 2023 lattice) | whole-detector 1D peak − Bragg: RMS / spread | Shift_z / Rot_y at 2θ₀ = 0 |
  |---|---|---|---|---|---|
  | none | 0.717° | 0.039° | 0.662 / 0.649° | 0.689 / 0.162° | 0 / 0 |
  | GUI | 0.0229° | 0.040° | 0.0231 / 0.0600° | 0.023 / 0.007° | −44.4 mm / 0.66° |
  | FitCalibration 2023 | 0.0381° | 0.053° | 0.0246 / 0.0577° | 0.034 / 0.023° | −51.3 mm / 1.09° |
  | FitCalibration 2020 | 0.0222° | 0.042° | **0.0167** / 0.0604° | 0.018 / 0.016° | −44.2 mm / 1.33° |
  | ring | **0.0065°** | **0.0040°** | 0.0251 / 0.0624° | **0.008 / 0.004°** | −44.3 mm / 0.36° |

* **#1018 success criteria:** λ not at a bound — met; strip drift flattened — met (10×); η-wedge residual
  ≤ FitCalibration's — **not met** (0.0251 vs 0.0167°). Each method wins its own metric; the independent
  whole-detector 1D peak positions favour the ring geometry (RMS 0.008 vs 0.018°, 4× less scatter, equal
  FWHM). The two disagree by up to 0.16° in pixel 2θ (mostly Rot_y and λ), with the same distance.
* **Geometry differs from 2023:** the 2024 detector is nearly square-on (Rot_y +0.34° with 2θ₀ = 0 and the
  distance free, against ~6° for HB2B_3510/3489) at Shift_z −44 mm (against −75 mm). With the vertical pitch
  locked at 0.29297 mm, 4971's statistics begin to separate distance and pitch: horizontal 0.302 ± 0.006 mm,
  Shift_z −13.7 mm, RMS 0.0064°; forcing Shift_z = 0 gives 0.3083 mm but RMS 0.0095°. Horizontal/vertical
  pitch ratio (distance-independent) 1.03 ± 0.02 (2023: 1.04 ± 0.05). Open: whether the detector was moved,
  serviced or re-tuned between cycles C501 and C510, and whether a radial collimator was used in 2023.

## η-wedge (texture) reductions share one 2θ axis: every wedge but the last is mislabelled (2026-10-28)

Found while explaining why the ring geometry scores worse than FitCalibration on FitCalibration's own
η-wedge residual (#1018). **A production bug in PyRS reduction — reported, not fixed.**

**Mechanism.** `ReductionManager.reduce_sub_run_texture` (used whenever `eta_step` is given) bins each η wedge
over the 2θ range of *that wedge's own pixels* (`generate_2theta_histogram_vector(..., mask_array=eta_mask)`),
then calls `HidraWorkspace.set_reduced_diffraction_data(sub_run, "eta_<c>", ...)`, which keeps **one 2θ axis
per sub-run** (`self._2theta_matrix[spec_id] = two_theta_array`). Each wedge overwrites it, so every wedge is
returned with the axis of the last one written (order 0, +3, +6, −3, −6 → η = −6°), and the intensities of the
other wedges sit on the wrong 2θ values. Wedge ranges differ by only 0.01–0.06°, but the bins are matched by
index, so peaks move by `x_last + (x − x_own)·(range_last / range_own) − x`.

**Size** (HB2B_4971, ring geometry, 512 bins, 3° wedges; `tests/scripts/eta_wedge_axis_check.py` on
`calibration-diagnostics`): wedge peak shifts +35 … −66 mdeg, RMS 25.6 mdeg, largest at high 2θ; the
prediction matches the observed shifts to 0.11 mdeg RMS. ≈ 40–500 µε of η-dependent apparent strain.

**Ruled out first** (`tests/scripts/peak_asymmetry_study.py`): the peaks are asymmetric (broader high-angle
side at every 2θ, split-PV width ratio 1.0–1.4, no sign change near 90°), but Gaussian ±1°, Gaussian + linear
±3° and pseudo-Voigt ±3° give the same centres within a few mdeg with no 2θ trend; normalising the wedge
histograms by pixel count (the reduction sums counts unless a vanadium is given) changes centres by < 2 mdeg.
On correctly-binned wedges the ring geometry's wedge residual is ~10 mdeg RMS and flat in η.

**Affects:** `FitCalibration` (its residual is these wedges, so every FitCalibration/GUI calibration was refined
against mislabelled patterns — this is why FitCalibration and the ring calibration differ by up to 0.16° in
pixel 2θ on HB2B_4971, and why #1018's "η-wedge residual ≤ FitCalibration's" criterion failed); any
η-resolved/texture reduction (`ReductionApp.reduce_data(eta_step=...)`, e.g. `reduce_texture.py`) and the
project files written from them (one 2θ matrix per sub-run). Not yet checked: reductions with several
user masks (ROIs), which go through the same `set_reduced_diffraction_data`.
## η-wedge (texture) reductions use one 2θ range per sub-run (2026-10, #1033)

`ReductionManager.reduce_sub_run_texture` binned each η wedge over the 2θ range of *its own* pixels when no
`min_2theta` / `max_2theta` was given, but `HidraWorkspace.set_reduced_diffraction_data` keeps **one 2θ axis per
sub-run**, which each wedge overwrote: every wedge but the last written (order 0, +3, +6, −3, −6 → η = −6°)
was returned on another wedge's axis. Wedge ranges differ by only 0.01–0.06°, but bins are matched by index,
so wedge peaks moved by up to 66 mdeg (RMS ~26 mdeg on HB2B_4971 and HB2B_3510; ≈ 40–500 µε varying with η).
This biased `FitCalibration` (its residual is these wedges) and every η-resolved/texture reduction without an
explicit 2θ range.

Fix (user-approved): when no range is given, take **one range per sub-run from the pixels the default detector
mask keeps** (`workspace.get_detector_mask(is_default=True)`), and bin every wedge on it. A user ROI does not
change the bins; callers that pass a range are unchanged. Effect: the four affected wedges move to their
correct 2θ (up to ±66 mdeg); the last wedge changes only by re-binning (≤ 0.5–1.6 mdeg). `FitCalibration`'s
gold file `HB2B_mantid_calib.json` changes only in its error estimates (`error_TTH_0` 1.775 → 1.449°,
`error_Lambda` 0.0352 → 0.0261 Å; all parameters unchanged at 1 evaluation per stage). Regression test:
`test_eta_wedges_are_binned_on_the_stored_axis` (`tests/integration/test_texture_reduction.py`). Not yet
checked: reductions with several user ROI masks, which share the same per-sub-run 2θ storage.

## HB2B_4971 re-scored with correctly binned η wedges (2026-10-28, #1018 after #1033/#1034)

FitCalibration re-run on HB2B_4971 with the single-2θ-range reduction (mask, per-parameter steps,
`max_nfev` 40, ~40 min each); all geometries re-scored (`calib_diagnostics/hb2b_4971/fixed_wedges/`).

| geometry | η-wedge RMS 2020 / 2023 lattice | ring-point RMS | strip drift mean \|d\| | 1D peak − Bragg RMS / spread | λ |
|---|---|---|---|---|---|
| GUI | 0.0246 / 0.0576° | 0.0229° | 0.040° | 23.1 / 6.8 mdeg | 1.53078 Å |
| FitCalibration 2023, corrected wedges | 0.0225 / **0.0565°** | 0.0210° | 0.034° | 21.8 / 5.9 mdeg | 1.5315 Å |
| FitCalibration 2020, corrected wedges | 0.0123 / 0.0600° | 0.0151° | 0.022° | 9.5 / 9.4 mdeg | 1.5313 Å |
| ring | **0.0091** / 0.0581° | **0.0065°** | **0.004°** | **7.8 / 4.2 mdeg** | 1.53129 Å |

* **#1018's η-wedge criterion is now met:** the ring geometry scores 0.0091° on FitCalibration's own metric,
  below FitCalibration's best (0.0123°, its own optimum 0.0121°). FitCalibration therefore stops short of the
  minimum of its own cost — its corrected-2020 result has Rot_x −1.78° and Rot_y 3.70° at 2θ₀ = 0 (ring 0.36°)
  — consistent with its noisy, re-reduce-every-step residual (#1030). Its λ now matches the ring calibration.
* **The 2023 lattice constants make every geometry fit equally badly on the wedges** (0.056–0.060°): they cannot
  describe the Ni/Fe ratio, so the residual is lattice error, not geometry (#1022).
* All three #1018 success criteria are met (λ not at a bound; strip drift flattened 10×; η-wedge residual below
  FitCalibration's).

## Strip fits handed Mantid σ² where it expects σ (2026-10-29, #1019 review)

**Cause.** `_fit_band_histograms` histograms each band per pixel (counts / efficiency) and stored
`max(summed, 1) / norm²`, i.e. **σ²**, in `HidraWorkspace.set_reduced_diffraction_data`'s
"variances" argument. Despite its name, that slot goes straight to Mantid as `DataE`
(`mantid_helper.generate_mantid_workspace`), and `DataE` is σ. Production reduction already stores σ
there (`reduce_to_2theta_histogram`: `data_var = np.sqrt(...)`, and `unp.std_devs` with vanadium).
`FitCalibration` takes the reducer's output, so **production peak fitting is not affected**; only
the ring extraction was. Because the normalised σ ≪ 1, E was far too small and the weights were
1/σ⁴. Mantid does not rescale parameter errors by χ², so nothing looked wrong.

**Effect (before the fix).**
- Reported centre errors were **20–200× too small**, and significances 20–65× too large (synthetic
  Poisson strips, reviewer's test). The regression test
  `test_center_error_matches_scatter` measured a ratio of reported error to actual scatter of
  0.06; it is now within [0.5, 2] at 0.1 and 1 count/pixel.
- `max_center_error` (0.05°) and `min_significance` (5) therefore rejected almost nothing. The
  1/σ⁴ weighting made sparse-strip centres about 1.5× noisier.
- This is the likely cause of two earlier observations: "Mantid centre uncertainty ~1000× too
  small" (sparse-strip entry) and "fits to pure noise reported 0.02–0.03°" (extraction entry).
  Both were blamed on Mantid or sparse data at the time.

**Fix.** Pass `sqrt(max(summed, 1)) / norm`. The gate defaults are unchanged so far. With honest
errors they now act as designed, so re-tune them only if real runs show a need.

**Re-runs with the fix** (driver defaults, mask `HB2B_Mask_12-18-19.xml`; outputs in
`calib_diagnostics/hb2b_4971/fixed_sigma/` and `calib_diagnostics/sigma_fix/`):

| run | ring points | ring RMS | λ (Å) | ring σ median | pixel 2θ change vs before |
|---|---|---|---|---|---|
| HB2B_4971 | 64 → 65 | 0.0065° | 1.53129 → 1.53135 | 0.00025 → 0.0038° | +5.7 mdeg mean, ≤ 12.6 mdeg |
| HB2B_3510 | 60 → 60 | 0.0175 → 0.0157° | 1.53058 → 1.53056 | — | — |
| HB2B_3489 | 61 → 59 | — → 0.0301° | 1.72077 → 1.72120 | 0.00016 → 0.0071° | +23 mdeg mean, ≤ 52 mdeg |

On HB2B_4971, scored as in the entries below (corrected η wedges, 2020 lattice, 65 new ring points):

| geometry | η-wedge RMS (FitCalibration's metric) | ring-point RMS | strip drift mean / max \|d\| |
|---|---|---|---|
| FitCalibration (2020, corrected wedges) | 0.0123° | 0.0138° | 0.021 / 0.039° |
| ring calibration, before the fix | 0.0091° | 0.0073° | 0.0051 / 0.0116° |
| **ring calibration, with the fix** | **0.0075°** | **0.0065°** | **0.0033 / 0.0102°** |

- The fix improves the ring calibration on every metric, including FitCalibration's own. #1018's
  conclusion stands, by a wider margin.
- On HB2B_3489 the optimiser ends at a stage bound both before (Shift_x −50 mm) and after (Rot_y
  +5.000°). That is the Shift_x/Rot_y/2θ₀ degeneracy line running into the box, not this fix.
  *Since 2026-10-29:* `refine` flags any parameter left at a bound (`least_squares`' `active_mask`)
  with error -2 and a warning. The driver shows it on 3489 ("FullCalibration: ['Rot_y'] stopped
  at a bound"). `estimate_errors` then re-fits with 2θ₀ held and bounds centred on the current
  values, so the written Rot_y error (±1.9°) is for that split. Written values are unchanged
  (≤ 3e-14 relative on 3489/3510/4971).

## Ring extraction does not need `HighBackground=False` (2026-10-29, #1019)

The draft switched FitPeaks to `HighBackground=False` (peak and background fitted together) for
the strip fits, with no recorded reason. The per-engine `high_background` option (see the
`HighBackground` entry above) exists only for that. Comparison of `True` (PyRS's default) against
`False`, with everything else the driver's defaults (5 strips, mask `HB2B_Mask_12-18-19.xml`).
Both end in the same joint peak + background fit. `True` starts it from a separate background
fit and peak fit, `False` from Mantid's estimates, so the difference is in the starting point.

**Synthetic images** (known geometry; mean of 6 noise draws; map error = max pixel-2θ error of the
refined geometry at arms 76/90/103°):

| case (counts/pixel) | HB | points | median / max \|col − true\| | cal RMS | map error mean (worst) |
|---|---|---|---|---|---|
| weak: 3 on 5 | True | 65.0 | 0.124 / 0.43 px | 0.0029° | 7.1 (10.7) mdeg |
| | False | 65.0 | 0.123 / 0.55 px | 0.0030° | 8.0 (14.8) mdeg |
| steep background: 3 on 20, 3/° | True | 63.7 | 0.258 / 2.07 px | 0.0074° | 18.9 (34.7) mdeg |
| | False | 46.8 | 0.546 / 3.69 px | 0.0194° | 83.9 (112.3) mdeg |
| sparse: 0.3 on 0.05 | True | 64.0 | 0.237 / 0.94 px | 0.0056° | 11.8 (18.7) mdeg |
| | False | 63.8 | 0.236 / 0.94 px | 0.0056° | 12.5 (21.9) mdeg |

(Strong rings, 40 on 5: identical.)

**Real runs** (`False` reproduces the committed results exactly):

| run | HB | points | RMS | λ (Å) | True vs False |
|---|---|---|---|---|---|
| HB2B_4971 | True / False | 65 / 65 | 0.0065 / 0.0065° | 1.53135 / 1.53135 | map ≤ 0.55 mdeg; columns ≤ 0.035 px |
| HB2B_3510 | True / False | 60 / 60 | 0.0166 / 0.0157° | 1.53061 / 1.53056 | map ≤ 7.1 mdeg; one point moved 1.56 px |
| HB2B_3489 | True / False | 58 / 59 | 0.0295 / 0.0301° | 1.72112 / 1.72120 | map ≤ 12 mdeg; `False` leaves Rot_y at its bound, `True` does not |

**Conclusions.**
- `True` is never worse on synthetic data, where the truth is known. It is much better when the
  background is strong and sloping: it keeps a third more points, and the geometry error is about
  4× smaller.
- On real data the two agree within the errors. HB2B_3510's RMS is 6% higher with `True`, and
  HB2B_3489's 2% lower. The λ differences (≤ 8e-5 Å) are below their errors (2–3e-4 Å).
- So nothing supports `False`.

**Switched (user decision, 2026-10-29).** Ring extraction uses PyRS's standard fit, and the
`high_background` option (`1883d1d6`) is removed, so `pyrs/peaks` is identical to `next`. The
driver's results are now the `True` rows above. HB2B_4971 scores 0.0077° on FitCalibration's
η-wedge metric (0.0075° before; FitCalibration 0.0123°), with ring RMS 0.0065° and strip drift
0.0033 / 0.0100°. Outputs: `calib_diagnostics/hb2b_4971/high_background_default/`. The HB2B_3510
integration-test reference was regenerated.

One unit test needed a looser intermediate check. With a starting wavelength 5 peak widths off
(1.572 Å against 1.5308 Å), the all-bands locate fit lands 0.08° from the band fits with `True`
(0.013° with `False`). FitPeaks' first, background-only fit masks only ±5 bins around the
*expected* position, so the real peak is fitted as background and the joint fit starts worse. At
realistic offsets (≤ 0.6°, the HB2B mono settings) the two settings agree exactly, and in every
case the ring points are within 0.10 px of the true ring.

*Aside (environment):* scripts that import `mantid.simpleapi` hung indefinitely in a background
session with no reachable display. Mantid's import loads matplotlib's Qt backend, which waits on
the display socket. `MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen` avoids it.

## The calibration's arbitrary 2θ₀ split is harmless on HB2B: the detector cannot move (2026-10-29)

At one sample–detector distance (L2), the ring fit determines only the set Shift_x / Rot_y / 2θ₀,
not each value (see the degeneracy entries above). Every member of that set gives the same pixel 2θ
*at that L2*. At another L2 they differ. The #1019 scientific review measured a near-uniform 2θ
change of 0.008° at 5 mm, 0.031° at 20 mm and 0.14° at 100 mm, and flagged that the calibration
JSON does not record its L2.

**Not an issue on HB2B (user, 2026-10-29): the detector is at a fixed location and cannot move.**
The software is consistent with that:
- No run logs an L2 (checked: HB2B_3489, 3510, 4971).
- `HidraWorkspace.get_l2` therefore returns None, and both reduction
  (`reduction_manager.py`, `build_instrument`) and the ring calibration use the instrument's
  nominal arm length, 0.985 m.

A calibration is therefore always applied at the distance it was made at, and the split has no
effect on reduced data.

**This would only change if** runs started carrying an `L2` log different from 0.985 m. The same
detector would then be described at two distances, and the split would matter. If that happens,
record L2 in the calibration file and check it in reduction.

## FitCalibration uses the 2020 lattice constants (2026-10, #1022)

`FitCalibration`'s powder lattice constants were the 2023-08 set (Ni 3.526314, Fe 2.865579, Mo 3.147664 Å,
commit `47d1e558`, "modified lattice based on newly calibrated data"). They were fitted from HB2B data, which
is circular for a calibration — only λ/a is measured — and their Fe/Ni ratio (0.81263) is ~0.1% off both the
2020 set and what HB2B shows (≈ 0.81335). They are now the 2020 set (Ni 3.523799438, Fe 2.8663982,
Mo 3.14719963 Å; `d6cd5e04`, common room-temperature reference values), user-approved.

**Effect:** η-wedge RMS of a full FitCalibration on HB2B_4971 0.0564 → 0.0121° (correctly binned wedges,
#1034), λ 1.5315 → 1.5313 Å; on HB2B_3510 (old binning) 0.0551 → 0.0261°; the ring calibration's RMS on
HB2B_3510 0.050 → 0.0175° with the Ni/Fe mean residuals going from +0.040 / −0.056° to −0.006 / +0.009°.
Calibrated λ changes by ~0.01–0.07% depending on the Ni/Fe balance of the run, and the geometry with it, so
calibrations made with the 2023 set differ from new ones for that reason alone. The `FitCalibration` gold files
still pass (`test_wavelength` λ 1.53002 vs gold 1.53030 Å; `test_all_refinements` only `error_Lambda`
0.02614 → 0.02315). Absolute λ rests on these constants; a certificate for the actual HB2B Ni/Fe standards
would be the final check. Pointing FitCalibration at `pyrs/calibration/powder_standards.py` (also adds FCC 400)
follows once that module (ring-based calibration) is merged.
## A project file keeps one 2θ axis for all masks: writes may not relabel masks already in it (2026-10)

`HidraProjectFile.write_reduced_diffraction_data_set` stores one 2θ dataset per file and used to replace it on
every write, while each mask's intensities are added beside it. Appending a reduction whose 2θ axis differs
(e.g. ROI masks, which `reduce_sub_run_diffraction` bins over the ROI's own pixels) left the masks already in
the file on the new axis: reducing HB2B_3510 sub-run 1 with a left-half ROI (67.91–76.11°), then a right-half
ROI (76.00–84.15°) in append mode, labelled the left-half pattern ~8° too high. No shipped workflow appends
(manual reduction, `reduce_HB2B.py`, `reduce_texture.py` save one reduction), so the risk was latent — the
file-level counterpart of #1033.

Guard (user-approved): the write raises `RuntimeError` when the new 2θ axis differs (beyond 1e-9°) from the
stored one *and* masks already in the file are not being rewritten. Single saves, re-saving a workspace that
rewrites all its masks, and appending data on the same axis are unchanged; no reduced data changes. Tests:
`tests/unit/pyrs/projectfile/test_reduced_data_2theta_guard.py`. Separate ROI reductions still get their own
axes; making them share one is a separate change.
## `MantidPeakFitEngine` left FitPeaks' positions table behind after every fit (2026-10-29, #1040)

`fit_peaks` passes FitPeaks four outputs. It deleted the two parameter tables after converting
them to a `PeakCollection`, and returned the fitted and difference workspaces, which the
peak-fitting and texture GUIs plot. The fourth, the peak-positions table
`fitted_peak_positions_<input>`, was neither returned nor deleted, and nothing in PyRS reads it.

- **Reused names** (GUI, `FitCalibration`) overwrote one table.
- **Fresh names** (the ring calibration's strip fits) left one table per fit in Mantid's workspace
  store: 26 after one synthetic 11-sub-run extraction.

It is now deleted with the parameter tables; no fitted value changes. The input, fitted and
difference workspaces remain the caller's to keep or delete. Regression test:
`test_fit_leaves_only_returned_workspaces` (fails without the fix).

## FitCalibration reads the shared powder standards (2026-10-30, #1022)

`FitCalibration` had its own copy of the calibration powders: names, `sy` positions, structures,
lattice constants and reflection lists. It now uses `pyrs/calibration/powder_standards.py`
(`HB2B_STANDARDS`, `standard_for_sy`), the same table as the ring-based calibration. User-approved,
including the unknown-`sy` change below.

**Differences from the old copy, and their effect:**
- **FCC 400 added.** FitCalibration fits only the lowest-angle reflection at least 1° inside each
  sub-run's pattern, so 400 matters only if it is the first in range. That needs the arm near 105°
  or above. On HB2B_3489/3510/4971 (33 sub-runs) every sub-run fits the same powder and reflection
  as before.
- **Lattice constants were silently float32.** The old code stored them in `np.zeros_like(sy)`, and
  the `sy` log is float32, so Fe was off by −3.6e-8 relative and Ni by −5.3e-9. The shared table
  keeps float64. Expected peak positions, which are the fit-window centres and the reference of the
  residual, move by ≤ 4.9e-6° (the residual's noise floor is ~1e-3°). Verified: the new code with
  float32-rounded constants reproduces the old selection bit for bit.
- **Unknown `sy`:** a sub-run whose `sy` matches no standard used to crash `get_diff_peaks` (the
  empty powder name is not a reflection key). It is now dropped from `_keep_subrun_list` with a
  warning, and `get_diff_peaks` raises a clear `ValueError` if that sub-run is used anyway.
- **Unreachable reflections** (λ > 2d) are left out instead of returning NaN. Selection is
  unchanged, because NaN never passed the range filter.

Tests: `tests/unit/pyrs/calibration/test_fitcalibration_powders.py` and the updated integration
test `test_reference_lattice_is_the_2020_set`.

## Calibration GUI: predicted-ring overlay on the 2D view (2026-10-30, #1039)

The Detector Calibration window's **2D** tab has a **Predicted rings** box. Its **Geometry** choice
is the nominal geometry at the mono wavelength, the calibration currently in the window, or a
calibration JSON (read with reduction's reader, TTH_0 included).
- **What is drawn:** the Bragg angles `2·asin(λ/2d)` of the sub-run's powder, unmodified, as
  contours of that geometry's pixel 2θ from `DetectorModel`, which is reduction's 2θ. Only
  unmasked pixels are drawn, and each ring is labelled by hkl. It is display only: no fit,
  reduction or parameter changes (user-approved GUI change).
- **Code:** `pyrs/calibration/pixel_calibration_plots.py`, `DetectorCalibrationModel.ring_overlay`,
  `DetectorView.set_ring_overlay`.
- **Every redraw removes the previous overlay.** The view adds a new image on each redraw without
  clearing, and contours draw above images, so an old overlay would otherwise stay visible.
- **Checked:**
  - pixel placement against the GUI's `imshow(np.rot90(counts))` (pixel `(i0, i1)` at
    `(i0 + 0.5, i1 + 0.5)`);
  - contour vertices lie on the ring (forward model at the fractional pixels);
  - by eye on HB2B_4971 and HB2B_3510. On 4971 the nominal ring is ~50 px off the Ni 311 band, and
    the ring calibration's ring lies on it.

**Running the GUI tier without a display.** matplotlib refuses its Qt backend when there is no
display *and* no `QApplication` yet ("Cannot load backend 'QtAgg' … 'headless'"). That happens
when an SSH-forwarded `DISPLAY` has died, and the GUI test modules then fail at import. Creating
the application offscreen before pytest imports anything works:

    env -u DISPLAY QT_QPA_PLATFORM=offscreen pixi run python -c "
    from qtpy.QtWidgets import QApplication; app = QApplication(['pytest'])
    import sys, pytest; sys.exit(pytest.main(['tests/ui', '-m', 'gui']))"

(11 GUI tests passed this way.)

## Calibration GUI: ring-based method and the new Export (2026-10-30, #1045)

The Detector Calibration window can run the ring-based calibration (user-approved GUI change).

**Method choice:** **FitCalibration** (the default, unchanged) or **Ring-based**. With Ring-based:
- **Fit Peaks** extracts the ring points from the loaded run, using its default mask and the
  starting calibration.
- **Strips** sets the strips per sub-run: an editable field, default 5, a whole number ≥ 1.
  2θ bin and η bin are greyed out.
- **The recipe** uses the same routine names as FitCalibration's stages, plus "iterate"
  (3 passes + polish). **Calibrate** runs it, then one `estimate_errors()` with 2θ₀ held, as the
  driver does.
- **Results** fill the same table, RMSE and Δ2θ fields. The window's refinement method and max
  nfev apply, a parameter left at a bound shows a warning, and the 2D overlay's "Current
  calibration" shows the result.
- **Checked:**
  - With the driver's settings (strips 5, "iterate", trf, the same mask and starting
    calibration), the window's calibration file on HB2B_3510 is **identical** to
    `pyrs-pixel-calibration`'s (`test_ring_based_calibration_matches_driver`).
  - The same holds headless on a synthetic run.

**Export, for both methods:**
- It always writes `<folder of the user's choice>/HB2B_CAL_<time>.json`.
- With **Also write the shared HB2B calibration**, ticked by default, it also writes
  `HB2B_Latest.json`, `HB2B_Latest_<Si…>.json` and `<Si…>/HB2B_CAL_<time>.json` in
  `/HFIR/HB2B/shared/CALIBRATION`.
- `<Si…>` is the run's monochromator setting from `mrot`, as FitCalibration names its archive
  folder. Si511 shares Si333's wavelength, and its name comes out as `Si333`, as before.
- All files are written with reduction's `write_calibration_to_json`, so TTH_0 is included and the
  −1/−2 error codes are kept.
- If the shared folder isn't writable, the user's copy is still written and an error says so.
  (FitCalibration's own `write_calibration` used to print a console message instead.)
- **Changed from before:** Export always wrote the timestamped archive copy, and `HB2B_Latest.json`
  only after a yes/no question. It never wrote `HB2B_Latest_<Si…>.json` and had no user folder.
  "Save Local Calibration" is removed, because Export's folder replaces it.
  `FitCalibration.write_calibration` itself is unchanged.

**Pitfalls found while testing:**
- **The GUI test's window is shared by the whole test session.** The first test's recipe
  (`tests/data/calibration_tests/test_ui_recipe_load.json`: max nfev 3, only sub-run 1 kept), the
  calibration it loads and its result columns all carry over. The ring-based test must set the
  driver's settings explicitly. The window rightly applies them to both methods.
- **FitCalibration's `residual_rmse` / `residual_sum` index an empty list until it has fitted.** The
  calibration progress timer can read them right after the method is switched. The GUI model now
  treats that as "no value yet".

## Calibration GUI: picked ring points and rejected strips on the 2D view (2026-10-30)

After the ring-based **Fit Peaks**, two tick boxes under the 2D view, **Ring points** and **Rejected
strips**, mark the extraction's picks for the shown sub-run (user-approved GUI change, display
only). The **Geometry** for predicted rings now defaults to **Current calibration**.
- **Ring points:** open white circles with a dark edge, exactly at the extracted `(index_0,
  index_1)`, i.e. drawn at `(i0 + 0.5, i1 + 0.5)` like the image and the rings.
- **Rejected strips:** grey ✕. A strip that was fitted but then rejected (e.g. as an outlier) is
  marked at its fitted 2θ. One never fitted is marked at the 2θ the extraction aimed for (the
  located ring, else the Bragg angle), mapped to the strip's centre row with the extraction's
  geometry (`ring_extraction.column_at_two_theta`). A strip whose ring misses the row is not marked.
- **Hovering** a mark shows its reflection and strip, plus the fitted 2θ ± error or the rejection
  reason.
- **Diagnostics:** `extract_rings` now keeps the per-strip diagnostics. Passing a diagnostics list
  does not change the extraction: the driver output is still bit-identical on HB2B_3489/3510/4971.
- **Seen on HB2B_3489 sub-run 12:** Ni 311 has 4 picks and 1 outlier strip; Ni 222 has all 5 strips
  rejected as not significant (1.6–1.7σ).

## PR #1046 review: fixes to the calibration GUI and Export (2026-10-30)

A five-agent review of PR #1046 (code, tests, comments, silent failures, type design) found no
problem in the physics. The forward model, refinement and extraction stand, and there are no
Constitution issues. It found several ways a wrong or unconverged calibration could reach
`HB2B_Latest.json`. All are fixed (user-approved):

**What could go wrong, and what changed:**
- **Loading a new run kept the previous run's ring-based result.** Export could write run A's
  calibration under run B's monochromator, and the 2D marks showed A's points.
  `DetectorCalibrationModel._reset_ring_state` now runs on every load.
- **Calibrate errors were swallowed.** This was in production code on `next`:
  `pyrs/interface/threading/worker_pool.py` called `traceback.print_exc(e)`, which treats the
  exception as a line *limit* and raises `TypeError`. So neither `result` nor `finished` was ever
  emitted: no message, buttons possibly left disabled, and a half-refined calibration still
  exportable. The worker now prints the traceback, emits a new `error` signal (shown as a message
  box) and always emits `finished`. A failing ring-based recipe is rolled back to its state before
  the recipe (`PixelCalibration.checkpoint` / `restore`).
- **Export now refuses** status -1 (not refined; FitCalibration had no check) and status 0 (a stage
  stopped at max nfev before converging), each with its reason.
  - A stage that stops at max nfev now warns ("stopped at max_nfev (n) before converging").
  - The CLI summary prints a warning line for status 0. The CLI also catches `OSError` (e.g. an
    unwritable `--output`) and exits with 1.
- **A partial recipe no longer moves other parameters.** `estimate_errors()` re-fits every parameter
  except 2θ₀, so after a recipe of just "wavelength" it would also have refined the geometry. After
  recipes that don't end in a joint refinement ("iterate", "full"), errors now come from
  `PixelCalibration.errors_at_current()`, which takes the Jacobian at the current values and moves
  nothing. After a joint refinement the driver's `estimate_errors()` is kept, so the GUI's result is
  still identical to the driver's.
- **Export writes are atomic and ordered:**
  - Every target is checked first: existing files must be writable, new ones need a writable folder.
  - Each file is written to a temporary file, given the normal umask mode, then renamed into place.
  - The order is the user's copy, then `<Si…>/HB2B_CAL_<time>.json`, then
    `HB2B_Latest_<Si…>.json`, then `HB2B_Latest.json` last. A failure never leaves a truncated file,
    and the production file is never newer than its archived copy.
  - On failure the message lists what was already written.
  - Archive names now include seconds (`%Y-%m-%dT%H:%M:%S`), so two exports in the same minute
    don't overwrite each other.
- **Controls are locked during Calibrate.** Method, Export and file loading are now locked with the
  other buttons while the worker refines.
- **Warnings are visible:**
  - Bound and convergence warnings are read from `PixelCalibration.warning_log`, not captured with
    `warnings.catch_warnings`, which is process-wide and not thread-safe in the worker thread.
  - Extraction warnings, such as a sub-run skipped because its `sy` matches no powder, are named in
    the status bar and shown in a message box.
- **Δ2θ means the same for both methods:** the signed sum of residuals (observed − expected). The
  ring-based field used to show a sum of squares.
- **Error handling:**
  - Only an error from the predicted-ring geometry now unticks "Predicted rings". Other 2D-view
    errors are reported as such.
  - Fit Peaks also catches table-parsing and missing-log errors.
  - An old recipe that names "iterate" switches to Ring-based instead of crashing.
- **`PixelCalibration`:**
  - `set_calibration_array` resets errors and status, which described the old values.
  - `set_refinement(method, max_nfev)` validates, replacing the GUI's writes to private attributes.

**Verification:**
- Driver output is bit-identical on HB2B_3489/3510/4971.
- The GUI ring-based result still equals the driver's, now compared against the real
  `run_calibration`.
- New tests cover each fix, and the worker tests fail on the old worker.

## PR #1046 re-review: Export permissions, worst status, failed FitCalibration recipes (2026-10-31)

A second review of the fixes (code reviewer and silent-failure hunter) confirmed most of them, and
found that two were wrong. This entry corrects the one above where they differ.

**Export permissions (user decision: anyone with write access to the folder may overwrite a calibration file).**
- **The problem:** the atomic replace created a new file with the exporter's ownership, group and
  umask mode (0600 under umask 077, unreadable by reduction run as other users). It briefly set the
  process-wide umask to read it. And its pre-check tested write access to the *file*, while a rename
  needs write and search access to the *folder*.
- **Now:**
  - Files are written as `rw-rw-r--` (`CALIBRATION_FILE_MODE = 0o664`) in the folder's group, where
    the user may set it, so group members may also overwrite them in place. The umask is not touched.
  - A created archive folder gets `rwxrwsr-x` (0o2775), so its files inherit its group.
  - The pre-check requires write and search permission on the folder. In a sticky-bit folder, the
    exporter must also own any file being replaced, or own the folder.
  - A read-only `HB2B_Latest.json` in a writable folder is replaced. A writable one in a read-only
    folder is refused before any shared file is written.
  - Per-file ACLs on a replaced file are not kept.

**Export checks the worst status, not the last.**
- **The problem:** a stage stopped at max nfev (status 0), a later stage converged, and Export
  accepted the result. For FitCalibration this was silent. The entry above overstated this ("Export
  now refuses status 0").
- **Now:**
  - `PixelCalibration.worst_status` keeps the lowest status since the last start; `set_calibration_array`
    resets it. The GUI model records FitCalibration's status after every recipe stage.
  - Export refuses if any stage did not converge.
  - The written `Status` is still the last stage's, so the driver output is unchanged.

**A FitCalibration recipe that fails partway can't be exported.** The stages before the failure
have changed the calibration. The GUI model marks it as incomplete, without touching FitCalibration,
until the next Calibrate.

**The Export dialog states what it will write:** the method, status, λ, shifts and 2θ₀. If the
calibration can't be exported, it shows why and disables OK.

**Smaller fixes:**
- Fit Peaks and Calibrate with no run loaded raise a clear error instead of an `AttributeError`
  escaping a Qt slot.
- The recipe boxes, strips field and keep table are locked during Calibrate. The result columns are
  labelled from the recipe.
- A failed ring recipe shows its own warnings.
- `Worker` documents that `result` is None on failure.
- FitCalibration's ten-branch `if/elif` uses the shared routine table (same names, same methods).

**Bugs already on `next`:**
- Fixed: saving a recipe kept only its last routine, because `save_json` overwrote the string on
  each pass.
- Reported, not fixed (production FitCalibration): it writes an error of **0** for undetermined
  parameters, because `-2 * np.zeros_like(...)` is zeros, not −2 (#1049). *Fixed 2026-11, see the #1049 entry.*

## Reduction places pixels with `DetectorModel`: one forward model (2026-11, #1011)

`ResidualStressInstrument.build_instrument` (reduction) now takes its lab-frame pixel positions from
`DetectorModel.lab_positions` in `pyrs/core/detector_model.py`, the forward model the ring-based
calibration refines. Before, reduction and the calibration each had their own copy of the geometry,
kept equal only by a test. Done as four separately approved production steps:

1. The rotation helpers moved into `detector_model.py` as `rotate_positions`, `rotation_matrix` and
   `rotation_matrix_x/y/z` (verbatim arithmetic; this also removed `DetectorModel`'s import of
   `ResidualStressInstrument`).
2. `_set_uncalibrated_pixels` used `DetectorModel.local_positions`.
3. `build_instrument` delegated to `DetectorModel.lab_positions`, keeping its input checks (a wrong
   calibration type is still an `AssertionError`), then computes 2θ and η as before.
4. The now-unused duplicates were removed: `_raw_pixel_matrix`, `_set_uncalibrated_pixels`,
   `rotate_detector_2theta`, `_rotate_detector`, `generate_rotation_matrix`,
   `_cal_rotation_matrix_x/y/z`. Their tests moved to `tests/unit/pyrs/core/test_detector_model.py`.

**Effect on results: none, to the bit.** After every step, a 177-entry SHA-256 golden file was
compared with the pre-#1011 baseline: `build_instrument` positions, rotation matrix and 2θ/η (1D and
2D) for 4 calibrations (none, zero, GUI-like, large) × 8 arm angles (−120° to 120°, including 0 and
5e-8, below the 1e-7 threshold for applying the 2θ rotation) × 3 L2, plus 80 reduced HB2B_3510
patterns (nominal and calibrated, two η steps). All were identical, as were the ring-calibration
driver output and ring points on HB2B_3489/3510/4971, and the unit, integration and GUI tiers.

**Tests.** The old bit-identity test (`DetectorModel` against `build_instrument`) cannot fail once
one calls the other, so it was replaced: `test_two_theta_eta_match_reference` and
`test_build_instrument_matches_reference` pin 2θ and η of six pixels (corners, centre, one
off-centre) in four geometries to values computed by the pre-#1011 reduction code (origin/next
7dde880b), the latter through reduction's 2D maps, its 1D pixel-ID order and `get_pixel_array`.
`test_build_instrument_non_square_matches_reference` does the same on a 4-row × 6-column detector
with 10 × 20 mm pixels, which catches swapped rows/columns or x/y pixel sizes. Tolerance: rtol 1e-12
plus atol 1e-9°, because `arccos` amplifies a last-bit change near 2θ = 0 (the arm-0 centre pixel,
0.012°) to ~3e-11°; the smallest geometry error the references catch is 8e-4° (reversed rotation
order). Verified: reversing the rotation order to `Rz @ Ry @ Rx` fails both; an independent review
injected 11 geometry mutations (transposed indices, flipped X, half-pixel offset, inverted rotation,
shift and 2θ₀ signs, shift after rotation, arm sign) and each failed by 4e-5° to 210°.

**Review (2026-11, user-approved fixes).** An independent differential test of 9,290 input
combinations (detector sizes down to 1 × 3, 2θ at the 1e-7 threshold, string/int/float32 inputs, bad
types) found the pixel map, shapes, dtype, pixel-ID order and exception types bit-identical for
Python-float/float64 calibrations. It found three latent differences on inputs no loader creates,
all fixed so the new code again matches the old:
- a `float32` 2θ₀ rotated the arm in float32 (≤ 1.6e-6° in 2θ) and a 1-element-array 2θ₀ raised:
  `lab_positions` now rotates by the Python-float sum, as reduction always did
  (`test_numpy_two_theta_0_rotates_in_float64`; bit-identical to pre-#1011 for float32, array and
  float);
  that sum is still float32-rounded for a float32 2θ₀ (as before #1011); making `DENEXDetectorShift`
  store float64 so the whole instrument is float64 for any caller is #1052;
  *since #1052 the stored 2θ₀ is float64 and the sum is exact; see the #1052 entry;*
- an arm length below 1 cm raised with `l2=None`, because `lab_positions` re-checked the resolved
  arm length against the 1 cm minimum for an explicit L2: `build_instrument` now passes `None` through
  (`test_build_instrument_default_l2_is_arm_length`);
- `build_instrument` keeps its own calibration type check so a wrong type is still an
  `AssertionError` (`test_build_instrument_bad_calibration_raises`).

**Found, not fixed (#1051):** `PyHB2BReduction.rotate_two_theta` calls a nonexistent
`ResidualStressInstrument.rotate_detector`; it and `build_instrument_prototype` have no callers, and
the latter builds `DENEXDetectorShift` with 6 arguments in the wrong order. `PyRsCore.reduce_diffraction_data`
(`pyrs/core/pyrscore.py`) also builds `DENEXDetectorShift(0, 0, 0, 0, 0, 0)` without `tth_0`, so a
calibration passed as a JSON file path raises `TypeError`. *All fixed 2026-11, see the #1051 entry.*

## FitCalibration's errors: scaled by the residual scatter, −2 when undetermined (2026-11, #1049)

`FitCalibration.FitDetector` (`pyrs/calibration/mantid_peakfit_calibration.py`) reported
`sqrt(diag(inv(JᵀJ)))`. Two defects:

- **No fit-quality factor.** Without `Σr²/(n−p)` the errors assume every residual (a peak-position
  difference in degrees) has σ = 1°. The real scatter is 0.03–0.2°, so every error was too large.
- **Undetermined parameters reported 0.** When `JᵀJ` was ~0 or singular, `-2 * np.zeros_like(...)`
  gave zeros, not −2, so a failed error estimate looked perfect. The brute-force path returned `[0]`.

**Fix (user-approved production change):** errors come from `pixel_refinement._parameter_errors`,
the ring calibration's formula `sqrt(diag(inv(JᵀJ)) · Σr²/(n−p))` with −2 (`UNDETERMINED`) for a
rank-deficient direction; the brute-force path reports −2 (it gives no covariance; it has no callers).
A zero Jacobian column flags *every* parameter −2, the ring calibration's existing conservative rule.

**Effect, HB2B_3510** (mask `HB2B_Mask_12-18-19.xml`, `max_nfev` 20; old = origin/next 7dde880b):
calibrated values bit-identical in every stage; only `error_*` changes.

| Stage | Parameter | Old error | New error | Old/new |
|---|---|---|---|---|
| `calibrate_wave_length` | λ | 0.00195 Å | 0.000443 Å | 4.4 |
| `CalibrateShift` | Shift_x / y / z | 2.09 / 3.97 / 36.2 mm | 0.054 / 0.10 / 0.93 mm | 38.9 |
| `CalibrateGeometry` (from nominal) | Shift_x / y / z | 32.2 / 5.71 / 40.7 mm | 3.82 / 0.68 / 4.83 mm | 8.4 |
| | Rot_x / y / z | 11.1 / 5.17 / 0.894° | 1.31 / 0.613 / 0.106° | 8.4 |
| | TTH_0 | 1.98° | 0.235° | 8.4 |

The ratio is constant within a stage: it is 1/√(χ²/dof), the residual scatter in degrees.

**Limit: FitCalibration does not flag the Shift_x/Rot_y/TTH_0 degeneracy.** In the ring
calibration that direction is ~4e-10 of the largest singular value of the column-scaled Jacobian and
is flagged −2. In FitCalibration's geometry stage the smallest relative singular value is 0.04
(others 0.31–0.98): the residual is noisy (peaks are re-binned and re-fitted at every evaluation,
#1030), so the degeneracy looks weak rather than exact and the 1e-6 rank test does not fire. Those
three parameters get large but finite errors (±3.8 mm, ±0.61°, ±0.24°). Meaningful errors for them
need 2θ₀ held, as the ring calibration's `estimate_errors` does — a separate change.

Tests: `tests/unit/pyrs/calibration/test_fitcalibration_errors.py` (constant fit gives s/√n, line
fit matches the covariance, errors scale with scatter, `lm` gives the same, degenerate pair → −2 with
the independent parameter finite, zero column → all −2, brute force → −2); all 7 fail on the old code.

**Gold files updated (user-approved):** `tests/data/calibration_tests/HB2B_mantid_calib.json` stored
the bug as the expected answer (`error_Shift_*`/`error_Rot_*` = −0.0); only the `error_*` fields of it
and `HB2B_mantid_calib_lambda.json` were replaced by the new output, values untouched
(`test_peakfit_calibration.py`; its round-trip test's hand-set `error_Lambda` follows the gold file).

## Broken, unreachable geometry paths removed or fixed (2026-11, #1051)

Found in the #1011 review; none could have affected a result, because each crashed before doing
anything (or had no caller). User-approved production changes:

- **Removed `PyHB2BReduction.rotate_two_theta` and `build_instrument_prototype`**
  (`pyrs/core/reduce_hb2b_pyrs.py`): no callers. The first called a nonexistent
  `ResidualStressInstrument.rotate_detector`; the second built `DENEXDetectorShift` with 6 arguments
  (7 are required since `tth_0` was added) and in the wrong order (the Z shift as X).
- **Fixed `PyRsCore.reduce_diffraction_data` with a calibration file path** (`pyrs/core/pyrscore.py`):
  it built a 6-argument `DENEXDetectorShift` (`TypeError`). Worse, with 7 it would have *silently
  applied a zero calibration*: `DENEXDetectorShift.from_json` → `set_from_dict` reads keys `"shift x"`,
  `"rotation x"`, …, but calibration files use `Shift_x`, `Rot_x`, … (the keys `convert_to_dict`
  writes), and it never reads `TTH_0`. It now uses the standard reader, as `ReductionApp` does:
  `read_calibration_json_file` + `check_calibration_status` (an uncalibrated file is refused). The
  file's wavelength is still not applied by this API, as before. `PyRsCore.reduce_diffraction_data`
  has no caller in `pyrs/`, `scripts/` or `tests/`. `from_json`/`set_from_dict` themselves are
  unchanged and still disagree with `convert_to_dict`; nothing else uses them.
- **Removed the ASCII geometry reader `import_calibration_ascii_file`**
  (`pyrs/utilities/calibration_file_io.py`). It was reachable — Manual Reduction and
  `ReductionApp.reduce_data` sent any calibration file not ending in `.json` to it — but always
  crashed on the 6-argument `DENEXDetectorShift`. Behind that it was wrong throughout: it matched
  `cal::shift_x` while its own docstring wrote `cal::shiftx`, `cal::rot_y` set `rotation_x`, `cal::arm`
  set an attribute nothing reads, and Shift_z and TTH_0 could not be set. Nothing writes the format.
  A non-`.json` calibration file now raises `ValueError("Calibration file must be a .json file, ...")`.

Tests (`tests/unit/pyrs/core/test_pyrscore.py`, `test_powder_pattern.py`, all failing on the old code):
a calibration file reaches reduction with every value including TTH_0; an uncalibrated file
(`Status` −1) is refused before reduction; a non-`.json` file raises the clear error.

## `DENEXDetectorShift` stores float64, so the instrument is float64 for any caller (2026-11, #1052)

`DENEXDetectorShift`'s setters converted with `to_float`, but `__init__` (and `set_from_dict`) stored
values as given. A calibration built from numpy float32 values therefore ran reduction's geometry in
float32 under numpy 2's type rules: `rotation_* * np.pi / 180` gave float32 angles, and the arm angle
`two_theta + tth_0` was rounded to ~4e-6° (pixel 2θ off by up to 1.6e-6°, measured at arm −35°,
`tth_0 = np.float32(0.123456)`). No code produced float32 (the JSON reader, both calibrations and the
GUI pass Python floats or float64 elements), so no result was affected.

**Fix (user-approved production change):** `__init__` and `set_from_dict` convert every field with
`to_float`, so each is a Python float (float64) whatever type it is given in (float32, int, string,
1-element array — the last with numpy's "array to scalar" deprecation warning). `DetectorModel.lab_positions`'
explicit conversion of the arm-angle sum (added in the #1011 review) is back to a plain sum.

**No range check in `__init__`, deliberately.** The setters reject rotations and 2θ₀ outside ±360°,
but the same class holds a calibration's *errors*, and a badly determined rotation's error can exceed
360°; checking ranges in `__init__` would make writing or reading such a calibration raise.
`test_error_above_360_degrees_constructs` guards this; the setters keep their checks.

**Effect:** real calibrations bit-identical (golden check, driver on HB2B_3489/3510/4971, FitCalibration
gold files). A float32 2θ₀ now gives the float64 sum of the arm angle and the exact stored value,
not the float32-rounded sum (`test_numpy_two_theta_0_rotates_in_float64` updated accordingly).
Tests: `tests/unit/pyrs/core/test_instrument_geometry.py`.

## Angle conversions use `np.deg2rad` / `np.rad2deg` throughout the detector geometry (2026-11, #1058)

The geometry converted angles two ways: `np.deg2rad` for the arm rotation and in
`DetectorModel.with_two_theta_0`, but by hand in 10 places — the calibration rotations
(`rotation_* * np.pi / 180.0`) and pixel 2θ/η (`* 180 / np.pi`) in `pyrs/core/detector_model.py`, and
the pixel 2θ/η maps and d-spacing in `pyrs/core/reduce_hb2b_pyrs.py`. `x * np.pi / 180` rounds twice
(x·π, then ÷180); `np.deg2rad` multiplies once by π/180, so the two differ in the last bit of ~27 % of
values. All 10 now use `np.deg2rad` / `np.rad2deg` (user-approved; **not bit-identical**, so kept out
of #1055/#1056).

**Effect** (pre-#1011 code vs this change; 96 geometries: 4 calibrations × 8 arm angles × 3 L2):

| quantity | largest change |
|---|---|
| pixel x, z / y | 4.4e-16 m / 0 |
| pixel η | 1.1e-13° |
| pixel 2θ | 2.2e-12° — one pixel at 2θ = 0.33° (arm 0, large calibration); next largest 7.4e-13° |
| reduced HB2B_3510 intensities and errors (80 patterns) | identical: no count changes bin |
| reduced 2θ axis (bin centres) | ≤ 2.8e-14° |

**Tolerance for 2θ: two roundings** (user decision, 2026-11-03): |Δ2θ| ≤ 2 × (one bit of the 2θ value
+ ε / sin(2θ) rad), ε = 2.2e-16 — i.e. the last bit of the result plus one bit of cos(2θ) magnified by
arccos. Measured over all 100.7 M pixels of the 96 geometries: 71.4 % unchanged, 99.9997 % within one
such unit, maximum 1.51. Two narrower criteria were tried first and were wrong: a fixed 1e-12° fails
near the beam (at 2θ = 0.33° one bit of cos(2θ) is 2.23e-12°, exactly the largest change observed),
and "one bit of cos(2θ)" alone fails for ~10 % of pixels at high 2θ because it leaves out the rounding
of the 2θ value itself (one bit of 120 is 1.4e-14°). Positions and η: 1e-15 m, 1e-12°.

**Ring calibration is not bit-identical either.** The ring points move by ≤ 1.6e-12 pixel, and the
iterative fit stops a hair further along the flat Shift_x / Rot_y / 2θ₀ direction. On the driver runs
(HB2B_3489 / 3510 / 4971, `Status` 1 unchanged) every refined value changes by ≤ 1e-3 of its own
uncertainty; largest: Shift_x on 3510 (1.0e-3 σ); Rot_y and TTH_0 move together by 1.6e-5° on 3510
(TTH_0 is set by the virtual swing, not fitted). Accepted (user, 2026-11-03).
FitCalibration gold-file tests pass unchanged (tolerance 5e-3).

**Golden baseline.** Bit-identity checks of later refactors must compare against a baseline recorded
*after* this change; the #1011 baseline (pre-#1011 code) differs from it in the last bits.

## Calibration files record how they were obtained: the `provenance` block (2026-11, #1053 part A)

Every calibration JSON written by `FitCalibration`, the ring calibration (`PixelCalibration`, the
`pyrs_pixel_calibration.py` driver) or the calibration GUI's Export now carries a `"provenance"` key
beside the values (`write_calibration_to_json(..., provenance=)`; `read_calibration_provenance()`
returns it, or None for older files). User-approved production change; **values, errors and `Status`
are bit-identical** to before (all keys except `provenance` compared on the ring driver for
HB2B_3489/3510/4971 and on FitCalibration for HB2B_3510), and a file written without provenance is
byte-for-byte what the old writer gave. PyRS's reader only looks up the keys it needs, so the extra
key is safe for it; check the deployed PyRS release's reader before the shared `HB2B_Latest.json` gains it.

**Schema 1** (`pyrs/calibration/provenance.py`): `created` (ISO 8601 with time zone), `pyrs_version`,
`user`, `host` (user decision: record them), `method` (`"peak-fit"` / `"ring"`); `inputs` (run and mask
with SHA-256, powders, monochromator, settings; the GUI adds its recipes and extraction settings);
`start`; `steps`, one per refinement routine *since the current start* (routine, free parameters, values,
errors, status, n, RMSE, nfev, the optimizer's message, why any parameter is −2); `fit_quality` at the
final calibration (n, dof, RMSE, χ²/dof, scatter, residuals per sub-run and reflection — and η mask for
peak-fit; in the JSON by user decision, ~50 entries); `error_model`; `held_parameters` (ring); and
`undetermined` with reasons. Non-finite numbers are written as `null`, so the file stays standard JSON.

Found while prototyping on HB2B_3510, and handled:
- **Step names.** FitCalibration's single-parameter routines go through `calibrate_single`; each public
  routine now passes its own name to `FitDetector`, so the step reads `calibrate_wave_length`, not the helper.
- **Reflections.** FitCalibration's residual is one peak per sub-run and η mask, the lowest reflection
  in the detector's range; `fit_peaks` records which (e.g. `"Ni 111"`) without changing the choice.
- **Why −2.** The ring calibration records *rank-deficient Jacobian* or *stopped at a bound*; after
  `FullCalibration` at one L2 the rank test flags Shift_x, Rot_y, TTH_0 and also Shift_y, Shift_z,
  Rot_z (see the degeneracy entry above). `estimate_errors` records TTH_0 as **held**.
- **A new start.** The GUI lets the user set a new starting calibration and Calibrate again; earlier
  steps (still kept for the GUI's plots) are not part of the new calibration, so the provenance lists
  only the steps, recipes and final residual since the latest start.
- **Robustness.** The provenance must never stop a calibration from being written: unknown inputs
  (e.g. a run without the `mrot` log) are recorded as null, and if the GUI cannot build the block the
  file is still exported with `{"error": ...}` in its place (and a warning).

**Keys by method (schema 1).** Both: `schema, created, pyrs_version, user, host, method, inputs, start
{source, values}, steps[], fit_quality, error_model, undetermined[]`. `inputs` — ring (driver):
`runs, mask, monochromator, monochromator_error, wavelength_start_A, extraction {strips, sub_runs},
iteration, driver, powders, ring_points, refinement`; ring (GUI): `interface, runs, mask, monochromator,
extraction {strips, sub_runs, starting_calibration, strip_fits, strip_fits_rejected {reason: count}},
recipes, powders, ring_points, refinement`; peak-fit: `[interface, recipes,] runs, mask, vanadium,
powders, sub_runs_used, monochromator, reduction {two_theta_bins, eta_slice_deg}, refinement`.
Ring only: `held_parameters`. Peak-fit `fit_quality` adds `residual_source`. A step that could not be
recorded is `{step, routine, error}`; a block that could not be built is `{schema, error}`. The reader
(`read_calibration_provenance`) warns on any other schema and returns the block as stored.

**Review fixes (2026-11-03, user-approved F1–F9).** Three independent reviews found no change to a
calibrated value, but the "never block" promise was not fully kept and some records could mislead:
- **F1 — one guard for every writer.** `provenance.safe_build` builds the block, converts numpy types
  (`jsonable`) and test-serializes it; any failure writes `{"schema", "error"}` and warns. Used by
  `PixelCalibration.write_calibration`, `FitCalibration.write_calibration` and the GUI; before, only the
  GUI was guarded and a numpy `bins` could still fail at serialization, outside it.
- **F2 — nothing fragile inside the fit.** `fit_peaks` records plain numbers per residual (sub-run,
  powder, structure, h²+k²+l², η mask) behind an index guard; the text label is built in
  `provenance()`. Step recording in `FitDetector` cannot lose the refinement's result. `hkl_label` moved
  to `powder_standards.py` (pure numpy; `pixel_calibration_plots` re-exports it), so no matplotlib
  import on the calibration path.
- **F3 — a recipe is recorded only after it ran** (a failed, rolled-back one is not listed).
- **F4 — FitCalibration's fit quality is evaluated at the solution** (user decision): the last peak fit
  of a refinement is a Jacobian probe, so its labels need not describe `out.fun`. After each routine,
  `fit_peaks` runs once more at `out.x` (called directly, so the GUI's residual history is unchanged;
  afterwards the workspace holds the solution rather than the last probe). If it fails, `out.fun` is kept
  without labels and `residual_source` says so. A label/residual mismatch is noted
  (`by_reflection_and_run_unavailable`), never guessed.
- **F5 — stale state:** `get_archived_calibration` and the brute-force path clear the previous fit
  quality; the ring `checkpoint()` includes the start, so `restore()` brings it back.
- **F6 — visible fallback:** the GUI status bar adds "(provenance incomplete: …)".
- **F7 — reasons beside nulls:** `sha256_error` for an unreadable file, `monochromator_error` in the
  driver (which now also catches a multi-valued `mrot`).
- **F8 — schema pinned** (keys above; reader warns on another schema). **F9** — both methods build the
  block with one `provenance.assemble`.
- The ring start records *who* set it (`set_calibration_array(..., source=)`; the GUI passes "starting
  calibration set in the GUI" or "nominal geometry, monochromator wavelength"), found when the GUI/driver
  test was strengthened to compare the whole block except time, user, host and inputs.
- Found, not fixed (outside provenance): `FitCalibration.set_calibration_array` keeps the previous
  `_caliberr`, so values set as a new start are exported with the errors of the earlier fit. *Fixed in #1060 (entry below).*

Part B (reduced project files record the calibration applied) has its own entry below.

## `FitCalibration.set_calibration_array` resets errors and status (2026-11, #1060)

`set_calibration_array` replaced the values but kept `_caliberr` and `_calibstatus` from the previous
refinement, unlike the ring calibration's, which resets both "so a calibration is not written with
errors of values it no longer has". Investigation showed it was **not reachable** from PyRS: the GUI's
Fit Peaks and Calibrate (FitCalibration method) call `set_reduction_param()` first, which builds a new
`FitCalibration` (errors −1, status −1), and only then set the start; `scripts/pyrs_calibration.py`
loads an archived start into a fresh object before refining. Only code calling it on an already refined
object and then `write_calibration` got new values with old errors and status.

Fixed anyway (user decision): `set_calibration_array` resets errors and status to −1, as the ring
calibration does. No change to the GUI's or the script's output (their objects are at −1/−1 at that
point); `test_new_start_resets_errors_and_status` fails without the fix, and
`test_archived_start_then_refinement_is_unchanged` pins the script's flow on HB2B_3510.

Note on staged recipes (by design, unchanged): each FitCalibration routine updates the errors of the
parameters it refined only (`set_shift` writes `_caliberr[0:3]`), so a file's errors can come from
different routines of one recipe; parameters no routine refined stay −1.

## Reduced project files record the calibration applied (2026-11, #1053 part B)

**The instrument object owns the calibration** (user design, 2026-11-05). `HidraSetup`
(`pyrs/core/instrument_geometry.py`, "anything to do with HB2B ... including geometry, wavelength and
calibration") now stores the calibration a reduction applies: the shift (`DENEXDetectorShift`) and its
errors, λ and its error, the status, and where it came from — `source` (`file` / `workspace` / `none`),
the calibration file and its SHA-256, `created` with `created_source`, `lambda_source`, and the
provenance block (#1053 part A). `load_calibration_file()` and `use_workspace_calibration()` fill it;
`calibration_dict()` gives it in the calibration JSON's own keys (`calibration_file_io.calibration_to_dict`,
the same builder `write_calibration_to_json` uses) and `calibration_record()` everything.
`ReductionApp.reduce_data` holds a `HidraSetup` and passes its shift to reduction exactly as before;
`save_diffraction_data` passes the instrument to `HidraProjectFile.write_reduction_provenance(instrument=)`,
which **reads the record from it**. No parameter names or values are copied in `powder_pattern.py` or the
writer. User-approved production change; **the reduced data are bit-identical** (every dataset outside
the calibration group, HB2B_3510 uncalibrated, calibrated and η-sliced).

In the project file's `instrument/calibration`: attributes `calibration_source`, `calibration_file`,
`calibration_file_sha256`, `calibration_created`, `calibration_created_source`,
`calibration_lambda_source` (and `calibration_record_error`), plus two string datasets (no ~64 kB
attribute limit): `calibration_json` — the calibration JSON's dictionary (`Shift_x` ... `Lambda`, `error_*`,
`Status`; a `codes` attribute: −1 not refined, −2 undetermined) — and `provenance_json`.
`read_reduction_provenance()` returns the record with `calibration_record()`'s keys;
`read_reduction_instrument()` rebuilds the `HidraSetup`, so a run can be **reduced again with exactly
the recorded calibration** (verified: shift, errors, λ and status identical to the calibration file).

**The geometry lives within the calibration** (user decision, 2026-11-05: the detector pixel size and
spatial distortion are to be calibrated later, so the geometry is part of the calibration, not separate).
Every calibration JSON carries a `geometry` block — `detector_size` [rows, columns], `pixel_dimension`
[x, y] (m), `arm_length` (m) — written by `FitCalibration` (its run's geometry), the ring calibration (its
detector model's), its driver and the GUI Export (`write_calibration_to_json(..., geometry=)`;
`read_calibration_geometry()` returns it, or None for older files). `HidraSetup` holds it
(`calibration_geometry`, `geometry_source`): from the calibration file, or labelled
`"run (the calibration file has no geometry)"` for an older file, or `"run"` for a workspace shift. **If the
calibration's geometry differs from the run's, the record flags it** (`geometry_note`) and keeps the
run's geometry (`run_geometry`), because **reduction still uses the run's geometry exactly as before** —
applying a calibrated geometry is the future step. In the project file the geometry travels inside
`calibration_json` (attributes `calibration_geometry_source`, `calibration_geometry_note`,
`calibration_run_geometry`), so `read_reduction_instrument()` rebuilds the whole `HidraSetup` — geometry
and calibration — from the file alone, with the geometry the reduction used. Reduction does **not**
write the separate `instrument/geometry setup` group (user decision: one source of truth); that group
keeps what the NeXus conversion writes. Like `provenance`, `geometry` is an extra key that PyRS's own
reader ignores; check the deployed release's reader before the shared `HB2B_Latest.json` gains either.

For a workspace shift only the values are recorded (`calibration_json` without `error_*` or `Status`);
λ only if the workspace marks it **calibrated** and every sub-run agrees, else null with
`lambda_source` saying why — a workspace's plain wavelength may be the monochromator's nominal value, and
2θ₀ and λ are never inferred from the monochromator. A reduction's record **replaces the earlier one
entirely** (append mode), including an old calibration file name, SHA-256 and vanadium run.

**Never stops a reduction or the save:** the calibration group is created if a file lacks it; building
the record (`calibration_record()`) and writing it are guarded — on failure nothing is half-written, the
record is `source = "unknown"` with every reason, and the failure is warned about and logged with its
traceback; a provenance that cannot be read is noted in the record, not raised. Integration tests assert
no `error` and no `unknown` source, so a regression fails a test.

Not covered (reported, not changed): "Combine runs" (`combine_runs_model`) writes a merged project
file without its source files' calibration records (fixed by #1063, below);
`HB2BReductionManager.save_reduced_diffraction` had no callers (removed by #1064, below).

## Combine runs refuses different calibrations and flags historic files (2026-11, #1063)

"Combine runs" (`CombineRunsModel`) merged reduced project files and wrote no calibration record at
all. It now **checks the sources' records (#1053 part B) before anything is loaded**, by
`file_object.combine_reduction_records` (user decisions, 2026-11-05):

- **Different calibrations are refused** (`CalibrationMismatchError`): "the same calibration" means
  the same applied calibration — its values, errors, status and **geometry** (the record's
  `calibration`, i.e. `HidraSetup.calibration_dict()`) — and the same calibration-file fingerprint. The message
  groups the files by calibration (file name, λ, shift); the GUI shows it as "Runs reduced with
  different calibrations" and nothing is combined.
- **The same calibration with different vanadium runs asks the user** (`VanadiumMismatchError`, now
  `ReductionMismatchError`, which also covers masks and binning: #1068, below; a Yes/No dialog listing which run normalized which files). If approved, the files are combined, the
  merged record carries no single `vanadium_run` (each source keeps its own) and its `warning` says the
  difference was approved by the user.
- **Historic files (reduced before #1053, no record) are allowed but flagged**, whether all files are
  historic or only some: a warning dialog ("Calibration not checked") names them, and the merged
  record's `warning` says so. With no recorded source the merged file records `source = "not recorded"`.

The merged file's `instrument/calibration` carries the shared record (the sources' calibration file
and SHA-256 written verbatim, not recomputed), the PyRS version, `warning`, and `sources_json`: every
source file with its SHA-256 and its own record. Merged data are bit-identical to before (the four
test project files, 362 sub-runs, every dataset outside the calibration group).

Also fixed in this path: the GUI's "Run Numbers" entry called `saveFileDialog` with an argument it
does not take (a `TypeError`, so no failure was ever shown); it now goes through the same checks and
messages as the file dialog. `tests/ui/test_merge_projectfiles.py` combines historic files, so the
"Calibration not checked" warning is a modal dialog there: the test records the message instead (and
asserts it), otherwise it blocks until the 300 s timeout.

## A reduced file records its mask and binning (2026-11, #1068)

Before this, a reduced project file stored the applied mask only as a per-pixel array
(`mask/detector/_DEFAULT_`) and its binning only implicitly (dataset names such as `eta_-3.0` and the
2θ vectors). The mask file's name was known only to `NeXusConvertingApp`, and production masks
(`/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml`) are replaced over time, as `HB2B_Latest.json` is.

- **Where it comes from.** `NeXusConvertingApp.convert()` records the mask's source on the workspace
  (`HidraWorkspace.set_detector_mask_source`): `{"source": "file", "file": <absolute path>, "sha256": ...}`
  from `provenance.file_record`, or `{"source": "none"}`. `ReductionApp.reduce_data` hands it to its
  `HidraSetup` (`mask_record`): the instrument owns the data and the writer reads from it (#1053's design).
  `NeXusConvertingApp.save()` writes it too (`HidraProjectFile.write_mask_source`), so converting to a
  file now and reducing it later keeps it. A workspace loaded from a project file reads it back, so
  re-reducing a saved file keeps it; a file converted or reduced before #1068 gives
  `{"source": "not recorded"}`, as does a record that cannot be read (it never stops a file loading).
- **Settings.** `ReductionApp.reduce_data` records `num_bins`, `min_2theta`, `max_2theta`,
  `delta_2theta`, `eta_step`, `eta_min`, `eta_max`, `mask_id` and `extra_mask` (the mask passed to
  `reduce_data` on top of the default: its name if named, `pixels_masked`, and the SHA-256 of which pixels
  it keeps, so two masks with the same name or size but different pixels differ; never the array). They
  are recorded once the reduction succeeded, as asked: `None` for a 2θ limit or step means computed for each sub-run, and
  the bins themselves are saved with the data.
- **Where it is stored.** `instrument/reduction/reduction_json` (JSON text: `mask`, `settings`), a sibling
  of `instrument/calibration` (user decision: mask and binning are not calibration, and the calibration
  record stays unchanged). A re-reduction replaces it; a write failure is warned about and leaves
  `reduction_record_error`, never stopping the save. `read_reduction_provenance()["reduction"]` returns
  it; `read_reduction_instrument()` restores `mask_record`.
- **No vanadium mask is recorded:** `reduce_to_2theta_histogram` applies the reduction's mask to the
  vanadium pixels as well (`reduce_hb2b_pyrs.py`, the `vanadium_counts_array[mask == 1]` line), so the
  vanadium's own conversion (always unmasked, `load_vanadium`) does not change which pixels count.
- **Combine runs** (user decision): the same calibration with a different mask (file SHA-256, none, or
  extra mask) or binning is asked about, Yes/No, together with different vanadium runs in one question
  (`ReductionMismatchError`, `accept_reduction_differences`; renamed from #1063's
  `VanadiumMismatchError`/`accept_vanadium_differences`). An approved difference leaves the shared
  `reduction` (or `vanadium_run`) out of the combined record; each source keeps its own under `sources`.
  Files whose mask is unknown (no record, `"not recorded"`, or a record that failed to write, named as
  such) are combined and named in the warning (dialog "Reduction not fully checked"): nothing is known to
  differ, so nothing is asked. What the sources agree on is kept separately (an approved binning
  difference keeps the shared mask), and binning compares numbers as floats (`3` equals `3.0`).
- **Effect on data:** none. HB2B_3510 reduced with and without a mask, calibrated, and η-sliced (3°) is
  bit-identical to `next` in every dataset outside `instrument/calibration` and `instrument/reduction`, and
  the calibration record's attributes are unchanged. The record adds about 0.4 kB.
## `HB2BReductionManager.save_reduced_diffraction` removed (2026-11, #1064)

The method wrote a project file through `HidraWorkspace.save_reduced_diffraction_data` without the
reduction record that `ReductionApp.save_diffraction_data` writes (`write_reduction_provenance`), so any
caller would have produced a file with no calibration or vanadium record. `grep -rn
"save_reduced_diffraction(" pyrs scripts tests docs` found no caller in the package, scripts or tests
(its last caller, `PyRsCore.save_diffraction_data`, was removed earlier; see "Second gap" above), so it was
deleted rather than wired into provenance. `HidraWorkspace.save_reduced_diffraction_data` stays: it is
the writer `ReductionApp` and "Combine runs" call. Save reduced data through
`ReductionApp.save_diffraction_data`, which writes the record.

## Peak-range JSON from the peak-fitting UI lost d0 and could not be loaded (2026-11-09, #925)

Two bugs, both in the peak-fitting UI (the texture-fitting UI has its own, working writer):

- `PeakFittingModel.save_peak_range_json` wrote the **whole** d0 list on every peak instead of that
  peak's value. Loading then crashed in `GuiUtilities.fill_peak_range_table` (`"{:.6f}".format(list)`).
- The viewer saved `self.list_peak_d0`, which was set to `[]` when the window opened and changed only
  when a file was loaded. The table's d0 values actually live in
  `PeakFitSetupView.list_fit_peak_d0` (kept in sync by `_update_fit_peak_ranges_plot`). So a normal
  save wrote `"d0": []` for every peak. The stale attribute is gone, and save/load use the widget's
  list.

`save_peak_range_json` now appends `.json` when the name has no extension (the non-native Qt dialog
does not). `load_peak_range_json` still reads files written before the fix. A d0 list with one entry
per peak resolves to that peak's value, which is exact. Any other list (in practice `[]`) falls back
to 1.0, the d0 the UI gives a newly drawn range.

**Keep the widget's three per-peak lists in step.** `list_peak_ranges`, `list_fit_peak_labels` and
`list_fit_peak_d0` are parallel lists. Any code that adds or removes a peak must update all three.
`remove_peak_range_table_row` used to skip d0. Removing row 0 then paired Peak1 with Peak0's d0.
Before #925 nothing read that list, so the mismatch was harmless. Once save started using it, the
wrong d0 was saved without any warning. Found in review and fixed in the same PR.
`QTableWidget.removeRow` emits no `cellChanged`, so the table-to-list sync in
`_update_fit_peak_ranges_plot` does not run after a removal.

**A `QObject` alone does not make a test GUI-tier.** `PeakFittingModel` is a `QObject` with signals.
It constructs, connects and emits with no `QApplication` and no display (verified with `DISPLAY`,
`QT_QPA_PLATFORM` and `WAYLAND_DISPLAY` unset). Only widgets need one. Its tests therefore live in
`tests/unit/pyrs/interface/peak_fitting/`.

Also found: the texture-fitting save/load dialogs, and still the calibration GUI's export/recipe
dialogs (`detector_calibration_viewer.py`), pass the file filter as `QFileDialog`'s *directory*
argument, so the filter never applies. The texture ones are fixed. The calibration ones are not.

`tests/ui/test_peak_fitting.py::test_peak_fitting` and
`tests/ui/test_texture_fitting.py::test_texture_fitting_viewer` fail on `main` (04102098) in a
workstation run, independent of this fix. CSV export builds a name like `HB2B_<path>.csv` that does
not exist. Not investigated further. The `FileNotFoundError` is raised inside the Qt event loop. The
session-scoped `my_qtbot` fixture (`tests/conftest.py`) reports it when it is torn down, so pytest
shows it as an ERROR on whichever GUI test runs **last**. That test is not at fault. In
`tests/ui/test_peak_fitting.py` that is now `test_remove_peak_range_row_drops_its_d0`.

Tests:
- `tests/unit/pyrs/interface/peak_fitting/test_peak_fitting_model.py`: round trip (in memory and on
  disk), file extension, int and `np.float64` d0, legacy files.
- `tests/ui/test_peak_fitting.py::test_save_load_peak_range_keeps_table_d0`.
- `tests/ui/test_peak_fitting.py::test_remove_peak_range_row_drops_its_d0`.

## Ring-based calibration vs FitCalibration as the default (2026-11, #1038)

Analysis only (no production change), on the three calibration runs available locally (HB2B_4971, HB2B_3510,
HB2B_3489), with `tests/data/HB2B_Mask_12-18-19.xml` (the production mask was not reachable) and the 2020
lattice constants. Script: `tests/scripts/default_calibration_study.py`; results in
`calib_diagnostics/1038/` (outside the repository). Pixel 2θ is compared over the detector at each run's arm
angles, never parameter by parameter: **Rot_y, Shift_x and 2θ₀ are exactly degenerate** (the ring cost is
flat to 1e-5 deg along Rot_y from -1 to +3.2 deg with the others free), so FitCalibration's Rot_y of 3.7 deg on
HB2B_4971 is not by itself a sign of failure.

- **Both costs have their minimum near the ring geometry.** FitCalibration's own η-wedge residual is lower at
  the ring calibration than at FitCalibration's own result on every run (HB2B_4971 7.5 vs 9.1 / 12.5 mdeg for
  the staged / full recipes; HB2B_3510 19.0 vs 23.8 / 21.7; HB2B_3489 28.4 vs 32.0 / 32.9). Started from the
  ring geometry, FitCalibration lowers its cost further (7.0, 17.4, 27.5 mdeg) and stays within 10-24 mdeg of
  the ring geometry (≤ 250 µε spread); from nominal it ends 30-200 mdeg away.
- **FitCalibration stops short because of its start and its rough cost, not its iteration limit.** It reports
  convergence (status 3) after about 20 evaluations per stage; an `lm` polish moves it by ≤ 3 mdeg; along the
  line from the ring geometry to its result its cost rises steadily with ±0.3-0.7 mdeg of roughness (peaks
  re-binned and re-fitted at each evaluation, #1030).
- **The ring refinement is start-independent; its extraction is not.** From 14 perturbed starts (shifts
  ±2-5 mm, rotations ±0.5-1 deg, λ ±0.2-0.3%) with the ring points held, every refinement ends at the same
  geometry (0.00 mdeg). Extracting the rings at a perturbed start does change the result: up to 7 mdeg on
  HB2B_4971, 42 on HB2B_3510, and on HB2B_3489 one start found 31 of ~60 points and ended 0.53-0.70 deg off
  with a *better* RMS (0.0127 deg) and no failure status (a silent failure). Extracting again at the refined
  geometry (3 rounds) removes it: worst / median spread 0.24 / 0.14 mdeg (HB2B_4971), 4.4 / 2.4 (HB2B_3510),
  22 / 10 (HB2B_3489, consistent with its 0.030 deg RMS over ~60 points: ≈ 11 mdeg).
- **The mask does not change the ring calibration** (the extraction already skips 40 pixels at each edge).
- **Strain consequence (C4).** Relative to the 3-round ring geometry, FitCalibration from nominal implies a
  strain varying across the detector by up to 260-440 µε (HB2B_4971), 1100-1240 µε (HB2B_3510) and
  1900-2100 µε (HB2B_3489) at one arm angle (95% range), plus a uniform offset of up to -400 µε (which a d0
  measured with the same calibration cancels; the variation across the detector it does not). The
  whole-detector 1D peak positions cannot tell the geometries apart (differences within their scatter): the
  difference is in the ring shape over the detector height (strip drift 20-38 vs 3 mdeg on HB2B_4971).
- **Cost (C5).** Ring calibration, 3 rounds, HB2B_4971: 36 s wall (22 s of computation, 3.8 GB); FitCalibration:
  about 40 min. Both write errors and a provenance block (#1053).

Recommendation (a production change would be a separate, approved proposal): make the ring-based calibration
the default **with the extraction repeated at the refined geometry** (at least 2 rounds; 3 tested), keep
FitCalibration available (started from the ring geometry it agrees within the data's precision), and flag a
round whose ring-point count drops or whose geometry moves by more than the data's precision. Limits: three
runs, one monochromator setting per run as recorded, a test mask.

## The ring calibration extracts the rings again until converged (2026-11, #1038 step 1)

Refining fixed ring points ends at the same geometry from any start (0.00 mdeg from 14 starts on three runs),
but where `extract_ring_points` looks for the rings depends on its starting geometry. From a poor start one
extraction can find only part of the rings: on HB2B_3489, from shifts of 5 mm, rotations of 1 deg and λ +0.3%,
it found 31 of about 60 points and the result was 0.53-0.70 deg off with a *better* RMS and a converged status.
`pyrs.calibration.ring_rounds` (re-exported from `pixel_pos_calibration`) therefore repeats the extraction where
the refined geometry predicts the rings:

- **Rule (user-approved, from the settling study below):** round 1 extracts at the start and refines
  (`PixelCalibration.iterate`); each further round extracts at the previous round's geometry and refines; stop
  after round k ≥ 2 once that round's **step** is less than **half the statistical precision** σ = RMS·√(8/N)
  (8 parameters, N ring points); **at most 10 rounds** (user decision), reaching which warns (`warning_log`,
  `warnings`, `extraction_converged = False`); nothing is corrected.
- **Step (user decision, after review):** the larger of the change of the detector's pixel 2θ over the whole
  detector (`pixel_two_theta_change`: geometry only) and the change of the predicted ring positions at the
  ring points (`ring_position_change`, the change of their 2θ residuals: geometry *and* wavelength). The next
  extraction looks for each ring at its Bragg angle, so a round that mainly moves λ moves the search windows
  too: in the settling study λ alone moved the rings by up to 1.6σ / 4.2σ / 2.7σ between rounds (HB2B_4971 /
  3510 / 3489), which the geometry-only step does not see. The ring-position step alone would stop about one
  round earlier (2-3 / 2-4 / 2-9 rounds) with the same spread between starts (0.4 / 4.1 / 17.9 mdeg).
- **Why relative to σ:** on sparse runs one or two ring points drop in and out between rounds (60-63 on
  HB2B_3489), so the geometry keeps stepping by about σ instead of settling. A fixed tolerance (1 mdeg) was
  reached by only 6 of 15 starts on HB2B_3510 and 2 of 15 on HB2B_3489 within 10 rounds.
- **Settling study** (10 rounds from nominal and 14 perturbed starts per run; step = largest pixel-2θ change
  between consecutive rounds over the detector at the run's arm angles):
  HB2B_4971, σ 2.3 mdeg: steps fall to ~0.1 mdeg (converges), spread between starts 0.19 mdeg, 3 rounds;
  HB2B_3510, σ 6.2 mdeg: steps jitter at 1-3 mdeg, spread 3.5 mdeg, 3-5 rounds by the rule;
  HB2B_3489, σ 10.6 mdeg: steps jitter at 1-22 mdeg, spread 17.8 mdeg, 3-9 rounds (median 4).
  The spread between starts matches σ: it is the calibration's precision on that run, not a failure.
- **Where it applies:** `calibrate_rings` (what `pyrs-pixel-calibration` runs; `--max-rounds`, default 10; `1`
  is the single extraction from before) and `continue_rounds` (the GUI's ring-based "iterate"; "Rounds" field,
  saved with a recipe). Comparisons are by pixel 2θ (`pixel_two_theta_change`), never parameter by parameter
  (Rot_y, Shift_x and 2θ₀ are degenerate).
- **Record:** `PixelCalibration.extraction_rounds` (per round: ring points, RMS, σ, both steps and the deciding
  one, values; the first round of each call also its `start`, because later rounds start where the previous
  one ended) goes into the calibration's provenance as `extraction_rounds`; a calibration from a single extraction carries no such key,
  so earlier provenance is unchanged. The GUI's provenance also records `inputs.extraction.rounds` and
  `max_rounds`, and its ring-point marks show the last round's points.
- **Effect (from nominal, against one extraction, final rule):** HB2B_4971 converges in 3 rounds (steps 2.2,
  0.5 mdeg; σ 2.3), HB2B_3510 in 4 (9.6, 5.5, 2.9; σ 6.3), HB2B_3489 in 5 (19.4, 12.1, 12.0, 4.1; σ 10.6); the
  calibration moves by 1.7 / 9.3 / 12.4 mdeg of pixel 2θ from the single extraction's. A round takes about 7-10 s
  (HB2B_3510, 4 rounds: 28 s unloaded).
- **Reference file:** `tests/data/calibration_tests/HB2B_3510_ring_reference.json` was made with one extraction.
  `test_driver_calibrates_hb2b_3510` therefore pins it with `--max-rounds 1` (2e-5 A, 2 mdeg), and
  `test_driver_rounds_agree_with_the_reference` checks the default (repeated) result against it within its
  errors: λ moves by 4e-5 A (its error 2e-4 A) and the pixel 2θ by about 9 mdeg (3σ ≈ 19 mdeg).

## The ring-based calibration is the GUI's default (2026-11, #1038 step 2)

User-approved, after the C1-C5 evaluation and step 1 (repeated ring extraction) above:

- **GUI:** the calibration window opens on "Ring-based" (`DEFAULT_METHOD`, listed first in
  `CALIBRATION_METHODS`); FitCalibration stays one click away. With the ring-based method selected and an empty
  recipe, the first recipe row is "iterate" (Fit Peaks, then Calibrate); a recipe is never changed by this.
- **Old recipes keep their method.** A recipe saved before the method choice existed (#1045) has no
  `calibration_method`: it is FitCalibration's, the only method then, unless it contains "iterate", which only
  the ring-based method has. It must not run with the new default, so `load_json` sets the method explicitly
  (`tests/data/calibration_tests/test_ui_recipe_load.json` is such a recipe).
- **Command line unchanged:** `pyrs-calibration` (FitCalibration, JSON input) behaves as before; its docstring
  and usage text point to `pyrs-pixel-calibration` as the default. (Reported, not changed: its usage text says
  the default `"method"` is shifts+rotations, but `REFINE_METHOD` defaults to "full".)
- **Docs:** a "Detector Calibration" section in the user guide (`basics.rst`) and the ring calibration in the API
  pages; the user docs now load `sphinx.ext.napoleon`, as the developer docs do, so Google-style docstrings
  render (user-docs build warnings 38 -> 3).
- **Effect:** a calibration made with the GUI's defaults moves by 30-200 mdeg of pixel 2θ from FitCalibration's
  from the nominal geometry (260-2100 µε across the detector) on the three runs studied. Existing calibration
  files, including `HB2B_Latest.json`, are untouched until someone calibrates again; reduction never chooses a
  method. Every calibration file records its method in its provenance (`method`: "ring" or "peak-fit").
