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

Two dead ends worth recording so they aren't retried:
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

**Why this matters going forward:** if the `tests` job fails again with
a `Segmentation fault` (exit code 139) *after* a passing pytest summary
line, that's a process-teardown crash, not a test regression. This
class of bug is inherently hard to verify locally under `offscreen`;
matching CI's real display server (`xvfb-run` with the `xcb` platform,
not `QT_QPA_PLATFORM=offscreen`) is likely required to reproduce it
directly — I was not able to reproduce this specific segfault locally
even once. When a fix like this can't be verified against the actual
failure, verify what you *can*: no regressions in the full local suite,
and (critically, since this is the actual defect the two dead ends
above introduced) that the summary output and coverage report are still
present in the log after the change.

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
at all, so it needed no change. If another caller of
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
  piece of code defines the detector, is the follow-up #1011/#1020.

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
