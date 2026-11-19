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

## Detector rotation matrices are float64; float32 limited angles to ~3e-6° (2026-10)

`ResidualStressInstrument._cal_rotation_matrix_x/y/z` (`pyrs/core/reduce_hb2b_pyrs.py`) built their
matrices as float32 (~7 significant digits), so every rotation angle — arm 2θ, 2θ₀, Rot_x/y/z —
resolved only to ~3e-6°. For reduction that is irrelevant, but calibration differentiates its
residual numerically: scipy's default finite-difference steps (1.5e-8 for 2-point, ~6e-6 for
3-point) sit at or below that resolution, giving zero or noisy angle derivatives (found in the
ring-based calibration work on `detector_calib_v3`, commit 095c7980 there). The matrices are now
float64 (user-approved). Effect on reduction: pixel 2θ/η move by ≤ 4.5e-7°; unit, reduction and
calibration integration tests unchanged. `test_rotation_matrix_resolves_tiny_angle_change` fails
with float32.

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

## Zero-count pixels are given σ = 1 before 2θ histogramming (2026-11-12)

Found while scoping the pixelation-uncertainty study
([plan](planning/PixelationUncertainty_Plan.md), [board #5](https://github.com/orgs/neutrons/projects/5)).
No code changed.

- **Per pixel:** `PyHB2BReduction.histogram_by_numpy` (`pyrs/core/reduce_hb2b_pyrs.py`) sets
  σᵢ = √Nᵢ, then replaces σᵢ = 0 with 1. A 2θ bin of *n* pixels, *n₀* of them empty, therefore
  gets σ²_bin = Σ Nᵢ + n₀ rather than the Poisson Σ Nᵢ. The absolute excess n₀ grows with the number
  of pixels in the bin (bin width, η range, mask). The *relative* inflation
  (Σ Nᵢ + n₀) / Σ Nᵢ ≈ 1 + mean(e^(−λᵢ)) / mean(λᵢ) depends only on the counts per pixel λ, so
  changing the bin width or η wedge at fixed λ does not change it (measured below).
- **Masked pixels** are removed before histogramming and do not add to *n₀*. **Empty bins** get
  intensity 0, σ = 1. **Vanadium:** pixels with < 0.9 vanadium counts are dropped, and vanadium σ
  uses the same 0 → 1 substitution.
- **`variances` holds σ, not σ².** The array (`variances_vector`, `HidraWorkspace._var_data_set`) is
  passed unchanged to Mantid as `DataE` (`mantid_helper.generate_mantid_workspace`), which Mantid
  reads as σ. The fit weights are therefore consistent with σ; only the name is wrong.
- **Fitting** uses `FitPeaks`' default least-squares cost, weighting each bin by 1/σ²_bin.

For a uniform Poisson mean λ per pixel, the reported variance is (1 + e^(−λ)/λ) times the true
variance, i.e. 3.2× σ at λ = 0.1 and 1.17× at λ = 1.

**Measured (2026-11-12, Phase 0, [#1106](https://github.com/neutrons/PyRS/issues/1106)).** Run 5344
(Si422 arm scan, 12 sub-runs, 0.5–1.1 counts per pixel) was reduced by the current reduction code with
the IPTS-36131 autoreduction settings: 2024-07-09 calibration, 2019-11-18 mask, vanadium 4501, 720 bins
(~0.023°, ~1,270 pixels per bin; ~230 per 3° η wedge). The stored autoreduced file
(`shared/autoreduce/HB2B_5344.h5`, made in 2025-09 by an older PyRS) is *not* reproduced by today's
code: its 2θ axis is wider (78.02–95.24° in 0.0239° steps for sub-run 7, against 78.90–95.08° in
0.0225°), and its intensities differ by ~2 % (median). Production's own `reduce_to_2theta_histogram` supplied the intensities and σ,
and the formula σ²_rel = (ΣN + n₀)/ΣN² + 1/ΣV + 1/V_max reproduces that σ to 4×10⁻¹⁶.
- **The inflation is exactly the Poisson prediction.** Per bin, observed (ΣN + n₀)/ΣN over
  1 + Σe^(−λᵢ)/Σλᵢ has a median of 0.996–1.005 in every sub-run and wedge, where λᵢ comes from the
  same-day vanadium 5339 times the bin's scale. The vertical efficiency stripes raise it only ~1.5 %
  above the uniform-λ formula.
- **Background bins** (the lower half by λ̄, per sub-run): variance inflated 2.4–3.1× (median), so
  **σ is reported 1.5–1.8× too large**. **Peak bins** (top 1 % by λ̄, λ̄ ≈ 3–15): 1.00–1.03. Over all bins, the median σ_reported / σ_Poisson is
  1.4–1.6 (90th percentile 1.7–1.9, max 2–4.5). 80–94 % of bins have variance inflated above 1.5×.
- **η wedges** (3°, centres 0, ±3, ±6) show the same inflation as the full detector, confirming
  that it is set by λ, not by the number of pixels. So does the bin width: with 360, 720 and
  1,440 bins (2,532 to 634 pixels per bin) the median inflation is 2.39–2.40, and the median
  σ_reported / σ_Poisson 1.53–1.54.
- **The vanadium terms barely matter:** σ_reported / σ_Poisson ≈ √((ΣN + n₀)/ΣN), because ΣV is large.
- Consequence for fitting (Phase 1, [#1107](https://github.com/neutrons/PyRS/issues/1107)): the fit
  weights background bins 2–3× too little relative to peak bins.

**Arm-angle sign.** `ReductionManager.setup_reduction_engine` passes **−(logged `2theta`)** to
`PyHB2BReduction.set_experimental_data`, converting DAS to PyRS convention. Building the instrument
with the logged sign mirrors the pixel 2θ map along x (column 0 then reads 78.65° instead of
61.35° at a 70° arm) and puts η near ±180°, so reduction's η wedges (centred on 0°) select no pixels.
The Phase 0 zero-count results in the next entry were rechecked with the correct sign: observed/expected zeros
moved by ≤ 0.002, and the row/column mismatches by ≤ 0.3 %.

## `FitPeaks` reports unscaled covariance errors; zero-count inflation enlarges weak-peak errors (2026-11-12, #1107)

Phase 1 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.
This was tested on 300 Poisson realisations of a Gaussian on a linear background (5344's bin width
0.0226°, 4° window, 172 dof). The fits used the `FitPeaks` arguments of `MantidPeakFitEngine.fit_peaks`
(`HighBackground=True`, `RawPeakParameters=True`, default least-squares cost, 500 iterations).

- **Parameter errors are not rescaled by the reduced χ².** Doubling every σ (`DataE`) doubles every
  reported error. With σ = 1 everywhere, the errors fall to ~1 % of the real scatter. The errors
  are √(diag covariance) for the σ supplied, so **any error in σ passes straight into the reported
  peak uncertainties**.
- **The `chi2` column is the reduced χ²:** 1.00 with σ = √N, 0.25 with 2√N.
- **With σ = √N, the errors are right:** reported error over the real scatter is 0.96–1.05 for
  centre, height, width and background.
- **With reduction's σ = √(N + n₀)** (background 0.63 counts per pixel, 1,268 pixels per bin, as
  in 5344), the fitted values do not change: scatter and bias are within 1 % of the Poisson fit. But
  the reduced χ² drops to 0.57–0.68, and the errors grow as the peak weakens:

  | Peak top, counts per pixel | Centre | Height | Width | Background |
  |---|---|---|---|---|
  | 10 | 1.05 | 0.97 | 1.05 | 1.33 |
  | 2.2 | 1.15 | 1.04 | 1.18 | 1.37 |
  | 0.95 | 1.27 | 1.20 | 1.28 | 1.36 |

  The table gives reported error over the real scatter. Strong peaks carry their information in bins
  where the inflation is ~1. Weak peaks rely on the background bins, where σ is inflated 1.5–1.8×.
  Peak-centre (strain) uncertainties of weak peaks are therefore over-reported by up to ~30 %.
- **Real data shift the fitted values (2026-11-16).** Each 5344 sub-run's strongest peak (Si, 2θ
  75–98°) was refitted with PyRS's `FitPeaks` arguments (pseudo-Voigt, mixing start 0.6, linear
  background). The fits used the intensities reduced as in the previous entry, once with
  production σ and once with Poisson σ (σ²_data = ΣN). Changing σ from production to Poisson
  moves every sub-run the same way:
  - peak centre +0.65 ± 0.19 mdeg on average (11 of 12 positive; 0.1–1.7 σ each; ≈ 6 µε at 90°);
  - FWHM smaller in all 12 (−0.3 to −0.95 σ);
  - integrated intensity larger in all 12 (+0.5 to +2.1 %, +0.6 to +1.7 σ);
  - centre and FWHM errors 2–9 % smaller, intensity errors 13–29 % smaller;
  - reduced χ² 0.8–3.4 → 1.3–4.2.

  On the synthetic peaks above, the values did not move. The difference is that χ² > 1 here: the
  pseudo-Voigt + linear model does not fully describe the measured peaks, so re-weighting the
  background moves the best fit. Since `FitPeaks` does not rescale by χ², the Poisson-σ errors are
  themselves too small by √χ² for this model misfit; production's inflated σ partly masks that.

## A stale vanadium biases η-wedge (texture) intensities by up to ~10 % (2026-11-16, #1107)

Phase 1 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.
Each 5344 sub-run was reduced into 3° η wedges (centres 0, ±3, ±6°) by the current reduction code
(2024-07-09 calibration, 2019 mask, 720 bins), once with the production vanadium 4501 (2024-07) and
once with the same-day flood 5339. The sub-run's strongest peak was fitted in each wedge with
PyRS's `FitPeaks` arguments.

- **The wedge-to-wedge ratio I₄₅₀₁ / I₅₃₃₉, divided by its wedge mean, spreads by 0.6–11.7 %
  (relative std) depending on the sub-run.** The fit's own error is 1–4 %; both fits share the same
  counts, so that is an upper bound on the noise in the ratio. The worst sub-runs show a
  pattern symmetric in η: the central wedge reads low and the ±6° wedges high, e.g. 0.83 → 1.17
  (sub-run 12) and 0.84 → 1.12 (sub-run 5). Others (4, 6, 11) stay within ±1–3 %.
- **For texture** (pole figures from η-resolved intensities), a vanadium that does not match the
  sample's vertical efficiency pattern therefore distorts relative intensities by up to ~10–17 %.
  This is larger than the counting error. Whether the cause is time drift or the different
  collimator (C508 with collimator, against C511) is not separated.

## Pixel grouping: group vertically, not horizontally (2026-11-16, #1107)

Phase 1 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.
The 5344 sub-runs were rebinned in software: counts and vanadium summed over g_x × g_y blocks, with
x horizontal (2θ) and y vertical (η), and a coarse pixel kept only if all its fine pixels are
unmasked. Each grouping was reduced by the current code on a coarse `DENEXDetectorGeometry`
(pixels g·0.293 mm; coarse-pixel 2θ and η equal the block means to ≤ 10⁻⁵°), onto the native
720 bins, and the strongest peak was fitted in the native window. The strain is relative to the
1 × 1 fit (Δε = sin θ₁ₓ₁ / sin θ − 1), with σ_ε = ½ cot θ · σ₂θ. Medians over 12 sub-runs:

| g_x × g_y | Zeros | Counts per pixel | σ_rep/σ_Poisson | σ_ε (µε) | Strain shift RMS (mean), µε | χ² |
|---|---|---|---|---|---|---|
| 1 × 1 | 61.5 % | 0.74 | 1.53 | 7.3 | — | 1.75 |
| 1 × 2 | 40.6 % | 1.5 | 1.20 | 7.1 | 5.5 (−4.5) | 2.09 |
| 1 × 4 | 18.0 % | 3.0 | 1.04 | 7.1 | 7.7 (−5.9) | 2.26 |
| 1 × 8 | 3.7 % | 6.0 | 1.00 | 7.0 | 8.1 (−6.2) | 2.30 |
| 1 × 16 | 0.3 % | 11.9 | 1.00 | 7.1 | 6.1 (−3.8) | 2.26 |
| 2 × 1 | 41.3 % | 1.5 | 1.21 | 7.0 | 33 (+5.5) | 2.74 |
| 4 × 1 | 21.1 % | 3.0 | 1.05 | 8.6 | 426 (−88) | 6.22 |
| 8 × 1 | 6.9 % | 6.0 | 1.01 | fit fails (28 empty bins in window) | — | — |
| 2 × 2 | 20.2 % | 3.0 | 1.05 | 6.9 | 34 (+3.8) | 2.97 |
| 4 × 4 | 0.5 % | 11.9 | 1.00 | 8.6 | 494 (−109) | 6.31 |

1. **Zero pixels** fall the same way in either direction, set only by the pixels per group
   (λ × g): 61 % → ~20 % at 4 pixels, → < 7 % at 8.
2. **Uncertainties.** Variance inflation and σ_rep/σ_Poisson reach ~1 at ≥ 4 pixels per group.
   For this strong peak the strain error barely changes (7.3 → 7.0 µε, the ~5 % production-σ
   excess). The intensity error falls from 0.91 % to 0.73 %. Grouping removes the σ artifact; it
   adds no information.
3. **Strain.**
   - **Vertical grouping itself does not move strain.** With Poisson σ at every grouping, the
     shift against 1 × 1 is ≤ 0.6 µε RMS up to 1 × 8 on 5344 (2.1 µε at 1 × 16). It is ≤ 2 µε mean
     and ≤ 5.6 µε RMS up to 1 × 16 on the sparse 3489 (Si331, peaks at 73–108°, 0.066 counts per
     pixel). The −4 to −6 µε offset in the table is the σ artifact below, not the grouping.
   - **Production σ moves weak-peak strains (2026-11-16).** Fitting the native data with Poisson σ
     instead of production σ changes the strain by −5.7 µε mean (7.9 RMS) on 5344, and by
     **+40 µε mean (56 µε RMS)** on 3489. With production σ, grouping removes zeros and so moves the
     result step by step toward the Poisson one: +5, +14, +26, +34 µε at 1 × 2, 4, 8, 16 on 3489.
     Which weighting is unbiased is for the Phase 2 Monte Carlo; Poisson σ is the correct
     statistical weight.
   - **Production σ over-reports weak-peak strain errors about 2×.** On 3489, σ_ε is 62 µε with
     production σ and 30 µε with Poisson σ. Grouping brings the production value to 37 (1 × 4),
     32 (1 × 8) and 31 µε (1 × 16), while the Poisson value stays at 30 µε for every grouping:
     grouping fixes the reported error and adds no information. The intensity error falls from
     9.7 % to 5.1 % (1 × 4) and 3.9 % (1 × 8).
   - **Horizontal grouping biases strain unless the 2θ bin is at least as wide as the coarse
     pixel.** Reduction does not split pixels across 2θ bins, so a coarse pixel's counts land in one
     bin and a too-fine binning aliases the pattern: χ² rises and the peak moves. Strain shift, RMS,
     with Poisson σ:

     | g_x (pixel width) | 720 bins (0.0225°) | 360 bins (0.045°) | 180 bins (0.09°) |
     |---|---|---|---|
     | 2 (0.034°) | 34.5 µε | 3.6 µε | 3.7 µε |
     | 4 (0.068°) | 490 µε | 83 µε | 9.1 µε |
     | 8 (0.137°) | fit fails | 390 µε | 1,040 µε |

     Even when it works, horizontal grouping gains nothing. The native strain error is 7.3–7.4 µε
     at every bin width, while wider bins sample the peak more coarsely and raise χ² (1.75 → 2.85 →
     5.61). Vertical grouping removes the zeros without this constraint.
   - **Stress** is a linear combination of strains along several directions, so these strain
     shifts and errors carry into stress in proportion. No stress value was computed: 5344 and 3489
     are calibration runs of unstressed powder standards (see the sample note below).
   - 3489 used the earliest Si331 calibration (2024-02-09) and vanadium 3446 (2023-05-23, C501,
     Si422 monochromator). **That calibration file turned out to be uncalibrated (all zero); the
     re-check below supersedes the 3489 numbers above.**
   - **Re-check on Ni reflections with the matching calibration (2026-11-18).** 3489 was rerun with
     `PreC509/HB2B_Si331_2023-05-30T17-28.json` and summarised over the Ni sub-runs only (4–6
     Ni 220, 10–12 Ni 311; 1–3 and 7–9 are Al tube peaks). Each cell gives 1 × 2 / 1 × 4 / 1 × 8 /
     1 × 16:
     - grouping strain shift with Poisson σ, mean (RMS): +1.2 (3.6) / −0.3 (3.7) / +0.4 (5.4) /
       **−10.5 (11.7)** µε;
     - with production σ: +3.4 (4.6) / +8.9 (12.2) / +17.4 (25.7) / +13.8 (27.1) µε;
     - σ_ε with production σ: 62.5 / 48.7 / 41.4 / 38.3 µε (native 85.0). With Poisson σ: 37.1 /
       37.2 / 37.2 / 37.1 µε (native 37.3), so production σ over-reports 2.3× at 1 × 1;
     - intensity error with production σ: 13.6 / 10.3 / 8.4 / 7.5 % (native 18.7 %);
     - native production → Poisson shift: +27.6 µε mean (40.7 RMS) on Ni; +39.6 (47.9) over all 12
       sub-runs.

     The conclusions hold for 1 × 4 and 1 × 8. **New: 1 × 16 is not safe with this calibration.**
     It rotates the detector by Rot_z = 1.87° in its own plane, so rings cross the rows at an angle
     and averaging 16 rows (4.7 mm) shifts the fitted 2θ. The uncalibrated file had Rot_z = 0, which
     hid this. The safe group height therefore depends on Rot_z.
   - **Sample note (2026-11-18).** "Si422" / "Si331" in these run titles are the **monochromator**,
     not the sample.
     - **3489** is a calibration run of **Ni powder in a sealed Al tube** (IPTS-25855). Its fitted
       "strongest peaks" are Al 220 (73.5–73.9°, sub-runs 1–3), Ni 220 (87.1–87.5°, 4–6), **Al 222**
       (94.2–94.8°, 7–9) and Ni 311 (107.7–108.4°, 10–12) at λ = 1.719 Å.
     - **5344** is the "Powder Standards Assembly" (calibration powders in the Al rack). Its peaks are
       Ni 220, 311 and 222, and Fe 211 (81.6°), which at λ = 1.531 Å nearly coincides with Al 222
       (81.7°).

## Monte Carlo: no strain bias; Poisson σ gives honest errors, production σ over-reports them (2026-11-16, #1108)

Phase 2 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.

**Setup.** The truth is the real fit of 5344 sub-run 7: Si at 91.89°, pseudo-Voigt with FWHM 0.65°
and mixing 0.56, on a linear background, evaluated with Mantid's own fit functions. It was placed
on that sub-run's calibrated detector (2024-07-09 calibration, 2019 mask) as
λᵢ = k · Vᵢ · R(2θᵢ) / V_max. Vᵢ is the same-day vanadium 5339, used both as the true per-pixel
efficiency (stripes included) and for reduction. k sets the mean count per used pixel: 0.066 (weak,
like 3489) or 0.74 (5344). Each Poisson realisation was reduced by the current production code
onto the native 720 bins at 1 × 1, 1 × 4 and 1 × 8, and fitted with PyRS's `FitPeaks` arguments,
once with production σ and once with Poisson σ. The truth for each grouping is the fit to its
noise-free expected pattern; it reproduces the input centre to within 0.8 µε. With
`--resample-vanadium`, a fresh Poisson vanadium is drawn for every realisation.

**Weak level (0.066 counts per pixel), 600 realisations,** with fixed / resampled vanadium:

| | Centre pull width | Strain scatter (µε) | Reported σ_ε (µε) | Intensity pull width | χ² |
|---|---|---|---|---|---|
| 1 × 1, production σ | 0.58 / 0.61 | 20.1 / 21.1 | 34.6 | 0.41 / 0.44 | 0.19 |
| 1 × 1, Poisson σ | 0.96 / 1.01 | 19.1 / 19.9 | 19.8 | 0.99 / 1.05 | 0.98 |
| 1 × 4, production σ | 0.85 / 0.89 | 19.3 / 20.2 | 22.5 | 0.67 / 0.71 | 0.50 |
| 1 × 8, production σ | 0.92 / 0.96 | 19.1 / 19.9 | 20.7 | 0.80 / 0.84 | 0.68 |

(With 1 × 4 or 1 × 8, Poisson σ gives the same results as at 1 × 1.)

1. **No option biases strain.** Every strain bias is ≤ 1.8 ± 0.9 µε, and intensity biases are
   ≤ 0.3 %, when the model matches the data. So the +40 µε production-vs-Poisson shift seen on real
   3489 is not a statistical bias. It is the σ weighting interacting with the misfit of the
   pseudo-Voigt + linear model to the measured peaks (the real χ² > 1).
2. **Poisson σ gives honest errors** (pull width 0.96–1.05, χ² ≈ 1).
3. **Production σ over-reports** the strain error 1.7× and the intensity error about 2.4×. It is
   also slightly less precise: its strain scatter is ~5 % larger than with Poisson weighting.
4. **Vertical grouping repairs production σ.** At 1 × 4 the centre pull width is 0.85–0.89, and at
   1 × 8 it is 0.92–0.96, with the scatter at the Poisson optimum.

**Strong level (0.74 counts per pixel), 300 realisations:** every option is close to honest.
- **Centre pull width:** 0.87–0.90 with fixed vanadium, 0.93–0.95 when it is resampled. The
  remaining few per cent come from reduction's σ including the vanadium's own counting noise
  (1/ΣV + 1/V_max). That noise is real, but it does not appear when the same vanadium is reused.
- **Integrated intensity, resampled vanadium:** pull width 1.18–1.19, i.e. the error is
  under-reported by ~19 %. V_max is one normalisation constant for all bins, so its noise is common
  to every bin, while the fit treats bin errors as independent. Relative intensities normalised by
  the same vanadium (texture) are unaffected, because the common factor cancels.

**3489's own peaks (2026-11-16).** The Monte Carlo was repeated on 3489 sub-runs 7 (94.2°) and 10
(107.7°), with their own real fit as the truth, their real count level (0.083 and 0.081 counts per
pixel), vanadium 3446 resampled, and 400 realisations:
- **Production σ makes some weak-peak fits fail.**
  - **Sub-run 10:** 24 % of production-σ fits land more than 5 robust-σ from the truth (up to
    867 µε), so the scatter is 260 µε against a reported 65 µε (pull width 4.3). Poisson σ never
    does this: scatter 32 µε, reported 31 µε, pull width 1.03, bias +10 ± 1.6 µε. With 1 × 4
    grouping, production σ still fails in 8 % of fits.
  - **Sub-run 7:** neither fails; production σ pull width 0.65, Poisson 1.03.
- **Real 3489 shifts compared with the simulation's production − Poisson spread.**
  - **Sub-run 7: not statistical, so the real data hold a misfit the model lacks.** The simulated
    difference is 0 ± 7.5 µε (1 × 1) and never reaches the real +90 µε (12σ). The real Poisson
    χ² is 2.1. Candidates: peak asymmetry, a shoulder, or an efficiency mismatch (vanadium 3446 was
    taken with the Si422 monochromator, 3489 with Si331).
  - **Sub-run 10: consistent with one failed production-σ fit.** The real +113 µε is inside the
    heavy-tailed simulated difference (P(|Δ| ≥ 113 µε) = 0.25).
- **Background curvature is not the cause.** Refitting the real 3489 sub-runs with a quadratic
  background leaves the production − Poisson shifts unchanged in 11 of 12 (mean +40 → +16 µε, the
  change driven by one fit that went bad). On 5344 sub-run 7, the real fit's curvature moves a
  linear-background fit by only 1.6 µε.

**The real 3489 shifts come from peak shape (2026-11-17).**
- **Calibration.** The earlier 3489 reductions used `Si331/HB2B_CAL_2024-02-09T16:20.json`, which is
  **uncalibrated** (all shifts and rotations 0, `Status` −1). The calibration made on 3489's own
  date is `PreC509/HB2B_Si331_2023-05-30T17-28.json` (Rot_z 1.87°, λ 1.7190 Å; read correctly, with
  `two_theta_0` mapped to `TTH_0`). With it, sub-run 7's Poisson χ² falls from 2.12 to 1.27, but the
  production − Poisson shift remains: +39.6 µε mean (48 RMS) over the 12 sub-runs, and +32 / +34 µε
  even where χ² ≈ 1. The same reflection still moves with arm angle (94.43° → 94.55° → 94.77° in
  sub-runs 7–9), so even this calibration is not exact.
- **Not the vanadium.** The Si331 flood 3744 gives the same shift as the Si422 flood 3446 (+83 to
  +97 µε for sub-run 7 under the nominal geometry). **Not the fit window** (±1.0° to ±2.0°: +80 to
  +97 µε).
- **Not a wrong geometry by itself.** Placing simulated rings with the 2023 calibration and
  reducing with the nominal geometry doubles the spread of the production − Poisson difference
  (9 → 17 µε) but gives no mean shift (+0.4 ± 1.0 µε).
- **What sub-run 7 is.** It is **Al 222 from the sample's Al tube**, not a Ni reflection (see the
  sample note in the pixel-grouping entry). A tube scatters from two walls at different distances
  from the instrument centre, so its peaks are doubled or asymmetric and their apparent 2θ shifts
  with arm angle. That is the most likely source of the "shoulder", and of the drift 94.43° →
  94.77° over sub-runs 7–9, rather than an instrument-profile asymmetry. Sub-runs 4 and 10 (Ni 220,
  Ni 311) also needed a second component (21 % and 75 %), cause not established.
- **Peak asymmetry, yes.**
  - **Residuals:** the real residuals oscillate across the peak, with excess on the low-2θ flank.
    A second, low-angle pseudo-Voigt (a shoulder 0.28–0.54° below the peak, 10–75 % of the main
    intensity) removes the misfit: sub-run 7's χ² falls to 0.93–1.04.
  - **Simulation:** taking that two-component fit as the truth and fitting it with PyRS's single
    pseudo-Voigt reproduces the real shifts: sub-run 7 +62.0 ± 0.6 µε simulated against +62.6 µε
    real, sub-run 4 +36.1 ± 3.1 against +32.0. Sub-run 10 is heavy-tailed (failed production fits).
  - **Mechanism:** production σ down-weights the low-count tails relative to Poisson σ, so the two
    weightings balance core and tails differently when the shape is wrong.
- **Both single-peak fits are far from the main component.** Relative to the main peak of the
  two-component truth, the single pseudo-Voigt is off by +156 / +218 µε (sub-run 7, production /
  Poisson σ), +329 / +365 µε (sub-run 4) and +313 / +460 µε (sub-run 10). The shoulder varies
  between sub-runs, so this offset does not cancel between arm positions. For sub-run 7 the
  shoulder is most likely the second wall of the Al tube. In any case, peak-shape misfit, not
  counting statistics, dominates these real-data shifts. Container peaks are not a valid strain
  reference.
- **IPTS-36131's autoreduction calibration** (`Si422/HB2B_CAL_2024-07-09T12:10.json`) has all
  geometry parameters at 0 with `Status` 3; only λ (1.5306 Å) was refined.

Not covered: several peaks in the window, η wedges.

## Ring calibration with vertical pixel grouping: no gain, sometimes no convergence (2026-11-18, #1109)

Phase 3, tests C2 and C3 of the pixelation plan ([plan](planning/PixelationUncertainty_Plan.md)). No
production code changed. The ring calibration of `next` (snapshot `2944d0d6`: `calibrate_rings` +
`estimate_errors` with TTH_0 held, as `scripts/pyrs_pixel_calibration.py` runs it, latest mask) was run
on HB2B_3510 and HB2B_4971 at 1 × 1, 1 × 4 and 1 × 8. For the grouped cases, each sub-run's counts
were summed over blocks, the default mask kept a coarse pixel only if all its pixels were kept, the
instrument setup became the coarse `DENEXDetectorGeometry`, and the extraction's `edge` was scaled
from 40 to 40/g_y rows (same detector area).

- **C2: parameters and fit quality.**
  - **4971:** every parameter agrees with 1 × 1 within 0.7 σ; 65 ring points / 13 rings and RMS
    0.0065–0.0067° at all three groupings; the errors are the same within 3 %.
  - **3510:** 1 × 8 agrees with 1 × 1. At 1 × 4, Shift_x moved by +2.9 mm (10.9 σ) and the held TTH_0
    by −0.18°, roughly along the exact Shift_x / Rot_y / TTH_0 degeneracy (Rot_y did not follow it).
    RMS is 0.0166–0.0170° and ring points 58–59 at all groupings.
  - **Convergence:** 3510's ring extraction converged in 4 rounds at 1 × 1, but **stopped at the
    10-round limit without converging** at both 1 × 4 and 1 × 8. 4971 converged in 3–4 rounds at all
    groupings.
- **C3: pixel 2θ maps.** Each calibration was evaluated on the native 1024 × 1024 grid at arms
  60–120°, and its difference from the 1 × 1 calibration converted to strain per unmasked pixel,
  Δε = Δλ/λ − cot θ·Δθ. This is compared with each calibration's own uncertainty: covariance draws
  (TTH_0 held) propagated to the same strain map.

  | Run | Grouping | Δε vs 1 × 1, RMS at 60 / 90 / 120° | Own uncertainty, RMS at 60 / 90 / 120° |
  |---|---|---|---|
  | 4971 | 1 × 4 | 6.2 / 1.7 / 1.6 µε | 84 / 29 / 34 µε |
  | 4971 | 1 × 8 | 7.5 / 3.2 / 1.6 µε | 92 / 34 / 35 µε |
  | 3510 | 1 × 4 | 49 / 39 / 42 µε | 229 / 85 / 94 µε |
  | 3510 | 1 × 8 | 7.2 / 3.5 / 3.4 µε | 247 / 89 / 90 µε |

  The native calibrations' own uncertainty is 87 / 32 / 33 µε (4971) and 246 / 91 / 91 µε (3510).
- **Conclusion.** Grouping never moves the calibration beyond its own uncertainty (≤ 0.1 of it on
  4971, ≤ 0.5 on 3510). It does not improve the errors or residuals: calibration runs carry their
  information in the rings, where zero counts are rare. On 3510 it can stop the iteration
  converging. Grouping does not reduce the refined parameters' errors either: they grow by 1 % (1 × 4)
  and 3 % (1 × 8) on 4971, all by the same factor as the residual RMS (0.0065 → 0.0067°), and agree
  within ±2 % on 3510.
- **Decision (user, 2026-11-18): detector calibration operates on the native 1 × 1 grid.** A
  grouped reduction applies that native calibration, which is exact by C1. C4 (cross-application on
  a strained reference) has not been run.

## HB2B peaks are asymmetric, and the asymmetry depends on where they land on the detector (2026-11-18, #1108)

Phase 2 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.

**Setup.**
- **Calibration:** HB2B_5344 (2025, powder-standards assembly) and HB2B_4971 (2024) were calibrated with
  `next`'s driver `pyrs-pixel-calibration`, snapshot `bca6d8eb`, with 2θ₀ held at 0 per #1090. Both
  converged in 3 rounds:
  - 5344: 70 points / 14 rings, RMS 0.0066°, λ 1.53259 ± 0.00008 Å, Shift_z −59.0 mm, Rot_y 2.39°;
  - 4971: 65 / 13, RMS 0.0065°, λ 1.53133 ± 0.00008 Å, Rot_y 0.63°.
- **Reduction:** each sub-run was reduced with its new calibration, the latest mask and a vanadium
  (5339 for 5344, 4501 for 4971), for the full detector and for 3° η wedges.
- **Reflections:** the standard was identified from the sub-run's time-averaged `sy`, using only its
  own reflections. Reflections within 0.4° of an Al reflection (holder) were dropped.
- **Fits:** each reflection was fitted with Poisson σ twice: a symmetric pseudo-Voigt (as PyRS fits)
  and a split pseudo-Voigt (left / right widths). Asymmetry A = (w_R − w_L)/(w_R + w_L).

**Results.**
- **Only Ni is usable on this holder.** At λ ≈ 1.53 Å, Fe 211 and Fe 220 coincide with Al 222 and
  Al 400 (within 0.4°), so every Fe sub-run was dropped.
- **Every Ni peak has a high-angle tail**: A = +0.005 to +0.154 on the full detector, both below and
  above 2θ = 90°. **It is not axial divergence** (confirmed by the user, 2026-11-18): that would flip
  the sign at 90°. The split
  profile lowers χ² from 1.7–3.0 to 1.1–1.5 where A is large.
- **The same reflection changes shape with where it lands on the detector.** Its asymmetry grows as
  the ring moves to the high-2θ (right) side. For 5344 at ring offsets 2θ_Bragg − arm of −6.1°,
  −0.1° and +5.9°:
  - Ni 220: A = +0.005, +0.041, +0.092;
  - Ni 311 (offsets −6.7°, −0.7°, +5.3°): +0.049, +0.063, +0.077.

  4971 shows the same trend (Ni 220 +0.048 at −4.2° offset, +0.152 at −0.2°; Ni 311 +0.009 to
  +0.097).
- **The asymmetry also falls from the bottom of the detector to the top.** In 3° wedges on 5344 it
  decreases steadily from η = −6° to +6° in almost every reflection (e.g. Ni 220 at arm 82°: +0.086
  → −0.067). Axial divergence would be symmetric in ±η. 4971 has fewer counts per wedge but shows
  the same sign.
- **Consequence for strain.** The symmetric pseudo-Voigt centre lies 20–410 µε (in strain) away from
  the split-profile peak. That offset changes by about 350 µε for the same reflection as it moves
  across the detector (5344 Ni 220: −19 µε on the left, −377 µε on the right).
  - The ring calibration fits symmetric pseudo-Voigts per strip, so part of the offset is absorbed
    into the geometry: symmetric centres sit −164 to +191 µε from Bragg; split peaks sit +72 to
    +398 µε.
  - **A peak's apparent strain therefore depends on where it falls on the detector,** at the level of
    a few hundred µε.

**Vertical incident-slit scan (2026-11-18): the asymmetry does not depend on beam height.** HB2B_5344,
5345 and 5346 are identical (12 arm angles, `sy`, `sx`, `sz`, ω, horizontal gap 3.0 mm, ISD 100 mm,
no receiving optic) except for the incident slit's vertical gap: 10, 5 and 3 mm.
- **Calibration:** each run was calibrated with `next`'s driver. All parameters agree within their
  errors: Shift_x −2.59 / −2.52 / −2.55 mm, Rot_y 2.39 / 2.33 / 2.31°, Rot_z 0.260 / 0.262 / 0.258°,
  λ 1.53259 / 1.53254 / 1.53257 Å; RMS 0.0066 / 0.0064 / 0.0080°, with fewer counts at 3 mm.
- **Peak shape,** reduced with 5344's calibration and vanadium 5339 (10 / 5 / 3 mm):
  - the asymmetry A of each reflection at each arm agrees within the scatter (e.g. Ni 222 at arm
    93°: +0.154 / +0.155 / +0.151; Ni 311 at arm 93°: +0.063 / +0.063 / +0.059);
  - the symmetric − split offset is unchanged (e.g. Ni 222: −408 / −393 / −386 µε);
  - the mean FWHM is unchanged (0.64–0.71°);
  - the bottom-to-top change A(η = −6°) − A(η = +6°) stays at about +0.10 to +0.16.
- **Conclusion:** the incident beam's vertical extent does not affect the peak asymmetry or the
  calibration. The horizontal gap, sample position, sample transparency and the absence of a
  receiving optic were held constant, so they are not yet tested.

**Interpretation (not established).** An asymmetry that varies with both detector coordinates, and
does not change sign at 90°, and appears alike with 2 × 15 mm and 3 × 10 mm incident slits, with
and without a radial collimator (4971 / 5344), points to the detector's position response, e.g.
parallax from
conversion depth in the ³He gas volume at oblique incidence, rather than to the beam. That fits
`next`'s open item that the detector position response still has to be measured (slit scan,
`calibration-diagnostics`). The 3489 sub-run 7 shoulder was an Al container reflection and is a
separate effect.

## Peak-profile Monte Carlo: a split pseudo-Voigt removes the asymmetry bias; the symmetric fit does not (2026-11-18, #1109)

Phase 3 of the pixelation study ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.

**Setup.**
- **Truth:** real split pseudo-Voigt fits (Poisson σ) of four Ni peaks in 5344, reduced with its new
  `next` ring calibration and vanadium 5339: Ni 220 in sub-run 3 (A = +0.005) and sub-run 1
  (+0.092), Ni 311 in sub-run 7 (+0.077), Ni 222 in sub-run 8 (+0.154).
- **Simulation:** each truth placed on its sub-run's detector, λᵢ = k·Vᵢ·R(2θᵢ)/V_max, at the real
  count level (0.58–1.06 counts per pixel) and at a weak 0.066; 200 Poisson realisations each, reduced
  by production code.
- **Fits** with Poisson σ: the symmetric pseudo-Voigt with PyRS's `FitPeaks` arguments, a split
  pseudo-Voigt (scipy), and Mantid's `AsymmetricPearsonVII` through `FitPeaks`.
- **Measure:** bias against the true peak top (the split profile's mode).

| Truth (A) | Level | Symmetric PV: bias, scatter | Split PV: bias, scatter | AsymmetricPearsonVII: bias |
|---|---|---|---|---|
| Ni 220 (+0.005) | real / weak | −20, 10 / −15, 59 µε | +1, 27 / +13 ± 6, 78 µε | −17 / −16 µε |
| Ni 311 (+0.077) | real / weak | −237, 7.5 / −228, 23 µε | +3, 16 / −1, 58 µε | −210 / −207 µε |
| Ni 220 (+0.092) | real / weak | −377, 14 / −367, 56 µε | −2, 27 / +7, 73 µε | −325 / −318 µε |
| Ni 222 (+0.154) | real / weak | −414, 9 / −406, 45 µε | +3, 22 / +4, 70 µε | −376 / −373 µε |

- **The symmetric fit's bias grows with the peak's asymmetry**, from −20 to −414 µε. It reproduces the
  real-data offsets (−19 to −408 µε), so they are a profile effect. Because A changes with ring position
  on the detector, the symmetric fit gives a **position-dependent strain error of up to ~400 µε**,
  against a single-measurement scatter of 7–14 µε (real counts). For weak peaks its pull width is
  1.4–1.9, i.e. heavy tails and errors under-reported.
- **The split pseudo-Voigt is unbiased** (|bias| ≤ 7 µε in every case but one: +13 ± 6 µε for the
  nearly symmetric weak peak), with honest errors (pull width 0.77–0.96, reported errors slightly
  conservative). **Its price is precision:** the extra width parameter makes one measurement 2–2.6×
  noisier (16–27 µε at real counts, 58–78 µε weak).
- **`AsymmetricPearsonVII` removes only 10–15 % of the bias in this Monte Carlo,** with pull widths up
  to 2.6–2.75. **This is not a fair test** (user, 2026-11-18): the truth was itself a split
  pseudo-Voigt, so the Monte Carlo favours that profile by construction. See the real-data
  comparison below.
- **Real-data comparison, no synthetic truth (2026-11-18).** 24 Ni fits in 5344 / 5345 / 5346 (5344's
  calibration, vanadium 5339, Poisson σ):

  | Profile (free parameters) | χ²/dof median (max) | AIC mean rank | d spread across detector positions | run-to-run |
  |---|---|---|---|---|
  | Mantid PseudoVoigt, PyRS's arguments (6) | 1.55 (2.97) | 4.4 | 100 µε | 10 µε |
  | symmetric pseudo-Voigt, scipy (6) | 1.55 (2.97) | 3.6 | 100 µε | 10 µε |
  | Mantid AsymmetricPearsonVII: one width, left / right exponent (7) | 1.15 (2.12) | 3.1 | 91 µε | 8 µε |
  | split pseudo-Voigt: left / right width (7) | 1.02 (1.45) | 1.9 | 72 µε | 16 µε |
  | split Pearson VII: left / right width and exponent (8) | 1.02 (1.43) | 2.0 | 57 µε | 23 µε |

  The "d spread" is the std of d for the same reflection measured at its 3 ring positions within a
  run, averaged over reflections and runs.
  - **Fit quality:** the split pseudo-Voigt and the split Pearson VII describe the peaks equally well.
    Mantid's AsymmetricPearsonVII, with one width for both sides, improves on the symmetric fit but
    not to their level.
  - **Consistency:** the split Pearson VII gives the most position-consistent d, then the split
    pseudo-Voigt.
  - **None reaches the noise level** (single-fit precision ~20–30 µε). One likely reason: the
    calibration's strip fits are symmetric, so its geometry is tuned to symmetric centres. A fair
    judgement needs the calibration done with the same profile.
- **Recalibration with split Pearson VII strip fits makes it worse (2026-11-18).** The study replaced
  `ring_extraction._fit_band_histograms` (on the imported module; `next` unchanged) with a split
  Pearson VII + linear fit (same histograms and σ; centre = peak top), and recalibrated 5344 otherwise
  exactly as the driver does.
  - **The calibration moves well beyond its errors:** λ 1.53259 → 1.53204 Å (−360 µε, ~4σ); Rot_z
    0.260 → 0.043° (~13σ); Shift_z −59.0 → −60.5 mm; Rot_y 2.39 → 2.06°. The residual RMS rises from
    0.0066° to 0.0101°.
  - **The d spread across detector positions grows for every profile:** symmetric 100 → 171 µε,
    AsymmetricPearsonVII 91 → 167, split PV 72 → 97, split Pearson VII 57 → 95 µε.
  - **Likely reasons, not established:**
    - per-strip split fits (8 parameters on low-count strips) locate the peak top poorly, hence the
      higher RMS;
    - the stand-in significance (height / error) may admit weak fits;
    - with symmetric strip fits, the bottom-to-top change of asymmetry shifts the fitted centre
      strip by strip, which the refinement can absorb as detector spin. So the standard Rot_z ≈ 0.26°
      may partly be a peak-shape artefact.
  - **Next:** take each ring's shape from the high-count all-strips fit and fix it, fitting only
    position, height and background per strip.
- **Two-stage calibration (user's approach, 2026-11-18).** Stage 1 is the symmetric calibration;
  stage 2 starts from it and re-extracts with **fixed-shape** split Pearson VII strip fits. Each ring's
  shape comes from its free all-strips fit; per strip only the peak top, height and background are
  fitted. Refinement is as in `next`.
  - **Convergence:** 2 rounds, 70 points / 14 rings, RMS 0.0084° (symmetric 0.0066°, free split
    0.0101°).
  - **Parameters:** λ 1.53259 → 1.53221 ± 0.00010 Å (−250 µε); Shift_z −59.0 → −60.65 ± 0.26 mm; Rot_y
    2.39 → 2.11 ± 0.22°. **Rot_z stays at 0.268 ± 0.013°,** so the detector spin is real; the free
    split fit's 0.04° came from unstable strip fits.
  - **Ring shapes:** the right side is always wider (w_R 0.67–0.78° against w_L 0.57–0.64°), with
    exponents ~3–5.
  - **Full-detector d spread across positions:** split Pearson VII 98, split PV 102, symmetric 178 µε.
    That is worse than with the symmetric calibration (57 / 72 / 100 µε).
  - **The test may be unfair to the split calibrations.** The calibration locates peak tops in narrow
    strips, but the test fits the full-detector pattern. With asymmetry that changes bottom to top,
    that pattern is a blend whose top is not the strips' average. Under the split calibrations some
    reflections become very consistent (16–30 µε) while others worsen (up to 175 µε). A like-for-like
    test (strip level, or narrow η wedges, on 5345 / 5346 with 5344's calibration) is needed before
    choosing.
- **Like-for-like test (2026-11-18): the asymmetric profile does not improve the calibration.**
  - **Strip level.** Ring points were extracted from 5344 / 5345 / 5346 at each calibration's geometry,
    with either strip profile. Their residuals against Bragg were evaluated with the calibration held
    fixed. RMS for 5344 / 5345 / 5346, with the ring-to-ring spread of mean residuals in brackets:
    - symmetric calibration + symmetric strips: 60 (55) / 61 (54) / 73 (60) µε, mean ≈ 0;
    - two-stage calibration + fixed-shape split strips: 81 (78) / 80 (69) / 72 (53) µε, mean −27 to 0;
    - mismatched pairs: about **±300 µε mean offset** (symmetric calibration + split strips +280 µε;
      two-stage + symmetric −300 µε).

    Single-point precision is 17 / 26 / 36 µε.
  - **η wedges (3°, matched pairs):** the d spread across ring positions is 100 µε (median 112) for
    symmetric + symmetric, with single-fit error 36 µε. For two-stage + split Pearson VII it is 130 µε,
    but that fit's own error is 150 µε, so it is noise-limited and inconclusive.
  - **Conclusions.**
    1. **Calibration and peak fitting must use the same profile;** otherwise d is offset by ~±300 µε
       (absorbed into λ). The full-detector 57 µε "best" above was such a mismatched pair.
    2. **Matched, the symmetric approach is as consistent as the split one or better,** and far more
       precise in narrow wedges.
    3. **A systematic position dependence of ~55 µε (strips) to ~100 µε (wedges) remains with every
       profile,** well above the noise, so it is not a peak-profile effect. It is more likely in the
       geometry model or the detector's position response (cf. `next`'s open slit-scan item).
- **Implication:** the bias that changes with position (up to ~400 µε) is far larger than the split
  profile's extra scatter (~15–50 µε). A profile whose asymmetry is fixed in advance, as an instrument
  parameter determined from the standards as a function of detector position and η, could keep the
  split profile's zero bias at close to the symmetric profile's precision. That is not yet tested, and
  a new instrument parameter needs approval.

## Event-split test on real data: bin-level Poisson σ is honest; the per-pixel floor of PyRS / pyFAI is not (2026-11-19, #1109)

Pixelation study, Phase 3 ([plan](planning/PixelationUncertainty_Plan.md)). No production code changed.
No simulation.

**Setup.**
- **Slicing:** each HB2B_5344 sub-run's pulses were split into K interleaved slices (pulse number mod
  K; K = 10 and 2), giving K independent real measurements of the same pattern.
- **Reduction:** each slice was reduced with 5344's `next` ring calibration and vanadium 5339, on the
  native 720 bins, for the full detector and the η = 0° wedge, with I = (ΣN / ΣV)·V_max.
- **Models, each giving a bin σ:**
  - (A) **PyRS today, identical to pyFAI's `poisson` model:** per-pixel variance max(1, Nᵢ)
    (pyFAI `preproc`: `variance = numpy.maximum(1.0, signal)`);
  - (B) **bin-level Poisson:** σ² = ΣN (1 for an empty bin);
  - (C) **model-based:** the mean ΣN of the other slices. For a bin total a pixel-level (2D) model
    gives the same, since Var(ΣNᵢ) = Σλᵢ;
  - (D) **pyFAI `azimuthal`:** σ² = Σ(Nᵢ − Ī·Vᵢ)², the spread of the bin's pixels about the bin mean.
- **Score:** the observed variance of I across slices over the mean predicted σ², pooled by counts per
  pixel per slice. 1 = honest.

| Model | < 0.05 /px | 0.05–0.2 | 0.2–1 | > 1 |
|---|---|---|---|---|
| (A) PyRS / pyFAI `poisson` | 0.036 | 0.07 | 0.40 | 0.84 |
| (B) bin Poisson ΣN | 0.999 | 1.005 | 1.013 | 1.04 |
| (C) model-based | 0.999 | 1.005 | 1.013 | 1.04 |
| (D) pyFAI `azimuthal` | 0.997 | 1.000 | 0.98 | 0.98 |

These are for K = 10, full detector (5,318 / 2,779 / 509 / 22 bins). The η = 0° wedge and K = 2 agree:
(A) 0.15–0.89, (B) 0.96–1.03, (D) 0.94–1.01 except **0.82–0.85 above 1 count per pixel**.

- **The per-pixel floor overstates σ in sparse bins by up to 5×** (√(1/0.036)) on real HB2B data. It
  is pyFAI's `poisson` convention too, so a common convention, but wrong at these count levels.
- **σ² = ΣN is honest at every count level, in the full detector and in a 3° wedge.** HB2B's counts
  behave as Poisson. It needs no model: (C) adds nothing for binned data.
- **pyFAI's `azimuthal` is honest in sparse bins but too large (up to ~20 %) in high-count peak
  bins.** There, real pixel-to-pixel differences within a bin (the intensity gradient across the bin,
  efficiency mismatch) are counted as noise.
- **Fit level (K = 10, 8 peak groups): open.** The real scatter of fitted Ni centres between slices
  exceeds the reported error for every model: (A) 1.69, (B) 1.34, (C) 1.41, (D) 1.39 (median χ² 0.19
  for A, 1.16 for the others). The η = 0° wedge gives (A) 0.93, (B) 1.49. With 8 groups and plain
  variances this may be dominated by a few unstable fits, as with the weak-peak fit failures earlier.
  A robust re-analysis with more groups is needed. K = 2 has too few slices for a fit-level score.

## HB2B detector: pixel orientation, vanadium dead area, and vertical wire-period stripes (2026-11-12)

From Phase 0 of the pixelation study
([plan](planning/PixelationUncertainty_Plan.md), [#1106](https://github.com/neutrons/PyRS/issues/1106)). No production code changed.
Data: 2019–2023 runs in `tests/data`, and 2025 runs 5339, 5340 and 5344 from IPTS-36131.

- **Orientation.** `counts.reshape(1024, 1024)` is indexed `[x, y]` (pixel ID = x·1024 + y).
  x is horizontal: uncalibrated 2θ spans 17.3° along x and 0° along y at a 90° arm. Event-file
  `bank1_events/event_id` uses the same IDs: it reproduces HB2B_938's project-file counts in all
  but 215 pixels (+1 or +2 each).
- **Vanadium-dead pixels (< 0.9 counts, run 931) are the detector border.** That is 7.9 % of
  pixels; the active area is x 29–999, y 36–1000, with only 11 dead pixels inside it.
  `HB2B_Mask_12-18-19.xml` masks all of them plus 52,985 live pixels.
- **Zeros in live pixels are counting statistics.** Observed zeros match Σ exp(−λᵢ), with λ from
  2θ × η bins × vanadium efficiency, to 0.1–1 % (2.5 % in one high-count sub-run of 1017, where
  the vanadium-noise bias below is largest). This holds for runs 938–3489 and, per sub-run, for
  2025 run 5344.
- **Arm scans must be analysed per sub-run.** Runs 1017, 3489 and 5344 move the detector arm
  between sub-runs (by 27°–43°). Summing them and binning with one arm angle puts most pixels at
  the wrong 2θ. That alone produced an apparent 3–7 % zero deficit, e.g. 1017: 0.933 whole-run vs
  0.975–0.996 per sub-run. Splitting the events on `scan_index` as reduction does, and
  time-averaging the `2theta` log over each sub-run, reproduces the arm 2θ per sub-run that
  reduction stores in the project file (5344, all 12 sub-runs).
- **Vertical efficiency is striped with the delay-line wire period.** Vanadium row means vary
  ±52 % (0.5×–2.6×) with a 6.56-row (1.92 mm) period; columns vary only ±12 %, smoothly.
- **The stripe pattern drifts.** Measured/predicted counts per row have a mismatch beyond
  counting noise that grows with time since the vanadium: 2.2 % (938), 4.4 % (1017), 9.4 %
  (1118), 12.8 % (1431), 12.1 % (3489, 2023). Columns stay matched (≤ 0.3 %). With a vanadium
  from the same day in 2025 (5339), the row mismatch is 1.3 % for another flood (5340) and
  1.4–2.1 % per sub-run of 5344, so **~2 % is the floor** and larger values are drift. The wire
  period was still 6.56 rows in 2025. These mismatch values are the same whole-run and per
  sub-run.
- **Production mask, 2025.** `HB2B_Mask_Latest.xml` (2026-10-13) covers all but 353 of the
  94,541 pixels dead in vanadium 5339.
- **The production vanadium for IPTS-36131 is 4501** (2024-07-23, C508 with collimator, 8 h,
  28 counts per live pixel), 14 months before the runs, per `shared/hb2b_autoreduction.ini`.
  - **Dead pixels:** 19,466 pixels live in 4501 are dead in 5339 (and 3,216 the reverse). All but
    395 lie inside the 2019 mask that autoreduction uses, so they barely affect production.
  - **Fit to the 2025 runs:** with the newly dead pixels excluded, 4501 matches rows to 3.3–3.7 %
    (5344 sub-runs, 5340), against 1.2–1.9 % for the same-day 5339. Columns match either way
    (≤ 0.3 %).
  - **Cause not separated:** 4501 and these runs used different collimators (C508 with collimator,
    against C511), so the extra ~3 % cannot be split between time drift and collimator.
- **Wire planes (2026-11-16, confirmed by the user).** The x (horizontal) coordinate is encoded by
  a wire plane with about half the pitch of the y plane. Detrended vanadium column means show a
  period of **3.29 px = 0.963 mm along x**, against **6.60 px = 1.933 mm along y** (ratio 2.007).
  This holds in floods 931 (2019), 4501 (2024) and 5339 (2025). The x modulation is weak, 1.2 %
  relative std between columns, against 44–49 % between rows along y. It is why the vanadium's x
  profile looked smooth, and why the count-ratio residuals along x show a weak 3.29 px component.
  **The y wires sit behind the x wires** in the 2D grid, farther from the sample (stated by the
  user; not measurable from these data). So the rear plane carries the strong (44–49 %) stripe
  pattern, the one that drifts with vanadium age. Which plane is the anode and which the cathode
  is not established.
- **Noise bias.** Raw per-pixel vanadium used as an efficiency over-predicts zeros by about
  1 + λ²σ²/2 (σ² ≈ 1/N_van). Average it along x before using it.
