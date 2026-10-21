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

`MantidPeakFitEngine` (and `FitEngineFactory.getInstance`) take
`high_background: bool = True`, forwarded to Mantid `FitPeaks` as
`HighBackground`, alongside `FindBackgroundSigma=1` (Mantid's default, so
a no-op kept for explicitness).

- `True` — background fitted first, then the peak. Upstream default since
  `ab3dd7d5` (2019-10-15, "Peak fitting pytests are passed"); every normal
  PyRS fit (GUI, reduction, strain/stress) relies on it.
- `False` — peak and background fitted together. Used only by pixel-position
  calibration ([pixel_pos_calibration.py](../pyrs/calibration/pixel_pos_calibration.py)).

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
`test_scipy_default_step_gives_accurate_angle_derivative` fails if the matrices go back to float32.

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
- **Mantid's centre uncertainty is unreliable at low signal**: fits to pure noise reported centre
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
