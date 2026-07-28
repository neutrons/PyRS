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

## Migrated hand-written Sphinx docs from reStructuredText to MyST Markdown (2026-07)

Both doc trees ([docs/user/source](../docs/user/source) for Read the Docs,
[docs/developer/source](../docs/developer/source) for the developer/API
guide) were converted from `.rst` to `.md` using the
[MyST](https://myst-parser.readthedocs.io/) Sphinx extension. Both
`conf.py` files now set `source_suffix = {".rst": "restructuredtext",
".md": "markdown"}` (a dict, not the old bare string) and add
`"myst_parser"` to `extensions`, with `myst_enable_extensions =
["dollarmath", "colon_fence"]` for `$...$`/`$$...$$` math and `:::`
fences.

**`api/modules.rst` under `docs/developer/source` was deliberately left as
`.rst`, not converted.** It's regenerated on every build by the
`run_apidoc` hook in `docs/developer/source/conf.py` (calls
`sphinx.ext.apidoc.main`), and `sphinx-apidoc` only emits
reStructuredText — there's no Markdown output mode. Because Sphinx's
`source_suffix` dict lets `.rst` and `.md` coexist in the same project,
the auto-generated tree can stay `.rst` while every hand-written page is
now `.md`, with no special-casing needed. It's also gitignored (see
`docs/developer/source/api/.gitignore`), so this was never something to
convert in the repo anyway.

**RST citations (`` .. [NumPy] text `` defined once, referenced elsewhere
as `` [NumPy]_ ``) have no direct MyST equivalent** — MyST doesn't support
the RST citation-node/citation-reference pair. Replaced with plain MyST
cross-reference targets: each bibliography entry in
`bibliography.md` is preceded by an explicit target line, e.g. `(numpy)=`,
and referenced elsewhere as `` {ref}`NumPy <numpy>` ``. **The explicit
link text (`` <numpy> `` inside the role, not just `` {ref}`numpy` ``) is
required** — `{ref}` normally auto-generates link text from the target's
heading/figure caption, and a bibliography entry is a plain paragraph
with no caption, so the bare form fails the build with `WARNING: Failed
to create a cross reference. A title or caption not found`. This was
caught by actually building the docs (`pixi run build-docs`), not by
inspection.

**RST relies on underline-character *order of first appearance* per file
to infer heading depth** (whichever underline character is used for the
first heading in a document becomes H1, the next new character becomes
H2, etc., independent of the literal character used) — Markdown has no
such inference; heading level is the literal number of `#`. When
converting each file, this meant reading the whole `.rst` file's
underline sequence first to work out the intended depth before choosing
`#`/`##`/`###`, rather than mapping characters 1:1
(e.g. `=` isn't always H1 — in `docs/user/source/api/manual_api.rst` the
title used `=` but a *later* section used `#` for a still-higher-level
heading than its position in the file would suggest). Getting this wrong
is silent in some cases but was caught in one case by MyST's own
`WARNING: Document headings start at H2, not H1 [myst.header]` (in
`math/instrument_and_reduction.md`) — that warning class is worth
grepping the build log for after any future heading-structure edit, since
Sphinx has no other structural check for this.

**Verification:** `pixi run build-docs` and `pixi run build-dev-docs`
both build clean (zero warnings from converted content) after the fixes
above. The developer build's remaining warnings (`more than one target
found for cross-reference 'HidraProjectFile'`/`'SampleLogs'`, a duplicate
`pyrs.utilities.NXstress.NXstress` object description) come entirely from
the auto-generated `api/*.rst` tree and pre-exist this migration —
confirmed by grepping the warning list for any converted `.md` path (none
appear).
