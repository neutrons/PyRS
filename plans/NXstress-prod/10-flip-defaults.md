# 10 — Flip Defaults: NXstress Becomes Primary

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 6
**Depends on:** [09 — Fit spectrum & calibration fidelity (NXstress)](09-fit-spectrum-nxstress.md)

---

## Overview

Complete the phased-replacement decision (README Section 4, Q2): promote
NXstress to the default save format across all NXstress-aware viewers, and
demote `HidraProjectFile` (`.h5`) to a legacy-read-only path.

**Re-scoped: this is now a pure config-default flip, not a code change.**
Every NXstress-wired viewer (specs 02, 03, 05, 07) already reads
`nxstress.enable`/`legacy_io.enable` directly to decide which of its
save actions are enabled (grayed out when disabled, never hidden — see
each spec's "Config-driven enablement" notes) — that wiring was built once,
in each spec, specifically so this phase would not need to touch any
viewer's code. Entering Phase 6 means flipping the *shipped default* in
`pyrs/resources/application.yml`:

```yaml
legacy_io:
  enable: false   # was: true
```

`nxstress.enable` stays `true` (it already is, from Phase 1). No viewer's
`Save`/`Save as NXstress…` action *logic* changes code — the existing
"Save" (`.h5`) action simply becomes disabled by default, and users who
still need `.h5` output can re-enable it via their own config file
(per-user override, not a code path).

**One real exception to "no viewer code changes," found by checking the
actual GUI code:** only `PeakFittingViewer` currently has a status bar.
`TextureFittingViewer`, `CombineRunsViewer`, and `StrainStressViewer` are
all code-built `QMainWindow`s with **no status bar at all** — their only
user-facing messaging today is modal (`QMessageBox`/`show_failure_msg`/
`pop_message`). The deprecation hint needs a status bar to attach to, so
this spec adds one (`self.statusBar()`, which creates it lazily — cheap)
to the three viewers that lack one. This is new UI, not a pure config
flip; see Scope below.

> **⚠ REMINDER for whoever picks this spec up:** two artifacts this plan
> has referenced since spec 04b's open questions still need adding to the
> repo, and are needed for this spec's (and specs 02/09's) Verification
> steps to actually be runnable:
> 1. **The `nexusformat`-org NXstress validator.** Confirmed: this is
>    *not* part of the installed `nexusformat` PyPI package (1.0.8 has no
>    `validate` module or `nxvalidate` script) — it exists only as a
>    separate repository. Add a link to that repository here once you have
>    it (not fabricated in this planning pass — link intentionally left as
>    a placeholder below).
> 2. **`NXstress.xml`/`NXstress.html`** (the canonical schema doc) — add it
>    to the repo under `docs/developer/source/design/nexus/`, and add a
>    link to it from `docs/developer/source/design/nexus/IO_prototype.rst`
>    in that same directory (which already discusses the `nexusformat`
>    validator's known quirks, e.g. its `UPPERCASE`-group-name requirement
>    — see `IO_prototype.rst`'s Overview section — so the schema doc and
>    the validator note belong together there).
>
> **Validator repository link:** _TODO — add once available; not guessed
> at in this planning pass._

**Trigger criteria** for entering this spec (to be confirmed at the end of
spec 02 / start of spec 03):
- All NXstress round-trip tests pass.
- At least one real-data smoke test has been run by a scientist and signed off.
- No open correctness bugs against the NXstress writer or reader.

---

## Scope

**In scope:**
- Flip `legacy_io.enable` to `false` and `nxstress.use_production_names` to
  `true` in `pyrs/resources/application.yml`.
- Keep `.h5` in all *load* dialog filters (legacy read-only) — loading is
  unaffected by `legacy_io.enable`, which only gates the *save* action's
  enablement, per each viewer's spec.
- **Add a status bar** (`self.statusBar()`) to `TextureFittingViewer`,
  `CombineRunsViewer`, and `StrainStressViewer` — confirmed none of the
  three has one today (only `PeakFittingViewer` does). This is the one
  piece of genuinely new UI in this spec, not a pure config flip.
- Deprecation note: a one-time status-bar message (matching
  `PeakFittingViewer`'s existing `showMessage` pattern) when a user opens
  a `.h5` file, in all four viewers. For `StrainStressViewer` specifically,
  the "one-time" flag must be tracked once at the **window level** (a
  single boolean on the main window instance) — its three direction slots
  (e11/e22/e33) all share the same `filesSelected` handler
  (slots: `strain_stress_view.py:331-333`; the shared handler itself is `controller.filesSelected`, `controller.py:9`, invoked from `strain_stress_view.py:68`), so a per-call guard inside that
  handler would still fire up to three times per session.
- Release notes / migration guide for existing users, including how to
  re-enable `.h5` saving via a personal config override (setting the `env`
  OS environment variable to a `.yml` file with `legacy_io.enable: true`,
  or via `neutrons_standard`'s auto-loaded `~/.pyrs/pyrs-user.yml`) if
  still needed.

**Out of scope:**
- DetectorCalibrationViewer (not NXstress-wired; unchanged).
- Removing the `HidraProjectFile` code itself (a separate cleanup, not part
  of this plan).
- Any change to the *save-action enablement* logic — that wiring already
  exists (specs 02, 03, 05, 07); this spec only flips the config default it
  reads. (The new status bars above are UI scaffolding for the
  deprecation hint, not a change to that logic.)

---

## PyRS Changes

_None._

---

## NXstress / GUI Changes

### `pyrs/resources/application.yml`

```yaml
nxstress:
  enable: true
  extension: ".nxs"
  use_production_names: true    # flipped from false
legacy_io:
  enable: false                 # flipped from true
  extension: ".h5"
```

(`nxstress.discriminator_fields`/`merge_workspaces` — established in the
Phase 2/3 bridge, spec 04b — are unchanged by this flip and omitted from
the snippet above for that reason, not because they've been removed.)

### Deprecation hint

When a user successfully opens a `.h5` file in any of the four NXstress-wired
viewers with a Save action, display a one-time status-bar message:
> "This file is in the legacy HiDRA format (.h5). NXstress (.nxs) is now
> the default save format."

### `pyrs/interface/texture_fitting/texture_fitting_viewer.py`, `pyrs/interface/combine_runs/combine_runs_viewer.py`, `pyrs/interface/strainstressviewer/strain_stress_view.py`

Add a status bar (`self.statusBar()`) to each — confirmed none of these
three `QMainWindow`s has one today. Wire each viewer's existing `.h5`-load
entry point (one chokepoint each: `load_project_plot`/`controller.load_projectfile`
for Texture; `combine_runs_crtl.py` for CombineRuns, regardless of which of
its two call sites is used; `controller.filesSelected` for StrainStress,
shared by all three direction slots) to show the hint via
`self.statusBar().showMessage(...)`, guarded by a one-time flag tracked on
the window instance (window-level for all four, but StrainStress
specifically needs this since its handler fires up to three times per
session).

**No save-action *enablement logic* needs editing** — each viewer's save
action was already built to read `legacy_io.enable`/`nxstress.enable`
directly (specs 02, 03, 05, 07); flipping the shipped default is
sufficient to disable `.h5` saving and make NXstress the only enabled save
action by default. Only the deprecation-hint UI (status bars) above is new.

---

## Tests

- Regression suite: `pytest tests/` — full test run, all specs. No failures.
- Smoke test, all four viewers (not just PeakFitting): open a legacy `.h5`
  file; confirm it loads (load is unaffected by `legacy_io.enable`);
  confirm the deprecation hint appears exactly once per session on the new
  or existing status bar. For PeakFitting, Texture, and CombineRuns:
  confirm the existing "Save" (`.h5`) action is now disabled (grayed out,
  not hidden) and "Save as NXstress…" is enabled. **StrainStress has no
  `.h5` Save action to disable** (per spec 05 — its save paths are
  CSV/JSON) — confirm only that its "Save as NXstress…" action is enabled.
- StrainStress-specific: open `.h5` files into all three direction slots
  (e11/e22/e33) in one session; confirm the hint appears only once, not
  three times.
- Confirm a user's `env`-var-pointed override file (or an auto-loaded
  `~/.pyrs/pyrs-user.yml`) can override `legacy_io.enable: true` to restore
  `.h5` saving without a code change.

---

## Delivered Feature

> **For end users:**
> NXstress (`.nxs`) is now the default and only enabled save format across
> all PyRS viewers wired for it. In PeakFitting, Texture, and CombineRuns,
> the existing "Save" (`.h5`) action is grayed out by default; "Save as
> NXstress…" is the enabled action. StrainStress never had a `.h5` Save
> action — for it, "Save as NXstress…" simply becomes enabled.
>
> Existing `.h5` project files can still be opened (they will continue to
> work). When you do open a legacy `.h5` file, PyRS will remind you NXstress
> is now the default format.
>
> **Still need `.h5` output?** Set `legacy_io.enable: true` in an override
> file pointed to by the `env` environment variable (or in your auto-loaded
> `~/.pyrs/pyrs-user.yml`) to re-enable it — no code change required, and
> both formats can be enabled simultaneously if you want to keep producing
> `.h5` files alongside `.nxs` during your own transition.

---

## Verification

- `pytest tests/` — full suite passes.
- Open a `.h5` file in each of the four affected viewers (PeakFitting,
  Texture, CombineRuns, StrainStress) plus ManualReduction's automatic
  write path — confirm load succeeds and the deprecation hint is shown
  where applicable.
- For PeakFitting, Texture, and CombineRuns — the three with an existing
  `.h5` Save action to gate — confirm it's disabled by default, "Save as
  NXstress…" is enabled. **StrainStress has no `.h5` Save action** (per
  spec 05); confirm only that its "Save as NXstress…" action is enabled —
  the flip is a no-op for its (nonexistent) legacy save path. The
  *save-action enablement logic* is otherwise unchanged between Phase 1
  and Phase 6, only the config default; the new status bars (Texture,
  CombineRuns, StrainStress) are this spec's one piece of new,
  non-enablement UI.
- Open the written `.nxs` files in a NeXus browser and run the
  `nexusformat`-org NXstress validator — no errors, **once that validator
  and the `NXstress.xml`/`.html` schema doc are added to the repo** (see
  the reminder in this spec's Overview — neither exists yet as of this
  writing).
- `tests/scripts/cis_tests/NXstress_demo_script.py` — runs cleanly.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A4) — "only `PeakFittingViewer` currently has a status bar", and the
hint uses "`PeakFittingViewer`'s existing `showMessage` pattern".
- Referent: `pyrs/interface/`, probed by
  [`probes/a4_qtpy_qaction_statusbar.py`](probes/a4_qtpy_qaction_statusbar.py).
- Verdict: **true of the four viewers this spec enumerates, too strong
  unqualified — and the "existing pattern" is not the pattern this spec
  proposes.**

  ```console
    CLAIM   Only PeakFittingViewer has a status bar today
    RESULT  .ui files declaring a QStatusBar: ['manualreductionwindow.ui',
            'peakfitwindow.ui', 'pyrsmain.ui']; source files calling
            self.statusBar(): NONE
  ```

  Three `.ui` files declare a `QStatusBar`, so ManualReductionViewer and the
  launcher window have one too. And `self.statusBar()` is called **nowhere** in
  `pyrs/` today: `PeakFittingViewer` reaches its bar as `self.ui.statusbar`,
  because the bar comes from `peakfitwindow.ui`. So this spec's mechanism is
  **net-new**, not a continuation of an existing pattern.
- Verdict on safety: **the mechanism is nonetheless correct.** The obvious risk —
  that calling `self.statusBar()` on a window whose bar came from a `.ui` file
  would create a *second* bar — does not occur:

  ```console
    CLAIM   on a window whose status bar came from a .ui file, does self.statusBar()
            return THAT bar or create a second one?
    RESULT  statusBar() is the declared bar: True; QStatusBar children: 1

    CLAIM   self.statusBar() creates a status bar lazily on a bare QMainWindow
    RESULT  QStatusBar children before call: 0; after call: 1; returned QStatusBar

    CLAIM   calling it twice returns the SAME object (idempotent, so 'cheap' holds)
    RESULT  first is second: True
  ```
- Action (for the implementing PR): qualify the sentence — "of the four viewers
  this spec touches, only `PeakFittingViewer` has a status bar today; it comes
  from `peakfitwindow.ui` and is reached as `self.ui.statusbar`, not via
  `self.statusBar()`". Keep the `self.statusBar()` mechanism: it is lazy,
  idempotent, and returns a `.ui`-declared bar when one exists, so it works
  uniformly across all four. The same over-strong sentence is in
  `README.md:728` (Decisions item 16a) and is corrected there too.

**F1.2** (A3) — "its three direction slots (e11/e22/e33) all share the same
`filesSelected` handler (`strain_stress_view.py:331-333`)".
- Referent: `pyrs/interface/strainstressviewer/`.
- Verdict: the claim is true, but the citation points at the wrong thing.
  `:331-333` are the three `FileLoad("e11"/"e22"/"e33", ...)` constructions —
  correct for the *slots*. The shared handler is `controller.filesSelected`,
  defined at `controller.py:9` and invoked from `strain_stress_view.py:68`
  inside `FileLoad`. Since this spec's design point is that "a per-call guard
  inside that handler would still fire up to three times per session", an
  implementer needs the handler, and this citation does not lead there.
- Action: corrected in place above to cite the slots **and** the handler.

**F1.3** (A1) — `## Tests` names no test file.
- Referent: this document's own `## Overview`, which calls the status-bar work
  "the one piece of genuinely new UI in an otherwise pure config-flip spec".
- Verdict: the Tests section is four manual smoke-test bullets and zero paths.
  New UI — a status bar added to three viewers, plus a one-time
  per-session deprecation hint with a window-level flag — ships with no
  automated coverage named anywhere.
- Action (for the implementing PR): name
  `tests/ui/test_nxstress_roundtrip.py` (marked `gui` **and** `integration`, per
  `CLAUDE.md`) and add at least: the hint appears exactly once per session for
  each of the four viewers; it appears once, not three times, when all three
  StrainStress direction slots are loaded; and the `.h5` Save action is disabled
  but still visible under the flipped default.

**Invariants flagged, not written** (an audit flags; the implementing PR
writes). Both are net-new shapes for this repo, which `process.md` §5.1 is
explicit about:
1. **A Qt API-surface pin** — that `setEnabled(False)` leaves `isVisible()`
   true. Owner: this PR. Tier: `gui` + `integration`. Possible from spec 02
   onward, since that is when the first gated action exists.
2. **A convention scan** — that no viewer reintroduces `setVisible` for
   format-gated actions, which is the rule Decisions item 11 rests on. Owner:
   this PR. Tier: unit (source scan, no widget).
