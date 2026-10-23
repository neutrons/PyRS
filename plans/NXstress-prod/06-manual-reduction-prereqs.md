# 06 — Manual Reduction PyRS Prerequisites (not required for NXstress)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** — (independent PyRS cleanup; not on the NXstress critical path)
**Depends on:** — (independent of all NXstress specs; can be worked in parallel)

---

## Overview

**Status update: this spec no longer blocks spec 07 or anything else in
this plan.** It was originally drafted as "before NXstress can be wired
into the ManualReductionViewer, two PyRS-side gaps must be closed" — that
premise doesn't hold once traced against the actual code:

1. `ReductionController.save_project` (`manual_reduction/pyrs_api.py:190`
   — **not** `HB2BReductionManager`, which is a different, unrelated class
   in `pyrs/core/reduction_manager.py`) is indeed `NotImplementedError`.
   But it has **zero callers anywhere in the codebase** — not from the
   viewer, not from the model's own passthrough
   (`manual_reduction_model.py:179-181`, which calls it but is itself never
   called). The manual-reduction save path that actually runs today is
   `reduce_hidra_workflow`'s automatic `reducer.save_diffraction_data(...)`
   call (`pyrs_api.py:348`), which already works — it is not blocked on
   anything here. `save_project` is orphaned code, not a live gap blocking
   users.
2. The two `nexus_conversion.py` (L118, L374) `NotImplementedError`
   branches are unrelated input-parsing gaps (an unsupported
   `TimeSeriesProperty` log type; a non-`.xml` mask-file format), reachable
   from NeXus-to-workspace *conversion*, not from any save/output-format
   path. NXstress's hookup point (spec 07,
   [07-manual-reduction-nxstress.md](07-manual-reduction-nxstress.md))
   touches `reduce_hidra_workflow`'s save step specifically, which never
   goes near either branch.

This spec remains a legitimate, independent PyRS code-quality item
(implementing an orphaned stub that may be worth a real use case someday;
auditing two unrelated input-parsing gaps) — but it is **not** a
prerequisite for NXstress work, and is not scheduled in this plan's phased
implementation sequence. Keep it as a standalone backlog item if still
wanted; do not resume treating it as a spec-07 blocker.

---

## Scope

**In scope:**
- Implement `ReductionController.save_project` in
  `manual_reduction/pyrs_api.py:190`
- Audit `nexus_conversion.py:118` and `nexus_conversion.py:374` —
  determine whether either branch is reachable from any manual-reduction
  path (confirmed: neither is reachable from the save/output-format path —
  see Overview); resolve (implement or safely guard) whichever is reachable
  from anything
- Unit and integration tests for the newly-implemented save path

**Out of scope:**
- Any NXstress code
- The NXstress hookup for ManualReductionViewer (spec 07 — does **not**
  depend on this spec; see Overview)
- Manual-reduction load paths (already working)

---

## PyRS Changes

### `pyrs/interface/manual_reduction/pyrs_api.py` — `save_project` (L190)

Implement the method. The surrounding context (`ReductionController`) holds
a `HidraWorkspace`; the save should write it out via `HidraProjectFile` (the
existing pattern used by all other viewers). Since nothing currently calls
this method, implementing it doesn't fix a live bug — it's forward-looking
groundwork for a possible future explicit "Save"/"Save As" action on
ManualReductionViewer (see the deferred `Viewer` ABC / viewer-symmetry idea
noted in the config-schema design session). The method signature and
expected output path convention should match what `ManualReductionViewer`
expects, if/when such an action is added.

`ReductionApp.save_diffraction_data` (`powder_pattern.py:231`, called from
`reduce_hidra_workflow` at `pyrs_api.py:348`) is a reasonable pattern to
reuse for `save_project`'s own implementation — but since `save_project`
has no current callers, there's no double-invocation risk either way;
delegate, wrap, or reimplement, whichever is cleanest, without needing to
preserve any existing caller's behavior.

### `pyrs/core/nexus_conversion.py` — L118 and L374

Audit each `NotImplementedError`:
- If the branch is reachable from the manual-reduction save path: implement
  it as part of this spec.
- If the branch is unreachable from the save path but reachable from other
  paths: file a separate issue and add a clear comment pointing to it.
- If the branch is dead code: remove it.

---

## NXstress / GUI Changes

_None._

---

## Tests

- `tests/unit/pyrs/interface/manual_reduction/test_pyrs_api.py` (new or
  extend): unit test `save_project` with a minimal `HidraWorkspace` —
  assert a valid `.h5` file is written.
- `tests/integration/test_manual_reduction_save.py` (new): run a minimal
  reduction workflow end-to-end and confirm the output file is loadable by
  `HidraProjectFile`.

---

## Delivered Feature

> **For contributors, not end users:**
> `ReductionController.save_project` — previously an unimplemented stub with
> no caller anywhere in the codebase — now has a working implementation.
> This doesn't change any current user-facing behavior: reduction's
> existing automatic save (via `reduce_hidra_workflow`) already works today
> and is unaffected. This is forward-looking groundwork, not a bug fix for
> anything currently reachable.

---

## Verification

- Unit test: call `ReductionController.save_project()` directly (no GUI
  action exists to trigger it) with a minimal `HidraWorkspace`; confirm a
  valid `.h5` file is written.
- `pytest tests/unit/pyrs/interface/manual_reduction/` — all pass.
- `pytest tests/integration/test_manual_reduction_save.py` — all pass.
- Confirm no regression: `reduce_hidra_workflow`'s existing automatic save
  continues to work unchanged (this spec doesn't touch it).

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A3) — "`ReductionApp.save_diffraction_data` (`powder_pattern.py:224`,
called from `reduce_hidra_workflow` at `pyrs_api.py:348`)".
- Referent: `pyrs/core/powder_pattern.py`.
- Verdict: drifted. `def save_diffraction_data` is at **:231**; `:224` is
  `sub_runs = [sub_run_number]` inside `plot_reduced_data` — plausible,
  unrelated code, which is what makes this class of defect survive re-reading.
  **The prose claim is entirely true**; only the pointer rotted.
- Action: corrected in place above to `powder_pattern.py:231` (a pointer, not a
  belief). `pyrs_api.py:348` was verified correct and is unchanged.

**F1.2** (A2) — This spec's files appear in no `README.md` §5 ownership row.
- Referent: `README.md`'s `## 5. Files to be Modified`.
- Verdict: correct by design but unrecorded. Decisions item 13 re-scoped 06 to
  "an independent, optional PyRS cleanup item, not scheduled in this plan's
  phases", and §5's table is organised by **phase** — so an unphased spec has
  nowhere to appear. The consequence is that `pyrs/core/nexus_conversion.py` is
  claimed by this spec and owned by nobody in the table, while
  `pyrs/interface/manual_reduction/pyrs_api.py` is listed in §5's Phase-4 row as
  spec 07's, even though this spec also edits it.
- Action (for the implementing PR, if 06 is ever scheduled): add an explicit
  `— (unscheduled)` row to §5 covering 06's two files, so the table stays
  exhaustive. Until then this is recorded, not fixed — the table is accurate
  about the *plan*, and 06 is deliberately outside it.

**Checked and accurate — no action.** `pyrs_api.py:190` (`def save_project`),
`manual_reduction_model.py:179-181`, `pyrs_api.py:348`, and both
`nexus_conversion.py:118` / `:374` `NotImplementedError` branches all land
exactly on their claimed targets, and the Overview's central corrections — that
the class is `ReductionController` and that `save_project` has zero callers —
hold as written.

**A4/A5 exemption, recorded rather than assumed.** This spec asserts nothing
about third-party library behaviour and nothing about how another module treats
its output. No probe was needed. (Written down because afterwards "no probe was
needed" and "no probe was written" are indistinguishable.)
