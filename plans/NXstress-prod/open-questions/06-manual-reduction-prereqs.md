# Open Questions — 06 Manual Reduction PyRS Prerequisites

**Spec:** [06-manual-reduction-prereqs.md](../06-manual-reduction-prereqs.md)
**Blocking:** No — this spec is no longer on the NXstress critical path at
all (see the spec's Overview); both questions below are now about an
independent PyRS cleanup item, not about anything spec 07 needs.

---

## Q1 — RESOLVED: `save_project` has no callers, so delegate/wrap/replace is moot

Originally: *"Specifically check how `ReductionApp.save_diffraction_data`
(L348) is currently called and whether `save_project` should delegate to
it, wrap it, or replace it."* Traced directly against the code: `save_project`
(the real class is `ReductionController`, not `HB2BReductionManager` —
that name belongs to a different, unrelated class in
`pyrs/core/reduction_manager.py`, used internally by `ReductionApp`) has
**zero callers anywhere in the codebase**, including the model's own
passthrough (`manual_reduction_model.py:179-181`), which is itself never
called. There is no double-invocation risk to weigh, because nothing
currently invokes `save_project` at all — implement it however is cleanest.

**Chris:** There was a dead code instance of `save_diffraction_data` that has been deleted in a recent PR.

**Correction:** this refers to a different, already-removed call site — the
one at `pyrs_api.py:348`, inside `reduce_hidra_workflow`, is still live and
actively used, both there and in 7+ integration test call sites.

---

## Q2 — RESOLVED: `nexus_conversion.py:118`/`:374` are unrelated to any save path

Originally framed as a three-way branch (implement if reachable from save,
track separately if reachable elsewhere, remove if dead). Traced directly:
L118 is an unsupported `TimeSeriesProperty` log type during NeXus log
conversion; L374 rejects non-`.xml` mask files. Both are reachable from
`reduce_hidra_workflow`'s NeXus-*conversion* step
(`converter.convert()`, `pyrs_api.py:319-320`) — **not** from any
save/output-format path. They have no bearing on NXstress's hookup (spec
07, which touches `reduce_hidra_workflow`'s save step specifically) and no
longer block anything in this plan.

**Chris:** `ManualReductionViewer` is a realatively old codebase. The `ManualReductionViewer` should use `HB2BReductionManager.save_project`.

**Correction:** this named the wrong class — `HB2BReductionManager` isn't
part of `ManualReductionViewer` at all; the real class there is
`ReductionController`. Regardless, this doesn't change the finding above:
neither `nexus_conversion.py` branch is on the save path.
