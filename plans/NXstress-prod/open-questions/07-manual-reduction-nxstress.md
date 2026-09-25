# Open Questions — 07 ManualReductionViewer NXstress Hookup

**Spec:** [07-manual-reduction-nxstress.md](../07-manual-reduction-nxstress.md)
**Blocking:** No.

---

## Resolved — write mode and overlap-conflict policy don't apply here

The original pre-split spec 07 bundled append support with the
ManualReductionViewer hookup, and this file previously carried two
questions about append's write-mode API and overlap-conflict policy. Both
are resolved for *this* spec: `reduce_hidra_workflow` always writes fresh
(it has no notion of an existing entry to extend), so append is never
invoked from this pathway. The underlying questions (write-mode API and
overlap-conflict policy) still matter for the append capability itself;
they both live in `open-questions/04c-nxstress-append.md` Q1 (Q2 there is
a separate matter — append's family scope — not either of these two).

**Chris:** My inital thought is that we an data reduction pathway should create a fresh file.

---

## RESOLVED — Does the suffix-dispatch branch live in `pyrs_api.py` or a separate model layer?

Originally hedged on this, pending spec 06's delegate/wrap/replace question.
Both resolved by tracing the actual code: the branch lives directly inside
`reduce_hidra_workflow` (`pyrs/interface/manual_reduction/pyrs_api.py`) —
the one place a save currently happens. `ReductionController.save_project`
(spec 06's target — not `HB2BReductionManager`, an unrelated class) has no
callers at all and was never the right place to hook in. This spec no
longer depends on spec 06.

**Chris:** Should `HB2BReductionManager` should coordinate how data are written into the NX file

**Correction:** this named the wrong class — no `HB2BReductionManager`
involvement exists in `ManualReductionViewer`'s save path at all (that name
belongs to an unrelated class in `pyrs/core/reduction_manager.py`, used
internally by `ReductionApp`). The actual coordination point, per the code,
is `reduce_hidra_workflow`, in `ReductionController`'s module
(`pyrs/interface/manual_reduction/pyrs_api.py`).

---

## RESOLVED — write mode / mismatched-extension handling

Superseded by the config-schema design session: `reduce_hidra_workflow`
writes once per currently-enabled format (`nxstress.enable`,
`legacy_io.enable`), never both automatically merged into ambiguity and
never gated by a caller-supplied extension. See spec 07's Overview for the
full reasoning — an explicit `project_file_name`'s extension is stripped
and ignored entirely, never validated against, never a reason to raise.
