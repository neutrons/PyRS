# Open Questions — 04c, raised during the implementation pass

**Spec:** [04c-nxstress-append.md](../../04c-nxstress-append.md)
**Planning-stage questions:** [open-questions/04c-nxstress-append.md](../04c-nxstress-append.md)
**Blocking:** No — the code is correct in every respect the implementation
could establish. What is open is a convention PyRS contradicts itself about,
which no amount of care inside NXstress can settle.

Questions here were raised *after* implementation, by the human PR review of
04 + 04b + 04c, rather than while the spec was being written. They are kept
separate from the planning-stage file because the two have different
standing: a planning question is answered before code exists, and one of
these is a defect in shipped code that could not be fixed without a decision
from outside this subspec.

---

## Q1 — Which axis does `two_theta_zero` rotate about, and does it belong in the chain at all?

*Raised 2026-10-15, PR review batch 27. Recorded as D55.*

`_instrument.py` writes `two_theta_zero` into the detector's
`NXtransformations` chain with axis vector `ex` — about **X**. The reduction
disagrees: `reduce_hb2b_pyrs.py::rotate_detector_2theta` applies two-theta
about **Y** (`_cal_rotation_matrix_y`), and it does so *after* folding
`two_theta_0` into the detector arm angle as a scalar offset
(`reduce_hb2b_pyrs.py:139`) rather than applying it as a transformation of its
own.

**This is not the rotation-*order* question.** That one is closed: the written
sub-chain composes to `Rx @ Ry @ Rz`, matching
`generate_rotation_matrix`, proven by
[`probes/a5_transformations_chain.py`](../../probes/a5_transformations_chain.py)
and now asserted on every commit by
`test_instrument.py::TestDetectorRotationOrder`, which composes the written
chain against that function rather than against a restatement of it. The
`# TODO: check order of rotations here!!!` comment that stood in the source
since the first implementation has been removed.

**Why it matters, and why NXstress cannot decide it.** Underneath sits a
contradiction inside PyRS itself, which this writer inherited rather than
introduced:

* `instrument_geometry.py:166` documents `two_theta_0` as *"in rotation of
  detector arm about X"*;
* `reduce_hb2b_pyrs.py` rotates about Y.

The code is the authority, so the comment is the likelier error — but
"likelier" is not a basis for writing a geometry into a file that external
readers are meant to trust, and the NXstress writer followed the comment. A
wrong axis here mis-orients the detector for any consumer that traverses the
chain, silently, in exactly the way Follow-up 7 F7.5's unreachable chain did.

Two questions, and the second only matters once the first is settled:

1. **Which is right — the comment or the reduction code?** Fix whichever is
   wrong, in PyRS, not in NXstress.
2. **Should `two_theta_zero` be a chain link at all?** The reduction treats it
   as an offset *added to* the arm angle before a single Y rotation, not as an
   independent transformation. A chain that models it as its own link may be
   describing a composition the instrument does not perform, even with the
   axis corrected.

**Next step:** owned by **spec 08** (resolve the contradiction; the scope item
is recorded there) and **spec 09** (emit the link on the settled axis, or stop
emitting it). Both carry it in their `## Scope`. Nothing in the detector chain
can be called verified until question 1 is answered.

**Not blocking 04c**, which is append and does not touch the detector chain;
blocking any claim that the written instrument geometry is correct.
