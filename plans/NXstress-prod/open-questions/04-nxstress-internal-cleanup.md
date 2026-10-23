# Open Questions — 04 NXstress Internal Cleanup (Phase 2 TODOs)

**Spec:** [04-nxstress-internal-cleanup.md](../04-nxstress-internal-cleanup.md)
**Blocking:** No. (Corrected: this header previously referenced a "Q5"
blocking the production-names default flip — no such question exists in
this file, only Q1–Q4 below, none of which claim to block anything. The
production-names flip is spec 10's concern, not this spec's — spec 04
itself lists it as out of scope. This was a stale leftover, likely from
before specs 04/10 were split apart.)

---

## Q1 — Are the legacy log-name FIXMEs (`file_object.py:404,494`) actually wrong?

The spec's own PyRS Changes section states the audit outcome isn't known
yet: *"No code changes required if the audit confirms NXstress already uses
the correct names; otherwise patch the relevant `_sample.py` or `_peaks.py`
key lookups."*

**Why it matters:** determines whether this item is a no-op or requires
changes to two other NXstress modules (`_sample.py`, `_peaks.py`) beyond
what's listed in this spec's own NXstress Changes section.

**Next step:** read `file_object.py:404` and `:494` in full, trace what log
keys they touch, and compare against what `_sample.py` / `_peaks.py` key off
of today.

**Response from Chris:**

`file_object.py:404` has been updated to return `two_theta_vec = self._project_h5[HidraConstants.REDUCED_DATA][HidraConstants.TWO_THETA][()]`

`file_object.py:494` has been resolved

---

## Q2 — What is the correct detector rotation order?

`_instrument.py:165` has an open TODO on rotation order in
`NXtransformations`. The spec requires cross-checking against
`DENEXDetectorGeometry`'s convention in `instrument_geometry.py`, but does
not state what the correct order actually is — that's the audit's job.

**Why it matters:** flagged in the plan README as a correctness risk with no
visible symptom: *"A wrong order silently mis-orients the instrument in the
file"* (README.md:123-125). A silent bug is the worst kind to leave
unresolved past this phase.

**Next step:** the spec's own Tests section requires a regression test that
round-trips a geometry and checks rotation components numerically — write
that test first (expected to fail against the current order) as the way to
discover the right answer, then fix the writer to match.

**Response from Chris:** The detector rotation order is defined in `file_object.py:generate_rotation_matrix` as `rot_x_matrix * rot_y_matrix * rot_z_matrix`. But this has been updated `rot_x_matrix @ rot_y_matrix @ rot_z_matrix` to ensure that the correct matrix multiplication is applied.

---

## Q3 — What are the actual `SampleLogs` key names for sx/sy/sz?

`_peaks.py:235-239` has commented-out code assuming keys `'sx'`, `'sy'`,
`'sz'`. The spec explicitly flags this as unconfirmed: *"Reconcile the
log-key names against what the reduction pipeline actually stores in
`SampleLogs`. If the keys differ ... use the correct names."*

**Why it matters:** if the guessed keys are wrong, restoring the
commented-out block would either raise a `KeyError` or (worse) silently
return `NaN` again if a bare fallback is used without knowing it should have
found real data.

**Next step:** grep the reduction pipeline (`nexus_conversion.py` and
whatever populates `SampleLogs`) for the actual key names used for sample
x/y/z position before uncommenting.

**Response from Chris:** `'sx'`, `'sy'`, and `'sz'` are teh specific `SampleLogs`
keys for the stage motor positions.  But, these enteries not used as part of the
StressStrain workflow. Instead, the `'vx'`, `'vy'`, and `'vz'` are the critical keys.
The stress/strain calculator pulls these enteries from the HIDRAWorkspace.

---

## Q4 — What's the complete disallowed-character set for `allowed_identifier`?

`_definitions.py:229` is flagged as having "incomplete" coverage. The spec
says to extend it to cover "at minimum `$`, whitespace, and any other
characters disallowed by the NXstress/HDF5 group-name rules" — the phrase
"any other characters" signals the full set isn't enumerated anywhere yet.

**Why it matters:** without a canonical list, "complete coverage" can't be
verified — later production data with an unanticipated character in, e.g., a
mask name or peak tag could still slip through and produce an invalid HDF5
group name.

**Next step:** find or write down the definitive HDF5-group-name character
restrictions (and any additional NXstress schema restrictions) as a single
reference list, rather than accumulating disallowed characters ad hoc as
they're discovered.

**Response from Chris:** We can define a specific schema.

---

## Q5 — In what reference frame are `peaks/(sx, sy, sz)` expressed?

*Raised 2026-09-29, during the implementation pass. Not blocking: the values
written are correct; what is undefined is the frame a reader should interpret
them in.*

The NXstress schema requires `peaks/sx`, `sy`, `sz` and documents each as
*"component of the sample position in the **sample reference frame**. The sample
reference frame is defined by the sample transformations"* — meaning the
`NXtransformations` group under `SAMPLE_DESCRIPTION`.

**PyRS does not write that group.** `_sample.py` writes `name`,
`chemical_formula`, `scan_point`, `vx`/`vy`/`vz`, optionally `temperature` and
`stress_field`, and a `logs` collection — no `NXtransformations`, and no
`depends_on`. So the frame the schema points at does not exist in our files.

**Why it matters:** for PyRS-to-PyRS round trips this is invisible — we write and
read the same convention implicitly. For an external NXstress consumer, which is
the entire point of writing the format, the positions are numerically present and
semantically unanchored. It is the same class of defect as the detector
`depends_on` chain (Follow-up 7 F7.5), differing only in that the detector chain
was *wrong* whereas this one is *absent*.

**Why it is not resolved here:** writing the group means asserting a goniometer
and sample-stage convention — which axes, in which order, relative to what — and
this spec has no referent for that. The detector side had one
(`reduce_hb2b_pyrs.py::generate_rotation_matrix`); the sample side has no
equivalent that has been located. Guessing would produce a file that is
confidently wrong rather than honestly silent.

**Next step:** determine whether HB2B's sample stage has a recorded convention
(and whether `mrot`, `phi`, `chi`, `omega` are the relevant axes), then decide
whether `SAMPLE_DESCRIPTION/NXtransformations` belongs in spec 08/09's
calibration-fidelity work rather than here.

---

### Update, 2026-10-14 — the chain is written, and the convention was recoverable after all

*Added during the subspec 04 PR review. Everything above is left as written: it
is the record of what was believed. Two of its claims are now false, and this
section says which and why.*

**Falsified claim 1 — "PyRS does not write that group."** It does now.
`_sample.py` writes `SAMPLE_DESCRIPTION/transformations` and the group's
`depends_on`, as a per-scan-point chain. The entry point names the **last** link,
so traversal reaches every one of them — the F7.5 defect asserted against on the
sample side before it could happen.

**Falsified claim 2 — "the sample side has no equivalent [referent] that has
been located."** It has one: `PoleFigureCalculator.rotate_project_q`
(`pyrs/core/polefigurecalculator.py:186`) applies all three stage angles. It was
passed over because it implements the closed-form pole-figure projection of Bob
He, *Two-Dimensional X-ray Diffraction*, ch. 8.3 — sines and cosines, no matrix
anywhere in it — so there is no axis in it to read off, and the first reading was
that nothing could be recovered from it.

That reading was wrong, and the correction came from the stakeholder question
*"I'm not understanding how we can be presented with a 3-angle transformation, in
an applicable form, that can't simply be mathematically decomposed?"* A rotation
composition is a rotation composition however it is spelled. Driving
`rotate_project_q` through its public signature with its probe vector aimed along
each basis vector in turn reads off the matrix it implements, as a black box; of
2880 candidate compositions over the six signed axes, exactly one survives.

**A first attempt at this decomposition did report an ambiguity — a handedness
sign shared by `omega` and `chi` — and that was an artefact of the attempt, not a
property of the code.** At `omega = chi = phi = 0` the projection does not reduce
to the identity but to `diag(1, -1, -1)`, a constant 180-degree rotation about
`+x`. A search over stage rotations alone cannot match a composition carrying a
constant factor, so it returned only partial matches that disagreed. With the
offset in the search the result is exact and unique.

**Probe:** [`probes/a5_sample_rotation_axes.py`](../probes/a5_sample_rotation_axes.py),
run with `pixi run python plans/NXstress-prod/probes/a5_sample_rotation_axes.py`:

```
STEP 1 -- recover the matrix, through the public signature only

at omega = chi = phi = 0 the formula gives
    [+1.000000e+00 +6.123234e-17 +1.518722e-32]
    [+0.000000e+00 -1.000000e+00 +2.480262e-16]
    [+3.037445e-32 -2.480262e-16 -1.000000e+00]
  identity?                      False
  180 degrees about +x?          True
  ^ a constant frame offset. A search over stage rotations alone cannot match it.

  every recovered matrix orthogonal: True, det +1: True
  ^ so the discarded third row was reconstructible, not lost.

STEP 2 -- identify it, over 40 random (omega, chi, phi) triples

  candidates considered: 2880
  exact matches:         2

  omega about +y, chi about -x, phi about +z
      lab <- sample reading:  M = R_phi @ R_chi @ R_omega @ M0
  omega about -y, chi about -x, phi about -z
      lab <- sample reading:  M = M0 @ R_phi @ R_chi @ R_omega

  The survivors are conjugate by M0 (a 180-degree rotation about x, which
  negates the y and z axes), so they are the same physical rotation written
  with the constant factor on either side -- not a genuine ambiguity.

STEP 3 -- the form NXtransformations needs, checked on 200 fresh triples

  NXtransformations describes sample -> lab, which is the transpose of the
  lab -> sample matrix recovered above.

  sample -> lab  ==  M0 @ R_omega(0,-1,0) @ R_chi(1,0,0) @ R_phi(0,0,-1)
  worst elementwise error over 200 triples: 5.818e-14

  Write order is composition order for a NeXus chain entered at its last link,
  so `_sample.py` writes frame_offset, omega, chi, phi in that order, with
  `depends_on` naming phi. Nothing here is left to choose.
```

**What is written.** NXtransformations describes **sample -> lab**, the transpose
of the lab -> sample matrix the projection applies. Write order is composition
order for a chain entered at its last link, so `TRANSFORMATION_CHAIN` is:

| # | link | type | vector | source | note |
|---|---|---|---|---|---|
| 1 | `translation_x` | translation | `(1, 0, 0)` | `vx` | mm |
| 2 | `translation_y` | translation | `(0, 1, 0)` | `vy` | mm |
| 3 | `translation_z` | translation | `(0, 0, 1)` | `vz` | mm |
| 4 | `frame_offset` | rotation | `(1, 0, 0)` | *constant* `180` | `M0` |
| 5 | `omega` | rotation | `(0, -1, 0)` | log `omega` | |
| 6 | `chi` | rotation | `(1, 0, 0)` | log `chi` | |
| 7 | `phi` | rotation | `(0, 0, -1)` | log `phi` | entry point |

The rotation links are written **all together or not at all**, gated on every
input carrying all three angle logs. `frame_offset` is gated with them although
it is constant and therefore always computable: alone it would assert the stage
was at zero, which an entry without the angle logs gives no grounds for. Every
real HB2B file in `tests/data` carries the three; the synthetic unit fixtures do
not, and get a translations-only chain. The raw angle logs are left untouched
under `logs/`, so the interpretation this group applies is purely additive.

**Verified end to end**, writing `tests/data/HB2B_1327_with_instrument.h5`
(150 scan points) through `NXstress.write` and reading the result back with
h5py: all seven links reachable from `depends_on`, all 150 rows long, and the
chain composed from the **on-disk** vectors and values reproduces
`rotate_project_q` to a maximum elementwise error of `1.11e-16`.

**Pinned by tests** in `tests/unit/pyrs/utilities/NXstress/test_sample.py`
(`TestSampleTransformationChain`), including
`test_the_rotations_compose_in_write_order`, which composes what was written and
compares it against `rotate_project_q` itself — so an edit to an axis *or to the
order* fails against the referent rather than against a restatement of it.

**What remains open.** Nothing about the convention. Two things are worth a
stakeholder's eye:

1. **Is `frame_offset` physical or an artefact of the projection's output
   convention?** It is real in the sense that the chain needs it to reproduce
   `rotate_project_q`. Whether it reflects how the sample is mounted, or a
   choice internal to the ch. 8.3 pole-sphere construction, the code cannot say.
   Either way the written chain is correct; only the *name* would change.
2. **`mrot`** appears in every real HB2B file alongside the three angles and is
   not in the chain, because `rotate_project_q` does not take it. If it moves the
   sample, the chain is incomplete.

**Response from Chris:**
