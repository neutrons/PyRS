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
