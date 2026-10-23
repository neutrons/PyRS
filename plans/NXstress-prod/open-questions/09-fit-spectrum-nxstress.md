# Open Questions — 09 Fit Spectrum & Calibration Fidelity (NXstress)

**Spec:** [09-fit-spectrum-nxstress.md](../09-fit-spectrum-nxstress.md)
**Blocking:** No — all items below are resolved.

---

## Q1 — RESOLVED: Does the NXbeam profile get populated, or closed as "not applicable"?

This spec's NXstress Changes section originally stated the branch as an
either/or, pending spec 08's trace of whether a beam-intensity data path
exists anywhere in PyRS.

**Chris:** We handle within `peak_collections.py` using the `self._exclude_list`

**Resolved by spec 08's investigation** (this Chris quote appears to answer
a different, adjacent question — see Q2 below, where it applies directly;
preserved here verbatim as originally recorded). The actual resolution:
no beam-intensity data path exists anywhere in PyRS, confirmed directly.
Rather than a binary "populate vs. close as not applicable" branch, PyRS
now encodes the "uniform beam" assumption as real data —
`HidraWorkspace.beam_intensity_profile` (spec 08), defaulting to a
documented uniform/constant value. This spec's `_instrument.py` reads that
property **unconditionally**; there is no branch to choose between
anymore, and no "not applicable" closing note is needed, since the
property always has a value.

---

## Q2 — RESOLVED: Where and how should the "sub-run did not converge" `NaN` fallback be logged?

Spec text: *"If the method is unavailable for a given scan point (e.g., fit
did not converge), fall back to `NaN` for that point only with a logged
warning."* The spec didn't specify the logging channel, level, or whether
the warning should be surfaced to the GUI.

**Chris:** We handle within `peak_collections.py` using the `self._exclude_list` this list is also exported to the csv files for record keeping

**Resolved, exactly as Chris proposed, via spec 08.** The "no representable
variance" case (the actual trigger for `model_intensity`/`model_variance`
being unavailable — see spec 08's fit-engine item) is now folded into the
existing `_exclude_list` mechanism: `PeakCollection.__set_fit_status`
extended so `bad_params` (unrepresentable parameter error) also sets
`_exclude_list[i] = True`. This mechanism is already GUI-visible (checkbox
toggles in `fit_table.py`) and already exported to CSV summaries for
record keeping, exactly as Chris describes — so the visibility concern
this question raised is addressed by reusing existing, already-wired
infrastructure rather than inventing a new logging channel. No separate
GUI-visible warning or backend-log-only decision is needed beyond this.

---

## Q3 — RESOLVED: cited line numbers and code shapes verified, three corrections made

A full research pass against the actual current code (following the same
verification this plan applied to specs 03, 06, 07, and 08) found the
spec's cited line numbers were close but imprecise, and two real design
gaps in the original draft:

- **Line numbers corrected**: `_fit.py:425-429` (placeholder comment),
  `:430-438` (field creation) — not the original "426-429, 486-487" (the
  latter pointed at an unrelated docstring note in a different class).
  `_instrument.py:126` (NXbeam construction, TODO at `:127`);
  `:132-150` (the calibrated/uncalibrated branch; the specific TODO +
  `distance = arm_length` line is correctly at `:138-140`).
- **`diffractogramFromNexus`'s return type**: decided to switch to a
  `NamedTuple` (`DiffractogramData`) rather than extending the current
  plain 4-tuple to six positional values — self-documenting, avoids
  positional ambiguity if more fields are ever added.
- **`instrumentFromNexus`'s reader-side scope was vague** ("update...
  correspondingly so the round-trip is lossless") — now concrete: it must
  read the new `arm_shift_applied` flag and use it to decide whether to
  call `apply_shift` or use the raw arm-length value, or a round-trip will
  double-shift or under-shift.
- **Simplification confirmed**: `_Diffractogram.init_group`'s `peakss`
  parameter already exists in the signature and is simply unused today —
  no new parameter needed to reach per-scan-point fit results.

---

## Q4 — RESOLVED: this spec's own multi-workspace signature was stale relative to 04b

A full-tree consistency scan (checking every spec against 04b/04c's
revised text) found that this spec's Q3 correction above — while
accurate about line numbers — still described
`_Diffractogram.init_group(ws, maskName, peakss)` with a single
`ws: HidraWorkspace`. That's the *pre-04b* signature. Spec 04b lands in
the Phase 2/3 bridge, well before this Phase 5 spec, and already
generalizes `init_group` to `(wss: list[HidraWorkspace], maskName,
peakss)`. This spec's own fit-spectrum/NXbeam work needed to be written
against that already-generalized signature, not the one that predates it.

**Resolved, three corrections made:**
1. `fit`/`fit_errors` must be resized to the *total* concatenated
   scan-point count across all of `wss`, not one workspace's count.
2. Placing each `PeakCollection`'s reconstructed spectrum into the right
   `fit`/`fit_errors` row requires a scan_point-*value* lookup against the
   concatenated `dg["scan_point"]` array — `peakss`'s discriminator-first
   order (04b) is not guaranteed to match the scan-point family's plain
   `wss`-order concatenation, so position-based alignment between the two
   families would be wrong in general.
3. `beam_intensity_profile` (NXbeam) is per-scan-point, like wavelength —
   per 04b's own Q7 clarification (found in the same full-tree scan) — so
   it's concatenated across `wss`, never validated for cross-workspace
   equality.

No change to this spec's scope or the items it delivers — `STRESS_FIELD`
remains blocked, the calibration/NXbeam/fit-spectrum items proceed as
before — only the description of *how* the writer reaches the right rows
under the signature that will actually exist by this phase.
