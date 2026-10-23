# 09 — Fit Spectrum & Calibration Fidelity (NXstress)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 5 (NXstress side)
**Depends on:**
- [07 — ManualReductionViewer NXstress hookup](07-manual-reduction-nxstress.md)
- [08 — Fit spectrum & calibration fidelity (PyRS)](08-fit-spectrum-prereqs.md)

---

## Overview

Wire the PyRS capabilities added in spec 08 into the NXstress writer and
reader, replacing NaN placeholders with real data — **except
`STRESS_FIELD`, which stays a documented gap** (spec 08 investigated it and
found the item genuinely blocked on missing data/clarification from the
instrument-science team, not merely unimplemented; see spec 08's Overview
and `open-questions/08-fit-spectrum-prereqs.md` Q3). This spec's other
three items are unaffected by that block and proceed normally.

---

## Scope

**In scope:**
- Populate `diffractogram/fit` and `diffractogram/fit_errors` from the fit
  engine's new reconstructed-spectrum method. Placeholder comment at
  `_fit.py:425-429`, actual field creation at `_fit.py:430-438` (corrected
  from an earlier draft's imprecise "426-429, 486-487" — the latter range
  is an unrelated docstring note in a different class, not `init_group`).
  These fields are currently **zero-sized resizable datasets**
  (`np.empty((0,0))`, `fillvalue=np.nan`), not pre-shaped NaN arrays — the
  writer must resize them to the real shape before populating, not
  index-assign into an already-correctly-shaped array (same
  `maxshape=(None,...)` resizable pattern used elsewhere in this codebase,
  e.g. `_peaks.py`).
  **By this phase, `_Diffractogram.init_group` already takes
  `list[HidraWorkspace]`, not a single `ws`** (spec 04b lands in the
  Phase 2/3 bridge, well before this Phase 5 spec) — the real shape to
  resize to is the *total* concatenated scan-point count across all input
  workspaces, not one workspace's count. See "Multi-workspace scope" below.
- Fix the L2 arm-shift round-trip in `_instrument.py:132-150` (the
  calibrated/uncalibrated branch; the specific TODO + `distance =
  geom.arm_length` line is at `:138-140`, unchanged from the earlier
  citation) using the new `DENEXDetectorGeometry.arm_shift_applied` (and
  the retained pre-shift arm-length value). Note: the double-counting risk
  this item originally worried about is **already resolved at the source**
  in spec 08 — `DENEXDetectorGeometry` itself now distinguishes base
  distance from applied shift, so this spec's fix is a straightforward
  consumer update, not a workaround. Confirmed still live today: both
  `translation_z = shift.center_shift_z` (`:135`) and `distance =
  geom.arm_length` (`:140`) are written, which is exactly the double-count
  spec 08's fix addresses.
- Populate the `NXbeam` intensity profile (constructed at
  `_instrument.py:126`; the TODO comment itself is at `:127`) by reading
  `HidraWorkspace.beam_intensity_profile` (spec 08) **unconditionally** —
  no "if the workspace carries one" branching; the property always has a
  value (a documented uniform/constant default when no real measurement
  exists), so this spec's writer code has no uniform-specific logic of its
  own. Per 04b's Q7 clarification, this is a **per-scan-point** field, like
  wavelength — it gets concatenated across all input workspaces alongside
  the rest of the scan-point family, not validated for cross-workspace
  equality.
- Update user-facing docs and release notes to remove NaN-placeholder
  warnings added in spec 02 — **except** for `STRESS_FIELD`, whose warning
  stays in place; it remains a real, current limitation.

**Out of scope:**
- `_sample.py:107` `STRESS_FIELD` dimensions — **blocked**, carried over
  unresolved from spec 08 (see Overview). Do not attempt to guess a shape
  here; leave the existing TODO in place, updated only if spec 08's
  blocker resolves in the meantime.
- Any remaining `NotImplementedError` in `fields.py` not exercised by the
  test suite at this point.
- GUI changes (all viewers already wired; no new actions needed).

---

## PyRS Changes

_None_ — all PyRS-side changes were made in spec 08.

---

## NXstress Changes

### `pyrs/utilities/NXstress/_fit.py` — fit spectrum (L425-429, L430-438)

**Multi-workspace scope, corrected from an earlier draft:** these line
numbers are cited against the current (pre-04b) codebase, where
`_Diffractogram.init_group(ws, maskName, peakss)` still takes a single
`ws: HidraWorkspace` (`_fit.py:398`). **By this phase, that's no longer
true** — spec 04b (Phase 2/3 bridge, well before this Phase 5 spec)
already generalizes `init_group` to `(wss: list[HidraWorkspace], maskName,
peakss)`, concatenating the scan-point family across all N inputs. This
spec's own work is written against that already-generalized signature,
not the single-`ws` one an earlier draft assumed:

In `_Diffractogram.init_group(wss, maskName, peakss)`:
- The `peakss: list[PeakCollection]` parameter **already exists in this
  method's signature** (`_fit.py:398`) but is currently unused in the body
  — no new parameter is needed, just use the argument already being passed
  in (caller: `_fit.py:555`). By this phase `peakss` is 04b's combined,
  multi-workspace peak index (discriminator-first sorted).
- Call the new fit-engine method (from spec 08) on each `PeakCollection` in
  `peakss` to obtain `model_intensity` and `model_variance` for each
  `(mask, scan_point)`.
- **Resize** the `fit`/`fit_errors` fields to the real shape before writing
  — they start as zero-sized resizable datasets, not pre-shaped NaN arrays.
  The real shape is the **total concatenated scan-point count across all
  of `wss`**, not one workspace's count — matching whatever size
  `dg["scan_point"]` ends up after 04b's concatenation.
- **Row placement is by scan_point *value*, not position.** `peakss`'s
  order follows the peak-index family's discriminator-first `sort_key`
  (04b), which is not necessarily the same order the scan-point family was
  concatenated in (plain `wss` order). For each `PeakCollection`'s
  scan points, look up their positions in the concatenated
  `dg["scan_point"]` array by value (the same value-based approach 04b's
  Q7 uses to split the scan-point family back into N workspaces on read)
  before writing into `fit`/`fit_errors` — never assume the two families'
  row orders align.
- If the method is unavailable for a given scan point (e.g., fit did not
  converge), that scan point is now excluded upstream in `PeakCollection`
  itself (spec 08's `_exclude_list` extension) — write `NaN` for excluded
  points; no separate logging decision is needed here (see
  `open-questions/09-fit-spectrum-nxstress.md` Q2).

In `_Diffractogram.diffractogramFromNexus(dg)`:
- **Change the return type to a `NamedTuple`** (e.g. `DiffractogramData`
  with fields `scan_points, two_theta, diffractogram, diffractogram_errors,
  fit, fit_errors`) rather than extending the current plain 4-tuple
  `(scan_points, two_theta, diffractogram, diffractogram_errors)` to six
  positional values. Self-documenting at the call site and avoids
  positional ambiguity if more fields are ever added later. Update the one
  existing caller (`NXstress.py:218`) to match.
- Read `fit` and `fit_errors` fields (not read at all today) so callers can
  use the reconstructed spectrum (e.g., for residual visualization in a
  future viewer feature).

### `pyrs/utilities/NXstress/_instrument.py` — L2 arm-shift (L132-150)

Use `DENEXDetectorGeometry.arm_shift_applied` (from spec 08) to determine
whether to write the raw L2 or the arm-shift-corrected L2 into the
`NXdetector`, and the retained pre-shift arm-length value where the raw
distance is needed alongside the shift amount. No double-counting guard is
needed in the writer — `DENEXDetectorGeometry` already keeps the two
values distinct (spec 08).

**`instrumentFromNexus` (`_instrument.py:223`) needs concrete, not vague,
reader-side work.** It currently reads `trans["distance"]` directly as
`arm_length` and constructs `DENEXDetectorGeometry(..., arm_length,
calibrated)` (`:253-258`), with no concept of "was a shift already folded
into this distance." Once the writer emits `arm_shift_applied` plus the
unshifted arm-length value, the reader must read that flag and use it to
decide whether to call the (spec-08-fixed) `apply_shift` or use the raw
value directly — otherwise a round-trip either double-shifts or
under-shifts. This replaces the earlier, vaguer "update
`instrumentFromNexus` correspondingly so the round-trip is lossless."

### `pyrs/utilities/NXstress/_instrument.py` — NXbeam profile (L126-127)

Read `HidraWorkspace.beam_intensity_profile` (spec 08) and write it into
`NXbeam` unconditionally — the property always returns a value (a
documented uniform/constant default, or real data if a future reduction
change ever populates it), so there is no "if present" branch and no
"not applicable" closing note needed. Remove the existing TODO marker.
**Per-workspace handling (04b's Q7):** like wavelength, this is a
per-scan-point field, not a single entry-wide value — concatenate it
across all of `wss` in the same order as the rest of the scan-point
family; do **not** validate it for cross-workspace equality the way
geometry/shift are (above).

### `pyrs/utilities/NXstress/_sample.py` — STRESS_FIELD shape (L107) — BLOCKED

**Do not touch this code in this spec.** Spec 08 investigated this item and
found it genuinely blocked — no file anywhere in the repository contains a
usable example, and the shape/dtype remain unverified (see
`open-questions/08-fit-spectrum-prereqs.md` Q3). Leave the existing TODO
comment in place; update it only to note that this was actively
investigated and remains blocked, not left untouched by oversight.

---

## Tests

`tests/unit/pyrs/utilities/NXstress/test_fit.py` (extend):
- Write a `.nxs` file with a fitted workspace where the fit engine returns
  a reconstructed spectrum; assert `diffractogram/fit` is not all-NaN and
  has shape `(n_sub_runs, n_2theta)` (confirming the resize-before-write
  behavior, not a pre-shaped array left over from initialization).
- Write a workspace where one sub-run is excluded (via spec 08's extended
  `_exclude_list`, not a separate NXstress-side convergence check); assert
  that sub-run's `fit` row is NaN but others are not.
- Assert `diffractogramFromNexus` returns the new `NamedTuple` type
  (`DiffractogramData`), with `fit`/`fit_errors` populated when present.
- Multi-workspace fit-spectrum placement: write two workspaces (distinct
  discriminator values, non-overlapping scan points, both with real
  `PeakCollection`s per 04b's Q7 invariant) whose peak-index order
  (discriminator-first) does not match their scan-point-family
  concatenation order; assert each `PeakCollection`'s reconstructed
  spectrum lands in the `fit`/`fit_errors` row matching its own
  `scan_point` value, not a row implied by position alone. Confirms the
  value-based row-lookup requirement above actually matters, not just a
  theoretical concern.

`tests/unit/pyrs/utilities/NXstress/test_instrument.py` (extend):
- Round-trip a calibrated `DENEXDetectorGeometry` through NXstress;
  assert the read-back geometry has `arm_shift_applied == True` and the
  L2 value matches the original.
- Round-trip `beam_intensity_profile`: assert the written `NXbeam` field
  matches whatever value the workspace carried (the documented uniform
  default, in the absence of a real measurement).
- Multi-workspace `beam_intensity_profile`: write two workspaces with
  *different* `beam_intensity_profile` values (not just two copies of the
  uniform default); assert both concatenate into `NXbeam` correctly and
  neither is rejected by a cross-workspace consistency check — confirms
  this field is treated like wavelength (concatenated), not like geometry
  (validated-for-equality).

---

## Delivered Feature

> **For end users and downstream NXstress consumers:**
> NXstress files produced by PyRS are now populated with real data for
> everything PyRS actually computes:
>
> - **Reconstructed fit spectra** (`diffractogram/fit`, `diffractogram/fit_errors`)
>   are written for every sub-run where the peak fit converged. These can be
>   used by external tools for residual analysis and quality assessment.
>   The propagated variance is a documented approximation that neglects
>   parameter correlations (spec 08) — not presented as exact.
> - **Calibrated detector geometry** (L2, arm shift) round-trips correctly —
>   a calibrated measurement read back from NXstress gives the same geometry
>   as the original.
> - **Beam-intensity profile** is written as a documented uniform/constant
>   value, reflecting PyRS's existing assumption explicitly rather than
>   leaving `NXbeam` silently empty.
>
> **Known limitation, carried forward from spec 08:** `STRESS_FIELD`
> remains a documented gap, not a NaN placeholder quietly left behind —
> spec 08 investigated it directly and found it genuinely blocked pending
> real data or clarification from the instrument-science team.

---

## Verification

- `pytest tests/unit/pyrs/utilities/NXstress/` — all pass.
- `pytest tests/integration/` — all pass (no regression in earlier specs).
- Write a `.nxs` from a real HB2B dataset and inspect in a NeXus browser:
  confirm `diffractogram/fit` contains a plausible model profile.
- Run the `nexusformat`-org's NXstress validator (with
  `nxstress.use_production_names = true`) on the output file — no errors,
  **once the validator and schema doc are available** (confirmed not yet
  present in this repo or in the installed `nexusformat` package as of
  this writing — see README's Decisions Log and spec 10's reminder to add
  both to the repo).
- Cross-check with `tests/scripts/cis_tests/NXstress_demo_script.py` —
  update it to use the newly-populated fit-spectrum, calibration, and
  beam-profile output and confirm it runs cleanly. `STRESS_FIELD` remains
  unaddressed by design; confirm the demo script doesn't assert against it.
- Confirm `_sample.py:107`'s TODO comment reads as investigated-and-blocked,
  not silently unchanged from spec 08.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A3) — **No defects found.** All twenty-one citations in this document
were resolved against the current code and every one lands exactly on its
claimed target, including the four that carry design weight:
- `_fit.py:425-429` → the `##` placeholder comment block, and `_fit.py:430-438` →
  the `dg[GROUP_NAME.DGRAM_FIT] = NXfield(...)` / `DGRAM_FIT_ERRORS` creations,
  verbatim `np.empty((0, 0))`, `maxshape=(None, None)`, `fillvalue=np.nan` —
  exactly as this spec describes them;
- `_instrument.py:135` → `tz = float(shift.center_shift_z)` and `:140` →
  `distance = float(geom.arm_length)`, the two writes whose co-existence is the
  double-count this spec addresses;
- `_fit.py:398` → `def init_group(cls, ws: HidraWorkspace, maskName: str,
  peakss: list[PeakCollection])`, confirming **both** that it takes a single
  `ws` today **and** that the `peakss` parameter already exists unused;
- `NXstress.py:218` → `scan_pts, two_theta, data, errors =
  _Diffractogram.diffractogramFromNexus(child)`, confirming the 4-tuple and its
  single caller.

Decisions item 15 recorded a round of line-number corrections against this
spec; that round demonstrably held. This document is also the series' best
example of the distinction the rest of the plan blurs: its "Multi-workspace
scope" note states explicitly that its numbers are "cited against the current
(pre-04b) codebase", while the Scope section describes the post-04b state —
rather than conflating the two.

**F1.2** (A4) — "the writer must resize them to the real shape before
populating, not index-assign into an already-correctly-shaped array".
- Referent: `h5py` 3.16.0 / `nexusformat` 1.0.8.
- Verdict: **confirmed by probe**
  ([`probes/a4_h5py_nexusformat_append.py`](probes/a4_h5py_nexusformat_append.py)).
  Zero-sized resizable datasets are created as such, and `resize` followed by a
  tail slice-assign works on a reopened, file-backed `NXfield`:

  ```console
    CLAIM   zero-sized resizable datasets are created as such
    RESULT  initial shapes: {'scan_point': (0,), 'h': (0,), 'phase_name': (0,),
            'center': (0,), 'fit': (0, 0)}

    CLAIM   resize(cur+N); arr[cur:] = ... works on a ZERO-SIZED reopened dataset
    RESULT  3 row(s) appended at offset 0; read back {'scan_point': [1, 2, 3],
            'phase_name': ['Fe', 'Fe', 'Fe'], 'center': [1.1, 1.2, 1.3]}
  ```
- Action: none. The claim stands, now with evidence rather than inference.

**A5 gap, recorded rather than assumed.** This spec's post-04b claims — that
`_Diffractogram.init_group` will take `list[HidraWorkspace]`, and that the
resize target is the total concatenated scan-point count across workspaces —
**cannot be probed**, because 04b is unimplemented. A5-**uncovered**, not
A5-verified.
