# 08 — Reconstructed Fit Spectrum & Calibration Fidelity (PyRS)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 5 (PyRS side)
**Depends on:** — (independent of NXstress; can be worked in parallel with specs 01–07)

---

## Overview

This spec closes PyRS-side gaps that block spec 09 from replacing the final
NaN placeholders in NXstress files. Four items, each grounded against the
actual code (the original draft's assumptions were checked and, in two
cases, corrected — see the notes under each item):

1. **Reconstructed fit spectrum** — PyRS's peak-fit engine returns parameter
   values but does not currently produce a full-length reconstructed model
   profile on the original two-theta grid. NXstress requires
   `diffractogram/fit` and `diffractogram/fit_errors` datasets; without a
   model spectrum these are NaN. **Real new work** — the one existing
   function that evaluates a model at arbitrary two-theta points is dead,
   buggy code, not a usable starting point (see PyRS Changes).
2. **Calibrated instrument geometry round-trip** — `DENEXDetectorGeometry`
   accepts a `calibrated` flag but discards it, and the arm-shift-application
   code path is dead and would raise if ever invoked. **Corrected scope**:
   the FIXME originally cited at `file_object.py:510` doesn't exist there;
   the real one is in NXstress's own `_instrument.py`, and this spec's fix
   stays scoped to `DENEXDetectorGeometry` (see PyRS Changes).
3. **Beam-intensity profile** — confirmed: no data path anywhere in PyRS
   carries this from reduction to `HidraWorkspace`/`SampleLogs`. PyRS
   encodes the "uniform beam" assumption as real, documented data instead
   of NXstress hardcoding it.
4. **`STRESS_FIELD` shape verification** — **blocked, not resolved.** The
   three example files initially offered as verification data exist but
   don't contain what's needed (see PyRS Changes) — this item cannot be
   completed from anything currently in the repo.

Like spec 06, this is **PyRS-only** and can be developed independently.

---

## Scope

**In scope:**
- Fit engine: new method returning a reconstructed model spectrum per
  `PeakCollection` × mask × scan_point, built on the underlying peak-shape/
  background evaluator functions (not on the existing dead-code wrapper).
- Extend `PeakCollection.__set_fit_status` so an unrepresentable parameter
  error also triggers exclusion (unifying "no representable variance" with
  the existing exclusion mechanism).
- `DENEXDetectorGeometry`: store the `calibrated` flag it already accepts;
  add `arm_shift_applied`; fix `apply_shift`'s existing bug; retain the
  pre-shift arm-length value so it's recoverable after a shift is applied.
- `HidraWorkspace`: new `beam_intensity_profile` property (get/set),
  defaulting to a documented uniform/constant value.
- `STRESS_FIELD` shape: **blocked** — document the blocker explicitly;
  do not guess a shape.

**Out of scope:**
- Any NXstress code (that is spec 09).
- Any change to `HidraProjectFile`'s `.h5` binary format (see "Future work,
  not decided against" below).
- GUI changes.

---

## PyRS Changes

### Fit engine — reconstructed spectrum

Files: `pyrs/peaks/peak_collection.py`, `pyrs/core/peak_profile_utility.py`.

**Starting point is not what the original draft assumed.**
`peak_profile_utility.py::calculate_profile` (the one existing function
that evaluates a peak+background model at arbitrary two-theta points) is
**dead code with zero callers anywhere in the repo**, and has real bugs:
leftover debug `print()` statements, it raises for any background other
than `Linear` (hardcoding the `Quadratic` background's extra term to zero),
and an off-by-one windowing issue. It is not a usable wrapper to build on
as-is.

The new method should instead call the underlying, otherwise-usable
module-level evaluators directly — `gaussian(x, a, sigma, x0)`,
`pseudo_voigt(x, intensity, fwhm, mixing, x0)`, and the background
functions — vectorized across scan points, given the native fit parameters
already stored in `PeakCollection._params_value_array`. Only `Gaussian` and
`PseudoVoigt` peak shapes, and `Linear`/`Quadratic` backgrounds, need
support (PyRS's only supported combinations).

Add a method (on `PeakCollection`, or a fit-engine base class) that, given
the workspace's original two-theta axis (`HidraWorkspace._2theta_matrix`,
confirmed unchanged from the original assumption), evaluates the fitted
model and returns:
- `model_intensity: np.ndarray` — shape `(n_scan_points, n_2theta)`.
- `model_variance: np.ndarray` — same shape, propagated from the stored
  per-parameter fit errors via the `uncertainties` package (already used
  elsewhere in the codebase — `_object_uarray`, `get_strain`, `fields.py`
  — for the same first-order propagation pattern).

**Document as a known limitation, not presented as exact:** no covariance
matrix is retained anywhere in PyRS — Mantid's fit-error output provides
diagonal standard errors only — so `model_variance` **neglects parameter
correlations** by construction. This should be stated explicitly in the
method's docstring.

**`model_variance` is NaN only when the fit was already unsound, not as an
unlucky edge case.** A converged fit's parameter errors come from the same
computation that determines whether the fit succeeded — if those errors
are themselves NaN, zero, or non-finite, the fit was never sound in the
first place, regardless of what a naive convergence check reported.
`PeakCollection.__set_fit_status` (`peak_collection.py:399-419`) already
computes exactly this condition (`bad_params`) and labels it
`"did not refine all parameters"`, but currently only acts on it via the
status string — it does not add the scan point to `_exclude_list` (only a
non-finite `chi2` does that today). **Extend `__set_fit_status` so
`bad_params` also sets `_exclude_list[i] = True`.** This makes "no
representable variance" simply a subset of the existing, already-persisted,
already-GUI-wired exclusion mechanism (checkboxes in `fit_table.py`, strain
masking, CSV export) — there is no case where a scan point has a genuinely
good fit yet an uncomputable variance, so extending this mechanism has no
unwanted side effect on unrelated outputs.

### `pyrs/core/instrument_geometry.py::DENEXDetectorGeometry`

**The FIXME originally cited (`file_object.py:510`) does not exist there.**
The only comment at that location is about a return-type annotation, not
calibration state. The real gap is entirely within this class:

- `__init__` accepts `calibrated: bool` but only validates it, never
  stores it (`instrument_geometry.py:88,107`). Store it.
- Add a boolean `arm_shift_applied` property reflecting that stored state
  (or an equivalent — the exact name is an implementation detail).
- `apply_shift` (`instrument_geometry.py:109-116`) is currently dead code
  that would raise `AttributeError` if ever called — it reads
  `DENEXDetectorShift.calibration_file` off the *class* rather than the
  *instance*. Fix this (`shift.calibration_file`, not
  `DENEXDetectorShift.calibration_file`) so the method actually works,
  since this spec is already touching the class to add the flag above.
- **`apply_shift` currently destructively overwrites `arm_length` in
  place** (`self._arm_length += geometry_shift.center_shift_z`), with no
  way to recover the pre-shift value afterward. Retain it — e.g. store
  `_unshifted_arm_length` alongside `arm_shift_applied` — so any caller
  that needs "the base distance" and "the shift amount" as two separate
  values can get both, without the class having already discarded the
  distinction. This is the fix for a double-counting risk that was
  originally (incorrectly) framed as an NXstress-side consumer bug — the
  real cause is this destructive overwrite, so the fix belongs here, not
  in `_instrument.py`.

**Confirmed: this arm-shift path is never actually exercised in production
today** — `apply_shift`'s only caller (`HidraSetup.get_instrument_geometry`)
is itself only called from three sites, all of which currently pass
`calibrated=False`. The real reduction pipeline applies the arm shift
directly to the pixel matrix instead (`reduce_hb2b_pyrs.py`), bypassing
this class entirely. This means today's NXstress round-trip is
self-consistent only by accident (nothing calibrated ever flows through
this class) — this spec makes the class itself correct, closing that gap
before it's ever hit in practice.

**Future work, not decided against — deferred, tracked explicitly:** the
`.h5` project-file format itself (`HidraProjectFile`) never stores a
calibration flag at all (its writer persists only L2/size/pixel), so
nothing round-trips through it either. Adding a calibration field to the
`.h5` binary format is a real, larger need but is explicitly **out of
scope for this pass** — a back-compat-sensitive format change deserving
its own future planning, not silently folded in here.

### Beam-intensity profile

**Confirmed absent, not merely undocumented:** no beam-intensity, monitor,
flux, or proton-charge concept exists anywhere in PyRS's reduction
pipeline, `HidraWorkspace`, or `SampleLogs`. The raw-NeXus loader
explicitly does not even load monitors
(`nexus_conversion.py: LoadEventNexus(..., LoadMonitors=False)`), and
reduction normalization is a flat-field vanadium correction only — no
per-scan-point flux or duration scaling is ever applied. A stakeholder's
"assume the beam-intensity profile is uniform" is confirmed as the code's
existing *implicit* assumption (incident beam treated as interchangeable
by omission), not a tracked-then-approximated quantity.

**Decided: encode the assumption as real PyRS-side data; NXstress just
writes whatever it's given, unconditionally.** Add a
`beam_intensity_profile` property (get/set) to `HidraWorkspace`, following
the same convention as `direction` (spec 05) — a plain instance attribute
behind a `@property`. Its current default, since no real measurement path
exists, is a documented uniform/constant value (exact shape — scalar vs.
per-scan-point array — is an implementation detail; document whichever is
chosen). This keeps the "uniform" assumption visible and inspectable, and
means if a real per-scan-point beam monitor is ever added to the reduction
pipeline in the future, it populates this same property with real data —
NXstress's writer code (spec 09) needs no change at all when that happens,
since it never special-cases "uniform" itself.

### `STRESS_FIELD` shape — BLOCKED, not resolved

**This item cannot be completed from anything currently in the repository.**
The three example files initially offered as verification data
(`example/HB2B_2246.h5`, `HB2B_2247.h5`, `HB2B_2251.h5` — confirmed real
and git-tracked) do **not** contain a `HidraConstants.STRESS_FIELD`
("stress field") log at all. What they carry is a different log,
`StrainDirection` (a string label — `"Longitudinal Direction"`,
`"Tranvserse Direction"` [sic], `"ST Direction"`) — confirmed by loading
each file directly. No file anywhere in the repository (40+ scanned,
including a second matched triple already wired into tests,
`tests/data/3393_PWHT-TD.h5`/`3394_PWHT-ND.h5`/`3395_PWHT-LD.h5`) has any
log with "stress" or "field" in its name.

**`StrainDirection` is not a valid substitute and should not be treated as
one, regardless of what a stakeholder may have intended by "the 3 strain
directions."** Strain and stress are physically distinct quantities by
definition (stress ≈ strain × elastic modulus) — a direction *label* for
strain measurements cannot stand in for a numeric stress-field *value*,
and conflating them would be a real physics error, not a naming
inconsistency.

`_sample.py:107`'s shape/dtype question for `STRESS_FIELD` **remains fully
open**. This spec cannot resolve it without new information: either a real
dataset containing a genuine numeric `"stress field"` sample log, or
clarification of what `STRESS_FIELD` is actually meant to represent (e.g.,
whether it's expected as a raw-data log at all, versus a value the
stress/strain calculator computes downstream that NXstress should instead
receive from a different source entirely). **Do not guess a shape to
unblock this — leave `_sample.py:107`'s existing TODO in place, updated to
reflect that this was actively investigated and remains blocked**, not
merely "never gotten to."

---

## NXstress / GUI Changes

_None._ (Spec 09 consumes the new `arm_shift_applied`/`_unshifted_arm_length`
accessors and the `beam_intensity_profile` property; it is not blocked on
the `STRESS_FIELD` item, which remains its own tracked gap.)

---

## Tests

- `tests/unit/pyrs/peaks/test_peak_collection.py` (extend): after fitting a
  synthetic diffractogram, call the new reconstructed-spectrum method and
  assert the output shape is `(n_sub_runs, n_2theta)` and values are
  finite; assert the docstring's covariance-neglect caveat is reflected in
  a test comparing propagated variance against a case with known
  correlated parameters (documenting the expected discrepancy, not
  asserting exactness).
- `tests/unit/pyrs/peaks/test_peak_collection.py` (extend): construct a fit
  result with a non-finite parameter error; assert `_exclude_list` is now
  `True` for that scan point (previously only non-finite `chi2` triggered
  this).
- `tests/unit/pyrs/core/test_instrument_geometry.py` (extend): construct a
  `DENEXDetectorGeometry`, apply a shift, assert `arm_shift_applied` is
  `True` and the pre-shift arm length remains recoverable; assert
  `arm_shift_applied` is `False` before the shift; assert `apply_shift` no
  longer raises `AttributeError`.
- `tests/unit/pyrs/core/test_workspaces.py` (extend): `beam_intensity_profile`
  property — default value is the documented uniform/constant value;
  get/set round-trip.

---

## Delivered Feature

> **Internal / contributor-facing:**
> The PyRS fit engine now exposes a reconstructed fitted spectrum (with a
> documented, correlation-neglecting variance approximation), making it
> available for export to NXstress and for other potential uses (e.g.,
> residual plots). `DENEXDetectorGeometry` now correctly tracks and
> preserves calibration state, with a previously-dead, broken code path
> fixed. `HidraWorkspace` now carries an explicit, documented
> beam-intensity-profile assumption rather than leaving it implicit. These
> are infrastructure improvements; the user-visible payoff arrives in spec
> 09 — except for `STRESS_FIELD`, which remains blocked pending new data
> or clarification from the instrument-science team.

---

## Verification

- `pytest tests/unit/pyrs/peaks/` — all pass including new spectrum and
  exclusion tests.
- `pytest tests/unit/pyrs/core/test_instrument_geometry.py` — all pass.
- `pytest tests/unit/pyrs/core/test_workspaces.py` — all pass.
- Manual inspection: fit a peak on a real dataset; print the reconstructed
  spectrum; overlay with the raw data to confirm it is plausible.
- Confirm `_sample.py:107`'s TODO comment is updated to reflect the
  investigated-but-blocked status, not left as if untouched.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A3) — "**The FIXME originally cited (`file_object.py:510`) does not
exist there.** The only comment at that location is about a return-type
annotation, not calibration state."
- Referent: `pyrs/projectfile/file_object.py:510`.
- Verdict: **the main claim is true; its supporting detail is not.** There is
  indeed no calibration FIXME at `:510` — but `:510` is
  `if "_var" not in mask_id:`, **not a comment at all**. The return-type
  annotation TODO (`*** TODO: actually this returns an instance of
  `DENEXDetectorGeometry`! ***`) is at **`:541`**, thirty-one lines away. A
  correction that fixes a location and then mis-describes what is actually there
  is the same defect class it was written to fix.
- Action (for the implementing PR): replace the second sentence with "There is
  no comment at that location at all; `:510` is a mask-id `_var` check. The
  nearest comment, at `:541`, is an unrelated return-type-annotation TODO." The
  same sentence appears in `README.md:726` (Decisions item 14a) and is corrected
  there too.

**F1.2** (A4) — "`model_variance` … propagated from the stored per-parameter fit
errors via the `uncertainties` package (already used elsewhere in the codebase)"
and "no covariance matrix is retained anywhere in PyRS — Mantid's fit-error
output provides diagonal standard errors only".
- Referent: the `uncertainties` package and Mantid 6.16.20260918.1225.
- Verdict: **not probed.** This is a genuine A4 claim about two third-party
  packages and it is recorded here as **uncovered**, not as verified. It was not
  probed because the code that would consume it does not exist yet, so a probe
  would test the packages in isolation rather than this design's use of them —
  but that is a scheduling reason, not an exemption. Whoever implements this
  spec should probe both halves before relying on them, because "Mantid returns
  only diagonal errors" is precisely the kind of claim that reads as settled and
  is not.
- Action: flagged for the implementing PR. No document change.

**Checked and accurate — no action.** Every other citation in this spec lands
exactly on its claimed target: `peak_collection.py:399-419` →
`def __set_fit_status`; `instrument_geometry.py:88,107` → `calibrated: bool,`
and its `check_bool_variable` call, which together confirm the central "accepts
but only validates, never stores" finding; `instrument_geometry.py:109-116` →
`def apply_shift`; `_sample.py:107` → the `STRESS_FIELD` dimensions TODO.

**A4/A5 exemption for the `STRESS_FIELD` item, recorded.** The blocked
`STRESS_FIELD` shape question is genuinely un-probeable: no file in the
repository carries such a log, which is this spec's own finding. Recorded as an
exemption with a reason rather than left silent.

---

## Follow-up 2 — 2026-09-26 (closing the A4 gap left open by Follow-up 1)

Follow-up 1 F1.2 recorded this spec's `uncertainties`/Mantid claims as **not
probed**, reasoning that the consuming code does not exist yet. That reasoning
was wrong: the claims are about an installed package and about existing PyRS
source, both executable today regardless of what consumes them. Probed now —
[`probes/a4_fit_error_propagation.py`](probes/a4_fit_error_propagation.py).

**F2.1** (A4) — **All five claims confirmed.**

```console
  CLAIM   uncertainties does first-order propagation, and PyRS already uses it
          via `_object_uarray` (claim 1)
  RESULT  uncertainties 3.2.3; (2.0±0.1)*(3.0±0.2) -> 6.0+/-0.5; closed-form
          first-order sigma = 0.500000

  CLAIM   `calculate_profile` is dead code with zero callers anywhere (claim 3)
  RESULT  defined at pyrs/core/peak_profile_utility.py:679; call sites found: NONE

  CLAIM   …with leftover debug print(s), a raise for unsupported backgrounds, and
          an exclusive-right-bound window (claim 3's three bugs)
  RESULT  print() at line(s) [736]; raise(s) [(758, "RuntimeError('Background type
          {} is not supported'…"), …]; windowing uses `[left_x_index:right_x_index]`
          (exclusive right bound): True

  CLAIM   only Gaussian/PseudoVoigt and Linear/Quadratic are supported (claim 4)
  RESULT  PeakShape members: ['GAUSSIAN', 'PSEUDOVOIGT'];
          BackgroundFunction members: ['LINEAR', 'QUADRATIC']

  CLAIM   the raw-NeXus loader explicitly does not load monitors (claim 5)
  RESULT  ['424: Filename=…, MetaDataOnly=True, LoadM…']
```

Claim 3's "hardcoding the `Quadratic` background's extra term to zero" is exact:
the `LINEAR` branch calls `quadratic_background(..., b2=0.0, b3=0.0)`, and any
other background raises.

**F2.2** (A4) — Claim 2, "no covariance matrix is retained anywhere in PyRS",
needs one refinement. A source scan does find two hits, and they are **not** a
counter-example — but a reader running the same grep would think they were:

```console
  RESULT  pyrs/core/peak_profile_utility.py:854: def calculate_chi2(covariance_matrix):
          pyrs/core/peak_profile_utility.py:857: :param covariance_matrix:

  CLAIM   …and those two hits are in `fit_peak`, a DEAD scipy path that computes a
          covariance and throws it away (claim 2, refined)
  RESULT  `scipy.optimize.curve_fit` returns one at peak_profile_utility.py:868,
          passed to a `# TODO` stub that ignores it and returns 1.0 (:854-861).
          Modules importing `fit_peak` from peak_profile_utility: NONE — dead code.
```

So the claim holds — nothing *retains* one — with two things worth stating:
the path that computes a covariance is **scipy, not Mantid**, and it discards it
into a `# TODO` stub. The documented approximation therefore stands, and this
notes where correlations would come from if a future spec wanted them.

**F2.3** (A6) — "the only comment there is an unrelated return-type-annotation
TODO" (Follow-up 1 F1.1 corrected its *location*; this corrects its *dismissal*).
- Referent: the archived draft's §2.2 against this spec's rewrite of it.
- Verdict: **"unrelated" is wrong.** The draft's claim at that spot was
  "`HidraProjectFile.read_instrument_geometry` returns `DENEXDetectorGeometry`
  even for calibrated instruments (`file_object.py:510` FIXME) — round-tripping
  through NXstress inherits this inaccuracy." The comment actually in the file,
  at `:541`, is `*** TODO: actually this returns an instance of
  DENEXDetectorGeometry! ***` — **the same observation**, attached to
  `read_instrument_geometry`'s docstring. It is a weaker claim than the draft's
  (a documented return-type mismatch rather than lost calibration state), but it
  is the same subject, not an unrelated one.
- Action (for the implementing PR): the dismissal should read "the nearest
  comment, at `:541`, records the same return-type mismatch the draft pointed
  at, but says nothing about calibration state — which is the gap this spec
  addresses, and it lives in `DENEXDetectorGeometry`, not here."
