# Open Questions — 08 Reconstructed Fit Spectrum & Calibration Fidelity (PyRS)

**Spec:** [08-fit-spectrum-prereqs.md](../08-fit-spectrum-prereqs.md)
**Blocking:** Partially — Q3 (`STRESS_FIELD`) genuinely blocks that one
item; Q1 and Q2 are resolved.

---

## Q1 — RESOLVED: Does a beam-intensity profile data path exist anywhere in PyRS?

Confirmed by direct investigation: **no.** No beam-intensity/monitor/flux/
proton-charge concept exists anywhere in PyRS's reduction pipeline,
`HidraWorkspace`, or `SampleLogs`. The raw-NeXus loader explicitly does not
load monitors; reduction normalization is flat-field vanadium correction
only, with no per-scan-point flux or duration scaling ever applied
(documented as deliberate in `docs/ground_truths.md`).

**Chris:** We make an assumption that the beam-intensity profile is uniform

**Resolved, and confirmed as the code's existing implicit-assumption
reading** — incident beam is already treated as interchangeable across
scan points by omission, not measured-then-approximated. Rather than
hardcode this inside NXstress's writer, PyRS now encodes it as real,
documented data: `HidraWorkspace` gains a `beam_intensity_profile`
property (get/set), defaulting to a uniform/constant value. NXstress
(spec 09) reads this unconditionally, with no uniform-specific logic of
its own — meaning a future real beam-monitor data path would populate the
same property and require no NXstress-side change at all.

---

## Q2 — RESOLVED: What does `model_variance` being "NaN if not computable" mean in practice?

**Confirmed: rare, and only when the fit was already unsound.** No
covariance matrix is retained anywhere in PyRS (only diagonal parameter
standard errors are ever stored), so `model_variance` uses first-order
`uncertainties`-package propagation of those stored errors — real, usable,
but it necessarily neglects parameter correlations by construction. This
is a documented approximation, not an exactness claim. `model_variance` is
NaN only when a stored parameter error is itself already NaN, zero, or
non-finite — which `PeakCollection.__set_fit_status` already detects
(`bad_params`, `peak_collection.py:406-408`) and already labels
`"did not refine all parameters"`.

**Chris:** We can extend `peak_collection.py:__set_fit_status` L399-L419 to automatically mask entereies when a `diffractogram/fit_errors` is `NaN`.

**Resolved, exactly as proposed** — and the "reuse existing exclusion
mechanism" side effect initially flagged as a concern (dropping a scan
point's row from CSV/strain outputs too, not just the diffractogram fields)
turned out not to be a real trade-off: there's no such thing as a fit that
genuinely succeeded but has unrepresentable variance — non-finite/zero
parameter errors are a symptom of an unsound fit, not an unlucky detail of
an otherwise-good one. `__set_fit_status` is extended so `bad_params` also
sets `_exclude_list[i] = True` (currently only non-finite `chi2` does).

---

## Q3 — BLOCKED: What is the actual (confirmed) `STRESS_FIELD` shape?

`_sample.py:107`'s comment was originally paraphrased here as *"No example
data existed when the writer was drafted"* — **correction: that's not the
literal comment text.** The actual comment reads: *"we don't have an
example of these entries, so the dimensions may not be correct!"* (verified
directly against the current file). Same substance, but the earlier
phrasing wasn't a verbatim quote despite being presented as one. The spec
required loading a real HB2B dataset and checking.

**Chris:** The examples folder has three files, `HB2B_2246.h5, HB2B_2247.h5, and HB2B_2251.h5`, that represent the 3 strain directions .

**Investigated — these files are real and git-tracked, but do not resolve
this question.** Loaded directly: none contains a
`HidraConstants.STRESS_FIELD` ("stress field") log. What they carry is a
different log, `StrainDirection` (a string direction label, not a numeric
stress value). No file anywhere in the repository (40+ scanned) has any
log with "stress" or "field" in its name.

**This is not resolvable by re-reading `StrainDirection` as a substitute.**
Strain and stress are physically distinct quantities by definition — a
direction label for strain measurements is not a numeric stress-field
value, regardless of what "the 3 strain directions" was intended to point
at. Treating it as equivalent would be a physics error, not a naming fix.

**Still needed, from Chris or the instrument-science team directly:**
either (a) a real dataset containing a genuine numeric `"stress field"`
sample log, so `_sample.py:107`'s shape/dtype can actually be verified, or
(b) clarification of what `STRESS_FIELD` is actually meant to represent —
e.g., is it expected as a raw-data log written during reduction at all, or
is it a value the stress/strain calculator computes downstream, which
NXstress should instead receive from a different source entirely (not a
`SampleLogs` entry populated at reduction time)?

**Next step:** this remains open and blocking for the `STRESS_FIELD` item
specifically — it does not block any other item in spec 08, and spec 08's
text has been updated to state this plainly as an investigated-but-blocked
item rather than an unstarted one.
