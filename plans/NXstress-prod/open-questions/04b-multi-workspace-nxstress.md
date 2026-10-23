# Open Questions — 04b Multi-workspace NXstress I/O

**Spec:** [04b-multi-workspace-nxstress.md](../04b-multi-workspace-nxstress.md)
**Blocking:** No — Q1 was originally a hard gate; downgraded below to a
tracked follow-up now that a precedent for the underlying mechanism is
confirmed to already exist in the codebase. Nothing here blocks starting
implementation.

---

<a id="q1"></a>

## Q1 — RESOLVED (downgraded): does `NXreflections` permit additional index columns at all?
Originally generalized `open-questions/05-strain-stress-viewer.md` Q1 into a
hard implementation gate, on the assumption that no precedent existed for
extending `NXreflections` beyond the schema's required columns.

That assumption doesn't hold: `_peaks.py::_init` already writes `mask`,
`scan_point`, `center`, `center_errors`, `center_type`, and `sx`/`sy`/`sz`
onto `NXreflections` (`_peaks.py:100-172`), and the module's own docstring
(`_peaks.py:37-40`) states only `h`/`k`/`l`/`phase_name` (plus the unused
`qx`/`qy`/`qz`) are schema-required — `mask` explicitly was not part of
`PeakCollection` before this implementation added it. A discriminator column
is the same category of extension `NXreflections` already tolerates in
practice.

**Status:** downgraded from a blocking gate to a tracked follow-up.
`NXstress.html` (the canonical schema doc) and the `nexusformat`-org
validator are being added to the repo separately; once both are present,
verify against them and record the outcome here and in the plan's Decisions
Log (README.md §4). If that verification later disagrees with the
`mask`/`scan_point` precedent, revisit — but implementation proceeds now
rather than waiting on it.

---

## Q2 — What is the exact discriminator field set? — mechanism now decided, names still deferred

Deliberately deferred at the story-scoping stage: "workspaces will be
separated by an explicit combined-index field (possible multiple fields) ...
It should be possible to write the spec without yet having the specifics."

The **mechanism** is now settled (see the spec's Overview, Scope, and
NXstress Changes sections): field names come from a new PyRS config key,
`nxstress.discriminator_fields: list[str]`, resolved via
`pyrs.utilities.config.Config["nxstress.discriminator_fields"]` inside
NXstress itself — not from
NXstress's own defaults and not from a per-`write()`-call argument. Values
are resolved per workspace via a property-or-`SampleLogs`-fallback resolver,
and carried internally name-keyed (not positionally), so config reordering
between write and read can't cause silent misattribution. What remains
**deferred** is only the actual field names any given deployment configures.

Known candidates so far:
- `direction` (spec 05: `"11"`, `"22"`, `"33"`) — the StrainStressViewer case.
- Something CombineRuns-shaped, if spec 03's export path adopts
  `NXstress.write([ws_a, ws_b, ...], peakss)` directly (see Q3) — possibly
  `run_number`, if CombineRuns' inputs are guaranteed disjoint by run.

**Why it matters:** the field-set decision determines whether one generic
discriminator column suffices, or whether the mechanism needs to support
multiple simultaneous discriminator columns (per the "possible multiple
fields" framing) — which affects `sort_key`'s ordering tuple and the
uniqueness-key logic in `validateNoDuplicatePeaks`. The mechanism already
supports multiple fields (name-keyed, not a fixed-arity tuple), so this is
now purely a config-value decision, not an implementation-shape one.

**Next step:** resolve at 04b implementation kickoff, informed by whichever
concrete consumer (spec 05's `direction`, or a CombineRuns discriminator) is
implemented first.

---

## Q3 — RESOLVED: does CombineRunsModel drop its in-PyRS pre-merge once 04b lands?

Spec 03 currently merges N `HidraProjectFile`s into one `HidraWorkspace` via
repeated `HidraWorkspace.append_hidra_project` calls
(`combine_runs_model.py:14-23`) *before* calling `NXstress.write`. Once
`NXstress.write` accepts `list[HidraWorkspace]` directly, that pre-merge step
is redundant for the NXstress export path — but `CombineRunsModel` might
still need the merged single-workspace form for its existing `.h5` export
path (`export_project_files`), which is unaffected by this spec.

**Decided: no, keep the pre-merge.** `HidraWorkspace.append_hidra_project`
already discards per-run boundaries the same way `nxstress.merge_workspaces:
true` would — same resulting semantics, just already implemented at the
PyRS layer. Since the `.h5` export path still needs the merged single
workspace regardless, dropping the pre-merge only for `.nxs` would mean
`CombineRunsModel` carrying two separate data paths (N raw workspaces for
`.nxs`, one merged workspace for `.h5`) to produce an *identical* resulting
`.nxs` file either way — a net complexity increase for no behavioral gain.
04b's own Scope item ("Update the spec-02/03 call sites... to pass/receive a
length-1... list") is therefore just that: **wrap the existing merged
workspace in a length-1 list**, not a restructuring of `combine_project_files`
itself. See `03-combine-runs-nxstress.md` ("Pre-merge stays" section) for
the full writeup.

---

## Q4 — What happens if a configured discriminator name collides with a reserved `NXreflections` column?

`_peaks.py::_init` already reserves `h`, `k`, `l`, `mask`, `scan_point`,
`center`, `center_errors`, `center_type`, `sx`, `sy`, `sz`, `qx`, `qy`, `qz`
as column names. If `nxstress.discriminator_fields` is configured with a
name that collides with one of these, the writer must not silently overwrite
or shadow the reserved column.

**Why it matters:** a silent collision would corrupt either the reserved
column's meaning or the discriminator's, with no error to signal it — the
kind of bug that only surfaces much later, when a round-trip produces
subtly wrong values.

**Next step:** implement a collision guard (raise at config-load time or at
first `write()` call, whichever is more useful to a caller who mistypes a
config value) and cover it with a unit test. Recommendation: raise — no
known use case needs a discriminator name to shadow a reserved column.

---

## Q5 — What should the property-or-log resolver helper be named / where does it live?

The spec's "Discriminator value resolution" section sketches the resolver's
logic but leaves its exact module and function name to implementation (it is
NXstress-internal — no new `HidraWorkspace` method, per the spec's PyRS
Changes section).

**Why it matters:** purely cosmetic — doesn't affect behavior — but worth
settling once so it isn't reinvented per call site inside `_peaks.py`,
`_sample.py`, etc.

**Next step:** decide at implementation time; a reasonable default is a
private function in `_peaks.py` (where discriminator values are consumed)
or a new small `_discriminator.py` module if the logic ends up shared across
more than one of `_peaks.py`/`_sample.py`/`_instrument.py`.

---

## Q6 — RESOLVED: discriminators as the most slowly varying sort-key coordinates

The user proposed making discriminator fields the most slowly varying
coordinates of the combined peak index (i.e. `sort_key` prepends them,
rather than appending or interleaving them), based on the observation that
sorting was never actually required by the NXstress reader — only a
"nice to have."

**Verified directly, and confirmed correct.** `_Peaks.peakCollectionRanges`
(`_peaks.py:246-338`) — the only reader-side splitter — enforces only
contiguity of each compound key's run and monotonic `scan_point` within a
run, never global order; the monotonic-`scan_point` invariant is itself
guaranteed upstream by `SubRuns.set` (`sample_logs.py:164-166`), not by
NXstress's `sorted()` calls. See `04c`'s `open-questions` Q5 for the full
verification writeup (shared across both specs).

**Resolved, adopted as the ordering rule** — `sort_key` returns
`(*discriminator_values, phase_name, h, k, l, mask)`. This makes each
input workspace's rows one contiguous super-block in every position-aligned
group, which in turn simplifies the read-side workspace split (a `groupby`
over already-contiguous ranges, not new indexing machinery) and keeps the
scan-point family's existing exact-match reader (`_input_data.py:70-72`)
working unchanged, since each workspace's slice of the concatenated array
equals its own `get_sub_runs()` verbatim.

**Confirmed orthogonal to Q2's name-keyed discriminator-value
representation:** name-keying governs value *attribution*; this rule
governs write-time row *order*. The reader detects boundaries by "the key
changed," not by re-deriving the writer's sort order, so
`nxstress.discriminator_fields` being reordered between a write and a
later read — the exact scenario name-keying (Q2) protects against — still
cannot corrupt anything under this ordering rule either.

**Also flagged for correction at implementation time:** `_peaks.py:44-45`'s
docstring currently states global lexicographic sorting as a format
guarantee ("the entire index set will be sorted lexographically prior to
output... makes the append operation more complicated, but provides
robustness against duplicates"). That framing is now known to be
inaccurate on two counts: the actual guarantee is the two invariants above
(not a specific global order), and the robustness against duplicates it
credits to sorting is actually delivered by `validateNoDuplicatePeaks`
and the contiguity check, independent of sort order.

---

## Q7 — RESOLVED: how does `read()` split the scan-point family back into N workspaces, and what does that require of `peakss`?

A full-tree consistency scan (checking every other spec against 04b/04c's
revised text) found a real gap in 04b itself, not just a downstream
propagation miss: the write side concatenates the **scan-point family**
(raw counts, sample logs, diffractogram, wavelength) across all N input
workspaces, but 04b never specified how `read()` recovers which rows of
that concatenated data belong to which workspace. The only on-disk
discriminator-value record is the PEAKS/`NXreflections` group, attached to
`PeakCollection` rows — the scan-point family carries no independent
marker of its own.

**Why it matters:** without a stated mechanism, this was an implicit
assumption load-bearing for the plan's *only* two multi-workspace-relevant
consumers — spec 05 (three real per-direction workspaces) needs sample
logs and diffraction data correctly split per direction for its
`StressField` reconstruction test to mean anything, and any future
consumer could violate the assumption with no error to signal it.

**Resolved, direct from the user:** (1) recover each workspace's scan-point
*set* from the already-split PEAKS ranges (union of that workspace's
`PeakCollection`s' scan points), then slice the scan-point family's
concatenated arrays by value-set membership, not position — robust
regardless of write-time concatenation order. (2) This requires every
input workspace to contribute **at least one `PeakCollection`** whenever
N>1 (spec 03's `peakss=[]` is unaffected — it's always `N == 1`, so the
whole discriminator mechanism never engages). (3) Per the user's explicit
follow-up — *"we also should validate it on the write side!"* — this isn't
merely documented: `write()` raises via the same
`_validateWorkspaceAndPeaksData` extension that already catches the
no-overlap violation, so a caller violating the invariant gets a clear
error immediately, not a silently-unsplittable file discovered later at
`read()`.

**Related, found while re-verifying `_instrument.py` directly against this
question:** the Scope/NXstress-Changes bullets describing
`_Instrument.init_group`'s "validate consistency across N inputs" claim
incorrectly grouped wavelength with geometry/shift/calibration-state.
Wavelength is per-scan-point (`_instrument.py:103`) and belongs to the
scan-point family's concatenation pattern instead — corrected in the spec
text; see the new "Reconstructing N workspaces from the scan-point family"
subsection.
