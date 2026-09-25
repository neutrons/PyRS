# 04b — Multi-workspace NXstress I/O

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 2/3 (bridges NXstress internal cleanup and the StrainStressViewer hookup)
**Depends on:**
- [01 — Config infrastructure & test framework](01-config-and-test-infra-PR.md)
- [04 — NXstress internal cleanup](04-nxstress-internal-cleanup.md)

---

## Overview

Generalize `NXstress.write` / `NXstress.read` from a single `HidraWorkspace`
to a `list[HidraWorkspace]`, round-trip symmetric: `write` accepts N
workspaces and merges them into one `NXentry`'s combined peak index; `read`
returns the same N workspaces back out.

This relies on the existing invariant that each input workspace covers only
unique scan points (and/or other index fields) — no overlap between inputs —
and requires NXstress to record enough on disk to recover the boundary
between inputs at read time. Workspace boundaries are recovered from
**explicit discriminator field(s) on the combined peak index**
(`_peaks.py::PeakIndex`), named by a new config key,
`nxstress.discriminator_fields: list[str]` (default `[]`) — see
[01](01-config-and-test-infra-PR.md), which this spec now depends on directly.
The specific field names in use for any given deployment are a config-level
policy decision, not a per-call argument, and are deliberately not fixed by
this spec — see `open-questions/04b-multi-workspace-nxstress.md` Q2.

**Ordering rule: discriminators are the most slowly varying coordinates of
the combined index.** `PeakIndex.sort_key` returns
`(*discriminator_values, phase_name, h, k, l, mask)` — discriminator
values first, existing columns unchanged and still last. This is a direct
consequence of checking what the reader actually requires, against the
current code:

- The only reader-side splitter, `_Peaks.peakCollectionRanges`
  (`_peaks.py:246-338`), enforces exactly two invariants: each compound
  key occupies one *contiguous* run (raises `"Interleaved blocks
  detected"` at `_peaks.py:313`/`:332` otherwise), and `scan_point`
  increases within a run (`_peaks.py:306`/`:326`). It never checks that
  the runs themselves are globally ordered — there is no
  `searchsorted`/`argsort`/binary search anywhere in the module. The
  three `sorted(peakss, key=_Peaks.PeakIndex.sort_key)` calls
  (`_peaks.py:184`, `_fit.py:87`, `_fit.py:287`) exist only to give the
  peak-index-family groups a **shared, deterministic block order** so
  their rows stay positionally aligned with each other — any commonly
  agreed order satisfies that, not specifically lexicographic.
- The monotonic-`scan_point`-within-a-run invariant is guaranteed
  upstream by PyRS itself, not by this sort:
  `SubRuns.set` (`pyrs/dataobjects/sample_logs.py:167-168`) already
  raises `"subruns are not sorted in increasing order"` unless
  `np.all(value[:-1] < value[1:])`, so every `PeakCollection.sub_runs`
  and `HidraWorkspace.get_sub_runs()` is strictly increasing by
  construction, independent of NXstress's `sorted()` call.
- Putting discriminators first means each input workspace's rows form
  one contiguous **super-block** in every position-aligned group. That
  is what makes the read-side split and the scan-point-family merge
  both trivial — see the corresponding Scope bullets below — rather than
  requiring new indexing machinery.
- Consequently, `_peaks.py:44-45`'s docstring (which currently states
  the index is *"sorted lexographically prior to output"* as a format
  guarantee, specifically to support append) should be corrected during
  implementation: the actual guarantee is the two invariants above, not
  a specific sort order, and the append cost the docstring warns about is
  addressed instead by [04c](04c-nxstress-append.md)'s tail-append-only
  scope.
- **Orthogonal to name-keyed discriminator values (below):** name-keying
  governs how a discriminator *value* is attributed to a field name;
  this ordering rule governs only *write-time row order*. The reader
  detects workspace boundaries by "the key changed," not by re-deriving
  the writer's sort order, so `nxstress.discriminator_fields` being
  reordered between a write and a later read cannot corrupt anything —
  the same property that already makes name-keying safe against config
  drift.

This spec retires `open-questions/05-strain-stress-viewer.md` Q2:
StrainStressViewer's "one `HidraWorkspace` per direction in; one `.nxs` file
with provenance out" is exactly this mechanism, with `direction` as one
instance of a discriminator field. Spec 05 depends on this spec and declares
`direction` as a discriminator rather than inventing its own index-extension
machinery.

> **Schema precedent already exists — not a hard blocker.** `_peaks.py::_init`
> already writes `mask`, `scan_point`, `center`, `center_errors`,
> `center_type`, and `sx`/`sy`/`sz` onto `NXreflections`
> ([_peaks.py:100-172](../../pyrs/utilities/NXstress/_peaks.py#L100)), and the
> module's own docstring
> ([_peaks.py:37-40](../../pyrs/utilities/NXstress/_peaks.py#L37)) states only
> `h`/`k`/`l`/`phase_name` (plus the unused `qx`/`qy`/`qz`) are
> schema-required — `mask` explicitly was not part of `PeakCollection` before
> this implementation added it. A discriminator column is the same category
> of extension `NXreflections` already tolerates in practice. Verify against
> `NXstress.html` (the canonical schema doc) and the `nexusformat`-org
> validator once both are added to the repo, and record the outcome — but
> that verification is confirmatory, not a precondition for starting this
> spec. See `open-questions/04b-multi-workspace-nxstress.md` Q1.

---

## Scope

**In scope:**
- Two new config keys under `nxstress` (land in `pyrs/resources/application.yml`,
  delivered by spec 01):
  - `discriminator_fields: list[str]` (default `[]`) — names of the fields
    that discriminate input workspaces from one another.
  - `merge_workspaces: bool` (default `false`) — see the empty-config policy
    below.
- Change `NXstress.write` to accept `list[HidraWorkspace]` in place of a
  single `HidraWorkspace` (a length-1 list remains valid, so existing
  single-workspace callers from specs 02/03 continue to work with a trivial
  call-site change).
- Change `NXstress.read` to return `list[HidraWorkspace]` in place of a
  single `HidraWorkspace`.
- A generic, NXstress-internal resolver that, for each configured
  discriminator field name, resolves a value from a given `HidraWorkspace`:
  prefer a matching `@property` on `HidraWorkspace` when one exists, else
  fall back to `HidraWorkspace.get_sample_log_value(name)`. See "Discriminator
  value resolution" under NXstress Changes below.
- Design and implement the discriminator-field mechanism in
  `_peaks.py::PeakIndex` (extend `sort_key`, `validateNoDuplicatePeaks`,
  `_init`, `init_group`, `peakCollectionsFromNexus` to be discriminator-aware,
  and split on read), with discriminator values carried **name-keyed**
  (e.g. a sorted tuple of `(name, value)` pairs), not positionally — see
  "Discriminator representation" below.
- Empty-config policy: `write()` raises if called with more than one
  workspace while `discriminator_fields` is empty, unless
  `merge_workspaces: true` — in which case the workspaces are silently
  merged into one combined index with no discriminator columns at all, and
  `read()` returns a single merged workspace rather than N. This flag is a
  no-op whenever `discriminator_fields` is non-empty; `N == 1` is unaffected
  by either key.
- Merge logic across workspaces in `_InputData.init_group` (concatenate raw
  counts, if present, across inputs), `_Sample.init_group` (concatenate
  sample logs), `_Instrument.init_group` (see the geometry-vs-wavelength
  split below — this is **not** a single uniform rule), and `_Fit.init_group`
  (concatenate reduced diffraction data
  per mask, filling `NaN` for scan points a given input doesn't contribute to
  a given mask, consistent with existing single-workspace behavior). This is
  **plain concatenation in workspace order** —
  `concat(ws0.get_sub_runs(), ws1.get_sub_runs(), …)` — not a merge-and-sort:
  `_Diffractogram.init_group` already writes
  `dg["scan_point"] = NXfield(ws.get_sub_runs())` verbatim (`_fit.py:410`),
  `_InputData.init_group` iterates `ws._raw_counts.keys()` in workspace
  order (`_input_data.py:37`), and `_InputData.readSubruns`'s exact-match
  check — `if ws.get_sub_runs() != scan_points: raise RuntimeError(...)`
  (`_input_data.py:70-72`) — keeps working unchanged specifically *because*
  each workspace's slice of the concatenated array equals its own
  `get_sub_runs()` as-is. A global-`scan_point` merge order would have
  interleaved workspaces with interleaving scan ranges (e.g. `[1,3,5]` and
  `[2,4,6]`) and broken that exact-match check; the slowest-varying
  discriminator rule above avoids that case entirely.
- Extend `_validateWorkspaceAndPeaksData` to validate across the full set of
  input workspaces (no-overlap invariant; required logs present in each).
- Update the spec-02/03 call sites (`PeakFittingModel`, `TextureFittingModel`,
  `CombineRunsModel`) to pass/receive a length-1 list. For `CombineRuns`
  specifically, this is a trivial wrap — `write([self._hidra_ws], [])` — not
  a restructuring: **decided** (see
  `open-questions/04b-multi-workspace-nxstress.md` Q3) that
  `combine_project_files`'s in-PyRS pre-merge stays, since it already
  produces the same indistinguishable-merge semantics
  `nxstress.merge_workspaces: true` would, and the `.h5` export path still
  needs the single merged workspace regardless.
- Round-trip test: write N workspaces, read back, assert the N reconstructed
  workspaces equal the N inputs (order-independent).

**Out of scope:**
- Append mode (spec 04c).
- Any GUI wiring beyond the call-site updates above needed to keep specs
  02/03 working as before.
- Deciding discriminator fields beyond what's needed to support `direction`
  (spec 05) and CombineRuns-style merges (spec 03) — further discriminators
  may be added by later specs using the same mechanism.
- Adding any dedicated `@property` accessor for a specific field name (e.g.
  `direction`) to `HidraWorkspace` — the resolver falls back to `SampleLogs`
  until/unless a later spec adds one, per the existing property convention
  (see PyRS Changes below).

---

## PyRS Changes

_None required._ `HidraWorkspace` and `PeakCollection` are consumed as-is, N
at a time instead of one at a time. This was reconsidered mid-design (an
earlier draft proposed a new `HidraWorkspace.get_discriminator_value(name)`
method) and reverted: the property-or-log resolution logic lives entirely in
NXstress's own code (see "Discriminator value resolution" below), which
already works against whatever `HidraWorkspace` exposes today with zero
changes to that class.

**Forward note for later specs:** `HidraWorkspace` already uses plain
`@property` for this exact shape of accessor — `name`, `hidra_project_file`,
`reduction_masks`, `calibration_file`, `sample_log_names`
(`pyrs/core/workspaces.py:55-1168`) are all properties, none are
`()`-called methods. If a later spec (e.g. 05, for `direction`) wants a
first-class dedicated accessor rather than relying on the `SampleLogs`
fallback, it should add a `@property` matching that convention — NXstress's
resolver picks it up automatically, with no change to NXstress itself.

---

## NXstress Changes

### `pyrs/utilities/NXstress/_peaks.py`

- Extend `PeakIndex` with a discriminator slot carrying values **name-keyed**
  — e.g. a sorted tuple of `(name, value)` pairs — not a bare positional
  tuple. This matters: `discriminator_fields` in config could in principle be
  reordered between when a file is written and when it's later read; a
  positional tuple would silently misattribute values in that case, while a
  name-keyed representation resolves correctly regardless of list order.
- `sort_key` prepends the discriminator tuple to the ordering key — i.e.
  `(*discriminator_values, phase_name, h, k, l, mask)` — so discriminator
  values are the most slowly varying coordinate and each input workspace's
  rows form one contiguous block (see the Overview's ordering-rule note).
- `validateNoDuplicatePeaks` treats the discriminator tuple as part of the
  uniqueness key, so the same `(phase, h, k, l, mask)` arriving from two
  different input workspaces is not flagged as a duplicate as long as their
  discriminator values differ (and their scan points don't overlap).
- `init_group` accepts the peak collections for all N input workspaces and
  writes the combined, sorted index (exact call shape — one flattened list
  with discriminator values attached, vs. one list per workspace — TBD).
  On-disk, each discriminator field becomes one `NXfield` on `NXreflections`,
  named via the existing `allowed_identifier()` sanitizer
  (`_definitions.py:221-231`).
- `peakCollectionsFromNexus` reconstructs the flattened index as today. The
  per-workspace split is a `groupby` over the discriminator-value prefix of
  the ranges `peakCollectionRanges` already returns — not a new indexing
  mechanism — because the slowest-varying ordering rule above guarantees
  each workspace's ranges are contiguous. The existing block-detection
  algorithm (`_peaks.py:246-338`) generalizes by *prepending* the
  discriminator columns to its key tuple; the contiguity/monotonicity
  checks it already performs are otherwise unchanged.
- Collision guard: raise if a configured discriminator field name collides
  with an existing reserved `NXreflections` column (`h`, `k`, `l`, `mask`,
  `scan_point`, `center`, `center_errors`, `center_type`, `sx`, `sy`, `sz`,
  `qx`, `qy`, `qz`) — see
  `open-questions/04b-multi-workspace-nxstress.md`.

### Discriminator value resolution (new, NXstress-internal)

A small, **bidirectional** resolver, living in `pyrs/utilities/NXstress/`
(exact module/name left to implementation — see open questions): a "get"
half used at write time, and a symmetric "set" half used at read time so a
caller can tell reconstructed workspaces apart by inspecting the same
field it originally supplied.

Get, used at write time for each configured discriminator field name and a
given `HidraWorkspace`:

```python
def _resolve_discriminator_value(ws: HidraWorkspace, name: str):
    if name.isidentifier() and isinstance(getattr(type(ws), name, None), property):
        return getattr(ws, name)
    return ws.get_sample_log_value(name)  # raises if missing or non-constant
```

Set, used at read time on each newly-reconstructed `HidraWorkspace`, mirroring
the get path exactly (property if the type defines a *settable* property,
else a sample log):

```python
def _apply_discriminator_value(ws: HidraWorkspace, name: str, value):
    prop = getattr(type(ws), name, None) if name.isidentifier() else None
    if isinstance(prop, property) and prop.fset is not None:
        setattr(ws, name, value)
    else:
        ws.set_sample_log(name, ws.get_sub_runs(), np.full(len(ws.get_sub_runs()), value))
```

- The `isinstance(..., property)` check (not a bare `hasattr`) matters: it
  guards against a discriminator name accidentally colliding with an
  unrelated method name already defined on `HidraWorkspace` (e.g.
  `save_experimental_data`) and being mismatched as an accessor, which would
  return a bound method instead of a value. The set half additionally checks
  `prop.fset is not None`, so a read-only property with the same name as a
  discriminator field falls back to the log path rather than raising on
  `setattr`.
- The get fallback reuses `HidraWorkspace.get_sample_log_value(name)`
  (`pyrs/core/workspaces.py:713-745`) as-is — it already returns the single
  value when every sub-run agrees, and raises otherwise. No new
  constancy-checking code is needed. The set fallback reuses the existing
  `HidraWorkspace.set_sample_log(name, sub_runs, values, units="")`
  (`pyrs/core/workspaces.py:1001`).
- This bidirectional shape exists so that any later spec adding a dedicated
  `@property` (get **and** set — see 05's `direction` property) gets
  round-trip behavior for free: `write()` reads the property off each input
  workspace, and `read()` writes it back onto each reconstructed one, with
  no NXstress-side special-casing per field name.
- `NXstress.py`/`_peaks.py` resolve `discriminator_fields` (and
  `merge_workspaces`) by reading `pyrs.utilities.config.Config` directly
  (e.g. `Config["nxstress.discriminator_fields"]`) — this module is PyRS's
  own implementation of NXstress and
  already imports `HidraWorkspace`, `PeakCollection`, and `SampleLogs`
  throughout, so a direct dependency on PyRS's config module is consistent
  with the existing coupling, not a new architectural boundary crossing.

### `pyrs/utilities/NXstress/_input_data.py`, `_sample.py`, `_instrument.py`, `_fit.py`

- `init_group` methods change from accepting one `ws` to accepting
  `list[HidraWorkspace]`; concatenate along the scan-point axis. Reuse the
  merge logic `HidraWorkspace.append_hidra_project` already implements
  in-memory (`pyrs/core/workspaces.py:517`) rather than re-deriving it.
- `_Instrument.init_group` treats two genuinely different kinds of field:
  - **Geometry, detector shift, and calibration state** are single,
    entry-wide values — validate they're consistent across all N input
    workspaces; raise a clear error if they are not (mixed-instrument or
    mixed-calibration merges are not supported by this spec).
  - **Wavelength is not** one of these, even though an earlier draft of
    this bullet grouped it with geometry. It's stored per-scan-point
    (`mono["wavelength"] = NXfield(wavelength, ...)`, `_instrument.py:104`)
    and can already legitimately vary *within* a single workspace under
    existing PyRS semantics (`HidraWorkspace.get_wavelength` can return a
    per-subrun dict). It belongs to the scan-point family's concatenation
    pattern (like `_Diffractogram`'s `scan_point`/`diffractogram`), not to
    a cross-workspace equality check — concatenate it in the same
    workspace order as everything else in that family. The same
    distinction will apply to spec 08/09's `beam_intensity_profile` once
    it exists: per-scan-point, concatenated, never validated-for-equality.

### Reconstructing N workspaces from the scan-point family (read side) — new invariant

The write side above concatenates the scan-point family (raw counts,
sample logs, diffractogram, wavelength) across all N input workspaces, but
there is no on-disk field that independently records which workspace a
given scan-point-family row came from — the *only* place a discriminator
value is ever written is the PEAKS/`NXreflections` group, attached to
`PeakCollection` rows. This has a direct, previously-unstated consequence
for `read()`:

- **Read-side mechanism:** first split the PEAKS group into per-workspace
  `PeakCollection` lists (already specified above — a `groupby` over the
  discriminator-value prefix of `peakCollectionRanges`'s contiguous
  ranges). For each resulting workspace, take the **union of its
  `PeakCollection`s' scan points** as that workspace's scan-point *set*.
  Then slice the scan-point family's concatenated arrays by testing
  membership in that set (`np.isin`-style boolean mask), not by position —
  this is robust regardless of what order `write()` happened to
  concatenate workspaces in, and regardless of whether that order matches
  the peak-index family's discriminator-sorted order.
- **New invariant, now made explicit: every input workspace must
  contribute at least one `PeakCollection` whenever N>1.** Without at
  least one `PeakCollection`, a workspace's discriminator value — and
  therefore its scan-point set — is not recoverable from anything on
  disk. Spec 05's three per-direction workspaces already satisfy this
  (every direction has real peak fits). Spec 03's `peakss=[]` is fine
  *only* because it's always `N == 1` (the discriminator mechanism never
  engages at all in that case — see
  `open-questions/04b-multi-workspace-nxstress.md` Q3). This was previously an
  implicit assumption, not a stated requirement.
- **This invariant is enforced, not just documented:** `write()` raises a
  clear error — via the same `_validateWorkspaceAndPeaksData` extension
  that already checks the no-overlap invariant — if `len(wss) > 1` and any
  input workspace contributes zero `PeakCollection`s. This is a
  correctness gate, not an optional convention: violating it would not
  fail loudly at write time, only produce a silently-unsplittable file
  discovered later at `read()` — exactly the kind of gap this plan's
  "raise a clear error" pattern exists to close elsewhere (see, e.g., the
  discriminator/reserved-column collision guard above).
- No new on-disk schema surface is needed for this — the fix is
  read-algorithm-plus-write-time-validation, not a new field.

See `open-questions/04b-multi-workspace-nxstress.md` Q7 for the full
writeup of how this gap was found.

### `pyrs/utilities/NXstress/NXstress.py`

- `write(wss: list[HidraWorkspace], peakss: ...)` — exact `peakss` shape
  (flattened vs. per-workspace) TBD in implementation.
- `read(entry_number) -> (list[HidraWorkspace], list[PeakCollection])`.
- `_validateWorkspaceAndPeaksData` extended to the N-workspace case,
  including the empty-`discriminator_fields`/`merge_workspaces` branch: raise
  on `len(wss) > 1` with no discriminator fields configured, unless
  `merge_workspaces` is `true`; **and** raise on `len(wss) > 1` if any
  input workspace contributes zero `PeakCollection`s (see "Reconstructing N
  workspaces from the scan-point family" above — without at least one
  `PeakCollection`, that workspace's discriminator value can't be
  recovered on read, so this must be caught at write time, not discovered
  later as a silent read-side misattribution).

---

## Tests

`tests/unit/pyrs/utilities/NXstress/test_peaks.py`, `test_NXstress.py` (extend):
- Discriminator round-trip: two minimal workspaces with disjoint scan points
  and distinct discriminator values; write, read, assert both are recovered
  exactly.
- No-overlap violation: two workspaces sharing a scan point under the same
  discriminator value raise a clear error at write time.
- Single-workspace back-compat: a length-1 list round-trips identically to
  the pre-04b `HidraWorkspace`-only behavior (regression guard for specs
  02/03).
- Empty-config policy: `write()` with N>1 workspaces and
  `discriminator_fields = []` raises by default; with `merge_workspaces =
  true`, it merges silently and `read()` returns one workspace, not N.
- Resolver behavior: a discriminator name matching a `HidraWorkspace`
  `@property` uses the property; a name matching an unrelated *method* name
  (not a property) falls through to the `SampleLogs` fallback rather than
  returning a bound method; a name absent from both raises via
  `get_sample_log_value`; a name present but non-constant across a
  workspace's scan points raises.
- Resolver symmetry: for a discriminator name backed by a get/set property,
  write then read recovers the value via the property on each reconstructed
  workspace; for a name with no matching property (or a read-only one),
  write then read recovers the value via `get_sample_log_value` instead.
- Discriminator-field-reorder regression: write with
  `discriminator_fields = ["a", "b"]`, then read with
  `discriminator_fields = ["b", "a"]` (simulating config drift between write
  and read) — assert values still resolve correctly by name, not position.
- Empty-`PeakCollection` invariant: `write()` with N>1 workspaces where one
  contributes zero `PeakCollection`s raises a clear error (not a silent,
  later-discovered read-side misattribution); confirm the same call with
  `N == 1` and zero `PeakCollection`s (spec 03's case) does **not** raise.
- Scan-point-family read-split correctness: write two workspaces whose
  scan-point *values* interleave numerically (e.g. workspace A has
  `[1, 3, 5]`, workspace B has `[2, 4, 6]` — their rows are still
  positionally contiguous per workspace in the concatenated array, since
  the write side never interleaves *positions*, but their values
  interleave when compared) — read back and assert each reconstructed
  workspace's sample logs, wavelength, and diffraction data contain
  exactly its own scan points, recovered via value-set membership, not by
  assuming the reader independently knows position boundaries.

`tests/integration/test_nxstress_viewer_roundtrip.py` (extend):
- CombineRuns regression: confirm `export_project_files`'s `.nxs` branch
  still works once wrapped in a length-1 list —
  `NXstress.write([self._hidra_ws], [])` — per the resolved Q3 (pre-merge
  stays; no restructuring of `combine_project_files`).

---

## Delivered Feature

> **For downstream NXstress consumers (not yet user-facing):**
> `NXstress` can now write and read multiple `HidraWorkspace` instances
> within a single `.nxs` `NXentry`, provided their scan points (and/or other
> index fields) don't overlap, and provided the deployment's config names at
> least one discriminator field (or explicitly opts into merging them
> indistinguishably). This is a library-level capability in this pass — no
> new GUI action is added; existing viewers continue to pass a single
> workspace. It is the foundation for spec 05 (StrainStressViewer,
> discriminating by `direction`). Spec 03 (CombineRuns) evaluated switching
> to this mechanism and decided against it — its existing PyRS-level
> pre-merge already produces equivalent indistinguishable-merge semantics,
> and is needed regardless for the unaffected `.h5` export path.

---

## Verification

- Cross-check against `NXstress.html` and the `nexusformat`-org validator
  once both land in the repo; record the outcome in
  `open-questions/04b-multi-workspace-nxstress.md` Q1. Not a precondition for
  starting implementation — see the schema-precedent note above.
- `pytest tests/unit/pyrs/utilities/NXstress/` — all pass, including new
  multi-workspace cases.
- `pytest tests/integration/test_nxstress_viewer_roundtrip.py` — all pass, no
  regression in specs 02/03.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A3) — "`SubRuns.set` (`pyrs/dataobjects/sample_logs.py:164-166`)
already raises `"subruns are not sorted in increasing order"` unless
`np.all(value[:-1] < value[1:])`".
- Referent: `pyrs/dataobjects/sample_logs.py`.
- Verdict: **the claim is true; the pointer lands on a different `RuntimeError`
  in the same method.** `:164-166` is the *"Cannot change subruns when
  non-empty"* raise. The guard and message quoted verbatim above are at
  line **167-168**. This is the most confusable possible miss: the right method,
  the wrong error, three lines apart — and it is the single sentence that
  Decisions item 17 rests two subspecs' design on, which is exactly the
  situation `process.md` §5.3 warns about.
  The claim itself is confirmed by probe
  ([`probes/a5_peakcollection_ranges.py`](probes/a5_peakcollection_ranges.py)):

  ```console
    CLAIM   SubRuns.set raises unless strictly increasing
    RESULT  RAISED RuntimeError: subruns are not sorted in increasing order
  ```
- Action: corrected in place to lines 167-168, here and at
  `04c-nxstress-append.md:41` and `README.md:729`.

**F1.2** (A3) — `pyrs/core/workspaces.py` citations are systematically stale, from
a single upstream cause.
- Referent: `pyrs/core/workspaces.py`.
- Verdict: one insertion in that file explains **seven** stale citations across
  two documents — roughly +2 lines before ~L463 and +20 after ~L500:

  | Cited (superseded) | Claimed symbol | Actually at |
  |---|---|---|
  | 693-724 | `get_sample_log_value` | **713-745** |
  | 981 | `set_sample_log` | **1001** |
  | 497 | `append_hidra_project` | **517** |
  | 55-1155 | the five `@property` accessors | `sample_log_names` is at **1168**, outside the range |
  | README §1.1, ×4 | `load_hidra_project`, `append_hidra_project`, `save_experimental_data`, `save_reduced_diffraction_data` | **465, 517, 1045, 1116** |

  **Every prose claim around them is still true.** This is the `process.md` §2.2
  pattern: the pointers rotted and the sentences did not, and the superseded 497 now lands
  on a plausible `except KeyError:` rather than on nothing, which is why
  re-reading never caught it.
- Action: all corrected in place, here and in `README.md`. The root cause is one
  edit, not seven independent errors — worth recording so the next pass checks
  `workspaces.py` citations as a *group* when that file moves.

**F1.3** (A3) — Two citations land on the comment above the code they quote.
- The citation at this spec's "merge logic" bullet gave `_input_data.py:36` for
  "iterates `ws._raw_counts.keys()`"; that line is the comment, and the iteration
  is at line **37**.
- The wavelength bullet gave `_instrument.py:103` and quoted `mono["wavelength"]
  = NXfield(wavelength, ...)`; that line is the `# wavelength by <sub run>?`
  comment, and the code is at line **104**.
- Action: both corrected in place.

**F1.4** (A5) — Decisions item 17's central premise, on which this spec's
discriminator-first `sort_key` depends.
- Referent: `pyrs/utilities/NXstress/_peaks.py`, probed by
  [`probes/a5_peakcollection_ranges.py`](probes/a5_peakcollection_ranges.py).
- Verdict: **fully confirmed.** The reader accepts a globally unsorted index so
  long as each compound key is contiguous, and enforces both stated invariants:

  ```console
    CLAIM   globally UNSORTED but each key contiguous -- 'locally sorted, globally
            segmented' is accepted
    RESULT  segmented: ACCEPTED -> 2 block(s) [(0, 3), (3, 6)]

    CLAIM   a key split into two runs is rejected (contiguity IS enforced)
    RESULT  interleaved: RAISED RuntimeError: Interleaved blocks detected for
            sub-index ('Fe', 1, 1, 1, 'm')

    CLAIM   descending scan_point within one run is rejected (monotonicity IS enforced)
    RESULT  non-monotonic: RAISED RuntimeError: scan_point values are not strictly
            increasing within PeakCollection block at ('Fe', 1, 1, 1, 'm'), indices [0, 3)

    CLAIM   no searchsorted/argsort/binary search anywhere in _peaks.py
    RESULT  occurrence counts: {'searchsorted': 0, 'argsort': 0, 'bisect': 0, 'np.sort': 0}
  ```
- Action: none — the design stands. Per `process.md` §5.3 the probe reproduces
  the counterparty's block rule **locally** rather than importing it, so it fails
  if `_peaks.py` changes its splitting rule instead of silently tracking the
  change. **Promote that property into the test the 04b PR writes** — a test that
  imports `_peaks.py`'s own rule would pass vacuously.

**F1.5** (A2) — This spec's `## NXstress Changes` heading lists four modules in a
single `###` where only the first is path-qualified:
`### \`pyrs/utilities/NXstress/_input_data.py\`, \`_sample.py\`, \`_instrument.py\`, \`_fit.py\``.
- Verdict: not wrong, but it is a fourth structural convention in a series that
  already has three (see the README's Follow-up 1, F1.6). Any mechanical
  ownership check must resolve the bare siblings against the first entry's
  directory or they read as unclaimed.
- Action: recorded, not changed — the heading is unambiguous to a human reader.

**Checked and accurate — no action.** `_peaks.py:313`/`:332` (both interleave
checks), `_peaks.py:306`/`:326` (both monotonicity checks), all three
`sorted(peakss, key=_Peaks.PeakIndex.sort_key)` sites (`_peaks.py:184`,
`_fit.py:87`, `_fit.py:287`) — the "three calls" count is exactly right —
`_peaks.py:44-45`, `_fit.py:410`, `_input_data.py:70-72`,
`_definitions.py:221-231`, `_peaks.py:246-338`.
