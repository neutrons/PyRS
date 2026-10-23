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
  (`_peaks.py:446-559`), enforces exactly two invariants: each compound
  key occupies one *contiguous* run (raises `"Interleaved blocks
  detected"` at `_peaks.py:534`/`:553` otherwise), and `scan_point`
  increases within a run (`_peaks.py:528`/`:548`). It never checks that
  the runs themselves are globally ordered — there is no
  `searchsorted`/`argsort`/binary search anywhere in the module. The
  three `sorted(peakss, key=_Peaks.PeakIndex.sort_key)` calls
  (`_peaks.py:325`, `_fit.py:109`, `_fit.py:309`) exist only to give the
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
- Consequently, `_peaks.py`'s module docstring, which stated the index is
  *"sorted lexographically prior to output"* as a format guarantee specifically
  to support append, had to be corrected during implementation: the actual
  guarantee is the two invariants above, not a specific sort order, and the
  append cost the docstring warned about is addressed instead by
  [04c](04c-nxstress-append.md)'s tail-append-only scope. **Done** --
  [_peaks.py:53-60](../../pyrs/utilities/NXstress/_peaks.py#L53) now states the
  two invariants, says in terms that a global lexicographic sort is *not*
  guaranteed, and cites Decisions Log item 17 for the change.
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
> ([_peaks.py:168-262](../../pyrs/utilities/NXstress/_peaks.py#L168)), and the
> module's own docstring
> ([_peaks.py:32-39](../../pyrs/utilities/NXstress/_peaks.py#L32)) states only
> `h`/`k`/`l`/`phase_name` are schema-required (the unused `qx`/`qy`/`qz` are
> recorded as required at
> [_peaks.py:196-198](../../pyrs/utilities/NXstress/_peaks.py#L196), not in that
> docstring) — `mask` explicitly was not part of `PeakCollection` before
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
  `dg["scan_point"] = NXfield(ws.get_sub_runs())` verbatim (`_fit.py:510`),
  `_InputData.init_group` iterates `ws._raw_counts.keys()` in workspace
  order (`_input_data.py:49`), and `_InputData.readSubruns`'s exact-match
  check — `if ws.get_sub_runs() != scan_points: raise RuntimeError(...)`
  (`_input_data.py:170-172`) — keeps working unchanged specifically *because*
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
(`pyrs/core/workspaces.py:68-1261`) are all properties, none are
`()`-called methods. If a later spec (e.g. 05, for `direction`) wants a
first-class dedicated accessor rather than relying on the `SampleLogs`
fallback, it should add a `@property` matching that convention — NXstress's
resolver picks it up automatically, with no change to NXstress itself.

### Updated by the PR review — the forward note above was wrong

**"NXstress's resolver picks it up automatically" held for reading and not for
writing**, and the asymmetry was silent. `resolve` does pick up any `@property`;
`apply` could only write one that had a *setter*, so a plain `@property` — which
is what the note recommends, and what all six existing ones are — was read from
the property and written back to a sample log. The value reached the file and came
back as the constructor's default. See Follow-up 3 F3.1; the files below are the
PyRS-side changes this section said were not required.

### `pyrs/utilities/restorable_property.py` (new)

`restorable_property`, a `property` subclass naming the backing attribute an I/O
reader may write, plus `restorable()` / `is_restorable()` / `restore()`. Read-only
is preserved — assignment raises — and `isinstance(…, property)` still holds, so
the read path is untouched.

### `pyrs/core/workspaces.py`

A `_direction` field and a `direction` `restorable_property` (backed by
`_direction`, **not** aliased to `_name`), and `hidra_project_file` marked
restorable. This supersedes the "None required" above, and delivers early what
[05](05-strain-stress-viewer.md) scoped as a settable `direction` property.

### `tests/unit/pyrs/utilities/conftest.py`

A `config_override` fixture that overrides config in place without swapping the
`Config` singleton, and a fix to `default_config`, which was leaking a new
singleton per test and orphaning every consumer module for the rest of the
session. See Follow-up 3 F3.3.

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
  (`_definitions.py:574-640`).
- `peakCollectionsFromNexus` reconstructs the flattened index as today. The
  per-workspace split is a `groupby` over the discriminator-value prefix of
  the ranges `peakCollectionRanges` already returns — not a new indexing
  mechanism — because the slowest-varying ordering rule above guarantees
  each workspace's ranges are contiguous. The existing block-detection
  algorithm (`_peaks.py:446-559`) generalizes by *prepending* the
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
  (`pyrs/core/workspaces.py:769-801`) as-is — it already returns the single
  value when every sub-run agrees, and raises otherwise. No new
  constancy-checking code is needed. The set fallback reuses the existing
  `HidraWorkspace.set_sample_log(name, sub_runs, values, units="")`
  (`pyrs/core/workspaces.py:1057`).
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
  in-memory (`pyrs/core/workspaces.py:574`) rather than re-deriving it.
- `_Instrument.init_group` treats two genuinely different kinds of field:
  - **Geometry, detector shift, and calibration state** are single,
    entry-wide values — validate they're consistent across all N input
    workspaces; raise a clear error if they are not (mixed-instrument or
    mixed-calibration merges are not supported by this spec).
  - **Wavelength is not** one of these, even though an earlier draft of
    this bullet grouped it with geometry. It's stored per-scan-point
    (`mono["wavelength"] = NXfield(wavelength, ...)`, `_instrument.py:227`)
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
  `04c-nxstress-append.md:49` and `README.md:729`.

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

**Checked and accurate — no action.** `_peaks.py:534`/`:553` (both interleave
checks), `_peaks.py:528`/`:548` (both monotonicity checks), all three
`sorted(peakss, key=_Peaks.PeakIndex.sort_key)` sites (`_peaks.py:325`,
`_fit.py:109`, `_fit.py:309`) — the "three calls" count is exactly right —
`_peaks.py:53-60`, `_fit.py:510`, `_input_data.py:170-172`,
`_definitions.py:574-640`, `_peaks.py:446-559`.

---

## Follow-up 2 — 2026-09-30 (implementation pass)

Eleven findings from implementing this spec. Four are corrections to claims the
body makes, three are consequences the body did not anticipate, and four are
citation drift from 04 landing. Probes are under
[`probes/`](probes/); each is run with
`pixi run python plans/NXstress-prod/probes/<name>.py`.

**F2.1** (A2/A3) — **the two config keys were never delivered.** Scope says they
"land in `pyrs/resources/application.yml`, delivered by spec 01".
- Referent: `pyrs/resources/application.yml`,
  [`01-config-and-test-infra-PR.md`](01-config-and-test-infra-PR.md).
- Verdict: **false.** 01 has landed; the file carried `enable`, `extension`,
  `use_production_names`, `instrument_name` and `instrument_short_name` and
  neither of these two, and the string `discriminator` appears nowhere in 01.
  This is the shape the session prompt warns about in the opposite direction —
  not work already done upstream, but work *assumed* done upstream and never
  scheduled anywhere.
- Action: shipped by this PR, together with type-checking in
  `validate_config()`. `neutrons_standard.Config` has no defaulting mechanism
  (a missing key raises `KeyError`), so a key that is not in the shipped
  `application.yml` is not "empty by default", it is an error at every call.

**F2.2** (A3) — **`append_hidra_project` cannot be reused as this spec's merge.**
The `init_group` bullet says to "reuse the merge logic
`HidraWorkspace.append_hidra_project` already implements in-memory
(`pyrs/core/workspaces.py:574`)".
- Referent: `pyrs/core/workspaces.py`, probed by
  [`probes/a5_subruns_nonmonotonic.py`](probes/a5_subruns_nonmonotonic.py).
- Verdict: **the pointer is right and the claim is wrong, twice.** It takes a
  `HidraProjectFile`, not an in-memory workspace; and it *renumbers* the
  appended subruns rather than preserving them — which is precisely what this
  spec's exact-match reader depends on:

  ```console
    CLAIM   append_hidra_project takes an in-memory HidraWorkspace (claim 4, part a)
    RESULT  signature (self, hidra_file); type-checked against HidraProjectFile in
            body: True; accepts a workspace: False

    CLAIM   append_hidra_project preserves each input's own scan points (claim 4, part b)
    RESULT  renumbers via `append_list = np.arange(len(hidra_file.read_sub_runs()))
            + 1 + self._sample_logs.subruns.size`: True -- so appended subruns become
            1..N of the merged workspace, not their original values
  ```
- Action: bullet struck; merging is raw-array concatenation inside NXstress.
  Decisions row 31.

**F2.3** (A5) — **the merged scan-point axis cannot be held by `SubRuns` at all,
so no merged `HidraWorkspace` can exist as an intermediate.** Unstated by this
spec, and load-bearing for its own specified `[1,3,5]`/`[2,4,6]` test.
- Referent: `pyrs/dataobjects/sample_logs.py`, `_sample.py`, probed by
  [`probes/a5_subruns_nonmonotonic.py`](probes/a5_subruns_nonmonotonic.py).
- Verdict: **confirmed, and it forces the shape of both sides.**

  ```console
    CLAIM   04b's own specified test case, as an array (not as its first few values)
    RESULT  A=[1, 3, 5] B=[2, 4, 6] -> concatenated [1, 3, 5, 2, 4, 6]; n=6 min=1
            max=6 ptp=5 distinct=6 strictly_increasing=False

    CLAIM   SubRuns accepts the workspace-order CONCATENATION of the two (claims 1, 2)
    RESULT  concatenated: RAISED RuntimeError: subruns are not sorted in increasing order

    CLAIM   the rejection is value-dependent, not intrinsic to concatenating
    RESULT  A ++ [7,8,9]: ACCEPTED -> [1, 3, 5, 7, 8, 9]

    CLAIM   _sample.py:733's `logs.subruns = SubRuns(scan_point)` over an unsplit
            axis (claim 3)
    RESULT  sampleLogsFromNexus shape: RAISED RuntimeError: subruns are not sorted
            in increasing order
  ```
- Action: write concatenates `NXfield`s directly; read splits the axis *before*
  constructing any `SampleLogs`, which is why `sampleLogsFromNexus`,
  `instrumentFromNexus`, `diffractogramFromNexus` and `readSubruns` each gained
  a row-selection argument. Decisions row 31.
- Note the probe also records that `SubRuns.append` bypasses the guard
  `SubRuns.set` enforces, so the invariant is a property of one entry point
  rather than of the type. Nothing here relies on that, but 04c's append work
  might.

**F2.4** (A4) — **Q1 is now answerable, and the answer is yes.** Verification
defers the schema cross-check until "both land in the repo"; the schema landed
with Decisions row 27(b), the validator did not.
- Referent: `docs/developer/source/design/nexus/NXstress.nxdl.xml`, probed by
  [`probes/a4_nxstress_extra_columns.py`](probes/a4_nxstress_extra_columns.py).
- Verdict: **the precedent argument is confirmed from the schema itself.**

  ```console
    CLAIM   the definition does NOT carry `restricts`, so it states a minimum,
            not a closed set
    RESULT  restricts=None; attributes present: ['category', 'extends', 'name', 'type']

    CLAIM   PyRS already ships `peaks` columns the schema does not declare (claim 1)
    RESULT  undeclared-but-written = ['scan_point', 'mask'];
            declared-but-unwritten = ['lattice', 'space_group']

    CLAIM   04b's reserved-column list matches what `_init` writes (claim 3)
    RESULT  written-not-in-guard = []; guard-not-written = []; equal=True
  ```
- Action: Q1 answered in the affirmative; a discriminator column is the same
  category of extension already shipping. The validator half of Verification
  remains unrunnable and stays recorded as such — see F2.10.

**F2.5** (A4, for spec 05) — **the schema has a first-class
`measurement_direction` field**, `NXstress.nxdl.xml:185-196`, `NX_CHAR`,
`minOccurs="0"`, enumerated {radial, longitudinal, normal, tangential,
multiple}, at `NXentry` level.
- Verdict: not a substitute for a per-row discriminator — one value per entry,
  and PyRS's directions are `"11"`/`"22"`/`"33"`, outside the enumeration — but
  no document in this series mentions it, and a three-direction entry arguably
  ought to set it to `"multiple"`.
- Action: **flagged for spec 05**, not implemented here. Recorded so that
  "nobody considered it" and "considered and deferred" stay distinguishable.

**F2.6** (A4) — **a string-*valued* sample log cannot be written at all**, which
this spec's own motivating case produces.
- Referent: h5py via `nexusformat`, probed by
  [`probes/a4_string_log_dtypes.py`](probes/a4_string_log_dtypes.py).
- **This is about log values, not log names**, and the two are independent
  mechanisms. `allowed_identifier` (Decisions rows 23 and 25) is total,
  injective and reversible over arbitrary text: there is no PV-log *key* it
  cannot encode, and the probe's claim 5 re-confirms that rather than leaving
  it to be assumed — `HB2B:Mot:sz_real`, `a b/c$d` and `__weird__` all encode,
  round-trip and write. What had never been checked is the NumPy dtype of the
  *array of values* stored under the encoded name. A field with an entirely
  plain name still fails when its values are `<U`.
- Verdict: **two defects, and the second is the dangerous one.**

  ```console
    CLAIM   a NumPy fixed-width unicode array is writable (claim 1)
    RESULT  <U: WRITE RAISED TypeError: No conversion path for dtype: dtype('<U2')

    CLAIM   a bytes array is writable -- why the existing fixtures work (claim 2)
    RESULT  |S: wrote dtype |S2 -> read dtype |S2, element np.bytes_(b'11')
            (type bytes_), equal to input: True

    CLAIM   the h5py variable-length UTF-8 dtype is writable (claim 3)
    RESULT  vlen utf-8: wrote dtype object -> read dtype object, element b'11'
            (type bytes), equal to input: False
  ```

  A `direction` log *value array* built as `np.array(["11", "11", "11"])` is
  dtype `<U2` and crashes the save, under any name at all. The existing fixtures never hit this because every string
  log they build (`start_time`, `end_time`, `Filename`) is deliberately bytes.
  And even the writable vlen **UTF-8** dtype reads back as `bytes`, so a value
  does not survive a round trip as the same Python object — which for a
  discriminator would not raise, it would split one workspace into two.
- Action: `_Sample._writable` coerces `<U` logs to the vlen dtype, and
  `_discriminator._as_text` normalises a resolved value to `str` on both sides,
  so a discriminator compares equal to itself across a round trip. Covered by
  `test_multi_workspace.py::TestScanPointFamilySplit`.

**F2.7** (A1) — **`read()`'s declared return type contradicts round-trip
symmetry.** The `NXstress.py` bullet declares
`read(entry_number) -> (list[HidraWorkspace], list[PeakCollection])` — flat —
while `write`'s `peakss` shape is left "TBD in implementation". A flat read
cannot say which collection belongs to which of the N workspaces it returns
alongside.
- Action: both sides are per-workspace,
  `list[list[PeakCollection]]`. Decisions row 28.

**F2.8** (A3) — **`HidraWorkspace` defines no property *setters*.** The
forward note is right that its accessors are properties, but none has an
`fset`, so `_apply_discriminator_value`'s settable-property branch has no live
subject until spec 05 adds `direction`.
- Referent: probed by
  [`probes/a5_discriminator_resolution.py`](probes/a5_discriminator_resolution.py).

  ```console
    CLAIM   HidraWorkspace's @property accessors, and which are settable (claim 4)
    RESULT  6 properties: ['calibration_file', 'hidra_project_file', 'name',
            'reduction_masks', 'sample_log_names', 'sample_logs_for_plot'];
            settable (fset is not None): NONE
  ```
- Verdict: every other branch of both resolver halves is confirmed against the
  real class, including that `save_experimental_data` resolves as a bound
  method under a bare `hasattr` and falls through correctly under the
  `isinstance(..., property)` test the spec specifies.
- Action: the branch is written anyway — 05 needs it — and tested against a
  stub subclass rather than asserted of a class that has none. The probe also
  records that with a settable property present, the set half writes the
  property and leaves the underlying log stale; the two can disagree, and the
  property wins on both halves.

**F2.9** (A3) — **`get_sub_runs()` returns `SubRuns`, not an array**
(`pyrs/core/workspaces.py:458-471`), so the Overview's
`concat(ws0.get_sub_runs(), …)` needs `.raw_copy()`. Cosmetic; noted because
the sentence reads as if it were already array-valued.

**F2.10** (A4) — **one Verification step remains unrunnable, and that is
recorded rather than skipped.** "Cross-check against `NXstress.html` and the
`nexusformat`-org validator once both land in the repo": the schema has landed
and is cross-checked (F2.4); the **validator has not**. `nexusformat` 1.0.8
still ships none, and the NeXus-org validator lives in a separate repository —
the state [`probes/a4_nexusformat_validator.py`](probes/a4_nexusformat_validator.py)
already tracks and the gate Decisions row 27 leaves open. Not a blocker for
this PR; the schema half is what Q1 needed.

**F2.11** (A1) — **`merge_workspaces` merges the scan-point family, not the peak
index.** Scope says that with the flag set "the workspaces are silently merged
into one combined index with no discriminator columns at all". It does not say
what happens when two inputs contribute the same `(phase, h, k, l, mask)`.
- Verdict: **they cannot be merged.** With no discriminator, two such
  collections become two blocks of the *same* compound key, which
  `peakCollectionRanges` rejects as interleaved — so the file would be written
  and then be unreadable. The existing `validateNoDuplicatePeaks` already
  catches it at write time, loudly, which is the right behaviour; what was
  missing was the statement that it *will*.
- Action: documented here and pinned by
  `test_multi_workspace.py::TestEmptyConfigPolicy::test_merging_inputs_that_share_a_compound_key_raises`.
  `merge_workspaces` is therefore usable when the inputs' peak collections have
  distinct compound keys, or none at all (spec 03's shape) — not as a general
  "combine anything" switch.

### Invariants written by this PR

| Invariant | Where | Tier |
|---|---|---|
| `_peaks.py`'s splitter enforces contiguity and monotonic `scan_point` **and nothing more**, with the rule reproduced locally so it fails rather than tracks a change | `test_peaks_read.py::TestSplitterEnforcesOnlyContiguityAndMonotonicity` | unit |
| `RESERVED_PEAK_COLUMNS` equals what `_Peaks._init` writes, by iterating the real group | `test_discriminator.py::TestReservedColumns::test_reserved_columns_matches_peaks_init` | unit |
| The identifier encoding is injective over discriminator names, which is why guarding the *encoded* name is sufficient | `test_discriminator.py::TestReservedColumns::test_only_a_reserved_name_encodes_onto_a_reserved_column` | unit |

The first is the promotion this PR owed, from
[`probes/a5_peakcollection_ranges.py`](probes/a5_peakcollection_ranges.py) via
`review/findings.md` §5 row 1.

### One new accepted-residue row in `check_ownership.py`

`README.md  in-table-but-unclaimed  pyrs/utilities/NXstress/_discriminator.py`.
The file is in §5 because this PR creates it; no subspec heading *claims* it
because this spec's heading for it — `### Discriminator value resolution (new,
NXstress-internal)` — deliberately names no path, the module's location having
been left to implementation by `open-questions/04b` Q5. That is the fifth
structural convention in the series, the same shape Follow-up 1 F1.5 recorded
and did not change. Now that the module exists and is called
`_discriminator.py`, the resolution is recorded here rather than by rewriting a
heading in the body. Expected residue, not an unfixed finding; total goes 9 → 10.

### Citations corrected in place

All from 04 landing; every surrounding claim re-read and still true.

| Claim | Superseded | Now |
|---|---|---|
| `peakCollectionRanges` | 246-338 | `_peaks.py:446-559` |
| interleave raises | 313 / 332 | `_peaks.py:534` / `:553` |
| monotonicity raises | 306 / 326 | `_peaks.py:528` / `:548` |
| the three `sorted(...)` sites | 184, 87, 287 | `_peaks.py:325`, `_fit.py:109`, `_fit.py:309` |
| the sort-order docstring | 44-45 | `_peaks.py:53-60` |
| `_init` writes non-required columns | 100-172 | `_peaks.py:168-262` |
| "only h/k/l/phase_name required" | 37-40 | `_peaks.py:32-39` |
| `allowed_identifier` | 221-231 | `_definitions.py:574-640` |
| `dg["scan_point"] = ...` | 410 | `_fit.py:510` |
| iterating `_raw_counts.keys()` | 37 | `_input_data.py:49` |
| the exact-match check | 70-72 | `_input_data.py:170-172` |
| `mono["wavelength"] = ...` | 104 (103 in open-questions) | `_instrument.py:227` |

Several of these moved again within this PR, since it rewrote the same
functions; the values above are as of this Follow-up.

**Superseded — see Decisions Log item 41 and Follow-up 12.** This paragraph
originally left Follow-up 1's closing "Checked and accurate" list untouched on
the grounds that an earlier Follow-up is never edited, and treated the resulting
drift as the expected cost of an append-only record. The rule has since been
narrowed: a citation *value* is a pointer, not a belief, so it is **not** covered
by the append-only rule and is updated wherever it appears. Follow-up 1's
pointers have been re-resolved along with everything else. Superseded values are
recoverable from git history, which is the only place they are worth anything.

---

## Follow-up 3 — 2026-10-07 (PR review)

Changes made during the human review of `eb5457b1..cda22352`, per
[`plans/PR-review-process/review-process.md`](../PR-review-process/review-process.md).
The review conversation is in [`plans/PR_review/04b-comments.md`](../PR_review/04b-comments.md);
only the changes are recorded here.

**F3.1** — **A read-only `HidraWorkspace` property used as a discriminator did not
round-trip, and nothing raised.** `_discriminator.resolve` prefers a property over
a sample log; `apply` used the property only when it had a setter, and otherwise
wrote a log. All six `HidraWorkspace` properties are read-only, so such a field was
read from the property and written back to a log: the value reached the file
correctly, and `read()` returned a workspace whose property still answered the
constructor's default. `write(read(f))` therefore stored something different from
`f`. Not hypothetical — `strainstressviewer/model.py` constructs
`HidraWorkspace(direction)`, so the viewer this series' spec 05 targets already
carries direction in a read-only property.

Resolved with a **`restorable_property`**: a `property` subclass that names the
backing attribute an I/O reader may write, declared on the property itself rather
than in a registry that would drift.

- **New** [`pyrs/utilities/restorable_property.py`](../../pyrs/utilities/restorable_property.py).
  Assignment still raises `AttributeError`, so nothing becomes publicly writable;
  `isinstance(…, property)` still holds, so `resolve`'s read path is unchanged;
  and a sweep over `vars(cls)` partitions properties into restorable and not,
  which is what makes the invariant below total.
- `pyrs/core/workspaces.py` gains `_direction` and a `direction`
  `restorable_property`, and marks `hidra_project_file` restorable.
  **`direction` is backed by `_direction`, deliberately not aliased to `_name`** —
  the viewer happens to put the direction in the name, but `CombineRunsModel`
  puts `"Combined Project Files"` there. Its getter falls back to the `direction`
  sample log when `_direction` is unset: a compatibility shim, which spec 05
  retires (05's Follow-up 2).
- `_discriminator.apply` gains an `is_restorable` branch.
- `_discriminator.field_names` **rejects, at configuration time**, a field naming
  a read-only property that is neither settable nor restorable. That is what
  makes the asymmetry unreachable rather than merely repaired: a field that
  cannot round-trip is refused before any file exists.
- `strainstressviewer/model.py` records the direction on the workspace, in
  `load_hidra_project_file` — the single funnel all three directions and both
  entry points pass through.

**F3.2** — **`_discriminator` was the only module in the codebase bypassing
`config.py`'s documented access rule**, and the comment defending it reasoned from
a true premise to a wrong conclusion. A bound `Config` name *does* go stale when
the `default_config` fixture reloads — but that override capability exists for
tests, so the accommodation belongs in a fixture, not in the shape of production
code. Now `from pyrs.utilities.config import Config`, as every other consumer does.

**F3.3** — **The fixture that made the deviation look necessary was itself
leaking.** `default_config`'s teardown reloaded to *yet another* new singleton, so
any test using it orphaned every consumer module for the remainder of the session.
It now restores the pre-test instance. A new `config_override` fixture deep-merges
into the live singleton's `_config` and restores a deep copy, never swapping the
instance, so an override reaches bound names and module attributes alike.

Four measurements behind that design, none of them previously recorded:

```console
  Config.__getitem__ -> _find reads _config LIVE        no cache, no rescan needed
  a shallow assignment drops sibling keys               KeyError on nxstress.enable
  loadEnv mutates _config in place                      rebound? False
  _Config.validate() is a no-op                         literally `pass`
```

**Side effect worth naming:** `_instrument.py` was unoverridable in tests —
`_instrument_names()` returned the shipped default whatever a test configured, so
any test asserting a configured instrument name was passing vacuously. Fixed by
the same change, without touching `_instrument.py`.

### Invariants written by this round

- `test_discriminator.py::TestEveryPropertyRoundTripsOrIsRefused` — **sweeps
  `vars(HidraWorkspace)`** and asserts the partition is exhaustive: every
  restorable or settable property round-trips through `apply`→`resolve`, and every
  plain read-only one is refused by `field_names()`. It names no property, so one
  added later is covered without editing the test — which is the point, since the
  defect was a property nobody had thought about.
- `test_config.py::TestConfigOverrideReachesBoundNames` — that an override reaches
  production modules which bound `Config` the documented way, that sibling keys
  survive a partial override, and, requesting **no** config fixture, that
  `_instrument`'s and `_discriminator`'s bound `Config` is still the live
  singleton. That last one fails if any fixture anywhere in the run swaps the
  singleton without restoring it.

### Verification, as run

`pixi run test-unit` **421 passed** · `test-integration` 104 passed, 28 skipped,
2 xfailed · `test-gui` 16 passed · ruff clean · mypy at its 11-error baseline ·
`restorable_property`'s doctests pass.

---

## Follow-up 4 — 2026-10-08 (PR review, batch 3)

Changes from the review of `_peaks.py`. Conversation in
[`plans/PR_review/04b-comments.md`](../PR_review/04b-comments.md); only the
changes are recorded here.

**F4.1** — **A string discriminator column was written as `float64` when the entry
held no peak collections.** `discriminator_dtypes` infers each column's dtype from
the values present; with none, `np.asarray([]).dtype` is `float64`, whose kind is
not in `("U","S","O")`, so the string branch was skipped. Reachable in production:
`CombineRunsModel` exports via `write([ws], [[]])` (Decisions row 28) and spec 05
configures `direction`.

```console
  peaks/direction   dtype=float64  shape=(0,)     <-- should hold strings
  peaks/phase_name  dtype=object                  (for comparison)
  read back: 1 workspace(s), [0] collection(s)    <-- harmless to PyRS itself
```

**Decided: emit no column at all** when there is nothing to discriminate, rather
than defaulting the dtype. An entry with no peak collections has no discriminator
values, and `names_for_read` already reads a column-less entry back as a single
workspace. Such an entry is consequently **non-appendable** except by writing a new
`NXentry` — which is correct, and which 04c's Case-A precondition 2 already
enforced with that exact message.

**F4.2** — **Passing the wrong discriminator names to `peakCollectionRanges` gave a
silent wrong answer, not an error.** The expected failure was "Interleaved blocks
detected"; the actual one was worse:

```console
  peakCollectionRanges(peaks) with the argument FORGOTTEN:
    no error; returned 1 range -- WRONG, should be 2 workspaces
```

Both workspaces shared `('Fe',1,1,0,'_DEFAULT_')`, so their adjacent blocks
collapsed into one range, and the monotonicity check passed because the
concatenated scan points were still increasing. Which failure you get depends on
the data.

New `_Peaks._validateDiscriminatorNames` runs first in `peakCollectionRanges` and
compares the caller's names against the columns actually on the group. Distinct
from `_discriminator.names_for_read`, which compares *configuration* against the
file; this compares what the caller passed. The parameter stays `()` rather than
becoming `None`: with the group authoritative the two would behave identically,
and the alternative reading of `None` — "derive them yourself" — would give a pure
function over a group a hidden dependency on configuration, and would mask the
mistake instead of surfacing it.

**F4.3** — **"Merged" was the wrong word, and this document helped spread it.**
NXstress performs no merge. `merge_workspaces` is a config key appearing in no
method signature, and its entire effect is permissive: it lifts
`_validateMultiWorkspace`'s refusal so rows may be concatenated with no boundary
recorded. The merging that produces a genuinely merged `HidraWorkspace` happens in
PyRS before NXstress is called — `HidraWorkspace.append_hidra_project`, via
`CombineRunsModel.combine_project_files`, which then passes a length-1 list
(Decisions rows 10 and 31 both say so; the prose elsewhere did not follow).

An empty discriminator-name tuple therefore means **the entry records no workspace
boundary**, not "the entry is merged". Corrected in
`_discriminator.names_for_read`, `_peaks._validateDiscriminatorNames` and the new
tests.

**A narrowing worth recording.** Even with `merge_workspaces: true`, two
workspaces sharing a compound key are still refused by `validateNoDuplicatePeaks`,
because with no discriminators their sort keys are identical. The escape hatch
works only when the inputs' compound keys already differ — so the ordinary
multi-direction case, two workspaces measuring the same peak, cannot use it at all.

### Invariants written by this round

`test_peaks_read.py::TestDiscriminatorNamesAreCheckedAgainstTheGroup` — omitted
names raise, names for a column the group lacks raise, the correct names still
split into two ranges, and a column-less group still reads with no names. The
third is the one that keeps the guard honest: it fails if the check is made too
strict.

`test_multi_workspace.py::...test_an_entry_with_no_peak_collections_carries_no_discriminator_column`
— asserts the column is absent *and* that the entry still reads back as one
workspace with zero collections.

### Verification, as run

`pixi run test-unit` **426 passed** · `test-integration` 104 passed, 28 skipped,
2 xfailed · `test-gui` 16 passed · ruff clean · mypy at its 11-error baseline ·
`a5_peakcollection_ranges`, `a5_append_preconditions` and
`a5_scan_point_family_resizable` all still run.

---

## Follow-up 5 — 2026-10-08 (PR review, batch 4)

Changes from the review of `_fit.py`. Conversation in
[`plans/PR_review/04b-comments.md`](../PR_review/04b-comments.md).

**F5.1** — **The NaN fill leaked into the reconstructed workspace's mask set.**
One `DIFFRACTOGRAM` group spans the whole entry, so an input that did not reduce a
mask has its rows NaN-filled to keep the scan-point axis aligned. That part is
correct and deliberate. The read side then handed every group to every workspace,
so a workspace came back carrying a mask it never reduced:

```console
  before write:      ws0 [None, 'mask_a']    ws1 [None]
  after round trip:  ws0 [None, 'mask_a']    ws1 [None, 'mask_a']   <-- all-NaN
```

That matters because `reduction_masks` is **counted**, not just listed, by
`texture_fitting_crtl.py:142` and `mantid_peakfit_calibration.py:240`, and drives
the texture viewer's out-of-plane-angle setup.

`_workspaceFromNexus` now omits a mask that is entirely NaN across *this
workspace's* rows. The accepted cost, recorded at the code: a reduction that
genuinely produced only NaN is indistinguishable from one that never ran, and is
dropped too — preferred over silently changing a workspace's mask set.

**Two things deliberately NOT changed**, both checked rather than assumed:

- **A mask defined but never reduced is already handled correctly.** `mask_keys`
  comes from `_diff_data_set`, not `_mask_dict`, so no diffractogram is written
  for it, while the mask array still reaches `instrument/masks` and round-trips
  into `_mask_dict`. Verified end to end. No warning is warranted: that is a mask
  defined and not used, not an anomaly.
- **Dropping such a mask at *write* time would be wrong for the case above**, where
  the mask does have reductions from another input. Whether a workspace has a mask
  is a per-workspace fact, resolvable only on read.

**F5.2** — **A workspace whose reduced data and sub-runs disagreed wrote a
misaligned file, silently.** NXstress writes `scan_point` from the sub-runs and
the diffractogram rows from `_2theta_matrix`; nothing checked that they agreed.
Pre-existing, but N workspaces amplify it from local corruption to a shift of
every later input.

Guarded in **two** places, because one does not cover it.
`HidraWorkspace.set_reduced_diffraction_data_set` now rejects a non-2-D matrix, a
mask array disagreeing with it, and a row count contradicting known sub-runs —
but that setter is one of **five** paths that write `_2theta_matrix`, and the
other four bypass it: `_load_reduced_diffraction_data`,
`_append_reduced_diffraction_data`, the row-wise `set_reduced_diffraction_data`,
and direct assignment. So `_Diffractogram._concatenated_diffraction` carries the
second guard at the write boundary they all funnel through, naming **which**
input is short — which matters precisely because the damage is not local.

`test_fit.py::TestReducedDataRowCountIsGuarded` pins both, and its docstring
lists those five paths so a later reader can see what each guard is for. Two of
its four tests force the mismatch *past* the setter, the way the other paths do.

### Verification, as run

`pixi run test-unit` **431 passed** · `test-integration` 104 passed, 28 skipped,
2 xfailed · `test-gui` 16 passed · ruff clean · mypy at its 11-error baseline.

---

## Follow-up 6 — 2026-10-09 (PR review, batch 3 reopened: `IndexedPeaks` was invisible to `mypy`)

Reopens batch 3. Found by questioning a phrase in the batch 4/5 verification
report — "mypy at the 11-error baseline" — which measured against the state at
the *start of the review* rather than against `next`. Since the review's subject
is 04 + 04b + 04c, those are different frames only if the errors are inherited,
and these are not:

```console
next   : Success: no issues found in 8 source files
HEAD   : Found 11 errors in 1 file (checked 10 source files)
```

`git blame` attributes all 11, in [_peaks.py](../../pyrs/utilities/NXstress/_peaks.py),
to `7799a6738` — this subspec's implementation commit. **The word "baseline" was
doing the hiding**, not the measurement: the number was right and the referent
was wrong. Recorded here because the same mistake is available on any later
subspec — a type checker, a linter and a test count are all baselined against
the merge target, never against the branch.

### F6.1 — one line made ten of the eleven

```python
class _Peaks:
    # Re-exported so callers can reach it as `_Peaks.IndexedPeaks`; it lives at
    # module scope because `@validate_call_` resolves annotations while the class
    # body is still executing, when a nested name does not yet exist.
    IndexedPeaks = IndexedPeaks
```

At runtime this is a harmless alias. To `mypy` it binds a class **variable** that
shadows the module-level `NamedTuple` for the remainder of the class body, so
every subsequent annotation named a variable rather than a type:

- `Variable "…_Peaks.IndexedPeaks" is not valid as a type` × 6 (`:140, :265, :307, :334, :600, :632`)
- and the knock-on `IndexedPeaks? has no attribute "discriminators" / "collection" / "logs"` × 4 (`:299, :339, :375, :399`)

So `IndexedPeaks` — the type the whole multi-workspace split is expressed in, and
the one Follow-up 2 F2.3 introduced to carry discriminators alongside a
`PeakCollection` — was the one type in this subspec that `mypy` could not check.

Note what the comment actually justifies. The rationale is true and worth
keeping, but it explains why the class is defined at **module scope**; it does
not justify the alias, and the alias is the shadowing. The comment now sits on
the definition, where it applies, and the re-export is gone. Nothing consumed it:
`_Peaks.IndexedPeaks` appeared in exactly two places repo-wide, both docstrings
in this same file, now naming the class directly.

### F6.2 — the eleventh: an implicit `Optional` contradicting the field it fills

`_Peaks.indexed(cls, peakss, logs: SampleLogs = None)` declares a non-optional
parameter with a `None` default, while the field it populates is declared
`logs: SampleLogs | None = None` — and most call sites omit the argument
entirely. Now `SampleLogs | None`, matching the field.

### Verification

`mypy` over `pyrs/utilities/NXstress`, `restorable_property.py` and
`pyrs/core/workspaces.py`: **no issues in 11 source files** — the first time this
branch has matched `next`'s clean result. 443 unit / 104 integration / 16 GUI
pass, `ruff check` and `ruff format --check` clean, and all probes run except the
deliberately-retired `a5_nxstress_roundtrip`.

**Not adopted:** adding `mypy` to `.pre-commit-config.yaml`, which would have
prevented this. It is declared in `pyproject.toml`'s dependencies but has no hook
and no pixi task, so nothing ran it; enabling it would surface whatever the rest
of `pyrs/` carries, which is out of scope for a PR review. Left as a deliberate
decision rather than an oversight.

---

## Follow-up 7 — 2026-10-09 (PR review: the probe this subspec retired was retired in one file only)

04b retired [`probes/a5_nxstress_roundtrip.py`](probes/a5_nxstress_roundtrip.py)
"to record" — it pinned the pre-04b single-`HidraWorkspace` `write` signature,
which is precisely what this subspec changed, so re-pointing it would erase the
baseline it exists to preserve. That decision stands. Its *execution* was
incomplete in a way that matters now that 04c has also landed.

### F7.1 — an expected failure must be declared in the file, not only in the index

The disposition was recorded only in `probes/README.md`'s table. The probe opened
as a live A5 probe and ended with `Run: pixi run python …`. Since an audit re-runs
the probe directory ([`process.md` §7.5](../audit-process/process.md)), that makes
its traceback indistinguishable from a regression — and the directory is run by
nothing *between* audits, which is how `a5_transformations_chain.py` stayed broken
from this subspec until the 04 PR review.

The file now carries a `RETIRED` block naming the exact failure to expect — a
pydantic `ValidationError` at `nxs.write(ws, peaks)`, the bare workspace against
this subspec's list signature — and the sentence **any other failure is a real
finding**, without which the annotation would excuse every future breakage too.
It also sets `AUDIT_STATUS = "retired"` at module level so a sweep can honour the
retirement without parsing prose.

Generalised, since this will recur: **`process.md` §7.5 gains item 9**, stating
what a retired probe must carry, and `probes/README.md` gains a sweep that keys on
the marker rather than the filename.

### F7.2 — two of its claims have since become false, and one left probe coverage

A retired probe's claims are not rewritten — §7.5 item 6, a log preserves what was
believed. But a claim that has become *false* must be marked as false at the claim,
or the file reads as a current description of the system:

- **Claim 4** asserted that `NXstress.py` raises on any operation extending an
  existing `NXentry` and that "each write must currently be a fresh entry". 04c
  inverted this (Decisions row 33). Marked **NO LONGER TRUE** in place.
- The *"Deliberately NOT probed"* paragraph said 04b's round trip, discriminator
  resolution and 04c's append path "cannot be probed: none of that code exists
  yet". All three have landed and all three are now probed.
- The one statement that still executes, `inspect.signature(NXstress.write)`,
  prints the **post**-04b signature under the claim heading asserting the pre-04b
  one — so the live measurement contradicts its own heading. The record of the old
  signature is the prose, not the measurement, and the file now says so.

§7.5 item 6 also requires the successor to be named in **both** files. The index
named `test_multi_workspace.py` — a *test*, and for one claim of five. The
per-claim map is now in both:

| claim | successor |
|---|---|
| 1 — I/O types | `a5_discriminator_resolution.py`, `a5_subruns_nonmonotonic.py` |
| 2 — read-back completeness, raw counts only with `input_data` | **none** |
| 3 — multiple `NXentry` per file | `a5_append_preconditions.py` |
| 4 — mode `"a"` cannot extend an entry | `a5_append_preconditions.py` |
| 5 — pre-04b `write` signature | nothing; history, and the reason the file is kept |

**Claim 2 is a real coverage change, not bookkeeping.** Read-back completeness is
no longer probed at all: `read()` is called by exactly one other probe, and only
for the append round trip. `README.md`'s A5 cell is already `~`, so the matrix is
not over-claiming — but the composition of that `~` has changed, which belongs in
`review/findings.md` §3 and is left for the batch that owns that file. "Covered by
tests" is not a successor probe, and item 9 now says so explicitly.

### Verification

The sweep documented in `probes/README.md`, run as written: **19 OK, 1 SKIPPED
(retired), 0 FAIL.**

---

## Follow-up 8 — 2026-10-10 (PR review, batch 5 completed: `_input_data.py`, `_sample.py`)

These two files were skipped. Appendix A's batch 5 names three — `_input_data.py`,
`_instrument.py`, `_sample.py` — and only `_instrument.py` was walked, because it
was pulled forward during batch 4 and written up under the heading "Batch 5",
after which the walk advanced on the label rather than the file list. Recorded
because the failure mode is a property of batch *labels*, not of these files: a
heading that names one file can stand for a batch that names three.

### F8.1 — the `stress_field` direction was written as an array into a scalar attribute

The direction is held as a per-scan-point sample log but the NXstress schema
defines it as a scalar attribute on `stress_field`. `init_group` stored
`logss[0][key]` — the whole array — and `sampleLogsFromNexus` broadcast it again:

```console
WRITE  direction attr value : ['z', 'z', 'z']
READ   recovered dtype/shape: <U1 (3, 3)
       original was         : ['z' 'z' 'z']
```

Nothing raised. `SampleLogs.__setitem__` validates the first axis only, and
`len((3, 3)) == 3`, so a 3-point direction became a 3×3 log on a round trip.

The same line bypassed the cross-input agreement that `name` and
`chemical_formula` get from `_scalar`, so two inputs could disagree and the first
would silently win.

Both now go through `_scalar`. The read side takes
`to_text(np.atleast_1d(attr)[0])`, which additionally recovers a file written
before this fix as a 1-D log instead of a 2-D one.

**`else "x"` is left standing**: when no direction log exists the writer still
invents `"x"` rather than omitting the attribute. Changing that alters what the
reader recovers, and the writer's own TODO records that no real example of these
entries exists. Flagged, not guessed at.

### F8.2 — `_scalar` compared normalised values and returned an unnormalised one, over first elements only

`distinct = {to_text(v) for v in values}` then `return values[0]`: the check
looked at one spelling and the write used another. Harmless once sample logs are
`str` in memory (04c Follow-up 5 F5.4), but it is the pattern that work removed
everywhere else.

It also took `[0]` of each input. These values are entry-wide but held
per-scan-point, so a log varying *within* one input is as wrong as two inputs
disagreeing — and `[0]` kept one of them silently. Every value is now compared,
and the agreed `str` is returned.

### F8.3 — the last two sites bypassing `growable`

`_input_data`'s `detector_counts` wrote `maxshape=(None, None), chunks=CHUNK_SHAPE(2)`
and `_sample`'s `scan_point` wrote `chunks=CHUNK_SHAPE(1), maxshape=(None,)`.
Literally equivalent to `**growable(2)`/`**growable(1)`, so no behaviour change —
but `growable` exists to be the one spelling, and a second spelling is how the
next field gets written with only half of it. Neither module imports
`CHUNK_SHAPE` any more.

### A pattern worth naming

Three defects in this PR were already understood by a test that worked around
them rather than reporting them:

| | the test | what it accommodated |
|---|---|---|
| D18 | `N_SCAN = 7`, with a comment reasoning about collisions with entry-wide array lengths | the length heuristic rejecting legal appends |
| D20 | a fixture built as `np.array([b"11", ...])` | a normalisation that handled `\|S` but not the `object` dtype HDF5 yields |
| D23 | `if isinstance(direction_val, list): assert direction_val[0] == "z"` | a scalar attribute holding an array |

Each comment is accurate about the hazard. In each case the accommodation was
written and the code was not changed. Worth a reviewer's attention wherever a
test explains *why* it is shaped unusually.

### F8.4 — three fragilities, raised as observations and then fixed

**An assertion's message text was load-bearing.** `_concatenated_pointlist`
detected non-finite sample positions by catching `AssertionError` from
`PointList` and matching `"some coordinates do not have finite values"`. A
reworded upstream assertion would stop the branch matching and let the error
escape part-way through a write. `_Sample._coordinates_are_finite` now asks
first, and `get_pointlist` is called only when it will succeed. `PointList`'s
other two assertions still propagate, which is correct: they report a malformed
coordinate set rather than an absent one. A **missing** coordinate log still
raises `ValueError` naming it, rather than being NaN-filled, which would hide a
workspace with no positions at all. No change to `sample_logs.py` was needed.

**`readSubruns` read the whole of `detector_counts` to keep one workspace's
rows.** Measured rather than assumed, on a 64 MB dataset with a 5% selection:

```console
  full .nxdata then mask     peak=   67.2 MB
  field[mask] then .nxdata   peak=    3.2 MB
```

The selection does reach h5py, so the field is now indexed with
`np.flatnonzero(rows)` before `.nxdata`. The saving scales with how small one
workspace is relative to its entry, which is exactly the multi-workspace case
this subspec introduced. Pinned by tests that check an index-list selection
matches the boolean mask it replaced, **including for non-contiguous rows**.

**A loop that exited on its first iteration.** `_InputData.validateAppend`'s
pixel check used `for point in ws._raw_counts: ... break`, which is correct --
one scan point is representative -- but reads like a defect. Now
`next(iter(...), None)`, with the empty case stepped over explicitly instead of
working because the loop never entered.

### Verification

507 unit (was 443 at the start of batch 6) / 104 integration / 16 GUI; `ruff` and
`mypy` clean; probe sweep 19 OK, 1 SKIPPED (retired), 0 FAIL.

---

## Follow-up 9 — 2026-10-10 (PR review, batch 7: an unresolvable discriminator field is a configuration error and must read as one)

Batch 7 covered `pyrs/resources/application.yml` and `pyrs/utilities/config.py`
([plans/PR_review/04b-comments.md](../PR_review/04b-comments.md)). It closes the
open defect carried since batch 2 — by disproving it.

### F9.1 — the open "whitespace" defect was neither about whitespace nor about identifiers

Batch 2 recorded that a configured discriminator field name containing stray
whitespace is accepted and produces a column that will not match once the typo is
corrected, and proposed either rejecting a name differing from its own `.strip()`
or normalising it.

Re-measured, the encoding is not the problem:

```console
field_names()  : (' direction ',)
column_name()  : '__20direction__20'
write          : ValueError: Sample log name  direction  is not found in
                 allowed value list ['vx', 'vy', 'vz', ...]
```

The encoding worked exactly as designed; the lookup failed because **no property
or sample log of that name exists**. `direciton` — a plain misspelling, and a
perfectly good identifier — fails identically.

**A non-identifier name is legitimate and works today**, which rules out the
stricter proposal raised during review (require `column_name(n) == n`):

```console
encode/decode round trip: True -> BL7__3ACS__3ASample__3ADir
on-disk column(s)       : ['BL7__3ACS__3ASample__3ADir']
read back               : 2 workspace(s), ['11', '22']
```

`BL7:CS:Sample:Dir` writes, splits and recovers. Rejecting escaped columns would
undo what Decisions row 23's `__` escape introducer was added for, and would make
a PV log usable as a retained log in `logs/*` but not as a discriminator.

### F9.2 — it cannot be caught at configuration time, so the fix is the message

Whether a field resolves depends on the workspace, which does not exist when the
configuration is read. `field_names()` already performs every check that *is*
available there — blank, duplicate, reserved-column collision, and (Follow-up 3
F3.1) an unrestorable property. Nothing further can be added without data.

`_discriminator.resolve` is therefore where this belongs, and it now names the
`nxstress.discriminator_fields` key, quotes the configured name with `repr` so
whitespace is visible without anyone having to guess at it, states which of the
two lookups were tried, notes when the name matches a *method* rather than a
property, and lists the logs present.

It tests `name in ws.sample_log_names` rather than catching
`get_sample_log_value`'s `ValueError`: the other failure that method raises, a
log varying across scan points, is an `AssertionError` that must keep
propagating, and re-describing one of two failures by catching an exception type
means depending on which is which — the same reasoning as Follow-up 8 F8.4's
assertion-message finding.

A "did you mean" suggestion was considered and **rejected by the stakeholder**:
*"Every time I've seen that attempted, the developers get it wrong, so it ends up
being extra code that isn't helping anyway."* Quoting the name exactly covers the
whitespace case without inferring intent.

Pinned by `test_discriminator.py::TestUnresolvableFieldIsReportedAgainstTheConfiguration`,
whose docstring records that a non-identifier name is legal — so the disproved
premise is not re-adopted by a later reader.

### F9.3 — `validate_config` checked four of the nine keys it claimed to

Its docstring said it raises "unless **every typed key** holds the type it
declares". `use_production_names`, `instrument_name`, `instrument_short_name`
and both `extension` keys went unchecked:

```console
instrument_name       : 123     -> validate_config() ACCEPTED
use_production_names  : 'yes'   -> validate_config() ACCEPTED
```

The replacement is a **schema**, at the stakeholder's direction: a first pass
used four hand-maintained key tuples, and *"could `VALIDATED_KEYS` be specified
somehow using a schema for the YAML config file?"* is the better shape, because a
key list maintained beside the thing it describes is what decayed to begin with.
`ConfigSchema` is a pydantic model mirroring `application.yml` -- one section
model per section, one field per key -- and `VALIDATED_KEYS` is **derived** from
it. `test_config.py::TestEveryShippedKeyIsValidated` compares the schema against
the shipped file in both directions.

**`strict=True` is load-bearing.** Pydantic's default would have made the schema
*worse* than the hand-rolled checks it replaced, by converting rather than
rejecting the exact values at issue:

```console
  Lax     flag='false' -> False
  Lax     flag='yes'   -> True
  Strict  flag='false' -> rejected
```

**And a schema has its own silent-pass mode.** `ConfigSchema` was first written
with only a docstring; a pydantic model with no fields validates anything, so
`VALIDATED_KEYS` came back empty and `validate_config()` passed vacuously. Caught
because the measurement printed the derived key list rather than only the
verdict, and now asserted by `test_the_schema_describes_some_keys_at_all`.

A blank `instrument_name` is now rejected at import rather than warned about and
replaced with `"HB2B"` at the point of use. The fallback remains as a backstop
for a config-read failure; silently writing `HB2B` into a file prepared for
another beamline is the outcome worth preventing.

### F9.4 — the configured extension carried its own separator

`application.yml` shipped `extension: ".nxs"`, and the first version of the
schema required the dot because `Path.with_suffix` raises
`ValueError: Invalid suffix 'nxs'` without one. The stakeholder's correction --
*"please do not put '.' in the extension; this needs to be implicit"* -- is the
right call: the dot is a `pathlib` separator rather than part of the extension,
and keeping it in the value leaves a reader unable to tell which.

The configuration now reads `"nxs"` / `"h5"`, the schema rejects a value
carrying a dot and names the value to write instead, and `impose_extension`
supplies the separator. It still tolerates a dotted *argument*, since it is also
called with literals; only the configured value is constrained. All five call
sites pass `Config[...extension]`, and every other `.nxs`/`.h5` in the viewers is
a hard-coded literal unaffected by this.

**Also recorded:** `nxstress.use_production_names` has no Python consumer at all
-- `grep` finds it only in plan documents. It is now at least type-checked, so a
typo in it fails loudly rather than sitting inert until spec 10 flips it.

### Deferred from this batch

`validate_config` runs once, at import, and nothing re-validates after
`Config.loadEnv(...)`. The gap belongs to `neutrons_standard.Config`, which
offers no validation hook and whose `_Config.validate` is a no-op, so it is
recorded as **DD-1** in
[plans/PR_review/deferred-defects.md](../PR_review/deferred-defects.md) rather
than worked around here. The runtime-override path is test-only by decision
(Follow-up 3 F3.2-F3.3), which is what makes deferring it safe.

### Verification

526 unit (was 507) / 104 integration / 16 GUI; `ruff` and `mypy` clean.

---

## Follow-up 10 — 2026-10-11 (PR review, batch 8: the viewers silently truncated what they could not show)

Batch 8 covered the four model classes
([plans/PR_review/04b-comments.md](../PR_review/04b-comments.md)) — four, not the
three Appendix A names, since `strainstressviewer/model.py` joined during batch 2.

### F10.1 — `wss[0]` discarded the rest of the file, and said nothing

This subspec changed `nx.read()` to return lists, and both reading viewers took
the first element under a comment asserting that "an entry written by this viewer
holds one". That is true of files the viewer wrote and says nothing about files a
user opens:

```console
entry holds 2 workspace(s): ['11', '22']
the viewer takes wss[0] -> '11'; the rest are discarded silently
```

Neither reader passed `entry_number` either, so `read()` took entry 1 of a
multi-entry file. Both are legitimate NXstress files — a multi-workspace entry is
what this subspec exists to produce, and 04c made multi-entry files routine — and
both loaded with no error and part of their content missing. A partial load can
be fitted and saved back, at which point the remainder is gone.

**Resolved by refusing, not by truncating.** New
`pyrs/interface/nxstress_load.py` provides `read_single_workspace` and
`UnsupportedNXstressFile`; the message names what the file holds and states that
nothing was loaded. `NXstress.entry_numbers()` is new and public — extracted from
the scan `_resolveTarget` did inline — because a caller that can present one
entry has to be able to ask how many there are.

The stakeholder's requirement was that the user be made aware *without*
terminating what they were doing. `UnsupportedNXstressFile` therefore subclasses
`RuntimeError`, to land in the viewers' existing "unable to load" handling, and
two things had to be repaired for that to be true:

- `peak_fitting_viewer.load_and_plot` popped its dialog and then **fell through**
  into code assuming a loaded workspace, turning a reported failure into an
  unreported one against the previously loaded file.
- `texture_fitting`'s load path had **no error handling at all**, so a raise
  escaped into Qt: a traceback on stderr, and a window that looked as though
  nothing had happened.

### F10.2 — the direction the user asserts, and the direction the file records

`strainstressviewer` tells the model which direction a file represents, and
Follow-up 3 F3.1 made that value stick to the workspace's `direction` property
for this subspec's discriminator to carry. If the project file records its own
`direction` log and the two disagree, one of them is wrong and neither the model
nor the viewer can tell which; it is now reported rather than silently resolved
in favour of the user's choice. The log is read directly, since `ws.direction`
would answer with the value just restored.

That model already had the reporting shape the other two needed —
`failureMsg.emit(...)` then `(None, {})` — so no new handling was required.
Covered for disagreement, agreement, and **no `direction` log at all**, which is
the ordinary case for a Hidra project file.


### `entry_numbers()` — why a method, and why both refusals sit inside the `with`

Raised by the reviewer after the fix landed, as a design-consistency question:
*"How does the following method work, design-consistency wise, with the NXstress
interface, which is generally accessed solely in terms of its context manager?"*

It is not an exception to that interface. It reads `self._root`, which only
`__enter__` sets, and carries the same guard and message as `read` and `write`:

```console
inside the context : (1, 2)
outside it         : RuntimeError: Usage error: only usage as context manager is supported!
```

It exists because `read`/`write` already *take* an `entry_number` and nothing
exposed which numbers are valid — so it completes an existing parameter rather
than introducing an idiom. The scan was already there, inline in `_resolveTarget`.

**Not a property**, on the measurement rather than on taste: the answer changes
within one instance's lifetime, so attribute syntax would imply a stability it
does not have.

```console
entry_numbers before write: (1,)
entry_numbers after  write: (1, 2)
```

Two lesser reasons point the same way — it raises outside the context, and a
property that raises makes an object awkward to inspect in a debugger or `repr`;
and `read`/`write` are methods, so a property would read as a different category
of thing.

**The question also exposed a documentation gap.** `entry_numbers` had `Returns:`
and no `Raises:`, so nothing stated the context requirement — and `read()` and
`write()` had the **same omission**, both raising that `RuntimeError` first thing
and documenting neither it nor (for `write`) the already-invalidated case. All
three now state it.

**And one about access.** Only the first of the helper's two refusals needs the
open file: everything `read` returns is detached, because every read path ends in
`.nxdata`, which materialises into numpy — measured by reading sub-runs, raw
counts, diffraction and peak parameters after the close.

**Reviewer:** *"Accessing within the with block is more consistent in any case,
regardless of whether or not we actually need to do that."* Taken as the rule:
the access *pattern* is the thing to be consistent about, not whether an
individual statement strictly requires the file. Both refusals sit inside.

`TestReadReturnsObjectsDetachedFromTheFile` pins the detachment anyway, since
the helper and both viewers depend on it and `NXstress` nowhere states it — and
the read path does index `NXfield`s lazily before materialising, so the
assumption is not idle.

### Verification

526 unit / **112 integration** (was 104) / 16 GUI; `ruff` and `mypy` clean.

---

## Follow-up 11 — 2026-10-15 (PR review, batch 25: the body is the live register, and it had stopped being true)

Raised in the human PR review of 04 + 04b + 04c (batch 25,
[plans/PR_review/04b-comments.md](../PR_review/04b-comments.md)), auditing this
document and its `open-questions` file.

### F11.1 — thirteen body citations, every one into a function this PR rewrote

Follow-up 2's `### Citations corrected in place` table fixed these as of
2026-09-30 and says so in terms: *"Several of these moved again within this PR,
since it rewrote the same functions; the values above are as of this
Follow-up."* The PR then moved them **again**, and the body was never re-synced.

| Claim | Was | Now |
|---|---|---|
| `peakCollectionRanges` | 419-529 | `_peaks.py:446-559` |
| interleave raises | 505 / 524 | `_peaks.py:534` / `:553` |
| monotonicity raises | 499 / 519 | `_peaks.py:528` / `:548` |
| the three `sorted(...)` sites | `_peaks.py:303`, `_fit.py:98`, `:298` | `_peaks.py:325`, `_fit.py:109`, `_fit.py:309` |
| "only h/k/l/phase_name required" | 30-37 | `_peaks.py:32-39` |
| `dg["scan_point"] = ...` | 466 | `_fit.py:510` |
| the exact-match check | 95-97 | `_input_data.py:170-172` |
| the five `@property` accessors | 55-1168 | `workspaces.py:68-1261` |
| `allowed_identifier` | 326-390 | `_definitions.py:574-640` |
| `get_sample_log_value` | 713-745 | `workspaces.py:769-801` |
| `set_sample_log` | 1001 | `workspaces.py:1057` |
| `append_hidra_project` | 517 | `workspaces.py:574` |
| `mono["wavelength"] = ...` | 205 | `_instrument.py:227` |

Two did not merely drift, they **misled**. `workspaces.py:517` landed inside
`get_wavelength` returning a per-subrun dict — entirely plausible beside a
wavelength discussion, and attached to the `append_hidra_project` claim.
`_input_data.py:95-97` landed on `appendableDatasets`' docstring and its
`return row_aligned_fields(...)`, which reads like a reader and is not one. A
citation that resolves to *something related* is worse than one that resolves to
nothing, because checking it does not feel like checking it.

One further inaccuracy in the same sentence: the module docstring at
`_peaks.py:32-39` lists `h`/`k`/`l`/`phase_name` and **not** `qx`/`qy`/`qz`,
which the prose attributed to it. Those are recorded at `_peaks.py:196-198`,
a code comment, now cited separately.

**Follow-up 1 and Follow-up 2 are untouched**, including Follow-up 2's table and
the stale pointers Follow-up 2 explicitly elected to leave in Follow-up 1
(:860-864). The corollary is what makes F11.1 a defect rather than more of the
same accepted cost: **if every Follow-up is frozen, the body is the only live
register there is**, and it is the one place a pointer has to be kept true.

### F11.2 — a claim that had come to say the opposite of its referent

The Overview bullet and `open-questions` Q6 both stated that
`_peaks.py`'s module docstring *"currently states the index is sorted
lexographically prior to output"* as a format guarantee, and that this
*"should be corrected during implementation"*.

It was corrected. `_peaks.py:53-60` now states the two invariants, says in terms
that a global lexicographic sort is **not** guaranteed, and cites this
document's Decisions Log item 17 for the change. So the citation resolved, passed
`check_citations`, and handed a reader text contradicting the prose that quoted
it — while also reading as outstanding work that was in fact done.

Both now record the correction as made and point at the new text. The superseded
wording is kept in the `open-questions` entry, which is the record of what was
flagged; the Overview bullet is live spec text and states the current position.

This is the same shape as spec 04's F-series finding on `check_citations`: the
tool verifies a citation **resolves**, never that the line says what the prose
claims. Two batches running, that gap has been where the defect was.

### Verification

`check_links` 0 findings across 25 documents; `check_citations` 3 problems out
of 425, the known `open-questions/06` bare-`Lnn` residue, none in these two
documents. Every corrected pointer was re-resolved mechanically against the
current source and asserted to contain the symbol or text its prose names — 21
checks, all passing, including one off-by-one caught and fixed that way
(`_peaks.py:195-197` → `:196-198`). No code changed in this batch, so no tiers
were re-run.

---

## Follow-up 12 — 2026-10-15 (batch 26 ruling applied retroactively: Follow-up pointers re-resolved)

Not a new audit of this document. Batch 25 (Follow-up 11) deliberately left this
document's Follow-up 1 and Follow-up 2 pointers stale, citing the policy in
Follow-up 2's closing paragraph. **That deferral was overturned during batch 26**
by the reviewer's ruling, now **README Decisions Log item 41**: a citation value
is a pointer, not a belief, and is updated wherever it appears.

Follow-up 2's closing paragraph is marked superseded in place, since it *states*
the retired policy rather than merely using it.

Re-resolved, all by symbol rather than by diff position:

- **Follow-up 1's "Checked and accurate" list** — interleave raises
  `_peaks.py:534`/`:553`; monotonicity raises `:528`/`:548`; the three
  `sorted(…)` sites `_peaks.py:325`, `_fit.py:109`, `_fit.py:309`; and
  `_peaks.py:53-60`, `_fit.py:510`, `_input_data.py:170-172`,
  `_definitions.py:574-640`, `_peaks.py:446-559`.
- **Follow-up 2's correction table** — the `Now` column only. The `Superseded`
  column is left exactly as written: there the old number *is* the content.
- `workspaces.py:517` → `:574` where Follow-up 2 quotes the body bullet batch 25
  corrected, so the quotation stays true to what it quotes.
- `_sample.py:272` → `:733` (`logs.subruns = SubRuns(scan_point)`), and
  `workspaces.py:403-415` → `:458-471` (`get_sub_runs`).

**Left as written, deliberately.** F1.1's `sample_logs.py:164-166` is the
pointer that finding *diagnoses* — its verdict is "the claim is true; the
pointer lands on a different `RuntimeError` in the same method", which would be
nonsense if the pointer were corrected. Likewise Follow-up 2's `_input_data.py:36`
and `_instrument.py:103`, both introduced by the phrase "the bullet **gave** …".
These are the exceptions Decisions item 41 names.

### Verification

`check_links` 0; `check_citations` 3 of 425 (known `open-questions/06` residue,
none here). A sweep of all 72 citations in this document leaves three landing on
structural lines, all three the `Was` column of Follow-up 11's own correction
table — historical by design. No code changed; tiers not re-run.

## Follow-up 13 — 2026-10-21 (phase-2 PR: `merge_workspaces` removed)

**F13.1** — **F4.3 is half right: NXstress performed no merge on write, but did on
read.** F4.3 says "NXstress performs no merge" and that `merge_workspaces` only lifts
`_validateMultiWorkspace`'s refusal. That is true of `write`. But an entry written
that way records no workspace boundary, so `read` can return only one workspace,
built from every row. NXstress therefore *produced* a merged `HidraWorkspace`, and a
by-hand demo (`tests/scripts/NXstress/multi_workspace.py`) met it: its `direction`
log varied, and `ws.direction` raised `get_sample_log_value`'s bare
`AssertionError`, "Single log direction (= ['11' '11' '11' '22' '22' '22']) is a np
array with multiple items".
- Verdict: the key contradicted this document's own round-trip requirement
  (README Decisions Log row 6). It was also narrower than PyRS's merge: F2.11
  showed that it refuses two workspaces measuring the same peak, and README row 44
  showed that a merged write in decreasing scan-point order could not be read.
- Action: the key is removed, README Decisions Log row 45. The Scope bullet
  "unless `merge_workspaces: true`" and F2.11's `merge_workspaces` usage note are
  superseded by that row.

**F13.2** — **What a merged workspace from PyRS does on write.** Measured before the
fix, with a workspace over sub-runs 1–6 whose `direction` log is `11,11,11,22,22,22`:

```
[] -> written; read back [1, 2, 3, 4, 5, 6] ['11', '11', '11', '22', '22', '22']
['direction'] -> AssertionError Single log direction (= ['11' '11' '11' '22' '22' '22']) is a np array with multiple items(shape = (6,))
```

- Verdict: the behaviour is right and the error is wrong. A discriminator needs one
  value per workspace, and this workspace has none. `resolve`'s docstring
  documented the `AssertionError` as expected.
- Action: `_discriminator.resolve` and `HidraWorkspace.direction` now raise
  `ValueError` and name the cause. After the fix:

```
ws.direction: ValueError Workspace 'test_workspace' has no single direction: its `direction` log varies across sub-
[] -> written; read back [1, 2, 3, 4, 5, 6] ['11', '11', '11', '22', '22', '22']
['direction'] -> ValueError NXstress: discriminator field 'direction' has no single value for workspace 'test_workspace'.
```

### Verification

Unit 612 passed, 15 xfailed; integration 115 passed, 28 skipped, 2 xfailed; GUI 31
passed, 2 xfailed; pre-commit clean. With the pre-fix `workspaces.py` and
`_discriminator.py` restored, the five new `ValueError` tests fail and the rest
pass. `tests/scripts/NXstress/multi_workspace.py` shows the merged-in-PyRS case end
to end.
