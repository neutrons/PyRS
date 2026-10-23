# 04c — NXstress Append Mode (library only)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 3
**Depends on:** [04b — Multi-workspace NXstress I/O](04b-multi-workspace-nxstress.md)

---

## Overview

Add the ability to extend an **already-written** `NXentry` with more scan
points, rather than requiring every write to create a fresh entry.

This is a different capability than it may first appear. **Before this spec**,
`NXstress.write` already accumulated multiple reductions into one `.nxs` file:
each call added a new, auto-numbered `NXentry`, and its only guard fired on an
accidental name collision. What was missing is the ability to grow *one*
`NXentry`'s data — e.g., a second reduction pass whose scan points belong to the
same combined index as an earlier write, not a new data-reduction condition.

**That starting point no longer describes the code**, and the paragraph above is
kept in the past tense for exactly that reason: under Decisions Log item 33 this
spec made `write()` dispatch on whether the *resolved entry already exists*
rather than always creating a new one, so an omitted `entry_number` now
**appends to the highest existing entry**. The four cases are tabulated in
`write()`'s own docstring
([NXstress.py:156-164](../../pyrs/utilities/NXstress/NXstress.py#L156)) and
restated under *Entry targeting* below; the collision guard survives as the
backstop for the one case still unimplemented, overwriting an existing entry's
contents ([NXstress.py:278](../../pyrs/utilities/NXstress/NXstress.py#L278)). Per the plan's own semantics
(README.md §2.4, "Multi-NXentry semantics"), separate conditions still get
separate `NXentry`s; append is for adding more of the *same* condition's data.

This was originally bundled with the ManualReductionViewer hookup (the old
spec 07) on the rationale that append and that hookup "share a common
primitive." That primitive is no longer shared: the reduction pathway is
decided to always create a fresh file (see
`open-questions/07-manual-reduction-nxstress.md`), so append gains **no GUI
entry point in this pass**. This spec delivers append as a tested library
capability only — usable by future callers (tests, scripts, or a future GUI
action) without a corresponding menu action anywhere in the GUI today.

### Relaxed sortedness: "locally sorted, globally segmented," not globally sorted

04b established that the reader (`_Peaks.peakCollectionRanges`,
`_peaks.py:446-559`) only ever required two invariants — each compound key
occupies one contiguous run (R1), and `scan_point` increases within a run
(R2, already guaranteed upstream by `SubRuns.set`,
`sample_logs.py:167-168`) — never global lexicographic order. Sortedness
beyond that was incidental, not a schema requirement.

This spec adopts that directly: a **single-step write** (all
`HidraWorkspace`/`PeakCollection` present at the `write()` call, 04b's
existing case) still produces a fully sorted combined index, exactly as
today — there's no reason to give that up when everything is available at
once. **Append does not re-sort the file.** It sorts only the incoming
batch internally (so the new data's own rows are self-consistent) and adds
it to the file without reordering what's already on disk. A file that has
been appended to is therefore "locally sorted, globally segmented" rather
than globally sorted — R1 and R2 still hold throughout, which is all the
reader has ever required.

**The non-overlapping-scan-points/index-rows invariant is unchanged and
still fully enforced** — relaxing global sortedness is not a relaxation of
that invariant; see Conflict policy below.

### Architecture: in-place tail-append, not read-merge-rewrite, and not general insertion

Two designs were considered for how append touches the *existing* entry:
1. Read the existing entry back into `HidraWorkspace`/`PeakCollection`
   objects (reusing `NXstress.read()`), append the new data to that list in
   memory, and rewrite the entry using 04b's N-workspace merge-and-write path
   wholesale.
2. Operate directly against the on-disk `NXfield` arrays (via `.nxdata`) to
   determine conflicts, without ever reconstructing the existing entry as
   `HidraWorkspace`/`PeakCollection` objects.

**Decided: (2).** The concern driving this is that a normal reduction/append
cycle should not have to round-trip the *existing* entry's data through
PyRS's outer types just to grow it — particularly raw detector counts, which
can be large.

Relaxed sortedness (above) further narrows what "(2)" needs to do. R1
(contiguity) splits append into two cases:
- **Case A — the incoming data introduces compound keys not already on
  disk** (the normal case: a new workspace, distinguished by a new
  discriminator value per 04b's ordering rule, or a genuinely new
  phase/hkl/mask). A **tail-append** — grow each affected array and write
  the new rows after the current end — satisfies both R1 and R2 without
  touching any existing row.
- **Case B — the incoming data extends a compound key already present on
  disk** (more scan points for a workspace/key that's already in the
  file). R1 forces those new rows into the middle of that key's existing
  run — a true insertion, not a tail-append.

**Decided (scope cut): this spec implements Case A only.** Case B is
detected (see Conflict policy) and rejected with `NotImplementedError` —
not silently mishandled, but explicitly deferred as a follow-up, since it
is a different and more invasive operation than what a normal
reduction/append cycle needs today. This falls directly out of relaxing
global sortedness (above): without a global-sort obligation, the common
case that actually needs to work — "append a new workspace's worth of
data" — reduces to code that already exists (see
`_Peaks._append_peak`, `_fit.py`'s `_PeakParameters._append_peak`, both
already written as `cur = shape[0]; resize(cur+N); arr[cur:] = …`, per the
`# TODO` that stood at `_peaks.py` lines 180-181, now resolved -- see Follow-up 2 F2.11). No new insertion-position machinery is
needed for the case this spec actually delivers.

The **new** data being appended is still supplied the normal way —
`list[HidraWorkspace]` + `list[PeakCollection]`, matching 04b's `write()`
signature — sorted internally among themselves as 04b already does; only
the *existing* side of the operation avoids both the round-trip and any
mid-array mutation.

### Scope: all position-aligned groups, not just input_data and peaks

An entry's on-disk groups fall into two position-aligned families, plus one
family that isn't position-sensitive at all:

| Family | Groups | Aligned by |
|---|---|---|
| Peak-index family | `_peaks.py` (`PEAKS`/`NXreflections`), `_fit.py::_PeakParameters`, `_fit.py::_BackgroundParameters` (`peak_parameters`/`background_parameters` under `FIT`) | The compound `PeakIndex` sort key. `_PeakParameters.init_group`/`_BackgroundParameters.init_group` build their rows via `sorted(peakss, key=_Peaks.PeakIndex.sort_key)` ([_fit.py:109](../../pyrs/utilities/NXstress/_fit.py#L109)) — **the exact same order as the peaks index**, so their rows are positionally aligned with it, not independently keyed. |
| Scan-point family | `_input_data.py` (`detector_counts`, `scan_point`), `_sample.py` (`SAMPLE_DESCRIPTION` per-scan-point logs **and** the `transformations` chain), `_fit.py::_Diffractogram` (`diffractogram`/`diffractogram_errors`), `_instrument.py::_Instrument` (`monochromator/wavelength`), `NXstress.py` (`entry/start_time`, `entry/end_time`) | `scan_point`. |
| Name-keyed, no insertion needed | `_instrument.py::_Masks` | Mask *name*, not position — `init_group` already accepts an existing `masks` group and extends it by name ([_instrument.py:558-567](../../pyrs/utilities/NXstress/_instrument.py#L558)); no new work needed here. **`_Masks` only** — `_Instrument` itself owns a scan-point field, one row above. |

The scan-point row above is the thing this spec's central guarantee is
quantified over, so it is enumerated exhaustively rather than by example. The
authority is each module's own `appendableDatasets`, which is what
`NXstress._validateAppendableShapes` actually consults — not this table:

| Member | Declared by |
|---|---|
| `input_data/detector_counts`, `input_data/scan_point` | [`_input_data.py:83`](../../pyrs/utilities/NXstress/_input_data.py#L83) |
| `SAMPLE_DESCRIPTION/*`, `SAMPLE_DESCRIPTION/logs/*`, `SAMPLE_DESCRIPTION/transformations/*` | [`_sample.py:485`](../../pyrs/utilities/NXstress/_sample.py#L485) |
| `FIT/DIFFRACTOGRAM/*` | [`_fit.py:544`](../../pyrs/utilities/NXstress/_fit.py#L544) |
| `instrument/monochromator/wavelength` | [`_instrument.py:378`](../../pyrs/utilities/NXstress/_instrument.py#L378) |
| `entry/start_time`, `entry/end_time` | [`NXstress.py:667`](../../pyrs/utilities/NXstress/NXstress.py#L667) |

Three of these were absent from this section as first written, and each is a
place the lockstep guarantee could have been violated silently: **wavelength**
(04b had already established it is per-scan-point, not entry-wide),
the **`transformations` chain** (added to `_sample.py` later in this work, and
scan-point-aligned by construction), and the **entry-level times** (the section
originally had no entry-level row at all). Recorded as Follow-up 6 F6.4.

**Decided:** this spec covers the full peak-index family and the full
scan-point family, coordinated so that append never leaves one group's rows
grown without the corresponding rows in every other group in its family.
A partial append (e.g., growing `detector_counts` without growing the
diffractogram) would leave the entry internally inconsistent — not an
acceptable interim state for a correctness-focused library feature. Per
the Case-A-only scope decision above, "grown" means **tail-appended**, not
inserted — both families' new rows are added after each dataset's current
end, in lockstep across every group in the family.

### Entry targeting

`NXstress(path, "a")` with no further argument targets the **last** existing
`NXentry` in the file — the common case (one entry being grown over time)
needs no extra argument. An optional `entry_number` targets a specific
earlier entry instead: `NXstress(path, "a", entry_number=N)`.

### Conflict policy, and Case A/Case B classification

The same pass that reads the existing on-disk key arrays to check for
conflicts also classifies each incoming compound key, at no extra cost:

- **New key relative to what's on disk (Case A):** no conflict — proceeds
  as a tail-append, **subject to two preconditions inherited directly from
  04b's Q7** (never previously carried into this spec's own text):
  1. **The new workspace must itself contribute at least one
     `PeakCollection`.** Exactly 04b's write-time invariant, restated for
     append: without one, this workspace's discriminator value can't be
     recovered on a later `read()`, and the resulting file would be
     silently unsplittable. Raise `RuntimeError` if violated, before any
     mutation — same failure mode as the fresh-write case, just checked
     here too.
  2. **The target entry must already have a discriminator scheme
     established** — i.e., its `PEAKS` group already carries discriminator
     column(s) from an earlier 04b-mechanism write. Appending a
     genuinely new, distinguishable workspace to an entry that was
     originally written as a bare `N == 1`/no-discriminator write would
     require adding a new on-disk column. That is mechanically possible —
     see [`probes/a4_h5py_nexusformat_append.py`](probes/a4_h5py_nexusformat_append.py)
     — but it is schema restructuring, which this spec deliberately
     excludes; a later spec could lift the restriction cheaply. Raise
     `RuntimeError` if the target entry has no discriminator columns at
     all. *(Reworded per Follow-up 1 F1.3 and Decisions Log row 22; the
     original wording, which implied a format limit, is quoted in F1.3.)*
- **Key already present on disk, new scan point(s) under it (Case B):**
  **rejected**, distinctly from a true duplicate — raise
  `NotImplementedError` (not `RuntimeError`; see below), since this isn't
  a corrupted-data condition, it's a real operation this spec deliberately
  doesn't implement (see the Architecture section above). The on-disk
  entry is left unchanged; the instance is *not* invalidated for this
  case specifically, since nothing was detected to be wrong with it —
  only unsupported.
- **Exact duplicate** (same key *and* same scan point, or the same
  peak-index row) already present in the target entry: the true conflict
  case. Raise `RuntimeError` — not the `ValueError`
  `validateNoDuplicatePeaks` raises for the analogous fresh-write duplicate
  case, since this represents a violated invariant on already-committed
  data, not an expected/anticipated bad-input case a caller would
  routinely hit and want to catch.

The classification/conflict check runs against **all** affected groups
before any resize/append call is made — including the two Case-A
preconditions above (≥1 `PeakCollection` for the new workspace; a
discriminator scheme already established in the target entry) — so a
rejected (`RuntimeError`) and an unsupported (`NotImplementedError`)
append are both true no-ops — the on-disk entry is left byte-for-byte
unchanged either way.

After a `RuntimeError` (an exact duplicate, or either of Case A's two
preconditions failing), the `NXstress` instance is left unusable for any
further `write()` call in the same session — the caller must re-open
rather than continue against a session that already detected a corrupted
assumption about the target entry's contents. A `NotImplementedError`
(Case B) does **not** invalidate the instance — the caller may still make
other, Case-A-only, append calls in the same session.

---

## Scope

**In scope:**
- Peak-index family: implement the **tail-append** path for `_peaks.py`'s
  compound index (the `# TODO` that stood at `_peaks.py` lines 180-181 already
  described code written in a form that allows this; see Follow-up 2 F2.11) and, in lockstep, for
  `_fit.py::_PeakParameters.init_group` / `_BackgroundParameters.init_group` —
  all three grow by the same new-row count, appended after their current end.
- Scan-point family: implement the tail-append path for
  `_input_data.py:62-79` (`detector_counts`, on both write and read),
  `_sample.py`'s per-scan-point logs, and `_fit.py::_Diffractogram.init_group`
  — all three grow by the same new-row count, appended after their current
  end.
- `NXstress.py`: remove the guard at `NXstress.py:278` for the append
  case; add entry targeting (`entry_number`, defaulting to the last entry);
  dispatch `write()` to the tail-append path when opened with mode `"a"`.
- Conflict/classification: compare new scan points / index rows against the
  existing on-disk arrays (read via `.nxdata`, not reconstructed as
  `HidraWorkspace`/`PeakCollection`) before any mutation; classify each
  incoming key as Case A (new — proceed, subject to its two preconditions:
  ≥1 `PeakCollection` for the new workspace, and a discriminator scheme
  already established in the target entry), Case B (extends an existing
  key — raise `NotImplementedError`), or exact duplicate (raise
  `RuntimeError`, invalidate the instance), per the policy above.
- Round-trip test: write, then append a new (Case A) workspace, into the
  same entry; verify all five groups' contents are combined and mutually
  consistent.

**Out of scope:**
- Any GUI menu action, file-dialog filter, or viewer wiring. No viewer in
  this plan calls append in this pass.
- ManualReductionViewer hookup (now a fresh-write-only spec: see
  [07 — ManualReduction hookup](07-manual-reduction-nxstress.md)).
- `_instrument.py::_Masks` — already append-capable, name-keyed, no change
  needed.
- **Case B (extending a compound key already present in the file with more
  scan points)** — genuinely different from Case A: it requires a true
  mid-array insertion, not a tail-append, since contiguity (R1) forces the
  new rows into the middle of that key's existing run. Raises
  `NotImplementedError`; not silently mishandled, but deferred as a
  follow-up should a future caller need it.
- Fit-spectrum data (spec 09).
- Detector-calibration fidelity fixes (spec 09).

---

## PyRS Changes

_None._ Append operates entirely within `pyrs/utilities/NXstress/`, against
on-disk NXstress structures and the new data's `HidraWorkspace`/
`PeakCollection` objects — no change to PyRS's own data-object classes.

---

## NXstress Changes

### `pyrs/utilities/NXstress/_peaks.py`

- `init_group` — implement the tail-append path described in the `# TODO`
  that stood at `_peaks.py` lines 180-181, reusing the existing `_append_peak`
  (`_peaks.py:331-404`)
  against an existing on-disk `PeakIndex` group instead of a freshly
  `_init`-ed one: read the existing on-disk `PeakIndex` arrays (via
  `.nxdata`, using 04b's name-keyed discriminator resolution to reconstruct
  each existing row's key) to classify the incoming, sorted batch as Case
  A/B/duplicate (see Conflict policy above); for Case A rows, `_append_peak`
  already does `cur = shape[0]; resize(cur+N); arr[cur:] = …` — no change
  to its resize/assign logic, only to what group it's called against.
- No insertion-position computation or cross-module position-sharing is
  needed — tail-append means every peak-index-family group grows by the
  same row count, from its own current end.

### `pyrs/utilities/NXstress/_fit.py`

- `_PeakParameters.init_group`, `_BackgroundParameters.init_group` — accept
  an existing `NXparameters` group; their existing `_append_peak` methods
  (already the same `resize(cur+N); arr[cur:] = …` shape, e.g.
  `_fit.py:~92-137`) tail-append the new (Case A) rows after the current
  end — same row count and order as `_peaks.py`'s append, so positional
  alignment is preserved automatically.
- `_Diffractogram.init_group` — accept an existing `NXdata` group; tail-append
  `diffractogram`/`diffractogram_errors`/`scan_point` after the current end.

### `pyrs/utilities/NXstress/_sample.py`

- `init_group` — accept an existing `NXsample` group; tail-append each
  per-scan-point log field after the current end.

### `pyrs/utilities/NXstress/_input_data.py`

- `init_group(wss, data=None)` — by this phase (Phase 3, after 04b's
  Phase 2/3 bridge) this already takes `list[HidraWorkspace]`, not a
  single `ws` — an earlier draft of this bullet cited the pre-04b
  signature. When `data` is an existing `NXdata` group (append mode),
  tail-append `detector_counts`/`scan_point` after the current end,
  rather than raising; the new data being appended is still `wss`, per
  04b's own signature (a length-1 list for the common single-new-workspace
  append case).
- `readSubruns(ws, data)` — this one keeps its single-`ws` signature
  unchanged, since it's called once per *reconstructed* workspace against
  an already-sliced `data` view (04b's Q7 value-set-membership split
  happens in the caller, before `readSubruns` is invoked per workspace —
  not inside this method). Unaffected by this spec's write-side scope, but
  confirm it still reads a post-append file correctly (the on-disk layout
  after append is a larger array, locally sorted per key but not globally
  re-sorted — no reader change should be needed, since the reader never
  required global order; see the Relaxed sortedness section above).

### `pyrs/utilities/NXstress/NXstress.py`

- `__init__(path, mode, *, entry_number: int | None = None)` — `entry_number`
  is only meaningful with `mode="a"`; defaults to the last existing entry.
- `write(wss, peakss)` — when opened in append mode, resolve the target
  entry, run the conflict/classification check across all affected groups
  first (Case A/B/duplicate per the policy above — raise `NotImplementedError`
  for Case B or `RuntimeError` and mark the instance invalid for a
  duplicate, before any mutation), then delegate to the append-aware
  `init_group` methods for both families, each simply tail-appending its
  own new rows. When opened in `"w"` mode, behavior is unchanged from 04b.

---

## Tests

`tests/integration/test_nxstress_append.py` (new):
- Write a minimal dataset to a new `.nxs` file; append a second (Case A —
  a new discriminator value/workspace), non-overlapping dataset to the
  same entry (default "last entry" target).
- Read back; assert combined scan-point count equals the sum of both writes;
  assert peak-index entries from both writes are present — file is "locally
  sorted, globally segmented" (each write's own rows sorted among
  themselves; the second write's block simply follows the first's, not
  interleaved into it) — and that `peak_parameters`/`background_parameters`/
  `diffractogram` rows are still positionally aligned with the peaks index
  after the tail-append.
- `entry_number` override: write two entries, append to the first (not the
  last) via explicit `entry_number`; confirm only the targeted entry grew.
- Case B rejection: attempt to append more scan points under a compound key
  (discriminator value + phase/hkl/mask) that's already present in the
  target entry; assert `NotImplementedError` is raised, the on-disk entry
  is unchanged (byte-for-byte, or field-by-field equality check), and the
  same `NXstress` instance can still make a subsequent Case-A append.
- Exact-duplicate conflict case: attempt to append a scan point (or
  peak-index row, under the same discriminator value) that duplicates one
  already in the target entry; assert `RuntimeError` is raised, the
  on-disk entry is unchanged, and a subsequent `write()` call on the same
  `NXstress` instance also raises.
- Case-A precondition 1 (empty `PeakCollection`s): attempt to append a
  new workspace with `peakss=[]` to an entry that already holds another
  workspace; assert `RuntimeError` is raised (its discriminator value
  would be unrecoverable on a later `read()`), the on-disk entry is
  unchanged, and the instance is invalidated (same treatment as an exact
  duplicate — a violated invariant, not an unsupported operation).
- Case-A precondition 2 (no discriminator scheme in target entry): write
  a fresh entry with `N == 1` (no discriminators engaged, per 04b's
  empty-config policy), then attempt to append a second, distinguishable
  workspace to it; assert `RuntimeError` is raised (the target has no
  discriminator columns to attach the new workspace's value to) rather
  than silently merging or corrupting the entry.

`tests/unit/pyrs/utilities/NXstress/test_peaks.py`, `test_fit.py`,
`test_sample.py`, `test_input_data.py` (extend):
- Tail-append correctness for each family, independent of the
  integration-level round-trip: appending to an existing (non-empty) group
  grows each dataset by exactly the new row count, with existing rows
  byte-for-byte unchanged and new rows correctly appended after them.
- Case A/B/duplicate classification, given a small existing on-disk index
  and various incoming batches.

---

## Delivered Feature

> **For downstream NXstress consumers (not yet user-facing):**
> An `NXentry` can now be incrementally grown: `NXstress(path, "a")` extends
> the last-written entry (or a specifically targeted one, via
> `entry_number`) with a new workspace's worth of scan points, rather than
> requiring a fresh file or a fresh entry per write. All five
> position-aligned data groups (raw counts, sample logs, diffractogram,
> peak index, and fit parameters) stay mutually consistent after an
> append, via a tail-append that never re-sorts what's already on disk.
> Growing an already-present workspace/key with more scan points (rather
> than adding a new one) is not yet supported (`NotImplementedError`) —
> that's a genuinely different, insertion-shaped operation, deferred as a
> follow-up. This pass ships the capability as a tested library feature
> only — no PyRS viewer currently exposes an "append" action. Wiring a GUI
> entry point (if a future use case needs one) is a separate,
> not-yet-scheduled follow-up.

---

## Verification

- `pytest tests/integration/test_nxstress_append.py` — all pass.
- `pytest tests/unit/pyrs/utilities/NXstress/` — all pass, no regression
  from 04b.
- Confirm (by inspection, not test) that no GUI file exists that calls
  `NXstress(..., "a")` — this spec is library-only by design.
- Manual check: append a new workspace, then open the resulting file with
  `nexusformat`/`h5dump` and confirm each group is "locally sorted,
  globally segmented" — the first write's rows sorted among themselves,
  the appended batch's rows sorted among themselves and following as a
  second contiguous block, not interleaved into the first — and that no
  group claims or implies a single global sort across the whole file.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A4) — **This spec's central mechanism is confirmed.** The claims that
tail-append "reduces to code that already exists" — `cur = shape[0];
resize(cur+N); arr[cur:] = …` — were, until now, asserted only from reading:
every landed use of that shape runs against **in-memory** `NXfield`s during
`init_group`, while append runs it against a **reopened, file-backed** group.
Those are different objects with different backing stores.
- Referent: `h5py` 3.16.0 / `nexusformat` 1.0.8, probed by
  [`probes/a4_h5py_nexusformat_append.py`](probes/a4_h5py_nexusformat_append.py).
- Verdict: **works, including the two cases most likely to have failed** — a
  genuinely zero-sized dataset, and variable-length UTF-8 string fields
  (`phase_name`, `mask`):

  ```console
    CLAIM   resize(cur+N); arr[cur:] = ... works on a ZERO-SIZED reopened dataset
    RESULT  3 row(s) appended at offset 0; read back {'scan_point': [1, 2, 3],
            'phase_name': ['Fe', 'Fe', 'Fe'], 'center': [1.1, 1.2, 1.3]}

    CLAIM   grows by exactly N, existing rows unchanged, new rows after them
    RESULT  2 row(s) appended at offset 3; scan_point now [1, 2, 3, 4, 5];
            existing rows preserved: True; vlen strings: ['Fe','Fe','Fe','Ni','Ni']
  ```
- Action: none. The design stands, now on evidence.

**F1.2** (A4) — "a rejected (`RuntimeError`) and an unsupported
(`NotImplementedError`) append are both true no-ops — the on-disk entry is left
byte-for-byte unchanged either way".
- Verdict: **confirmed**, for the stated precondition (the classification check
  runs before any resize):

  ```console
    CLAIM   opening 'rw' and raising BEFORE any resize leaves the file unchanged
    RESULT  sha256[:16] before=cf489122b1f81fd2 after=cf489122b1f81fd2
            identical=True (simulated conflict detected before any resize)
  ```
- Action: none — but note what this does **and does not** establish. It confirms
  that merely opening in `"rw"` and raising is byte-neutral. It does **not**
  establish atomicity across a *partial* append, which is why this spec's
  "check all affected groups **before any resize/append call is made**"
  ordering requirement is load-bearing and must survive implementation.

**F1.3** (A4) — "Appending a genuinely new, distinguishable workspace to an entry
… isn't possible without adding a new on-disk column, which this spec's
tail-append design does not do".
- Verdict: **the sentence is literally true and its framing misleads.** Adding a
  new column to an existing group *succeeds*:

  ```console
    CLAIM   adding a NEW on-disk column to an existing group -- 04c says its design
            does not do this; is it even possible?
    RESULT  SUCCEEDED -- group now has ['center', 'direction', 'fit', 'h',
            'phase_name', 'scan_point']
  ```

  So the restriction is a **scope decision, not a technical limit**. As written
  ("isn't possible without …, which this design does not do") a reader
  reasonably concludes the format forbids it and stops looking.
- Action (for the implementing PR): reword to "…would require adding a new
  on-disk column. That is mechanically possible — see
  `probes/a4_h5py_nexusformat_append.py` — but it is schema restructuring, which
  this spec deliberately excludes; a later spec could lift the restriction
  cheaply." This matters because it changes what a future spec can assume.

**F1.4** (A3) — the bare line reference in the peak-index bullet, preceding
"reusing the existing `_append_peak`".
- Verdict: that bare line reference had no resolvable antecedent — no filename
  appears anywhere in that paragraph, and the file (`_peaks.py`) was implied only by the
  enclosing section. The parallel prose at `04b:40-41` writes it explicitly.
  Reported rather than guessed, per the toolkit's rule.
- Action: qualified in place to `` `_peaks.py:180-181` ``.

**F1.5** (A3) — `sample_logs.py:164-166` corrected in place to `:167-168`; see
`04b-multi-workspace-nxstress.md`'s Follow-up 1 F1.1 for the evidence.

**Checked and accurate — no action.** `_peaks.py:180-181`,
`_input_data.py:44-46,63-72`, `NXstress.py:278`, `_peaks.py:331-404`,
`_peaks.py:446-559` all land exactly on their claimed targets. The premise at
`:14-21` — that `write` already accumulates entries and the guard "only fires on
an accidental name collision" — is confirmed against the landed library
([`probes/a5_nxstress_roundtrip.py`](probes/a5_nxstress_roundtrip.py)):

```console
  CLAIM   what mode 'a' actually does today: does it EXTEND the existing entry,
          or add a new one?
  RESULT  no exception raised; NXentry count 2 -> 3 ['entry', 'entry_2', 'entry_3'].
          So a write never extends an existing entry -- it appends a new one.
```

**A5 gap, recorded rather than assumed.** This spec's conflict classification,
Case-A/Case-B dispatch and `entry_number` targeting **cannot be probed** —
none of it exists yet. A5-**uncovered**, not A5-verified. The *mechanism* they
rest on is covered by F1.1–F1.3; the dispatch logic is not.

---

## Follow-up 2 — 2026-10-01 (implementation pass)

Ten findings. Two of them change the design, one of those decisively: the
mechanism this spec is built on was available for **half** the groups it covers
and for the other half was not available at all.

**F2.1** (A5) — **"Tail-append reduces to code that already exists" is true of
the peak-index family and false of the scan-point family.** Follow-up 1 F1.1
recorded this spec's central mechanism as "confirmed", on
[`probes/a4_h5py_nexusformat_append.py`](probes/a4_h5py_nexusformat_append.py).
That probe is sound and its verdict is correctly stated — but it built **its
own** fixtures, and gave every one of them `maxshape=(None,)` + `chunks`. It
therefore established that `resize(cur+N); arr[cur:] = …` works *on a resizable
dataset*, which is a different claim from "the datasets the NXstress writer
emits are resizable". An HDF5 dataset created without `maxshape` is contiguous
and cannot be extended by any mechanism.
- Referent: the writer's real output, built and traversed by
  [`probes/a5_scan_point_family_resizable.py`](probes/a5_scan_point_family_resizable.py).
  A grep over the source would have shown the same thing, but only if one
  already knew to look for an absent kwarg; building the artifact and walking it
  reports the state rather than the intent.
- Verdict against the writer as 04b left it: **86 datasets, 32 extendable, 54
  fixed-size.** The peak-index family was 22 of 24 extendable (the two
  exceptions, `peak_parameters/title` and `peaks/center_type`, are per-entry
  scalars and correct as they are). The scan-point family was almost entirely
  fixed:

  ```console
    CLAIM   claims 2-4: is every named scan-point-family dataset tail-appendable?
    RESULT  /entry/start_time: shape=(3,) maxshape=(3,) chunks=None -> FIXED
            /entry/end_time: shape=(3,) maxshape=(3,) chunks=None -> FIXED
            /entry/input_data/detector_counts: maxshape=(None, None) -> extendable
            /entry/input_data/scan_point: shape=(3,) maxshape=(3,) chunks=None -> FIXED
            /entry/instrument/monochromator/wavelength: maxshape=(3,) -> FIXED
            /entry/SAMPLE_DESCRIPTION/scan_point: maxshape=(None,) -> extendable
            /entry/SAMPLE_DESCRIPTION/vx: maxshape=(3,) chunks=None -> FIXED
            /entry/SAMPLE_DESCRIPTION/vy: maxshape=(3,) chunks=None -> FIXED
            /entry/SAMPLE_DESCRIPTION/vz: maxshape=(3,) chunks=None -> FIXED

    CLAIM   claim 4: _Diffractogram's datasets
    RESULT  /entry/FIT/DIFFRACTOGRAM/XAXIS: maxshape=(3, 20) -> FIXED
            /entry/FIT/DIFFRACTOGRAM/diffractogram: maxshape=(3, 20) -> FIXED
            /entry/FIT/DIFFRACTOGRAM/diffractogram_errors: maxshape=(3, 20) -> FIXED
            /entry/FIT/DIFFRACTOGRAM/scan_point: maxshape=(3,) -> FIXED
            /entry/FIT/DIFFRACTOGRAM/fit, fit_errors: maxshape=(None, None) -> extendable

    CLAIM   claim 3: _sample.py's retained per-scan-point logs
    RESULT  4 log field(s); all FIXED
  ```
- What this falsifies: the Architecture section's "reduces to code that already
  exists … No new insertion-position machinery is needed", the Scope bullet
  claiming a tail-append path for `_input_data.py` / `_sample.py` /
  `_fit.py::_Diffractogram`, and the NXstress Changes bullets for
  `_Diffractogram.init_group` and `_sample.py`'s `init_group`. All of them read
  as though those groups need only a new entry point. They needed a writer
  change first.
- Correction, and **why it is not read-merge-rewrite**: a new helper
  [`growable(rank)`](../../pyrs/utilities/NXstress/_definitions.py#L70) returns
  the `maxshape`/`chunks` kwargs, and is applied at every per-scan-point
  `NXfield(...)` — `NXstress.py:1093-1094`, `_instrument.py:227`,
  `_input_data.py:74`, `_sample.py:234`/`:247`/`:266`/`:287`,
  `_fit.py:510`/`:515`/`:519`/`:523`. A matching
  [`tail_append(field, values)`](../../pyrs/utilities/NXstress/_definitions.py#L109)
  carries the growth rule, and
  [`appendable(field)`](../../pyrs/utilities/NXstress/_definitions.py#L90) the
  pre-flight predicate. The alternative — read the fixed-size array back,
  concatenate, delete, rewrite — was rejected: it round-trips existing data
  through memory, which is exactly what the Architecture section's decision (2)
  excluded, and it interleaves deletes with writes, which would cost the
  byte-neutrality the Conflict-policy section depends on.
- **Consequence, recorded because it is a compatibility break:** a `.nxs` file
  written by a PyRS predating this change has contiguous per-scan-point datasets
  and **cannot be appended to**. `tail_append` refuses it by name rather than
  failing part-way. Nothing depends on this in practice — no viewer writes
  NXstress through an append path, and the GUI hookup is phases 4-6 — but a file
  already on disk is affected. Decisions Log row 32.
- Note the rank: `stress_field` is `(n_scan, 3)`, not 1-D, so `_sample.py` takes
  the rank from the data rather than assuming it. The existing
  `test_Sample_stress_field_present` caught the assumption immediately.

**F2.2** (A4) — **A fixed-width `|S` string column silently truncates on
append.** Found while checking F2.1's fix was safe for the string fields, which
`a4_string_log_dtypes.py` had already shown to be the delicate ones.
`_Sample._writable` converted NumPy `<U` to the variable-length UTF-8 dtype and
**passed `|S` bytes arrays through untouched** — correct, since they are
writable as they stand. But a fixed-width HDF5 string column is sized by the
longest value present *when it is created*, and an append is by definition later
than that.
- Referent: [`probes/a4_growable_string_fields.py`](probes/a4_growable_string_fields.py),
  claim 5.
- Verdict: **the value is truncated and nothing raises.**

  ```console
    CLAIM   claim 5 -- appending a LONGER value to a fixed-width |S column
    RESULT  column dtype is |S8 (sized by the longest value at WRITE time)
            appended b'a_considerably_longer_filename.h5' (33 bytes)
            read back: [b'short.h5', b'a_consid']
            round-tripped intact: False  <-- SILENTLY TRUNCATED, no exception raised
  ```

  `Filename`, `start_time` and `end_time` are all bytes logs, so this was reachable
  by appending a workspace whose project file has a longer name than the first's.
- Action taken: `_writable` now coerces `|S` as well as `<U` to the
  variable-length dtype ([`_sample.py:637`](../../pyrs/utilities/NXstress/_sample.py#L637)).
  The read-back property is unchanged — both dtypes return `bytes` — so no reader
  is affected. The same probe confirms all three forms the writer produces
  (`list[str]`, `|S`, vlen UTF-8) write, reopen and tail-append correctly with
  `maxshape`/`chunks`; bare `<U` still fails at write, which is what `_writable`
  exists to prevent.

**F2.3** (A1) — **The scan-point family has two members this spec's table omits**,
and they are the two that do not live in a subgroup: `entry/start_time` and
`entry/end_time`, written per scan point by `NXstress._init`. The "Scope: all
position-aligned groups" table lists only the five subgroups. An append that
grew those five would have left both arrays short, with no error.
- Action: `_init`'s time computation is extracted to `_entryTimes`, shared with
  [`_appendEntryTimes`](../../pyrs/utilities/NXstress/NXstress.py#L483), so the
  two paths cannot diverge. Pinned by
  `test_append.py::TestRoundTrip::test_every_position_aligned_group_grows_by_the_same_count`,
  which asserts a single length across all fourteen.

**F2.4** (A1) — **`write()` cannot mean both "add an entry" and "grow an entry",
and this spec asks it to mean both.** The Overview states, correctly and with
probe evidence, that `write` already accumulates entries and that mode `"a"`
adds a new one. The NXstress Changes section then says to "dispatch `write()` to
the tail-append path when opened with mode `"a"`" — which would leave "add
another `NXentry` to an existing file", documented in `NXstress.py`'s usage
comment (at lines 152-156 before this pass), with no spelling at all.
- Resolution (stakeholder decision, 2026-10-01): **dispatch on whether the
  resolved entry exists, not on the mode.** `entry_number` omitted, or naming an
  existing entry, appends; naming no existing entry writes a fresh one. Mode
  governs file access, `entry_number` governs targeting. Decisions Log row 33;
  implemented in
  [`_resolveTarget`](../../pyrs/utilities/NXstress/NXstress.py#L237).
- **Overwriting** an existing entry's contents is a third operation, neither
  append nor create, and stays unimplemented. Note what that means under this
  dispatch: no input selects it, so the former collision guard is **unreachable
  by construction** rather than, as this finding first said, "the backstop". It
  is kept at `NXstress.py:278` regardless — the cost of that reasoning being
  wrong is an entry silently replaced instead of grown. Caught by
  `check_citations.py` on the re-run, which flagged the stale pointer that led
  back to it.
- A gap is **rejected**: `entry_number` beyond `max + 1` raises `ValueError`
  naming the next free number. A number past the end is far likelier a typo than
  an intent, and its silent outcome — a stray entry instead of the append that
  was meant — is not one a caller would notice.
- **`entry_number` had to become a `write()` argument as well as a constructor
  one**, which neither the spec nor the decision anticipated. Under the rule
  above a bare second `write()` in one session appends to the entry the first
  one created, so a constructor-only kwarg would have made "write two entries in
  one session" unexpressible — a capability `test_NXentry_multiple` already
  covered and the old usage comment advertised. That test now passes
  `entry_number=2` explicitly.

**F2.5** (A1/A5) — **The Conflict-policy section enumerates three outcomes and
needs seven.** Case A / Case B / duplicate classify the *peak index*. They say
nothing about the other ways an incoming workspace can disagree with the entry
it is joining, each of which produces a readable and wrong file rather than an
error. Added to the pre-flight pass, each raising `RuntimeError`:
  1. **Scan points disjoint from the whole entry.** A genuinely new compound key
     (Case A) may still reuse a scan-point *value*, and the reader attributes
     rows by value — `_workspaceSelections` raises on read. Caught at write now.
  2. **Reduced-diffraction mask set matches.** A mask the entry has no
     `DIFFRACTOGRAM` for would need a new group with no rows for the scan points
     already on disk.
  3. **Retained sample-log set matches** (`_Sample.validateAppend`,
     [`_sample.py:511`](../../pyrs/utilities/NXstress/_sample.py#L511)), compared
     by encoded column name rather than raw PV key, since that is what the file
     indexes by.
  4. **Optional sample fields match** — `temperature`, `stress_field`: an entry
     has one for all its scan points or for none.
  5. **Raw counts loaded on both sides or neither** (`_InputData.validateAppend`,
     [`_input_data.py:83`](../../pyrs/utilities/NXstress/_input_data.py#L83)).
  6. **Instrument geometry, detector shift and calibration agree**
     (`_Instrument.validateAppend`,
     [`_instrument.py:378`](../../pyrs/utilities/NXstress/_instrument.py#L378)) —
     one `NXentry` describes one instrument configuration.
  7. **Peak profile and background function match** the entry's `title` scalars.
     `_append_peak` already raises on a mismatch, but only after earlier groups
     have grown.
- **Two of these were initially written at their point of use and were wrong
  there**, which is the finding worth keeping. Both raised the right exception
  and both left a changed file, because `_appendEntryTimes` had already run. The
  spec says the check must run "before any resize/append call is made"; that
  sentence is load-bearing exactly as Follow-up 1 F1.2 warned, and only a
  byte-level comparison detects its violation. They are now in
  [`_classifyAppend`](../../pyrs/utilities/NXstress/NXstress.py#L325), with the
  point-of-use raise retained as a backstop.

**F2.6** (A5) — all of the above is now probeable, and probed.
[`probes/a5_append_preconditions.py`](probes/a5_append_preconditions.py) builds a
real entry through `NXstress.write` and drives every rejection path, comparing
the file's sha256 before and after each — the measurement
`a4_h5py_nexusformat_append.py` explicitly declined to make ("It does **not**
establish atomicity across a *partial* append"). All confirmed:

```console
  CLAIM   claim 2: Case B -- more scan points under a key the entry already holds
  RESULT  NotImplementedError: ... which this entry already holds, is not supported.
          file byte-for-byte unchanged: True

  CLAIM   claim 3: exact duplicate -- same key, overlapping scan point
  RESULT  RuntimeError: ... scan point(s) [2, 3] are already present in this entry ...
          file byte-for-byte unchanged: True

  CLAIM   claim 4a: Case A precondition 1 -- the new workspace contributes no PeakCollection
  RESULT  RuntimeError: ... input workspace(s) [0] contribute no `PeakCollection`.
          file byte-for-byte unchanged: True

  CLAIM   claim 4b: Case A precondition 2 -- the target entry has no discriminator column
  RESULT  RuntimeError: ... the target entry carries no discriminator columns ...
          file byte-for-byte unchanged: True

  CLAIM   claims 2-3: Case B leaves the instance usable; a duplicate does not
  RESULT  after NotImplementedError: the later Case-A append SUCCEEDED -- instance stayed usable
          after RuntimeError:       the later append raised RuntimeError: ... no longer usable.
```

This closes the A5 gap Follow-up 1 recorded ("This spec's conflict
classification, Case-A/Case-B dispatch and `entry_number` targeting **cannot be
probed** — none of it exists yet"). The coverage matrix moves 04c's A5 from `~`
to `✓✓`.

**F2.7** (A3) — **`if dgram_name in fit.NXdata` can never be true**, so the
pre-existing guard against a duplicate `DIFFRACTOGRAM` group in `_Fit.init_group`
is dead code. `NXgroup.NXdata` returns a **list of `NXdata` objects**, not a
mapping of names, so a `str` is never a member of it:

```console
  children:   ['DESCRIPTION', 'DIFFRACTOGRAM', 'background_parameters', ...]
  NXdata attr: [NXdata('DIFFRACTOGRAM')]
```

Harmless as it stood — `mask_keys` is a `set`, so the duplicate it guards against
cannot arise on a fresh write — but the append path's equivalent check needed to
work, and inherited the bug before a smoke test caught it. Both now use
`in fit`. Found by implementation, claimed by no document.

**F2.8** (A1/A2) — **`_instrument.py` and `_definitions.py` are not in this
spec's ownership row.** `README.md` §5 lists 04c as touching
`pyrs/utilities/NXstress/{NXstress,_input_data,_sample,_fit,_peaks}.py`, and the
"Scope: all position-aligned groups" table places `_instrument.py` under
"Name-keyed, no insertion needed" on the strength of `_Masks`. That is right
about `_Masks` and wrong about the module: `monochromator/wavelength` is written
per scan point and is sliced by `rows` on read, so it is a scan-point-family
member and must grow. `_definitions.py` gains the three helpers. Both added to
§5; `check_ownership.py` would otherwise report them.

**F2.9** (A1) — Follow-up 1 F1.3's reword, which Decisions Log row 22 adopted,
**had not been applied**: the Conflict-policy section still read "isn't possible
without adding a new on-disk column, which this spec's tail-append design does
not do". F1.3 assigned the reword to the implementing PR, and it is applied now.
Noting the tension this sits in: the convention is that a subspec body is left as
the record of what was believed and Follow-ups carry the corrections, but F1.3's
`Action` is explicit and specific. The body is edited, the Follow-up records that
it was, and the original wording is quoted in F1.3 where it remains readable.

**F2.10** (A1) — **The test tier is unit, not integration.** This spec's Tests
section, `README.md` §5's inventory and `probes/README.md`'s Disposition column
all specify `tests/integration/test_nxstress_append.py`. All three predate 04b,
whose structurally identical round trip landed as
`tests/unit/pyrs/utilities/NXstress/test_multi_workspace.py` — unmarked, built on
`minimal_HidraWorkspace`, writing to `tmp_path`, touching no real data. `CLAUDE.md`
agrees: "a synthetic in-memory round trip through `tmp_path` that stays inside one
component's own public API … is unit". Three documents against one precedent and
the tier rule; the precedent and the rule win, and the three documents are
corrected. Shipped as
[`tests/unit/pyrs/utilities/NXstress/test_append.py`](../../tests/unit/pyrs/utilities/NXstress/test_append.py).
Decisions Log row 34.

### Invariants written by this PR

`review/findings.md` §5 item 2 — "tail-append grows each dataset by exactly N,
leaves existing rows unchanged, and a pre-resize abort is byte-neutral" — is
written, at unit rather than the integration tier recorded there (F2.10):
`test_append.py::TestTailAppendHelper` for the growth rule and the
fixed-size refusal, and the `digest(...)` assertion on every rejection test for
byte-neutrality. `probes/README.md` assigns the same promotion to
`a4_h5py_nexusformat_append.py`; it is satisfied by the same tests.

**One invariant not on anyone's list, and the most valuable one here.**
`test_append.py::TestWriterEmitsResizableDatasets` writes an entry, reopens it,
and **sweeps** every dataset whose first axis is the scan-point axis, asserting
each is extendable. It iterates rather than naming fields — the
[`test_definitions.py`](../../tests/unit/pyrs/utilities/NXstress/test_definitions.py)
idiom — so a per-scan-point field added later without `growable` fails this test
rather than failing a user's append months afterwards. F2.1 is precisely the
defect it would have caught, and nothing in the series was looking for it. Its
companion `test_the_sweep_notices_a_field_that_loses_growable` adds a fixed-size
field *locally*, not through the writer, so the detector is checked against
something other than the code under test.

### Verification, as run

- `pixi run test-unit` — **412 passed**, 150 deselected (was 360 after 04b). *(403 when Follow-up 2 was written; the nine added in the Follow-up 3 round.)*
- `pixi run test-integration` — **104 passed**, 28 skipped, 2 xfailed.
- `pixi run test-gui` — **16 passed**.
- "Confirm by inspection that no GUI file calls `NXstress(..., "a")`" —
  confirmed: every call in `pyrs/interface/` is `"w"` or `"r"`
  (`combine_runs_model.py:40`, `peak_fitting_model.py:234`,
  `texture_fitting_model.py:197`, and `nxstress_load.py:55`, which is the
  one `"r"`).
- "Manual check … confirm each group is locally sorted, globally segmented" —
  run as the last claim of `a5_append_preconditions.py` and pinned by
  `test_append.py::TestRoundTrip::test_file_is_locally_sorted_globally_segmented`,
  which appends a discriminator value sorting *before* the one on disk, so a
  re-sorting append would be detected rather than merely assumed absent.

**F2.11** (A3) — **The `# TODO` three of this spec's claims point at no longer
exists**, because this PR is what resolved it. `_Peaks.init_group` carried
"these code sections are implemented in a form that allows new scan-point data
to be appended / However, at present, appending data is not yet supported",
and the Architecture, Scope and NXstress Changes sections each cite it as
evidence that the mechanism is already half-built. It was, and the comment is
now replaced by the `data=None` branch it predicted. The three body citations
are repointed at `init_group` itself (`_peaks.py:305-329`) and reworded to the
past tense, since a pointer to a deleted comment resolves to whatever happens to
occupy those lines. Mentioned because this is the ordinary end-state of a `TODO`
cited as evidence: the citation outlives its referent by exactly one PR.

Other pointers corrected in place this pass, all pure drift with the claims
around them unchanged: `peakCollectionRanges`, `_append_peak`, the
`detector_counts` write path, and the overwrite guard (which this PR moved
rather than removed — see F2.4). The values they were given in *this* pass have
since drifted again and were re-resolved by symbol in Follow-up 6; the
superseded numbers this paragraph used to spell out are in git history, which is
where Decisions Log item 41 says they belong.

---

## Follow-up 3 — 2026-10-01 (sub-agent review round)

Follow-up 2 was written against code that passed 403 tests, three clean tiers
and the whole toolkit. Two review sub-agents — one on design, one on tests —
then independently found the **same defect**, and one of them demonstrated it
corrupting a file through the public API. That is the finding worth keeping:
the property Follow-up 2 reported as closed was closed for every path the
tests covered and open on the one they did not.

**F3.1** (A5) — **"A rejected append leaves the entry byte-for-byte unchanged"
was false.** F2.5 moved six checks into the pre-flight pass and recorded that
two of them had initially sat at their point of use. It missed that
`tail_append` itself makes two refusals — a non-resizable dataset, and a
trailing-axis disagreement — and that **neither was pre-flighted**. They
therefore fired during mutation, after earlier groups had grown.
- The giveaway, in hindsight: `_definitions.appendable` was written precisely
  so the pre-flight could ask "can this grow?", and **it had no production
  caller at all**. A predicate nothing asks is a design that did not land.
- Reproduced through `NXstress.write`, with the entry written at 20 two-theta
  bins and the appended batch reduced onto 25 — not a contrived input but the
  ordinary difference between two reduction passes:

  ```console
    before: start_time (3,)  SAMPLE/scan_point [1, 2, 3]  XAXIS (3, 20)
    RAISED: RuntimeError NXstress: cannot append to '/entry/FIT/DIFFRACTOGRAM/XAXIS':
            the incoming rows have trailing shape (25,), the existing data (20,)
    file unchanged? False
    after:  start_time (6,)  SAMPLE/scan_point [1, 2, 3, 4, 5, 6]  XAXIS (3, 20)
    read back OK: [[1, 2, 3, 4, 5, 6]]
  ```

  27 datasets at six rows, three at three, **and it reads back without error**
  as one workspace of six scan points, three of which have no diffraction data
  and no peaks. Silent corruption, which is worse than the crash it replaced.
- Why the tests did not catch it, which is the structural lesson: the refusal
  *was* tested, twice — at the `tail_append` level and at the
  `_Diffractogram.init_group` level. Neither went through `NXstress.write`, so
  neither could observe the 27 datasets that had already grown. **Every refusal
  reachable during an append needs one end-to-end test with a `digest()`
  assertion**, not a unit test of the function that raises. `pytest.raises`
  passed throughout; only the digest fails.
- Corrections, all in the pre-flight:
  - [`_validateAppendableShapes`](../../pyrs/utilities/NXstress/NXstress.py#L644)
    sweeps every per-scan-point dataset in the target entry and refuses the
    append as a whole if any cannot grow — the production counterpart of
    `TestWriterEmitsResizableDatasets`, reimplemented rather than shared with it
    so the test can fail independently of the code it guards.
  - [`_Fit.validateAppend`](../../pyrs/utilities/NXstress/_fit.py#L706) and
    [`_Diffractogram.validateAppend`](../../pyrs/utilities/NXstress/_fit.py#L519)
    check the two-theta bin count per mask.
  - [`_InputData.validateAppend`](../../pyrs/utilities/NXstress/_input_data.py#L83)
    gains the detector pixel count; it checked only *whether* counts existed.
  - The mutation phase is wrapped so that **any** escape sets `self._invalid`
    ([`NXstress.py:395`](../../pyrs/utilities/NXstress/NXstress.py#L395)). The
    policy was exactly inverted before: a pre-flight rejection, where the file
    is untouched, invalidated the instance; a mutation failure, the only case
    where the file is damaged, left the caller free to append onto the wreckage.
- Re-measured after the fix, through the same input:

  ```console
    RAISED: RuntimeError NXstress._fit: cannot append -- the incoming reduced
            diffraction for mask '_DEFAULT_' has 25 two-theta bin(s), the target entry 20.
    file unchanged? True
    after:  start_time (3,)  SAMPLE/scan_point [1, 2, 3]  XAXIS (3, 20)
  ```

  `probes/a5_append_preconditions.py` gains both cases (claim 6), and
  `test_append.py::TestPreconditions` gains seven end-to-end refusal tests, each
  with its `digest` assertion.

**F3.2** (A1) — **`_Sample._append_group` validated after mutating**, and its own
comment said so: "a mismatch found mid-append would already have grown
something" sat *below* a `tail_append` of `scan_point` and the three coordinate
axes. Same class as F2.5's two, missed in the same pass. The call is now the
first statement ([`_sample.py:594`](../../pyrs/utilities/NXstress/_sample.py#L594)),
and the coordinate arrays are built and shape-checked before anything is
resized rather than appended as they are computed.

**F3.3** (A3) — **An appended workspace's detector masks were silently
discarded.** `_Instrument.init_group`'s append branch grows `wavelength` and
returns; the mask arrays are entry-wide and were written fixed-size, so they
*cannot* grow. `_classifyAppend`'s mask check compares `ws._diff_data_set`
keys, which is a different set from `_Masks.mask_keys(ws)` — a workspace could
pass it carrying masks the entry has no record of, and `masksFromNexus` would
hand it the first write's masks on read. Now refused in
[`_Instrument.validateAppend`](../../pyrs/utilities/NXstress/_instrument.py#L377),
reusing the existing `_validate_masks_agree`.

**F3.4** (A4) — **`data: NXdata = None` is a trap under `@validate_call_`.**
Pydantic does not validate defaults, so omitting the argument works, but
passing it explicitly does not: `_InputData.init_group(wss)` succeeds where
`_InputData.init_group(wss, data=None)` raises `ValidationError: Input should be
an instance of NXdata`. The first caller to write `data=maybe_group` would hit
it. All seven `data` parameters are now `X | None = None`, and
`_Fit.init_group`'s had no annotation at all. **mypy cannot catch this**:
`nexusformat` is untyped and `ignore_missing_imports` makes `NXdata` resolve to
`Any`.

**F3.5** (A1) — smaller corrections from the same round:
- `if dgram_name in fit.NXdata` (F2.7) had a sibling: `_resolveTarget` re-derived
  entry-name parsing with `rsplit` instead of using `suffix_from_group_name`,
  the declared inverse of `group_naming_scheme`, and its `range(1, len(...)+1)`
  term reduced to `[1]` after its own existence filter — a comment reading
  "derived from the names present rather than from a count", immediately above
  code using a count.
- Appending to an entry written with `write([ws], [[]])` was refused for
  "disagreeing with the target entry's `_undefined_`" — the writer's own
  sentinel. Now refused for the real reason: such an entry records no fit model
  and no discriminator values, so nothing appended to it could be told apart.
- `_classifyAppend` reached 158 lines doing seven jobs, which is the wrong shape
  for the one method a reader must follow in full to believe the no-op property.
  Split into six named checks called in sequence.
- `_Sample.OPTIONAL_SCAN_POINT_FIELDS` is a parallel definition: the fresh-write
  path still spells each field out, because each carries its own attribute rule.
  Cross-referenced both ways rather than unified, with the failure mode named —
  a field added to one and not the other is written and silently never grown.
- `TestWriterEmitsResizableDatasets` kept a 17-name `ENTRY_WIDE` denylist of
  which **two names did not exist in any entry this writer produces** and 14
  were already excluded by the scalar check. Removed: the scalar filter does the
  work. Its `n_scan` moved from 3 to 7, because the sweep matches by length and
  the writer emits fixed-size arrays of length 1, 2 and 16 — a collision would
  have been a false failure, and nothing recorded the constraint.
- The lockstep test enumerated 15 dataset paths by hand, in the class whose
  docstring argues against exactly that. It now *discovers* the family from the
  pre-append file, so a field added later is covered without editing it.
- `createPeakCollection` gained a `mask` parameter, replacing a `collection._mask`
  poke in the new mask test.

### Invariants added in this round

Seven end-to-end refusal tests in `TestPreconditions`, each asserting the file's
sha256 is unchanged: two-theta width, detector pixel count, detector shift,
background function, detector mask set, a legacy fixed-size entry, and a
mutation-phase failure invalidating the instance. The last is reached by
monkeypatching `_sample.tail_append`, since every *known* cause is now
pre-flighted — the guarantee under test is the invalidation, not the trigger.

Plus `test_the_sweep_reaches_input_data` (the raw-counts datasets were outside
the resizability guarantee, because the shared fixture carries no counts and a
length-matched sweep drops zero-length arrays), and
`test_a_workspace_lacking_a_mask_is_nan_filled_on_append`.

### What this round says about the process

`process.md` §5.6 budgets a document-correction pass per PR. This was a
*code*-correction pass, found after the PR had passed its own `## Verification`,
all three tiers and the full toolkit — none of which could see it, because the
defect was in a path no test took and no probe built. The audit's own
anti-pattern list already names the mechanism twice: "a probe that reimplements
its counterparty is testing the reimplementation" (F2.1, the `maxshape` fixtures)
and "reading source tells you what is written, not what it means". **F3.1 is the
third instance and the first that reached shipped code.** The practice that
caught it was neither auditing nor testing but an adversarial read by someone
who had not written it, and the cheap generalisation is the rule in F3.1: a
refusal is only verified where it is reachable from the public API, with the
file compared byte-for-byte.

---

## Follow-up 4 — 2026-10-06 (PR review)

Changes made during the human review of `eb5457b1..cda22352`, per
[`plans/PR-review-process/review-process.md`](../PR-review-process/review-process.md).
The review conversation is in [`plans/PR_review/04c-comments.md`](../PR_review/04c-comments.md);
only the changes are recorded here.

**F4.1** — `tail_append`'s three error messages named the dataset `'unknown'` whenever
the field was not attached to a tree. `NXfield.nxpath` and `.nxname` both return that
literal string, so a message whose entire purpose is to name the offending dataset
instead reported a non-existent name. A new `_field_label(field)` returns
`dataset '<nxpath>'` or ``an unattached (3,) `NXfield` ``, and all three messages use
it. Unreachable from the append path, which is always file-backed; reachable from a
unit test or a future in-memory caller.

**F4.2** — `tail_append(field, <scalar>)` raised `IndexError: tuple index out of
range`. A 0-d array has no trailing axes, so it passes the trailing-axis check
vacuously, and `values.shape[0]` then fails naming neither the field nor the problem.
Guarded with an explicit `RuntimeError`, placed *before* the trailing-axis check with
a comment recording why that order matters.

**F4.3** — **`growable`'s unlimited trailing axes are a chunking artifact, not a
contract, and this is now written down.** Every dataset it creates goes to disk with
`maxshape` unlimited on *every* axis, so `diffractogram` tells a NeXus reader its
two-theta axis can grow — which `tail_append` and `_Fit.validateAppend` both refuse.
Pinning the trailing axes was considered and measured:

```console
  maxshape=(None, None)  chunks=(1, 100)  data (3, 20):  OK
  maxshape=(None, 20)    chunks=(1, 100)  data (3, 20):  ValueError:
        Chunk shape must not be greater than data shape in any dimension
  maxshape=(None, 20)    chunks=(1, 20)   data (3, 20):  OK
```

Pinning **works and HDF5 enforces it** — an axis-1 resize is then refused by the
library rather than only by our code. But h5py accepts a chunk wider than the data
only while that axis is unlimited, and `CHUNK_SHAPE` asks for 100 on the fast axis
unconditionally, so pinning requires a shape-aware `CHUNK_SHAPE`. It would also break
`DIFFRACTOGRAM/fit` and `fit_errors`, the `(0, 0)` placeholders **spec 09** resizes on
*both* axes.

**Decided: keep `(None, None)`** rather than change the on-disk layout a second time
in one PR. The `growable` docstring now carries the asymmetry, the h5py constraint,
the measurement above, and the spec-09 reason, and names the three places that enforce
the real rule. `tail_append`'s trailing-axis message gained the *why*: "Only the first
axis grows on an append; the trailing axes are fixed when the entry is written."

**A note for spec 09.** When it fills in `fit`/`fit_errors` it resizes them on both
axes. That is the one legitimate both-axis resize in the package, and it is why
pinning was rejected here — if 09 instead writes those fields at their final size, the
objection disappears and pinning becomes cheap.

---

## Follow-up 5 — 2026-10-09 (PR review, batch 6: the append pre-flight asked the wrong question)

Batch 6 of the PR review covered `NXstress.py`
([plans/PR_review/04c-comments.md](../PR_review/04c-comments.md)). Two defects, and
the second one invalidates a claim made in Follow-up 2.

### F5.1 — the diffractogram-key check still used the mask namespace

`_rejectMaskMismatch` was the last call site in the package passing
`_diff_data_set` keys to `nxstress_mask_names`, which injects `DEFAULT_TAG`
because every reduction uses the default *detector mask*. The on-disk side it
compares against is written by `_Fit.init_group`, which after 04's Follow-up 8
does not inject it. An eta-only workspace therefore presented one key more than
the entry could possibly hold, and **every texture append was refused** — through
`self._reject`, so the instance was invalidated too.

Renamed `_rejectDiffractogramKeyMismatch` and switched to
`nxstress_diffractogram_keys`. The rename is part of the fix: the defect was a
name under which the wrong function looked right.

Covered by `test_append.py::TestTextureEntryAppends`. Note what let this through:
all ~40 existing tests in that module reduce under `{None: ...}`, the one shape
where the two namespaces agree.

### F5.2 — `_perScanPointDatasets` rejected legal appends; **Follow-up 2's framing of it was wrong**

Follow-up 2 introduced the pre-flight sweep and its docstring argued that
identifying scan-point datasets by array length is safe because over-matching
"can only demand that something be resizable which need not be". That is the
defect, not a mitigation: the swept set feeds a resizability **requirement**, so a
false match *rejects a legal append*.

Raised by the reviewer from the specification alone, then measured:

```console
n_scan = 3;  43 dataset(s) swept in
  FIX  entry/instrument/masks/names   shape=(3,) maxshape=(3,)
APPEND: RuntimeError: ... 1 dataset(s) in this entry were written at a fixed size
        and cannot be extended: ['entry/instrument/masks/names'].
```

Three detector masks and three scan points. `masks/names` is entry-wide, is never
grown, and `_Instrument.validateAppend` *requires* it to stay constant — so the
sweep demanded growability of the one thing the design forbids growing, and the
message blamed a PyRS predating `growable`, which is both false and unactionable.

It also **under**-matched. The peak index is one row per (compound key, scan
point), so it equals `n_scan` only while each workspace fits exactly one peak. Fit
a second phase and `peaks/*`, `peak_parameters/*` and `background_parameters/*`
left the pre-flight entirely — restoring precisely the mid-mutation refusal
Follow-up 3 F3.1 had hoisted out of `tail_append`.

**The hazard was already written down, in the wrong place.**
`TestWriterEmitsResizableDatasets` picks `N_SCAN = 7` under a comment explaining
that a smaller value risks colliding with an entry-wide array length. The
collision was understood well enough to be designed around in a test fixture and
was never reported as a defect in the code it was testing.

**Resolution.** Each grouping module gains `appendableDatasets(group)`, declaring
the datasets *it* writes row-aligned, and `NXstress._appendableDatasets` composes
them. This is exact rather than heuristic **because the declarer owns the group's
layout**: inside a group NXstress writes, a field is either row-aligned — and
therefore `growable` — or an entry-wide scalar, so `_definitions.row_aligned_fields`
can enumerate a group without a maintained name list. `_Instrument` is the
exception and enumerates explicitly, because its non-scalar fields are mostly
entry-wide arrays — which is how this started.

Declared: **49** datasets, against 43 swept of which one was false and the
peak-index family was present only by coincidence.

**What the sweep gave for free, and where it went.** A field added to the writer
later was covered without editing the check. That is now stated directly instead
of emerging from a heuristic:
`TestAppendableDatasetsAreDeclaredByTheirOwner::test_every_growable_dataset_has_exactly_one_owner`
asserts over a written entry that every dataset carrying `maxshape[0] is None` is
declared by exactly one module, and that every declared dataset is resizable on
disk. That is a stronger guarantee than the sweep's, and it fails at commit time
rather than at the next append.

### F5.3 — `@validate_call_` restored on `_validateWorkspaceAndPeaksData`

Lost during 04b's signature rewrite, while `_init` and `init_group` kept theirs.
Re-added after measuring that the suite still passes with it, rather than assuming
it had been dropped for a reason — 04c's Follow-up 3 F3.4 records a genuine
`@validate_call_` incompatibility, so the question was real.

### Withdrawn during this batch

A concern that nothing checks the per-scan-point `start_time`/`end_time` arrays
against the scan-point count: `SampleLogs.__setitem__` already refuses a log whose
length differs from the subruns, so the counts cannot diverge. Recorded because an
unchecked worry and a checked one look identical afterwards.

### F5.4 — string sample logs: `bytes` on disk, `str` in memory

**This change reaches outside the plan series. It is flagged for explicit review
and is called out in the commit message.**

`_entryTimes` decoded every time value unconditionally, inside a `try` that
catches only `ValueError` — the clause that substitutes `NO_LOG` for an
unparseable timestamp — so a `str`-valued log raised `AttributeError` past the
fallback. On the append path `_appendEntryTimes` is the **first** mutation step,
so the instance was invalidated for a type mismatch.

The underlying cause is not NXstress's. Nothing in PyRS normalised a string log
at any boundary, so its dtype was an accident of provenance: `bytes` from
`HidraProjectFile.read_sample_logs` and from NXstress's own reader, `str` from a
workspace built in memory. Six consumers downstream, four tolerant and two not.

Both candidate directions were **built and measured** against all three tiers
before choosing, rather than argued:

| | `bytes` in memory | **`str` in memory** |
|---|---|---|
| unit | 2 failed | passed |
| integration | 1 failed | passed |
| what failed | the CSV exporter wrote `# string1 = b'a constant string'` into a user-facing header | 83 failures, all the *same* defect: the unconditional `.decode()`, in two places |

Writers that must encode are few and already do it; readers that must decode are
many and mostly untested. `str` also ratifies the two deliberate normalisations
already in the tree (`_discriminator._as_text`, `HidraWorkspace.direction`)
instead of reversing them.

Enforced by **one** conversion rather than per-site coercions, at the
stakeholder's direction: *"those sections need to go through the same conversion
mechanism — it must be centralised."* `to_text` and `to_text_array` live in
`pyrs/utilities/convertdatatypes.py`, beside the existing `to_int`/`to_float`,
and are called

- at the store boundary, `SampleLogs.__setitem__`. **Both** file readers
  therefore normalise without either one decoding for itself — which answers the
  stakeholder's original question ("what happens if we adjust both readers?")
  better than adjusting them would have, since `HidraProjectFile` needed no
  change and NXstress's reader needed the *opposite* of a decode;
- at the HDF5 boundary, `HidraProjectFile.add_sample_log`. Not optional: h5py has
  no conversion path for numpy's `<U` dtype, and 18 integration tests failed on
  exactly that until it was added;
- at the **nine** sites across five modules that previously hand-rolled
  `isinstance(v, bytes)`, which had drifted into four different behaviours —
  some handling `numpy.bytes_` and some not, some coercing the non-bytes branch
  to `str` and some returning it untouched. `_definitions.as_text`, added earlier
  in this batch, is gone; `_discriminator._as_text` delegates.

### F5.5 — two defects in F5.4, found after it had passed every tier

Both were caught by the stakeholder reading the claim rather than the test
results, and both are recorded because the *shape* of each mistake recurs.

**The dtype that matters is `object`, not `|S`.** The first version tested
`value.dtype.kind == "S"`, which catches only **fixed-width** byte arrays. HDF5
does not produce those: h5py and `nexusformat` both yield an **object** array of
Python `bytes` for the variable-length UTF-8 dtype, whose kind is `"O"`. So the
check passed 465 unit, 104 integration and 16 GUI tests while doing nothing at
all for `start_time`, `end_time`, `Filename` or any discriminator column:

```console
AFTER the first "fix", a round-tripped workspace:
    start_time    array-dtype=object   first=bytes    b'2024-01-15T10:00:00'
    direction     array-dtype=object   first=bytes    b'11'
    SampleName    array-dtype=<U5      first=str_     'steel'
```

Only `SampleName` and `chemical_formula` converted, and only because
`_Sample.sampleLogsFromNexus` broadcasts them through `np.array([b"steel"] * n)`,
which *is* `|S`. **The tiers were green for the wrong reason** — they pass because
the intolerant consumers had been made tolerant, not because logs had become
text. And the unit test passed because its fixture was `np.array([b"11", ...])`:
it tested the dtype the literal syntax gives, not the one the reader yields.

**Encode to variable-length, never to fixed-width.** The write boundary first
used `numpy.char.encode`, producing `|S`, whose width is fixed by the longest
value present at creation — the silent-truncation hazard this plan already
documented for NXstress. Since `HidraProjectFile` re-saves a workspace it
previously read, that would also have converted an existing file's
variable-length columns to fixed-width on every save. It now targets
`h5py.string_dtype(encoding="utf-8")`, the same dtype `_Sample._writable` uses.

Both are pinned: `test_convertdatatypes.py::TestToTextArray` constructs the
object array explicitly and says why, `test_sample_logs.py` (dataobjects) stores
one, and `test_sample_logs.py` (projectfile) asserts the stored dtype is
variable-length and that a longer value written later survives.

### F5.6 — the conversion raises on misuse rather than passing it through

`to_text_array` first returned a numeric array unchanged, on the reasoning that
`SampleLogs` calls it for *every* log and most logs are numbers.

**Stakeholder:** *"Why would `to_text_array` leave numeric types untouched? In
that case it's obviously being abused and should raise an exception -- that would
be a developer 'usage' error!"*

Correct, and the pass-through is the same mistake in miniature as the one F5.5
records: it makes a misdirected call indistinguishable from a working one, and
the symptom surfaces later, somewhere else, as a `bytes` value in a consumer that
did not expect one. So the dispatch moves to the caller, where the ambiguity
actually lives, and the conversion is strict:

- `is_text_array(values)` is public and is the single definition of "array of
  strings", including the element-wise inspection an `object` array requires.
- `to_text_array(values)` raises `TypeError` naming the dtype that arrived, and
  points at the predicate.
- The two call sites that may legitimately hold either kind guard with it:
  `SampleLogs.__setitem__`, whose logs are mostly numeric, and
  `_Peaks._decoded`, whose discriminator column is whatever the configured sample
  log holds -- so a numeric discriminator passes through untouched by intent
  rather than by accident.

`TestIsTextArray::test_it_agrees_with_what_to_text_array_accepts` pins the pair
together: a True must mean the conversion succeeds, a False must mean it raises.

**One case is accepted, not refused: an empty `object` array.**

**Stakeholder:** *"Behavior on an empty array should not necessarily raise --
that's going to cause problems... I can imagine a sample-log that will be an
array of bytes (on disk), but is presently empty -- it is still legitimate during
conversion, and during I/O."*

Measured, and the ambiguity turns out not to exist on the path that matters: an
empty HDF5 dataset keeps its own dtype on read, and **only** a variable-length
string comes back as `object`.

```console
empty_fixed_str    h5 |S4     -> numpy |S4
empty_float        h5 float64 -> numpy float64
empty_int          h5 int32   -> numpy int32
empty_vlen_str     h5 object  -> numpy object
```

So an empty object array is unambiguously an empty string column. It has no
values to misclassify, so only its dtype is at stake, and `<U` is both what the
invariant wants and what the data was. An empty array whose dtype still says
*numbers* stays an error: `float64` is not ambiguous merely for being empty.

This is already reachable, and both the old and the interim behaviour were wrong.
Writing the documented "no peak fits" shape (`write([ws], [[]])`) leaves
`peaks/phase_name` and `peaks/mask` as empty `object` columns:

```console
phase_name  read=object -> _decoded=<U1 (len 0)      # now
                        -> float64                   # the hand-rolled version
                        -> object                    # the interim predicate
```

The hand-rolled conversion returned `np.array([])`, which is **`float64`** — an
empty array of *numbers* standing in for a string column. `to_text_array` now
passes `dtype=np.str_` explicitly for exactly that reason. None of it was visible
because `peakCollectionRanges` does `if len(phase_name) == 0: return []` two lines
later.

Promoted to [docs/ground_truths.md](../../docs/ground_truths.md) and Decisions
row 40, since it is a property of PyRS rather than of this document. Pinned by
`test_sample_logs.py::TestStringLogsAreTextInMemory` (dataobjects) and
`::TestStringLogsAreBytesOnDisk` (projectfile), which cross-reference each other
so neither half reads as redundant.

**Incidental, not fixed:** `summary_generator.py:169` calls `value.decode()` and
discards the result, which is why bytes reached that CSV header at all.
Unreachable under the new policy; left for its own change.

**Also found:** nothing runs the doctests. There is no `--doctest-modules` in
`addopts` and no collector in `tests/`, so the examples in
`restorable_property.py` and `as_text` are documentation, not tests.

### Verification

**494 unit** (was 443) / 104 integration / 16 GUI; `ruff` and `mypy` clean; probe
sweep 19 OK, 1 SKIPPED (retired), 0 FAIL.

---

## Follow-up 6 — 2026-10-15 (PR review, batch 26: citations re-resolved by symbol, not by position)

Raised in the human PR review of 04 + 04b + 04c (batch 26,
[plans/PR_review/04c-comments.md](../PR_review/04c-comments.md)).

### F6.1 — the append-only rule does not cover citation values

Decided by the reviewer during this batch, recorded as **README Decisions Log
item 41**:

> it is OK to update the citations in a follow-up section; that is, the citation
> specifics themselves are *not* subject to the "do not edit follow-up" rules.
> Stale citation *values* are worthless because they will point to something
> that makes no sense; when required, the actual previous values can be
> recovered from the git history!

This narrows rather than overturns "an earlier Follow-up is never edited". A
Follow-up's **claims and verdicts** stay frozen, because those are the record of
what was believed. A **pointer** is not a belief. Two exceptions survive, both
because there the old number *is* the content: a correction record of the form
`was X → now Y` keeps its `X`, and a finding that diagnoses a wrong pointer
keeps the pointer it diagnoses (F1.1 in 04b is one).

Line numbers are expected to be replaced by semantic-id XREF keys in a later
pass, which removes the problem at its source rather than re-fixing it per PR.

### F6.2 — every citation in this document re-resolved, by symbol

Thirty-three citations, each re-resolved by looking up the **symbol or text the
prose names** and asserting the target line contains it — then checked
mechanically, 30 of 30 passing. The method matters, and is the finding:

**Batch 24 remapped four of this document's pointers by diff position**, and
positional remapping is only as good as the pointer it starts from. It
faithfully carried one that was already wrong: `_sample.py:312` was cited for
"the variable-length dtype" while landing on `_present_in_all`, and the remap
dutifully moved it to `:529`, where it landed on `_present_in_all` again. It now
reads `_sample.py:637`, which is `_writable`, the function the claim is about.
A positional remap preserves *where the text went*; only a symbol lookup
preserves *what the claim meant*.

Corrected here, grouped by what moved them:

| Claim | Now |
|---|---|
| `peakCollectionRanges` | `_peaks.py:446-559` |
| `_append_peak` | `_peaks.py:331-404` |
| `_Peaks.init_group` | `_peaks.py:305-329` |
| the `sorted(…)` site | `_fit.py:109` |
| `_Masks.init_group` extends by name | `_instrument.py:558-567` |
| the overwrite guard | `NXstress.py:278` |
| `growable` field sites | `NXstress.py:1093-1094`, `_instrument.py:227`, `_sample.py:234`/`:247`/`:266`/`:287`, `_fit.py:510`/`:515`/`:519`/`:523` |
| `_Sample._writable` | `_sample.py:637` |
| `_Sample.validateAppend` | `_sample.py:511`, called first at `:594` |
| `_Instrument.appendableDatasets` | `_instrument.py:378` |

`_instrument.py:317` deserves separate mention: it had come to land on the
**detector** `depends_on` assignment while being cited for the *mask* group's
extend-by-name behaviour. A citation that resolves to real, related,
plausible-looking code is worse than one that resolves to nothing, because
following it does not feel like a failed check.

### F6.3 — the viewer call-site enumeration was wrong in three of five

Follow-up 2's verification recorded *"every call in `pyrs/interface/` is `"w"` or
`"r"` (`combine_runs_model.py:37`, `peak_fitting_model.py:97`/`:234`,
`texture_fitting_model.py:52`/`:197`)"*. The **conclusion is still true**, and
the enumeration backing it was not: there are four `NXstress(...)` constructor
calls under `pyrs/interface/`, not five. `combine_runs_model.py:37` is off by
three; `peak_fitting_model.py:97` and `texture_fitting_model.py:52` are not
constructor calls at all; and `nxstress_load.py:55` — the only `"r"` in the set,
and therefore the one most load-bearing for the claim — was missing entirely,
having been added later in this review.

Now reads `combine_runs_model.py:40`, `peak_fitting_model.py:234`,
`texture_fitting_model.py:197`, `nxstress_load.py:55`. Re-confirmed by
`grep -rn "NXstress(" pyrs/interface/`, which is how the count should have been
taken in the first place: an enumeration written out by hand goes stale
silently, while the command that produced it can be re-run.

### F6.4 — the scan-point family table omitted three of its own members

`## Scope: all position-aligned groups` is where this spec says what append
covers, and its central promise — *"append never leaves one group's rows grown
without the corresponding rows in every other group in its family"* — is
quantified over the table beneath it. The table listed three members. The code
grows five.

| Missing | Why it matters |
|---|---|
| `instrument/monochromator/wavelength` | 04b had already established this is per-scan-point rather than entry-wide, and corrected its own Scope for exactly that. 04c's table then placed `_instrument.py` only under *name-keyed, no insertion needed*, which is true of `_Masks` and false of the module. |
| `SAMPLE_DESCRIPTION/transformations/*` | Added to `_sample.py` later in this review (spec 04 Follow-up 9); scan-point-aligned by construction, seven links per entry. |
| `entry/start_time`, `entry/end_time` | The section had no entry-level row at all, so these belonged to no family. |

Each is a place the lockstep guarantee could have been broken without the
document noticing. The table now enumerates the family exhaustively and, more
usefully, names the **actual authority** beside it: each module's own
`appendableDatasets`, aggregated by `NXstress._appendableDatasets`
([NXstress.py:667](../../pyrs/utilities/NXstress/NXstress.py#L667)) and consumed
by `_validateAppendableShapes`. A prose table is a restatement and goes stale;
pointing at the code that is actually consulted means the next member added
shows up in one place rather than needing to be remembered in two.

### F6.5 — the Overview's motivating claim had become false in the present tense

The Overview opened by observing that `NXstress.write` *"already supports
accumulating multiple reductions into one `.nxs` file **today** — each call adds
a new, auto-numbered `NXentry`"*. That was true as a statement of the
pre-04c starting point, and this spec then changed it: under Decisions Log
item 33, `write()` dispatches on whether the resolved entry already exists, so
an omitted `entry_number` now **appends to the highest existing entry** rather
than creating a new one. The sentence had come to assert the opposite of
`write()`'s own docstring table.

Rewritten into the past tense and explicitly labelled as the starting point,
with a following paragraph stating the current behaviour and pointing at
[NXstress.py:156-164](../../pyrs/utilities/NXstress/NXstress.py#L156).

This also releases the one pointer F6.2 had to leave unrepaired:
`NXstress.py:151-152` could not be corrected while the claim it served was
false, since repointing it at the targeting table would have made the citation
contradict the sentence citing it. **A stale pointer attached to a stale claim
cannot be fixed as a pointer** — that is worth separating from ordinary drift,
because the mechanical sweep that catches the rest will report this one as
merely out of date.

### Verification

`check_links` 0 findings across 25 documents; `check_citations` 3 of 425, the
known `open-questions/06` bare-`Lnn` residue, none in this document;
`check_ownership` unchanged accepted residue. Every corrected pointer asserted
to contain the symbol its prose names — 30 checks, all passing. A final sweep of
every citation in this document leaves none landing on a structural or blank
line. `NXstress.py:151-152`, left unrepaired when F6.2 was written because the
claim it served was false, was released by F6.5 and is gone. No code changed in
this batch; tiers not re-run.
