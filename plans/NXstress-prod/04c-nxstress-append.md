# 04c — NXstress Append Mode (library only)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 3
**Depends on:** [04b — Multi-workspace NXstress I/O](04b-multi-workspace-nxstress.md)

---

## Overview

Add the ability to extend an **already-written** `NXentry` with more scan
points, rather than requiring every write to create a fresh entry.

This is a different capability than it may first appear: `NXstress.write`
already supports accumulating multiple reductions into one `.nxs` file today
— each call adds a new, auto-numbered `NXentry`
([NXstress.py:149](../../pyrs/utilities/NXstress/NXstress.py#L149)), and the
guard at [NXstress.py:151-152](../../pyrs/utilities/NXstress/NXstress.py#L151)
only fires on an accidental name collision. What's missing is the ability to
grow *one* `NXentry`'s data — e.g., a second reduction pass whose scan points
belong to the same combined index as an earlier write, not a new
data-reduction condition. Per the plan's own semantics
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
`_peaks.py:246-338`) only ever required two invariants — each compound key
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
`# TODO` at `_peaks.py:180-181`). No new insertion-position machinery is
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
| Peak-index family | `_peaks.py` (`PEAKS`/`NXreflections`), `_fit.py::_PeakParameters`, `_fit.py::_BackgroundParameters` (`peak_parameters`/`background_parameters` under `FIT`) | The compound `PeakIndex` sort key. `_PeakParameters.init_group`/`_BackgroundParameters.init_group` build their rows via `sorted(peakss, key=_Peaks.PeakIndex.sort_key)` ([_fit.py:87](../../pyrs/utilities/NXstress/_fit.py#L87)) — **the exact same order as the peaks index**, so their rows are positionally aligned with it, not independently keyed. |
| Scan-point family | `_input_data.py` (`detector_counts`), `_sample.py` (`SAMPLE_DESCRIPTION` per-scan-point logs), `_fit.py::_Diffractogram` (`diffractogram`/`diffractogram_errors`) | `scan_point`. |
| Name-keyed, no insertion needed | `_instrument.py::_Masks` | Mask *name*, not position — `init_group` already accepts an existing `masks` group and extends it by name ([_instrument.py:317](../../pyrs/utilities/NXstress/_instrument.py#L317)); no new work needed here. |

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
     originally written as a bare `N == 1`/no-discriminator write isn't
     possible without adding a new on-disk column, which this spec's
     tail-append design does not do (no schema restructuring, only
     resize-and-append into existing datasets). Raise `RuntimeError` if
     the target entry has no discriminator columns at all.
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
  compound index (the `# TODO` at `_peaks.py:180-181` already describes
  code written in a form that allows this) and, in lockstep, for
  `_fit.py::_PeakParameters.init_group` / `_BackgroundParameters.init_group` —
  all three grow by the same new-row count, appended after their current end.
- Scan-point family: implement the tail-append path for
  `_input_data.py:44-46,63-72` (`detector_counts`, on both write and read),
  `_sample.py`'s per-scan-point logs, and `_fit.py::_Diffractogram.init_group`
  — all three grow by the same new-row count, appended after their current
  end.
- `NXstress.py`: remove the guard at `NXstress.py:151-152` for the append
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
  at `_peaks.py:180-181`, reusing the existing `_append_peak` (`_peaks.py:190-244`)
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
`_input_data.py:44-46,63-72`, `NXstress.py:151-152`, `_peaks.py:190-244`,
`_peaks.py:246-338` all land exactly on their claimed targets. The premise at
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
