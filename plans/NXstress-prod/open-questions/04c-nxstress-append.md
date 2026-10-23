# Open Questions — 04c NXstress Append Mode (library only)

**Spec:** [04c-nxstress-append.md](../04c-nxstress-append.md)
**Blocking:** No — all previously-open design questions are now resolved
(see below). What remains is implementation detail, not user-facing
behavior.

---

## Q1 — RESOLVED: write mode, entry targeting, and conflict policy

Originally three separate open items (carried over from the pre-split spec
07's Q1/Q2, plus a new entry-targeting question that surfaced once the
architecture was worked out). All three are now decided together, since they
shape one coherent API:

- **Architecture:** append resizes and inserts directly into the on-disk
  arrays of the target entry; it does **not** reconstruct the existing
  entry as `HidraWorkspace`/`PeakCollection` and rewrite via 04b's merge
  path. Driving concern: a reduction/append cycle shouldn't have to
  round-trip potentially large existing raw-count data through PyRS's outer
  types just to grow it.
- **Write mode:** always explicit — `NXstress(path, "a")` — no auto-detection
  based on whether the file/entry already exists. This was already the
  spec's leaning before this round; it's now fully consistent with entry
  targeting also being explicit.
- **Entry targeting:** `NXstress(path, "a")` with no further argument
  targets the **last** existing `NXentry`; `entry_number` is an optional
  override for targeting an earlier entry.
- **Conflict policy:** reject. Any overlapping scan point or peak-index row
  raises `RuntimeError` (not `ValueError` — deliberately distinct from
  `validateNoDuplicatePeaks`'s fresh-write duplicate case, since this
  represents a violated invariant on already-committed data, not an
  expected bad-input case). The check runs against all affected groups
  before any mutation, so a rejected append is a true no-op. After raising,
  the `NXstress` instance is invalidated for further writes in that
  session — the caller must re-open rather than continue.

**Superseded in part by Q5 below — do not take "resizes and inserts" or
the binary conflict policy above at face value.** By the time the
sortedness/append scope was revisited (Q5), append was cut to
**tail-append only**: "insert" here should be read as "tail-append," and
the conflict outcome is now three-way, not binary — a genuinely new
compound key (Case A) proceeds; a key already on disk needing more scan
points (Case B) raises `NotImplementedError` without invalidating the
instance; only an exact duplicate raises the `RuntimeError` described
above. This paragraph is left as originally written (a documentation
choice, not an oversight) so the history of how this spec's scope
narrowed is visible; the spec text itself (`04c-nxstress-append.md`)
reflects only the current, post-Q5 behavior.

See the spec's Overview for the full reasoning; nothing here blocks
implementation.

---

## Q2 — RESOLVED: append scope covers all position-aligned groups, not just input_data and peaks

The spec as originally drafted (inherited from the pre-split spec 07) only
listed `_input_data.py` and `_peaks.py` as needing append logic. That
undercounts what a consistent append actually requires:
`_fit.py::_PeakParameters`/`_BackgroundParameters` build their rows via the
exact same sort order as the peaks index (`_fit.py:87`), so they're
positionally aligned with it, not independently keyed — and
`_sample.py`/`_fit.py::_Diffractogram` are both indexed by `scan_point`,
same as `_input_data.py`'s `detector_counts`. A partial append that only
grew some of these would desynchronize the entry with no error to signal it.

**Decided:** full scope — both position-aligned families (peak-index family:
peaks + peak_parameters + background_parameters; scan-point family:
detector_counts + sample logs + diffractogram), inserted in coordinated
lockstep within each family. `_instrument.py::_Masks` needs no new work — it's
name-keyed, not position-sensitive, and already accepts an existing group to
extend.

**Still correct after Q5's tail-append-only cut** — "inserted" here should
now be read as "tail-appended," but the substance (full scope, both
families, in lockstep) is unchanged. Only Q1's binary conflict-policy
framing needed a real correction; this question's scope decision holds.

---

## Q3 — RESOLVED/OBSOLETE: shared insertion-position helper: one utility, or one per group?

Originally: the spec described computing insertion positions once per
family and applying them across every group in that family, but didn't
specify whether this should be a single shared utility or independently
duplicated per-group logic.

**Obsolete as of the relaxed-sortedness/tail-append-only decision (see Q5
below).** No insertion positions are computed at all under this spec's
now-scoped-down behavior: append only ever tail-appends new rows after
each dataset's current end (`cur = shape[0]; resize(cur+N); arr[cur:] =
…`, the shape `_peaks.py::_append_peak` and `_fit.py`'s
`_PeakParameters._append_peak` already implement). There is no "insertion
position" to compute, share, or risk diverging — the question this item
raised doesn't arise. Extending an existing key with more rows (the case
that *would* need real insertion) is out of scope for this spec
(`NotImplementedError`), not merely deferred to "implementation time" as
originally framed.

---

## Q4 — Does the conflict check need to handle a family-cross-check (e.g. a scan point present in `_input_data` but not yet in `_sample`)?

The conflict policy (Q1) assumes each family's groups are always mutually
consistent going into an append — i.e., that no prior write ever left, say,
`detector_counts` and `SAMPLE_DESCRIPTION` with different scan-point sets
for the same entry. That should hold given 04b's write path always writes
all groups in a family together, but append's conflict check doesn't
currently specify whether it should also *verify* that invariant on the
existing entry (defensively) before proceeding, or simply trust it.

**Why it matters:** if the invariant were ever violated (e.g. by a bug in
an earlier write, or a hand-edited file), an append that trusts it could
silently produce a still-inconsistent result rather than surfacing the
pre-existing problem.

**Next step:** decide at implementation time whether the conflict check
includes a defensive cross-group consistency check on the *existing* entry
before applying the new data, or documents that it assumes (without
verifying) the existing entry is internally consistent. Low priority —
does not block starting implementation.

---

## Q5 — RESOLVED: relax global sortedness; scope this spec to tail-append only

The user proposed relaxing the assumption (inherited from `_peaks.py:44-45`'s
docstring) that the combined peak index must be kept globally
lexicographically sorted on disk, since sorting has nothing to do with the
NXstress schema itself.

**Verified directly against the code, and confirmed correct.** The only
reader-side splitter, `_Peaks.peakCollectionRanges` (`_peaks.py:246-338`),
enforces exactly two invariants: each compound key occupies one
*contiguous* run (raises `"Interleaved blocks detected"` otherwise,
`:313`/`:332`), and `scan_point` increases within a run (`:306`/`:326`).
It never checks global order — no `searchsorted`/`argsort` exists anywhere
in the module — and the monotonic-scan-point invariant is guaranteed
upstream by `SubRuns.set` (`sample_logs.py:164-166`) regardless of
NXstress's own `sorted()` calls. Global sortedness was never a reader
requirement.

**Resolved, per the user's three-part proposal:**
1. [04b](../04b-multi-workspace-nxstress.md) makes discriminator fields
   the most slowly varying coordinates of the combined index, so each
   input workspace's rows form one contiguous super-block.
2. A single-step write (all data present at the `write()` call) still
   produces a fully sorted index, as today.
3. Append sorts only the incoming batch, and does not re-sort the file —
   "locally sorted, globally segmented," not globally sorted. The
   non-overlap invariant is unchanged and fully enforced; only the
   *ordering* guarantee is relaxed.

Contiguity (R1, above) is not relaxable, which splits append into two
cases: **Case A** (the incoming data introduces new compound keys — the
normal "append a new workspace" case) is satisfiable by a tail-append,
and is already effectively implemented (`_peaks.py::_append_peak`,
`_fit.py`'s `_PeakParameters._append_peak`, both already
`resize(cur+N); arr[cur:] = …`). **Case B** (the incoming data extends a
compound key already on disk) genuinely requires mid-array insertion.

**Decided (scope cut): this spec covers Case A only.** Case B raises
`NotImplementedError` rather than being silently mishandled or
implemented via the original insertion machinery. This removes the need
for computed insertion positions entirely (obsoleting Q3, above) and
turns this spec's implementation into plumbing over `init_group` methods
that already know how to grow themselves — no new positional-insertion
logic anywhere in `pyrs/utilities/NXstress/`.

---

## Q6 — RESOLVED: Case A never carried forward 04b's own preconditions for a new workspace

A follow-up full-tree consistency check (re-verifying this spec against
04b, not just other specs against this pair) found that Case A — "the
incoming data introduces new compound keys, proceed as a tail-append" —
never restated two things 04b's own write path already requires of any
new workspace:

1. **The new workspace must contribute at least one `PeakCollection`.**
   04b's Q7 established this as a write-time-enforced invariant for the
   fresh-write case (without it, a workspace's discriminator value can't
   be recovered on `read()`); Case A is exactly "adding a new workspace,"
   so the same invariant applies here, and needs the same enforcement —
   not merely an assumption a caller happens to satisfy.
2. **The target entry must already have a discriminator scheme
   established** — appending a distinguishable new workspace to an entry
   originally written as a bare `N == 1`/no-discriminator write isn't
   possible under this spec's tail-append-only, no-schema-restructuring
   design (there's no existing discriminator column to attach the new
   workspace's value to).

**Decided:** both are added to Case A's classification pass as explicit
preconditions, each raising `RuntimeError` (and invalidating the instance,
same treatment as an exact duplicate — these are violated invariants, not
unsupported-but-otherwise-valid operations like Case B). See the spec's
"Conflict policy, and Case A/Case B classification" section for the
current text.
