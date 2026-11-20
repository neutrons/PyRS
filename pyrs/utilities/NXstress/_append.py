"""
pyrs/utilities/NXstress/_append.py

Tail-append to an existing NXstress entry: the pre-flight that decides
whether an append may proceed, and the commit that grows every group.

An append is split in two so that its no-op guarantee can be read off the code:

1. `preflight(entry, wss, peakss)` runs every check, touching nothing, and
   returns the `AppendPlan` the commit needs. A refusal therefore leaves the entry
   byte-for-byte unchanged.
2. `commit(entry, wss, plan)` grows each group. Past its first resize there is no
   undo.

A violated invariant -- a duplicate row, an unmet precondition, a disagreement with
the entry -- is raised as `InvariantViolated`. `NXstress` takes that, and any
escape from the commit, as invalidating the instance: continuing would be writing
against an assumption already known to be false. An "unsupported" refusal
(`NotImplementedError`: Case B) leaves it usable.

The checks are named for what they raise:

- `reject_*` raises `InvariantViolated` -- including for the plain `RuntimeError`
  of a group's own `validateAppend` or of `validate_appendable_shapes`, which
  `reject_ungrowable_or_disagreeing_groups` re-raises through `reject`;
- `classify_compound_keys` raises either: a duplicate row is violated, more scan
  points under an existing key are unsupported.
"""

from typing import NamedTuple

import numpy as np
from nexusformat.nexus import NXdata, NXgroup
from nexusformat.nexus.tree import NeXusError

from pyrs.core.peak_profile_utility import BackgroundFunction
from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    GROUP_NAME,
    UNDEFINED_PEAK_TAG,
    appendable,
    appendable_dtype,
    nxstress_diffractogram_keys,
    suffix_from_group_name,
)
from ._fit import _Fit
from ._input_data import _InputData
from ._instrument import _Instrument
from ._peaks import _Peaks, IndexedPeaks
from ._sample import _Sample
from . import _discriminator, _runs, _validation
from ._workspace_key import WorkspaceKey


class InvariantViolated(RuntimeError):
    """An append refused for a violated invariant, which invalidates the `NXstress` instance.

    A `RuntimeError`, as these refusals always were; the subclass is what lets
    `NXstress` tell them from a `RuntimeError` that is merely "unsupported".
    """


class AppendPlan(NamedTuple):
    """What the pre-flight established, for the commit to use unchanged.

    Attributes:
        indexed: The incoming peak collections, each with its workspace's key.
        discriminator_names: The fields the entry records.
        run_boundary: Whether the entry's workspace key carries the run.
    """

    indexed: tuple[IndexedPeaks, ...]
    discriminator_names: tuple[str, ...]
    run_boundary: bool


# What a refused append raises before its first write, from a check or on the way
# to one; anything else there is a defect, not a refusal.
REFUSALS = (ValueError, RuntimeError, NeXusError)

_PREFIX = "NXstress: cannot append -- "


def refusal(error: BaseException) -> InvariantViolated:
    """`error` as an append refusal, keeping its message and naming the operation once."""
    message = str(error)
    return reject(message if message.startswith("NXstress: cannot append") else _PREFIX + message)


def reject(message: str) -> InvariantViolated:
    """Build the exception for a violated invariant.

    Returned rather than raised so each call site reads `raise reject(...)` and the
    traceback starts there.
    """
    return InvariantViolated(message)


def preflight(entry: NXgroup, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]) -> AppendPlan:
    """Decide whether an append may proceed, touching nothing.

    Every check happens here, before any resize, so a rejected append leaves the
    entry byte-for-byte unchanged. That includes the checks `tail_append` itself
    would make -- resizability and trailing-axis agreement -- which are asked up
    front rather than discovered part-way through, because by then earlier groups
    have grown and there is no undo.

    Args:
        entry: The target `NXentry`, file-backed and open for writing.
        wss: Workspaces to append.
        peakss: One list of `PeakCollection` per workspace, same order.

    Returns:
        The plan `commit` carries out.

    Raises:
        InvariantViolated: On a violated invariant -- a duplicate row, a missing
            precondition, or a disagreement with the entry.
        NotImplementedError: For Case B, extending a compound key already on disk
            with further scan points. That needs a mid-array insertion rather than
            a tail-append and is deliberately out of scope.
    """
    peaks_group = entry[GROUP_NAME.PEAKS]
    discriminator_names = _discriminator.names_for_read(peaks_group)

    # Asked before any key is built: an entry written before the run index has
    # no run column to grow, and one whose workspace key carries the run needs
    # every incoming workspace to hold exactly one.
    reject_run_index_mismatch(entry, wss)
    run_boundary = _runs.is_boundary(peaks_group)

    indexed = [
        IndexedPeaks(WorkspaceKey.of(ws, discriminator_names, carries_run=run_boundary), collection, ws._sample_logs)
        for ws, collections in zip(wss, peakss)
        for collection in collections
    ]

    run_checks(entry, wss, peakss=peakss, indexed=indexed, discriminator_names=discriminator_names)
    return AppendPlan(tuple(indexed), discriminator_names, run_boundary)


def commit(entry: NXgroup, wss: list[HidraWorkspace], plan: AppendPlan) -> None:
    """Grow every group below the entry by the incoming rows, as `preflight` planned.

    A **tail-append**: the incoming batch is sorted among itself and added after
    each dataset's current end. Nothing already on disk is read back into
    `HidraWorkspace`/`PeakCollection` form, reordered or rewritten -- particularly
    not the raw detector counts, which are the bulk of an entry. The file is
    therefore "locally sorted, globally segmented" afterwards rather than globally
    sorted, which is all the reader has ever required: `_Peaks.peakCollectionRanges`
    enforces that each compound key occupies one contiguous run and that
    `scan_point` increases within it, and nothing more.

    The entry-level `start_time`/`end_time` belong to `NXstress`, which grows them
    first.

    Every group grows through `_definitions.tail_append`. `NXstress` calls this
    inside `deferred_appends`, so nothing is written here: every row is computed
    and checked, and `apply_appends` then writes them all. A failure while
    computing therefore leaves the entry untouched.
    """
    logss = [ws._sample_logs for ws in wss]
    _InputData.init_group(wss, data=entry[GROUP_NAME.INPUT_DATA])
    _Instrument.init_group(wss, data=entry[GROUP_NAME.INSTRUMENT])
    _Sample.init_group(logss, data=entry[GROUP_NAME.SAMPLE_DESCRIPTION])
    indexed = list(plan.indexed)
    _Fit.init_group(wss, indexed, logss, data=entry[GROUP_NAME.FIT])
    _Peaks.init_group(
        indexed,
        logss[0],
        plan.discriminator_names,
        data=entry[GROUP_NAME.PEAKS],
        run_boundary=plan.run_boundary,
    )


def run_checks(
    entry: NXgroup,
    wss: list[HidraWorkspace],
    *,
    peakss: list[list[PeakCollection]],
    indexed: list[IndexedPeaks],
    discriminator_names: tuple[str, ...],
) -> None:
    """Run every pre-flight check on an incoming batch, in order, touching nothing.

    Runs to completion before the first resize. That ordering is what makes
    both rejection outcomes true no-ops, and it is the reason every check
    that could fail part-way through an append lives here rather than at the
    point of use -- including the ones `tail_append` would otherwise make
    during mutation. Kept as a readable list of calls for the same reason:
    this is the method a reader has to follow in full to believe the no-op
    property, so it is the one that must stay short.

    Raises:
        InvariantViolated: On any violated invariant.
        NotImplementedError: For Case B. Leaves the instance usable.
    """
    # Ordinary write-time validation applies to the incoming batch regardless
    # of where it is going.
    try:
        _validation.validate_inputs(wss, peakss, indexed)
    except ValueError as error:
        raise reject(f"NXstress: cannot append -- {error}") from error

    reject_unmet_case_a_preconditions(wss, peakss, discriminator_names)
    reject_unconfigured_columns(entry, discriminator_names)
    reject_discriminator_type_mismatch(entry, indexed, discriminator_names)
    reject_ungrowable_or_disagreeing_groups(entry, wss)
    classify_compound_keys(entry, indexed, discriminator_names)
    reject_existing_workspace_key(entry, wss, discriminator_names)
    reject_scan_point_collisions(entry, wss)
    reject_diffractogram_key_mismatch(entry, wss)
    reject_fit_model_mismatch(entry, indexed)
    reject_partially_covered_entry(entry)


def reject_run_index_mismatch(entry: NXgroup, wss: list[HidraWorkspace]) -> None:
    """The target entry must carry the run index, in a form the incoming workspaces fit.

    Three refusals, all before anything is resized, all invalidating the
    instance like every other violated precondition:

    - An entry written before the run index has no `run_number` column, on
      `peaks` or on `SAMPLE_DESCRIPTION`, so the incoming rows' runs would have
      nowhere to go -- as an entry written before its datasets could grow is
      refused (`validate_appendable_shapes`).
    - An entry whose workspace key does not carry the run
      (`_runs.BOUNDARY_ATTRIBUTE` false) holds one workspace merged from
      several runs, and anything appended would be read back as part of it.
    - An entry whose key does carry the run cannot take an incoming workspace
      merged from several runs: it has no one key to join under.
    """
    peaks_group = entry[GROUP_NAME.PEAKS]
    if _runs.RUN_NUMBER not in peaks_group or _runs.RUN_NUMBER not in entry[GROUP_NAME.SAMPLE_DESCRIPTION]:
        raise reject(
            "NXstress: cannot append -- the target entry was written before PyRS recorded run "
            "numbers, so it has no `run_number` column for the incoming rows.\n"
            "  Write a new file instead."
        )
    if not _runs.is_boundary(peaks_group):
        raise reject(
            "NXstress: cannot append -- the target entry holds a single workspace merged from "
            "several runs, so anything appended to it would be read back as part of that "
            "workspace.\n  Write a new file instead."
        )
    merged = _runs.merged_inputs([ws._sample_logs for ws in wss])
    if merged:
        raise reject(
            f"NXstress: cannot append -- input workspace(s) {merged} hold several runs, and this "
            "entry tells its workspaces apart by run.\n"
            "  Append one workspace per run."
        )


def reject_unmet_case_a_preconditions(
    wss: list[HidraWorkspace], peakss: list, discriminator_names: tuple[str, ...]
) -> None:
    """Case A's preconditions, which a multi-workspace write already imposes."""
    # -- Case A precondition 1: every incoming workspace must contribute at
    #    least one `PeakCollection`. `_validation.validate_multi_workspace` already enforces
    #    this, but only for N > 1; on append it binds at N == 1 too, because
    #    the entry it joins holds other workspaces whose rows must stay
    #    distinguishable from these.
    empty = [n for n, collections in enumerate(peakss) if not collections]
    if empty:
        raise reject(
            f"NXstress: cannot append -- input workspace(s) {empty} contribute no "
            "`PeakCollection`.\n"
            "  The peak index is the only place a workspace's identity is recorded, so one "
            "with no peaks could not be recovered from the appended entry."
        )

    # -- Every incoming workspace's `PeakCollection`s must cover all of its scan
    #    points, whatever `nxstress.allow_partial_peak_coverage` says: the entry
    #    holds other workspaces, so an uncovered point would be claimed by none
    #    of them on read.
    try:
        _validation.validate_peak_coverage(wss, peakss)
    except ValueError as error:
        raise reject(f"NXstress: cannot append -- {error}") from error

    # -- Case A precondition 2: the target must already have a discriminator
    #    scheme. An entry with none holds one workspace, as a write with no
    #    discriminator configured may (`_validation.validate_multi_workspace`,
    #    check 1) -- even though its key carries the run: the run tells apart the
    #    workspaces of an entry that records a discriminator, and does not make one
    #    on its own. Adding a column to an existing group is possible in h5py and
    #    nexusformat, but it is schema restructuring, which append excludes -- a
    #    scope decision, not a format limit.
    if not discriminator_names:
        raise reject(
            "NXstress: cannot append -- the target entry carries no discriminator columns, so it "
            "holds one workspace: with no discriminator, an entry holds a single workspace, as a "
            "write of several is refused.\n"
            "  Configure 'nxstress.discriminator_fields' and write the entry afresh; this pass "
            "does not add a column to an entry that lacks one."
        )


def reject_unconfigured_columns(entry: NXgroup, discriminator_names: tuple[str, ...]) -> None:
    """Every `peaks` column must be one the append knows how to grow.

    A read ignores a column it does not recognise; an append cannot. It grows
    the reserved columns and the configured discriminators, so any other
    column -- another producer's reflection field, or a discriminator the
    configuration does not name -- would be left short, and the group would
    no longer be row-aligned.
    """
    configured = {_discriminator.column_name(name) for name in discriminator_names}
    unknown = sorted(_discriminator.columns_on_disk(entry[GROUP_NAME.PEAKS]) - configured)
    if unknown:
        raise reject(
            f"NXstress: cannot append -- this entry's `peaks` group has column(s) {unknown}, "
            f"which are neither PyRS's own nor among the discriminator fields "
            f"{list(discriminator_names)}.\n"
            "  An append grows only the columns it knows, so these would be left short. "
            "Write a new entry instead."
        )


def reject_discriminator_type_mismatch(
    entry, indexed: list[IndexedPeaks], discriminator_names: tuple[str, ...]
) -> None:
    """Incoming discriminator values must fit the type of the column already on disk.

    The column's type was fixed by the entry's first write. A text value
    appended to a numeric column (or the reverse) fails inside `_Peaks`'
    append, after other groups have grown; a float appended to an integer
    column would be truncated without any error.
    """
    peaks_group = entry[GROUP_NAME.PEAKS]
    for name in discriminator_names:
        on_disk = peaks_group[_discriminator.column_name(name)].dtype
        incoming = np.asarray([dict(item.key.discriminators)[name] for item in indexed]).dtype
        if not appendable_dtype(incoming, on_disk):
            raise reject(
                f"NXstress: cannot append -- discriminator field '{name}' is stored as {on_disk} "
                f"in this entry, and the incoming value(s) are {incoming}.\n"
                "  A discriminator must keep the type the entry was first written with."
            )


def reject_ungrowable_or_disagreeing_groups(entry: NXgroup, wss: list[HidraWorkspace]) -> None:
    """Everything `tail_append` would refuse, and every per-group disagreement.

    Asked before the first resize rather than during mutation: a refusal
    discovered part-way through leaves the entry desynchronized, and such an
    entry reads back without error.
    """
    # -- Every dataset this append will grow must be able to grow, and must
    #    agree with the incoming rows on its trailing axes. Asked here rather
    #    than discovered inside `tail_append`, which runs during mutation:
    #    by then earlier groups have grown, the entry is desynchronized, and
    #    it reads back without error.
    #
    # -- Per-group agreement with the existing entry. Each of these lives on the
    #    group that owns the rule and is called here, before any resize, so that a
    #    mismatch is a true no-op rather than a half-grown entry.
    #
    # The groups raise a plain `RuntimeError`; re-raised through `reject` so that
    # these refusals invalidate the instance like every other violated invariant.
    try:
        validate_appendable_shapes(entry, wss)
        _Instrument.validateAppend(wss, entry[GROUP_NAME.INSTRUMENT])
        _InputData.validateAppend(wss, entry[GROUP_NAME.INPUT_DATA])
        _Sample.validateAppend([ws._sample_logs for ws in wss], entry[GROUP_NAME.SAMPLE_DESCRIPTION])
        _Fit.validateAppend(wss, entry[GROUP_NAME.FIT])
    except NotImplementedError:
        # A `RuntimeError` subclass, but "unsupported" rather than "violated",
        # and so not a reason to invalidate the instance.
        raise
    except RuntimeError as error:
        raise reject(str(error)) from error


def classify_compound_keys(entry: NXgroup, indexed: list[IndexedPeaks], discriminator_names: tuple[str, ...]) -> None:
    """Case A (proceed), Case B (unsupported) or duplicate (violated invariant)."""
    # -- Case A / Case B / duplicate, from the entry's existing peak index.
    existing_rows: dict = {}
    for workspace_key, key, start, end in _Peaks.peakCollectionRanges(entry[GROUP_NAME.PEAKS], discriminator_names):
        points = entry[GROUP_NAME.PEAKS]["scan_point"].nxdata[start:end]
        existing_rows[(workspace_key, key)] = set(np.asarray(points).tolist())

    for item in indexed:
        compound = (item.key, _Peaks.PeakIndex.sort_key(item.collection))
        if compound not in existing_rows:
            continue  # Case A -- a key the entry does not hold yet.
        incoming = set(item.collection.sub_runs.raw_copy().tolist())
        shared = incoming & existing_rows[compound]
        if shared:
            raise reject(
                f"NXstress: cannot append -- scan point(s) {sorted(shared)} are already present "
                f"in this entry under {compound[1]}{item.key.describe()}.\n"
                "  An appended row may not duplicate one already committed."
            )
        raise NotImplementedError(
            f"NXstress: appending further scan points to the compound key {compound[1]}{item.key.describe()}"
            ", which this entry already holds, is not supported.\n"
            "  Each key must occupy one contiguous run, so new scan points under an existing "
            "key would have to be inserted into the middle of it -- a different operation from "
            "the tail-append this implements. Appending a NEW workspace is supported."
        )


def reject_scan_point_collisions(entry: NXgroup, wss: list[HidraWorkspace]) -> None:
    """The `(run, scan point)` axis must stay partitioned across the whole entry."""
    # -- The reader attributes rows to a workspace by `(run, scan point)`, so a
    #    pair appearing twice in one entry cannot be attributed at all. A scan
    #    point alone may repeat across runs.
    sd = entry[GROUP_NAME.SAMPLE_DESCRIPTION]
    on_disk = set(_runs.pairs(_runs.axis(sd), np.asarray(sd["scan_point"].nxdata)))
    for n, ws in enumerate(wss):
        incoming = set(_runs.workspace_pairs(ws._sample_logs))
        collisions = on_disk & incoming
        if collisions:
            raise reject(
                f"NXstress: cannot append -- (run, scan point) pair(s) {sorted(collisions)} of input "
                f"workspace [{n}] are already in this entry.\n"
                "  The reader recovers each workspace's rows by (run, scan point), so a repeated "
                "pair cannot be attributed."
            )


def reject_existing_workspace_key(
    entry: NXgroup, wss: list[HidraWorkspace], discriminator_names: tuple[str, ...]
) -> None:
    """An incoming workspace may not share a workspace key with one already in the entry.

    The reader recovers each workspace by its workspace key -- its discriminator
    values, and its run where the entry's key carries the run -- so rows under a
    key already on disk would be read back as part of that workspace: two
    workspaces merged into one. NXstress never merges workspaces -- PyRS does that
    before writing, as CombineRuns does -- and the same rule binds a fresh write in
    `_validation.validate_multi_workspace`.

    Runs after `classify_compound_keys`, so that Case B -- the same key *and* the
    same peak, with further scan points -- is still reported as the unsupported
    operation it is. That one extends a single workspace, rather than merging two,
    and stays a deferred feature.
    """
    peaks_group = entry[GROUP_NAME.PEAKS]
    on_disk = {workspace_key for workspace_key, *_ in _Peaks.peakCollectionRanges(peaks_group, discriminator_names)}
    carries_run = _runs.is_boundary(peaks_group)
    for n, ws in enumerate(wss):
        key = WorkspaceKey.of(ws, discriminator_names, carries_run=carries_run)
        if key in on_disk:
            raise reject(
                f"NXstress: cannot append -- input workspace [{n}] has the workspace key "
                f"{dict(key.as_pairs())}, which this entry already holds.\n"
                "  Its rows would be read back as part of the existing workspace. NXstress does not "
                "merge workspaces: merge them in PyRS first, as CombineRuns does, or give this one a "
                "distinct discriminator value."
            )


def reject_partially_covered_entry(entry: NXgroup) -> None:
    """The entry's own scan points must all be covered by its peaks.

    A single-workspace entry written with `nxstress.allow_partial_peak_coverage`
    may hold scan points with no peak rows; it reads back whole. Appending to it
    makes a multi-workspace entry, which `read` splits by peak rows, so those
    points would belong to no workspace. The incoming workspaces are checked in
    `reject_unmet_case_a_preconditions`; this checks the entry they join.

    Compared by `(run, scan point)`, since a scan point alone may repeat across
    runs; `reject_run_index_mismatch` has already required both run columns.
    """
    sd, peaks_group = entry[GROUP_NAME.SAMPLE_DESCRIPTION], entry[GROUP_NAME.PEAKS]
    on_disk = set(_runs.pairs(_runs.axis(sd), np.asarray(sd["scan_point"].nxdata)))
    covered = set(_runs.pairs(peaks_group[_runs.RUN_NUMBER].nxdata, peaks_group["scan_point"].nxdata))
    uncovered = sorted(on_disk - covered)
    if uncovered:
        raise reject(
            f"NXstress: cannot append -- this entry's (run, scan point) pair(s) {uncovered} have no peaks.\n"
            "  It was written with 'nxstress.allow_partial_peak_coverage', and an appended entry "
            "recovers each workspace's scan points from its peaks, so these would belong to no "
            "workspace on read. Write a new entry instead."
        )


def reject_diffractogram_key_mismatch(entry: NXgroup, wss: list[HidraWorkspace]) -> None:
    """Every scan point in an entry needs a diffractogram under each key.

    Compares **diffractogram keys**, not detector-mask names. The two are
    different sets: a key is a reduction identifier -- `eta_-5.0`,
    `mask_a_eta_-5.0` -- that references a detector mask without being one,
    and only a reduction that used the default mask *alone* produces a bare
    `DEFAULT_TAG` key. So the incoming side is read through
    `nxstress_diffractogram_keys`, never `nxstress_mask_names`: the latter adds
    `DEFAULT_TAG` because every reduction uses the default mask, a key the
    writer never emits, which would refuse every texture append.
    """
    # -- The diffractogram key set must match. A key the entry has no group
    #    for would need a new group with no rows for the scan points already on
    #    disk, which is restructuring rather than growth.
    incoming_keys: set = set()
    for ws in wss:
        incoming_keys |= nxstress_diffractogram_keys(ws._diff_data_set.keys())
    on_disk_keys = {
        suffix_from_group_name(name, GROUP_NAME.DIFFRACTOGRAM)
        for name in entry[GROUP_NAME.FIT]
        if isinstance(entry[GROUP_NAME.FIT][name], NXdata)
    }
    if incoming_keys != on_disk_keys:
        raise reject(
            f"NXstress: cannot append -- the incoming workspaces' reduced-diffraction keys do "
            f"not match the target entry's.\n"
            f"  On disk:   {sorted(on_disk_keys)}\n"
            f"  Incoming:  {sorted(incoming_keys)}\n"
            "  Every scan point in an entry must have a diffractogram under each key."
        )


def reject_fit_model_mismatch(entry: NXgroup, indexed: list[IndexedPeaks]) -> None:
    """All peak collections in one entry share a single fit model."""
    # -- The peak profile and background function are entry-wide scalars that
    #    an append cannot change. `_append_peak` would raise on a mismatch, but
    #    only after earlier groups had already grown.
    fit_group = entry[GROUP_NAME.FIT]
    for group_name, label, incoming_titles in (
        (
            GROUP_NAME.PEAK_PARAMETERS,
            "peak profile",
            {str(item.collection.peak_profile).lower() for item in indexed},
        ),
        (
            GROUP_NAME.BACKGROUND_PARAMETERS,
            "background function",
            {str(BackgroundFunction.getFunction(item.collection.background_type)).lower() for item in indexed},
        ),
    ):
        on_disk_title = to_text(fit_group[group_name]["title"].nxdata)
        if str(on_disk_title) == UNDEFINED_PEAK_TAG:
            # The entry was written with no peak collections at all (the
            # CombineRuns export shape, `write([ws], [[]])`). Its fit model is
            # the writer's own sentinel, so comparing against it would reject
            # the append for apparently disagreeing with `_undefined_`. Say
            # what is actually wrong instead.
            raise reject(
                "NXstress: cannot append -- the target entry holds no `PeakCollection`s, so it "
                "records no fit model and no discriminator values for the rows already in it.\n"
                "  An appended workspace could not be told apart from them. Write a new entry."
            )
        if incoming_titles - {str(on_disk_title)}:
            raise reject(
                f"NXstress: cannot append -- the incoming `PeakCollection`s' {label} "
                f"{sorted(incoming_titles)} does not match the target entry's "
                f"'{on_disk_title}'.\n"
                "  All peak collections in one entry share a single fit model."
            )


def appendable_datasets(entry: NXgroup) -> dict:
    """Every dataset in `entry` that an append grows, declared by its owner.

    Each grouping module reports the datasets *it* writes row-aligned, because
    each one owns its group's layout. Inside a group NXstress writes, a field
    is either row-aligned -- and therefore `growable` -- or an entry-wide
    scalar, so a field added to such a group later is covered without editing
    anything here.

    Declared, not inferred from `shape[0] == n_scan`: length cannot tell the two
    kinds apart. An entry-wide array may happen to be `n_scan` long (three masks,
    three scan points), and the result is a resizability *requirement*, so it
    would refuse a legal append. And the peak index is one row per (compound
    key, scan point), so it is not `n_scan` long once a workspace fits more than
    one peak.

    Args:
        entry: The target `NXentry`.

    Returns:
        Path -> `NXfield`, for every dataset an append would extend.
    """
    found = {
        f"{GROUP_NAME.ENTRY}/start_time": entry["start_time"],
        f"{GROUP_NAME.ENTRY}/end_time": entry["end_time"],
    }
    for group_name, owner in (
        (GROUP_NAME.INPUT_DATA, _InputData),
        (GROUP_NAME.INSTRUMENT, _Instrument),
        (GROUP_NAME.SAMPLE_DESCRIPTION, _Sample),
        (GROUP_NAME.FIT, _Fit),
        (GROUP_NAME.PEAKS, _Peaks),
    ):
        if group_name in entry:
            found.update(
                {
                    f"{GROUP_NAME.ENTRY}/{path}": field
                    for path, field in owner.appendableDatasets(entry[group_name]).items()
                }
            )
    return found


def validate_appendable_shapes(entry: NXgroup, wss: list[HidraWorkspace]) -> None:
    """Every dataset the append will grow can grow, without growing any of them.

    Two distinct refusals, both of which `tail_append` would otherwise make
    during mutation:

    1. **Resizability.** A dataset written without `maxshape` is contiguous
       and can never be extended -- a file from a PyRS predating `growable`.
    2. **Trailing-axis agreement**, delegated to the groups that own the
       two-dimensional data, since only they know what the incoming rows
       would be shaped like.

    Which datasets those are is likewise asked of the groups that wrote them
    (`appendable_datasets`), not inferred from array length.

    Args:
        entry: The target `NXentry`.
        wss: Workspaces being appended.

    Raises:
        RuntimeError: Naming every dataset that cannot grow.
    """
    fixed = [path for path, field in appendable_datasets(entry).items() if not appendable(field)]
    if fixed:
        raise RuntimeError(
            f"NXstress: cannot append -- {len(fixed)} dataset(s) in this entry were written at a "
            f"fixed size and cannot be extended: {sorted(fixed)}.\n"
            "  Appending requires every per-scan-point dataset to have been created resizable "
            "(`maxshape`/`chunks`). A file written by a PyRS predating that change cannot be "
            "appended to -- write a new entry instead."
        )
