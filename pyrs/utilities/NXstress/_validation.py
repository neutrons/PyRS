"""
pyrs/utilities/NXstress/_validation.py

Write-time validation of the workspaces and peak collections given to `NXstress`.

Everything here runs before anything is written, and raises rather than writing a
file that would be readable and wrong. An append runs the same checks on its
incoming batch (`_append.run_checks`) before its own.

Typical workflow:

1. `validate_entry_number(n)` refuses an entry number that is not 1-based.
2. `validate_inputs(wss, peakss, indexed)` checks one write's inputs, and through
   `validate_multi_workspace` the invariants that bind only when an entry holds
   more than one workspace.
3. `validate_peak_coverage(wss, peakss)` refuses a workspace whose peak
   collections miss some of its scan points: always with several workspaces and
   on append, and for a single workspace unless
   `nxstress.allow_partial_peak_coverage` allows it.
"""

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.config import Config
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import REQUIRED_LOGS
from ._fit import _Fit
from ._peaks import _Peaks
from . import _discriminator, _runs, _workspace_key
from ._workspace_key import WorkspaceKey


def validate_entry_number(entry_number: int) -> None:
    """Refuse an entry number below 1: entries are numbered from 1."""
    if entry_number < 1:
        raise ValueError(f"NXstress: `entry_number` is 1-based; got {entry_number}.")


@validate_call_
def validate_inputs(wss: list[HidraWorkspace], peakss: list[list[PeakCollection]], indexed: list) -> None:
    """Check one write's inputs: their shapes, required logs, and peak collections against workspaces.

    Raises:
        ValueError: Naming the first violated requirement.
    """
    if len(peakss) != len(wss):
        raise ValueError(
            f"NXstress.write expects one list of `PeakCollection` per workspace: "
            f"got {len(wss)} workspace(s) and {len(peakss)} peak list(s)."
        )
    if not wss:
        raise ValueError("NXstress.write requires at least one `HidraWorkspace`.")

    for n, ws in enumerate(wss):
        logs = ws.sample_log_names
        for k in REQUIRED_LOGS:
            if k not in logs:
                raise ValueError(f"NXstress requires log '{k}', which is not present in workspace [{n}]")

    if len(wss) > 1:
        validate_multi_workspace(wss, peakss)
    elif not Config["nxstress.allow_partial_peak_coverage"]:
        validate_peak_coverage(wss, peakss)

    _Peaks.validateNoDuplicatePeaks(indexed)

    for ws, peaks in zip(wss, peakss):
        _Fit.validateWorkspaceAndPeaksData(ws, peaks)


def validate_multi_workspace(wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]) -> None:
    """The invariants that only bite when an entry holds N > 1 workspaces.

    These are write-time gates rather than documented conventions, because
    violating any of them produces a file that is *readable* and wrong --
    an unsplittable entry, inputs merged into one workspace, or rows
    attributed to the wrong workspace -- or one that is written correctly
    and then cannot be read back.
    """
    # 1. Without a discriminator, the boundary between inputs is not recorded
    #    anywhere. NXstress does not merge workspaces: that is PyRS's job,
    #    done before the write.
    if not _discriminator.field_names():
        raise ValueError(
            f"NXstress.write was given {len(wss)} workspaces, but "
            "'nxstress.discriminator_fields' names no field, so they could not be told apart "
            "when the file is read.\n"
            "  Configure a discriminator field, or merge the workspaces into one in PyRS "
            "before writing, as CombineRuns does."
        )

    # 2. A workspace contributing no `PeakCollection` has no discriminator
    #    value on disk, and therefore no recoverable scan-point set.
    empty = [n for n, peaks in enumerate(peakss) if not peaks]
    if empty:
        raise ValueError(
            f"NXstress.write: input workspace(s) {empty} contribute no `PeakCollection`.\n"
            "  With more than one workspace in an entry, every workspace must contribute at "
            "least one -- the peak index is the only place a workspace's identity is recorded, "
            "so one with no peaks cannot be recovered on read."
        )

    # 3. With several workspaces in one entry, each must hold a single run: the
    #    run is part of the workspace key (`_workspace_key.entry_carries_run`), so a workspace merged
    #    from several runs in PyRS has no one key to be read back under.
    merged = _runs.merged_inputs([ws._sample_logs for ws in wss])
    if merged:
        raise ValueError(
            f"NXstress.write: input workspace(s) {merged} hold several runs "
            f"({[sorted(_runs.run_numbers(wss[n]._sample_logs)) for n in merged]}).\n"
            "  A workspace merged from several runs can only be written on its own: alongside "
            "other workspaces, its rows could not be told apart from theirs on read."
        )

    # 4. The read side recovers each workspace by `(run, scan point)` membership,
    #    which cannot distinguish two workspaces sharing a run AND a scan point.
    #    A scan point alone may repeat -- every run numbers its scan points from 1.
    seen: dict = {}
    for n, ws in enumerate(wss):
        for point in _runs.workspace_pairs(ws._sample_logs):
            if point in seen:
                raise ValueError(
                    f"NXstress.write: (run, scan point) {point} appears in both input workspace "
                    f"[{seen[point]}] and [{n}].\n"
                    "  Input workspaces sharing an entry must not repeat a scan point within one "
                    "run: the reader recovers each workspace's rows by (run, scan point), so a "
                    "repeat cannot be attributed. Two workspaces with no `run_number` log share "
                    "the configured default run number."
                )
            seen[point] = n

    # 5. The reader recovers each workspace by its workspace key -- its
    #    discriminator values, and its run where the key carries the run -- so two
    #    inputs sharing the key would be read back as one: a merge. NXstress never
    #    merges workspaces; that is PyRS's job.
    discriminator_names = _workspace_key.entry_discriminator_names(wss)
    carries_run = _workspace_key.entry_carries_run(wss)
    owner: dict = {}
    for n, ws in enumerate(wss):
        key = WorkspaceKey.of(ws, discriminator_names, carries_run=carries_run)
        if key in owner:
            raise ValueError(
                f"NXstress.write: input workspaces [{owner[key]}] and [{n}] have the same workspace "
                f"key {dict(key.as_pairs())}, so they would be read back as one.\n"
                "  NXstress does not merge workspaces: merge them in PyRS first, as CombineRuns "
                "does, or give each a distinct discriminator value."
            )
        owner[key] = n

    # 6. On read, a workspace's scan points are the union of its collections'
    #    `sub_runs`, so a point no collection covers is claimed by no workspace.
    validate_peak_coverage(wss, peakss)


def validate_peak_coverage(wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]) -> None:
    """Every scan point of a workspace with peaks must be covered by one of them.

    A multi-workspace entry is split on read by peak-collection membership:
    each workspace gets the union of its collections' `sub_runs`, and a scan
    point outside every union raises then. Refused here instead, before
    anything is written. A workspace with no `PeakCollection` at all is not
    this rule's concern -- see `validate_multi_workspace`.

    Raises:
        ValueError: Naming each workspace and its uncovered scan points.
    """
    for n, (ws, peaks) in enumerate(zip(wss, peakss)):
        if not peaks:
            continue
        covered: set = set()
        for collection in peaks:
            covered |= set(collection.sub_runs.raw_copy().tolist())
        uncovered = sorted(set(ws.get_sub_runs().raw_copy().tolist()) - covered)
        if uncovered:
            raise ValueError(
                f"NXstress.write: scan point(s) {uncovered} of input workspace [{n}] are in none of "
                "its `PeakCollection`s.\n"
                "  Every scan point of a workspace with peaks must be covered by at least one: a "
                "multi-workspace entry recovers each workspace's scan points from its peaks, so an "
                "uncovered point belongs to no workspace on read. A single-workspace write may "
                "allow it with 'nxstress.allow_partial_peak_coverage'."
            )
