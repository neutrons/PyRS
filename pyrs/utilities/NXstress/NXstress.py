"""
pyrs/utilities/NXstress/NXstress.py

Primary service class for NeXus NXstress-compatible I/O.
"""

from datetime import datetime
from nexusformat.nexus import NXdata, NXentry, NXfield, NXFile, nxopen
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.peak_profile_utility import BackgroundFunction
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    DEFAULT_TAG,
    GROUP_NAME,
    appendable,
    appendable_dtype,
    group_naming_scheme,
    growable,
    nxstress_diffractogram_keys,
    suffix_from_group_name,
    tail_append,
    NO_LOG,
    UNDEFINED_PEAK_TAG,
    logger,
    REQUIRED_LOGS,
)
from ._input_data import _InputData
from ._instrument import _Instrument, _Masks
from ._sample import _Sample
from ._fit import _Fit, _Diffractogram
from ._peaks import _Peaks, IndexedPeaks
from . import _discriminator

import numpy as np


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

/<entryname>                               (NXentry, group)
│
├─ definition                                (dataset: "NXstress")
├─ start_time                                (dataset: ISO8601 string)
├─ end_time                                  (dataset: ISO8601 string)
├─ processingtype                            (dataset: string)
│
├─ instrument                             (NXinstrument, group)
│   ├─ name                                 (dataset: string)
│   ├─ source                               (NXsource, group)
│   ├─ detector                             (NXdetector, group)
│   └─ mask (optional)                      (NXcollection, group)
│
├─ sample                                 (NXsample, group)
│   ├─ name                                 (dataset: string)
│   ├─ chemical_formula (optional)          (dataset: string)
│   ├─ temperature (optional)               (dataset: string)
│   ├─ stress_field (optional)              (dataset: string)
│   └─ gauge_volume (optional)              (NXparameters, group)
│
├─ fit                                    (NXprocess, group)
│   ├─ @date                                (attribute: ISO8601 string)
│   ├─ @program                             (attribute: string)
│   ├─ description                          (NXnote, group)
│   ├─ peakparameters                       (NXparameters, group)
│   └─ diffractogram                        (NXdata, group)
│        ├─ diffractogram                     (dataset)
│        ├─ diffractogram_errors              (dataset)
│        ├─ daxis/xaxis                       (dataset)
│        ├─ @axes                             (attribute: string)
│        └─ @signal                           (attribute: string)
│
├─ peaks                                  (NXreflections, group)
│   ├─ h                                    (dataset)
│   ├─ k                                    (dataset)
│   ├─ l                                    (dataset)
│   └─ phase_name                           (dataset)
"""


class NXstress:
    ##################################################################
    ## Service class to write NXstress-compliant NXentries:         ##
    ##   the `write` method writes the next `NXentry` to the file.  ##
    ##################################################################

    ## Context-manager related methods:
    def __init__(self, file_path: Path, mode: str = "r", *, entry_number: int | None = None):
        """Open an NXstress file.

        Args:
            file_path: Path to the `.nxs` file.
            mode: `"r"` to read, `"w"` for a new file, `"a"` to open an existing
                one for writing.
            entry_number: Default target for `write`, 1-based; `write` takes the
                same argument per call. Mode governs file access and
                `entry_number` governs targeting, so it is meaningful with both
                `"w"` and `"a"`. Omitted, `write` targets the highest-numbered
                existing entry -- see `write` for the full resolution.

        Raises:
            ValueError: If `entry_number` is given for a read, or is not a
                positive integer.
        """
        if entry_number is not None:
            if mode == "r":
                raise ValueError(
                    "NXstress: `entry_number` selects which entry `write` targets and is "
                    "meaningless for a read; pass it to `read` instead."
                )
            NXstress._validateEntryNumber(entry_number)

        self._path = str(file_path)
        self._mode = mode
        self._entry_number = entry_number
        # Set when a write detects a violated invariant about the target entry's
        # contents. The instance is then unusable: continuing would be writing
        # against an assumption already known to be false.
        self._invalid = False
        self._nx = NXFile(self._path, self._mode)  # low-level handle
        self._root = None  # will *ONLY* be set in __enter__

    def __enter__(self) -> "NXstress":
        self._root = nxopen(self._path, self._mode)
        if self._root is None:
            raise RuntimeError(
                f"Unexpected `nexusformat` error opening '{self._path}' "
                f"for {'read' if 'r' in self._mode else 'write'}."
            )
        self._root.__enter__()

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._root:
            self._root.__exit__(exc_type, exc, tb)
            self._root = None

        # Do not suppress exceptions
        return False

    def write(
        self,
        wss: list[HidraWorkspace],
        peakss: list[list[PeakCollection]],
        *,
        entry_number: int | None = None,
    ):
        """Write a new `NXentry`, or grow one that is already in the file.

        Which of the two happens is decided by whether the targeted entry
        already exists -- not by the file mode:

        | `entry_number` | targets | effect |
        |---|---|---|
        | omitted | the highest-numbered existing entry | **append** |
        | an existing entry | that entry | **append** |
        | no existing entry, `<= max + 1` | a new entry there | fresh write |
        | omitted, file has no entries | entry 1 | fresh write |

        An append is a *tail-append*: the batch is sorted among itself and added
        after each dataset's current end, with nothing already on disk read
        back, reordered or rewritten. Every check runs before the first resize,
        so a refused append leaves the file byte-for-byte unchanged.

        Args:
            wss: Workspaces to write or append, in the order their rows go on disk.
            peakss: One list of `PeakCollection` per workspace, same order. A
                workspace with no peak fits is `[[]]`, not `[]`.
            entry_number: Per-call override of the instance default; 1-based.

        Raises:
            ValueError: If the inputs are inconsistent (mismatched list lengths,
                a missing required log, overlapping scan points between inputs),
                or if `entry_number` skips past the next free entry.
            RuntimeError: If called outside the context manager, where there is
                no open root to write to; if this instance was already
                invalidated; or, on an append, for a violated invariant -- a
                duplicate row, an unmet precondition, or a disagreement with the
                target entry. The last **invalidates this instance**: a later
                `write` on it raises rather than proceeding.
            NotImplementedError: On an append, for Case B -- extending a
                compound key the entry already holds with further scan points.
                That needs a mid-array insertion rather than a tail-append and
                is deliberately out of scope. Also, on a write or an append, if
                a workspace would be read back with scan points that are not
                strictly increasing (see `_refuseUnreadableScanPointOrder`).
                Nothing is written, and the instance stays usable.

        Example:
            >>> with NXstress(path, "a") as nxs:        # doctest: +SKIP
            ...     nxs.write([ws], [peaks])            # grow the last entry
        """
        # Write the _next_ NXentry to the file:
        #
        # -- multiple NXentry are allowed by the NXstress schema.
        # -- each NXentry includes:
        #
        #   -- [optional] input_data: raw detector counts, indexed by 'scan_point' (aka: 'subrun');
        #   -- the `NXinstrument`, including its `NXdetector`, applicable `NXtransformations`
        #      and detector and solid-angle masks;
        #   -- a canonical PEAKS instance:
        #
        #      -- peaks are indexed by: phase, (h, k, l), mask, <scan point>
        #         (no duplicate entries are allowed)
        #
        #   -- reduced 'diffraction_data' sections corresponding to the PEAKS entries:
        #
        #     -- peak-fit details, indexed as for the PEAKS indices;
        #     -- normalized and reduced data for each mask, indexed by 'scan_point';
        #     -- a calculated model spectrum: this section is still in progress.
        #

        #   -- more than one `HidraWorkspace` may share one NXentry: their rows are
        #      concatenated in workspace order, and the boundary between them is
        #      recovered on read from the discriminator columns named by
        #      `nxstress.discriminator_fields` (see `_discriminator.py`).

        ##########################################################
        ## Recommended usage:                                   ##
        ## -----------------------------------------------------##
        ## from pyrs/utilities/NXstress import NXstress         ##
        ## ...                                                  ##
        ## wss: list[HidraWorkspace]                            ##
        ## peakss: list[list[PeakCollection]]                   ##
        ##   -- one inner list per workspace, same order.       ##
        ## ...                                                  ##
        ## # To write the first (, or only) entry:              ##
        ## with NXstress(<file name>.nxs, 'w') as nxS:          ##
        ##     nxS.write([ws], [peaks])                         ##
        ## -----------------------------------------------------##
        ## # A workspace with no peak fits at all:              ##
        ##     nxS.write([ws], [[]])                            ##
        ## -----------------------------------------------------##
        ## # To GROW the last entry of an existing file:        ##
        ## with NXstress(<same file>.nxs, 'a') as nxS:          ##
        ##     nxS.write([ws], [peaks])                         ##
        ## -----------------------------------------------------##
        ## # To grow a specific earlier entry:                  ##
        ## with NXstress(<same file>.nxs, 'a',                  ##
        ##               entry_number=1) as nxS:                ##
        ##     nxS.write([ws], [peaks])                         ##
        ## -----------------------------------------------------##
        ## # To add a NEW entry, name the next free number      ##
        ## # explicitly -- as a constructor default, or per      ##
        ## # call, which is how one session writes two:          ##
        ## with NXstress(<file name>.nxs, 'w') as nxS:          ##
        ##     nxS.write([ws], [peaks0])                         ##
        ##     nxS.write([ws], [peaks1], entry_number=2)         ##
        ##########################################################

        # Which entry a write targets, and whether it grows or creates it, is
        # decided by `entry_number` alone -- not by the file mode. An existing
        # target is grown (spec 04c's tail-append); a target that names no
        # existing entry is created. Overwriting an existing entry's contents is
        # neither, and remains unimplemented.
        if self._root is None:
            raise RuntimeError("Usage error: only usage as context manager is supported!")
        if self._invalid:
            raise RuntimeError(
                "NXstress: this instance already detected a violated invariant about the target "
                "entry's contents and is no longer usable.\n"
                "  Re-open the file rather than continuing against an assumption known to be false."
            )

        if entry_number is not None:
            NXstress._validateEntryNumber(entry_number)
        entry_name, exists = self._resolveTarget(entry_number)
        if not exists:
            # Unreachable by construction -- `_resolveTarget` reports existence from
            # the same root this assigns into -- and kept because the cost of being
            # wrong is an entry silently replaced rather than grown. Overwriting an
            # entry's contents is a third operation, neither append nor create, and
            # is not implemented.
            if entry_name in self._root:
                raise RuntimeError(f"Not implemented: overwriting existing `NXentry` '/{entry_name}'.")
            entry = self.init_group(wss, peakss)
            self._root[entry_name] = entry
            return

        self._append(self._root[entry_name], wss, peakss)

    def entry_numbers(self) -> tuple[int, ...]:
        """The 1-based numbers of the `NXentry` groups this file holds, ascending.

        Derived from the names actually present, not from a count: the count is
        the highest number only when the numbering has no gaps, and a file this
        code did not write is not obliged to oblige. `suffix_from_group_name` is
        the declared inverse of `group_naming_scheme`, so this stays correct if
        the naming changes. Entry 1 is handled separately because its name
        carries no suffix for that inverse to find.

        Public because `read`/`write` take an `entry_number` and nothing else
        exposes which numbers are valid: a caller that can present only *one*
        entry has to be able to ask how many there are before it reads. See
        `pyrs.interface.nxstress_load`.

        Like every other operation on this class, it is only usable **inside the
        context manager** -- it reads the open root, not the path.

        A method rather than a property, deliberately: the answer is a live query
        against the open file and changes within one instance's lifetime, so
        attribute syntax would imply a stability it does not have.

            >>> with NXstress(path, "a") as nxs:     # doctest: +SKIP
            ...     nxs.entry_numbers()              # (1,)
            ...     nxs.write([ws], [peaks], entry_number=2)
            ...     nxs.entry_numbers()              # (1, 2)

        It also raises outside the context, and a property that raises makes an
        object awkward to inspect in a debugger or a `repr`.

        Returns:
            Ascending entry numbers; empty for a file with no entries.

        Raises:
            RuntimeError: If called outside the context manager, where there is
                no open root to inspect.

        Example:
            >>> with NXstress(path, "r") as nxs:      # doctest: +SKIP
            ...     if len(nxs.entry_numbers()) == 1:
            ...         workspaces, peakss = nxs.read()
        """
        root = self._root
        if root is None:
            raise RuntimeError("Usage error: only usage as context manager is supported!")

        numbers = {1} if GROUP_NAME.ENTRY in root else set()
        for name in root:
            if not str(name).startswith(f"{GROUP_NAME.ENTRY}_"):
                # A root may hold things that are not entries; `suffix_from_group_name`
                # raises rather than returning for those.
                continue
            suffix = suffix_from_group_name(str(name), GROUP_NAME.ENTRY)
            if suffix.isdigit():
                numbers.add(int(suffix))
        return tuple(sorted(numbers))

    @staticmethod
    def _validateEntryNumber(entry_number: int) -> None:
        if entry_number < 1:
            raise ValueError(f"NXstress: `entry_number` is 1-based; got {entry_number}.")

    def _resolveTarget(self, entry_number: int | None = None) -> tuple[str, bool]:
        """Name of the `NXentry` this write targets, and whether it already exists.

        Args:
            entry_number: Per-call override of the instance default.

        Returns:
            `(entry_name, exists)`.

        Raises:
            ValueError: If `entry_number` skips past the next free number. A
                number beyond that is far more likely a typo than an intent to
                leave a gap, and the silent outcome -- a stray entry instead of
                the append that was meant -- is not one the caller would notice.
        """
        # Derived from the names actually present, not from a count: the count is
        # the highest number only when the numbering has no gaps, and a file this
        # code did not write is not obliged to oblige. `suffix_from_group_name` is
        # the declared inverse of `group_naming_scheme`, so this stays correct if
        # the naming changes. Entry 1 is handled separately because its name
        # carries no suffix for that inverse to find.
        numbers = set(self.entry_numbers())
        root = self._root
        assert root is not None, "`entry_numbers` raises when there is no open root"
        highest = max(numbers) if numbers else 0

        requested = entry_number if entry_number is not None else self._entry_number
        if requested is None:
            requested = max(highest, 1)
        if requested > highest + 1:
            raise ValueError(
                f"NXstress: `entry_number={requested}` skips past the next free entry.\n"
                f"  This file holds {highest} entry(ies); append to one of them, or pass "
                f"`entry_number={highest + 1}` to start a new one."
            )

        entry_name = group_naming_scheme(GROUP_NAME.ENTRY, requested)
        return entry_name, entry_name in root

    def _append(self, entry, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]):
        """Grow an existing `NXentry` with more scan points (spec 04c).

        A **tail-append**: the incoming batch is sorted among itself and added
        after each dataset's current end. Nothing already on disk is read back
        into `HidraWorkspace`/`PeakCollection` form, reordered or rewritten --
        particularly not the raw detector counts, which are the bulk of an entry.
        The file is therefore "locally sorted, globally segmented" afterwards
        rather than globally sorted, which is all the reader has ever required:
        `_Peaks.peakCollectionRanges` enforces that each compound key occupies
        one contiguous run and that `scan_point` increases within it, and
        nothing more.

        Every check happens in `_classifyAppend`, before any resize, so a
        rejected append leaves the entry byte-for-byte unchanged. That includes
        the checks `tail_append` itself would make -- resizability and
        trailing-axis agreement -- which are asked up front rather than
        discovered part-way through, because by then earlier groups have grown
        and there is no undo.

        Args:
            entry: The target `NXentry`, file-backed and open for writing.
            wss: Workspaces to append.
            peakss: One list of `PeakCollection` per workspace, same order.

        Raises:
            RuntimeError: On a violated invariant -- a duplicate row, a missing
                precondition, or a disagreement with the entry. Invalidates this
                instance.
            NotImplementedError: For Case B, extending a compound key already on
                disk with further scan points. That needs a mid-array insertion
                rather than a tail-append and is deliberately out of scope. Also
                for an append that would leave a workspace's scan points out of
                order on read. The instance stays usable.
        """
        peaks_group = entry[GROUP_NAME.PEAKS]
        discriminator_names = _discriminator.names_for_read(peaks_group)

        indexed = [
            IndexedPeaks(_discriminator.key(ws, discriminator_names), collection, ws._sample_logs)
            for ws, collections in zip(wss, peakss)
            for collection in collections
        ]

        self._classifyAppend(entry, wss, peakss, indexed, discriminator_names)

        # Past this point the entry is being mutated group by group, and there is
        # no way to undo a partial one. Anything that escapes here has already
        # left the entry desynchronized -- rows grown in some groups and not
        # others -- which reads back without error and is worse than a crash. The
        # instance is therefore invalidated on ANY escape, not only on the
        # violated-invariant ones `_classifyAppend` raises: a caller must not be
        # able to append again onto a damaged entry.
        logss = [ws._sample_logs for ws in wss]
        try:
            self._appendEntryTimes(entry, wss)
            _InputData.init_group(wss, data=entry[GROUP_NAME.INPUT_DATA])
            _Instrument.init_group(wss, data=entry[GROUP_NAME.INSTRUMENT])
            _Sample.init_group(logss, data=entry[GROUP_NAME.SAMPLE_DESCRIPTION])
            _Fit.init_group(wss, indexed, logss, data=entry[GROUP_NAME.FIT])
            _Peaks.init_group(indexed, logss[0], discriminator_names, data=peaks_group)
        except BaseException:
            self._invalid = True
            raise

    def _classifyAppend(
        self,
        entry,
        wss: list[HidraWorkspace],
        peakss: list[list[PeakCollection]],
        indexed: list,
        discriminator_names: tuple,
    ) -> None:
        """Decide whether an append may proceed, touching nothing.

        Runs to completion before the first resize. That ordering is what makes
        both rejection outcomes true no-ops, and it is the reason every check
        that could fail part-way through an append lives here rather than at the
        point of use -- including the ones `tail_append` would otherwise make
        during mutation. Kept as a readable list of calls for the same reason:
        this is the method a reader has to follow in full to believe the no-op
        property, so it is the one that must stay short.

        Raises:
            RuntimeError: On any violated invariant; also sets `self._invalid`.
            NotImplementedError: For Case B, or for an out-of-order read-back.
                Leaves the instance usable.
        """
        # Ordinary write-time validation applies to the incoming batch regardless
        # of where it is going.
        try:
            self._validateWorkspaceAndPeaksData(wss, peakss, indexed)
        except ValueError as error:
            raise self._reject(f"NXstress: cannot append -- {error}") from error

        self._rejectUnmetCaseAPreconditions(peakss, discriminator_names)
        self._rejectUnconfiguredColumns(entry, discriminator_names)
        self._rejectDiscriminatorTypeMismatch(entry, indexed, discriminator_names)
        self._rejectUngrowableOrDisagreeingGroups(entry, wss)
        self._classifyCompoundKeys(entry, indexed, discriminator_names)
        self._rejectScanPointCollisions(entry, wss)
        self._refuseUnreadableAppendOrder(entry, wss, discriminator_names)
        self._rejectDiffractogramKeyMismatch(entry, wss)
        self._rejectFitModelMismatch(entry, indexed)

    def _reject(self, message: str) -> RuntimeError:
        """Build the exception for a violated invariant, and invalidate the instance.

        Returned rather than raised so each call site reads `raise self._reject(...)`
        and the traceback starts there.
        """
        self._invalid = True
        return RuntimeError(message)

    def _rejectUnmetCaseAPreconditions(self, peakss: list, discriminator_names: tuple) -> None:
        """Case A's two preconditions, inherited from 04b's write-time invariants."""
        # -- Case A precondition 1: every incoming workspace must contribute at
        #    least one `PeakCollection`. `_validateMultiWorkspace` already enforces
        #    this, but only for N > 1; on append it binds at N == 1 too, because
        #    the entry it joins holds other workspaces whose rows must stay
        #    distinguishable from these.
        empty = [n for n, collections in enumerate(peakss) if not collections]
        if empty:
            raise self._reject(
                f"NXstress: cannot append -- input workspace(s) {empty} contribute no "
                "`PeakCollection`.\n"
                "  The peak index is the only place a workspace's identity is recorded, so one "
                "with no peaks could not be recovered from the appended entry."
            )

        # -- Case A precondition 2: the target must already have a discriminator
        #    scheme. Attaching a new, distinguishable workspace to an entry written
        #    without one would mean adding an on-disk column. That is mechanically
        #    possible (see `probes/a4_h5py_nexusformat_append.py`) but it is schema
        #    restructuring, which this pass excludes -- a scope decision, not a
        #    format limit. See the plan's Decisions Log item 22.
        if not discriminator_names:
            raise self._reject(
                "NXstress: cannot append -- the target entry carries no discriminator columns, so "
                "an appended workspace could not be told apart from the one(s) already there.\n"
                "  Configure 'nxstress.discriminator_fields' and write the entry afresh; this pass "
                "does not add a column to an entry that lacks one."
            )

    def _rejectUnconfiguredColumns(self, entry, discriminator_names: tuple) -> None:
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
            raise self._reject(
                f"NXstress: cannot append -- this entry's `peaks` group has column(s) {unknown}, "
                f"which are neither PyRS's own nor among the discriminator fields "
                f"{list(discriminator_names)}.\n"
                "  An append grows only the columns it knows, so these would be left short. "
                "Write a new entry instead."
            )

    def _rejectDiscriminatorTypeMismatch(self, entry, indexed: list, discriminator_names: tuple) -> None:
        """Incoming discriminator values must fit the type of the column already on disk.

        The column's type was fixed by the entry's first write. A text value
        appended to a numeric column (or the reverse) fails inside `_Peaks`'
        append, after other groups have grown; a float appended to an integer
        column would be truncated without any error.
        """
        peaks_group = entry[GROUP_NAME.PEAKS]
        for name in discriminator_names:
            on_disk = peaks_group[_discriminator.column_name(name)].dtype
            incoming = np.asarray([dict(item.discriminators)[name] for item in indexed]).dtype
            if not appendable_dtype(incoming, on_disk):
                raise self._reject(
                    f"NXstress: cannot append -- discriminator field '{name}' is stored as {on_disk} "
                    f"in this entry, and the incoming value(s) are {incoming}.\n"
                    "  A discriminator must keep the type the entry was first written with."
                )

    def _rejectUngrowableOrDisagreeingGroups(self, entry, wss: list[HidraWorkspace]) -> None:
        """Everything `tail_append` would refuse, and every per-group disagreement.

        Asked before the first resize rather than during mutation: a refusal
        discovered part-way through leaves the entry desynchronized, and such an
        entry reads back without error. See 04c's Follow-up 3 F3.1.
        """
        # -- Every dataset this append will grow must be able to grow, and must
        #    agree with the incoming rows on its trailing axes. Asked here rather
        #    than discovered inside `tail_append`, which runs during mutation:
        #    by then earlier groups have grown, the entry is desynchronized, and
        #    it reads back without error. See 04c's Follow-up 3 F3.1.
        #
        # -- Per-group agreement with the existing entry. Each of these lives on the
        #    group that owns the rule and is called here, before any resize, so that a
        #    mismatch is a true no-op rather than a half-grown entry.
        #
        # The groups raise a plain `RuntimeError`; re-raised through `_reject` so that
        # these refusals invalidate the instance like every other violated invariant.
        try:
            self._validateAppendableShapes(entry, wss)
            _Instrument.validateAppend(wss, entry[GROUP_NAME.INSTRUMENT])
            _InputData.validateAppend(wss, entry[GROUP_NAME.INPUT_DATA])
            _Sample.validateAppend([ws._sample_logs for ws in wss], entry[GROUP_NAME.SAMPLE_DESCRIPTION])
            _Fit.validateAppend(wss, entry[GROUP_NAME.FIT])
        except NotImplementedError:
            # A `RuntimeError` subclass, but "unsupported" rather than "violated",
            # and so not a reason to invalidate the instance.
            raise
        except RuntimeError as error:
            raise self._reject(str(error)) from error

    def _classifyCompoundKeys(self, entry, indexed: list, discriminator_names: tuple) -> None:
        """Case A (proceed), Case B (unsupported) or duplicate (violated invariant)."""
        # -- Case A / Case B / duplicate, from the entry's existing peak index.
        existing_rows: dict = {}
        for discriminators, key, start, end in _Peaks.peakCollectionRanges(
            entry[GROUP_NAME.PEAKS], discriminator_names
        ):
            points = entry[GROUP_NAME.PEAKS]["scan_point"].nxdata[start:end]
            existing_rows[(discriminators, key)] = set(np.asarray(points).tolist())

        for item in indexed:
            compound = (item.discriminators, _Peaks.PeakIndex.sort_key(item.collection))
            if compound not in existing_rows:
                continue  # Case A -- a key the entry does not hold yet.
            incoming = set(item.collection.sub_runs.raw_copy().tolist())
            shared = incoming & existing_rows[compound]
            if shared:
                raise self._reject(
                    f"NXstress: cannot append -- scan point(s) {sorted(shared)} are already present "
                    f"in this entry under {compound[1]}"
                    + (f" for {dict(compound[0])}" if compound[0] else "")
                    + ".\n  An appended row may not duplicate one already committed."
                )
            raise NotImplementedError(
                f"NXstress: appending further scan points to the compound key {compound[1]}"
                + (f" for {dict(compound[0])}" if compound[0] else "")
                + ", which this entry already holds, is not supported.\n"
                "  Each key must occupy one contiguous run, so new scan points under an existing "
                "key would have to be inserted into the middle of it -- a different operation from "
                "the tail-append this implements. Appending a NEW workspace is supported."
            )

    def _rejectScanPointCollisions(self, entry, wss: list[HidraWorkspace]) -> None:
        """The scan-point axis must stay partitioned across the whole entry."""
        # -- The scan-point axis must stay partitioned: the reader attributes rows
        #    to a workspace by scan-point value, so a value appearing twice in one
        #    entry cannot be attributed at all.
        on_disk_points = set(np.asarray(entry[GROUP_NAME.SAMPLE_DESCRIPTION]["scan_point"].nxdata).tolist())
        for n, ws in enumerate(wss):
            collisions = on_disk_points & set(ws.get_sub_runs().raw_copy().tolist())
            if collisions:
                raise self._reject(
                    f"NXstress: cannot append -- scan point(s) {sorted(collisions)} of input "
                    f"workspace [{n}] are already in this entry.\n"
                    "  Workspaces sharing an entry must cover disjoint scan points: the reader "
                    "recovers each workspace's rows by scan-point value, so an overlap cannot be "
                    "attributed."
                )

    def _refuseUnreadableAppendOrder(self, entry, wss: list[HidraWorkspace], discriminator_names: tuple) -> None:
        """The appended entry must still read back with increasing scan points per workspace.

        An appended workspace sharing a discriminator value with rows already on
        disk is read back together with them, after them; so are all rows of an
        entry holding a single discriminator value. A tail-append of lower scan
        points is then unreadable. "Unsupported" rather than "violated", like
        Case B, so the instance stays usable.
        """
        points_by_key: dict = {}
        peaks_group = entry[GROUP_NAME.PEAKS]
        for discriminators, _key, start, end in _Peaks.peakCollectionRanges(peaks_group, discriminator_names):
            points_by_key.setdefault(discriminators, set()).update(
                np.asarray(peaks_group["scan_point"].nxdata[start:end]).tolist()
            )
        for ws in wss:
            points_by_key.setdefault(_discriminator.key(ws, discriminator_names), set()).update(
                ws.get_sub_runs().raw_copy().tolist()
            )
        axis = np.concatenate(
            [np.asarray(entry[GROUP_NAME.SAMPLE_DESCRIPTION]["scan_point"].nxdata)]
            + [ws.get_sub_runs().raw_copy() for ws in wss]
        )
        self._refuseUnreadableScanPointOrder(axis, points_by_key, discriminator_names, operation="append")

    def _rejectDiffractogramKeyMismatch(self, entry, wss: list[HidraWorkspace]) -> None:
        """Every scan point in an entry needs a diffractogram under each key.

        Compares **diffractogram keys**, not detector-mask names. The two are
        different sets: a key is a reduction identifier -- `eta_-5.0`,
        `mask_a_eta_-5.0` -- that references a detector mask without being one,
        and only a reduction that used the default mask *alone* produces a bare
        `DEFAULT_TAG` key. Reading this side through `nxstress_mask_names`, which
        injects `DEFAULT_TAG` because every reduction uses the default mask,
        therefore added a key the writer never emits -- and made every texture
        append impossible. See 04's Follow-up 8.
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
            raise self._reject(
                f"NXstress: cannot append -- the incoming workspaces' reduced-diffraction keys do "
                f"not match the target entry's.\n"
                f"  On disk:   {sorted(on_disk_keys)}\n"
                f"  Incoming:  {sorted(incoming_keys)}\n"
                "  Every scan point in an entry must have a diffractogram under each key."
            )

    def _rejectFitModelMismatch(self, entry, indexed: list) -> None:
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
                raise self._reject(
                    "NXstress: cannot append -- the target entry holds no `PeakCollection`s, so it "
                    "records no fit model and no discriminator values for the rows already in it.\n"
                    "  An appended workspace could not be told apart from them. Write a new entry."
                )
            if incoming_titles - {str(on_disk_title)}:
                raise self._reject(
                    f"NXstress: cannot append -- the incoming `PeakCollection`s' {label} "
                    f"{sorted(incoming_titles)} does not match the target entry's "
                    f"'{on_disk_title}'.\n"
                    "  All peak collections in one entry share a single fit model."
                )

    @classmethod
    def _appendableDatasets(cls, entry) -> dict:
        """Every dataset in `entry` that an append grows, declared by its owner.

        Each grouping module reports the datasets *it* writes row-aligned, because
        each one owns its group's layout. Inside a group NXstress writes, a field
        is either row-aligned -- and therefore `growable` -- or an entry-wide
        scalar, so a field added to such a group later is covered without editing
        anything here.

        This replaced a sweep of the whole entry for `shape[0] == n_scan`, which
        was wrong in both directions. It **over**-matched, and over-matching is
        not the harmless direction it was documented to be: the result feeds a
        resizability *requirement*, so any entry-wide array that happened to be
        `n_scan` long rejected a legal append. Three detector masks and three scan
        points was enough -- `masks/names` is entry-wide, is never grown, and is
        required by `_Instrument.validateAppend` to stay constant, yet the append
        was refused with a message blaming an out-of-date writer. It also
        **under**-matched: the peak index is one row per (compound key, scan
        point), so it equals `n_scan` only while each workspace fits exactly one
        peak. Fit a second phase -- the ordinary multi-peak case -- and `peaks/*`
        and both parameter groups dropped out of the pre-flight entirely, leaving
        exactly the mid-mutation refusal that 04c's Follow-up 3 F3.1 hoisted out.
        See 04c's Follow-up 5.

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

    @classmethod
    def _validateAppendableShapes(cls, entry, wss: list[HidraWorkspace]) -> None:
        """Every dataset the append will grow can grow, without growing any of them.

        Two distinct refusals, both of which `tail_append` would otherwise make
        during mutation:

        1. **Resizability.** A dataset written without `maxshape` is contiguous
           and can never be extended -- a file from a PyRS predating `growable`.
        2. **Trailing-axis agreement**, delegated to the groups that own the
           two-dimensional data, since only they know what the incoming rows
           would be shaped like.

        Which datasets those are is likewise asked of the groups that wrote them
        (`_appendableDatasets`), not inferred from array length.

        Args:
            entry: The target `NXentry`.
            wss: Workspaces being appended.

        Raises:
            RuntimeError: Naming every dataset that cannot grow.
        """
        fixed = [path for path, field in cls._appendableDatasets(entry).items() if not appendable(field)]
        if fixed:
            raise RuntimeError(
                f"NXstress: cannot append -- {len(fixed)} dataset(s) in this entry were written at a "
                f"fixed size and cannot be extended: {sorted(fixed)}.\n"
                "  Appending requires every per-scan-point dataset to have been created resizable "
                "(`maxshape`/`chunks`). A file written by a PyRS predating that change cannot be "
                "appended to -- write a new entry instead."
            )

    def _appendEntryTimes(self, entry, wss: list[HidraWorkspace]) -> None:
        """Extend the entry-level `start_time`/`end_time` arrays.

        These are per-scan-point, exactly like the groups below them -- the one
        member of the scan-point family that lives directly on the `NXentry`.
        """
        start_times, end_times = self._entryTimes(wss)
        tail_append(entry["start_time"], np.asarray(start_times, dtype=object))
        tail_append(entry["end_time"], np.asarray(end_times, dtype=object))

    def read(self, entry_number: int = 1):
        """Read back the workspaces and peak collections of one NXstress NXentry.

        Parameters
        ----------
        entry_number : int
            Which NXentry to read (1-based). Default is 1.

        Returns
        -------
        tuple
            `(list[HidraWorkspace], list[list[PeakCollection]])`, parallel and
            in the same order -- the same shape `write` accepts. An entry
            that records no workspace boundary yields one of each.

        Raises
        ------
        ValueError
            If `entry_number` is not a positive integer.
        RuntimeError
            If called outside the context manager, where there is no open root
            to read; if the entry's discriminator columns disagree with
            `nxstress.discriminator_fields`; or if the recovered scan-point
            sets do not partition the entry's scan-point axis.
        """
        # Verify context manager is active
        if self._root is None:
            raise RuntimeError("Usage error: only usage as context manager is supported!")
        # `group_naming_scheme` names entry 1 `entry` and every other number
        # `entry_<n>`, so 0 or a negative number would silently read entry 1.
        NXstress._validateEntryNumber(entry_number)

        # Resolve entry name
        entry_name = group_naming_scheme(GROUP_NAME.ENTRY, entry_number)

        # Access entry
        entry = self._root[entry_name]

        # Read peak collections FIRST: in a multi-workspace entry they are the
        # only record of where one workspace's rows end and the next begin, so
        # nothing else can be split until they have been read and grouped.
        indexed = []
        discriminator_names: tuple = ()
        if GROUP_NAME.PEAKS in entry:
            peaks_group = entry[GROUP_NAME.PEAKS]
            discriminator_names = _discriminator.names_for_read(peaks_group)
            if GROUP_NAME.FIT in entry:
                indexed = _Peaks.peakCollectionsFromNexus(peaks_group, entry[GROUP_NAME.FIT], discriminator_names)

        scan_point_axis = entry[GROUP_NAME.SAMPLE_DESCRIPTION]["scan_point"].nxdata
        selections = self._workspaceSelections(indexed, scan_point_axis, discriminator_names)

        # Instrument geometry, detector shift and masks are entry-wide.
        default_mask, mask_dict = _Masks.masksFromNexus(entry[GROUP_NAME.INSTRUMENT][GROUP_NAME.MASKS])

        workspaces = []
        peak_collections = []
        for rows, discriminators, collections in selections:
            ws = self._workspaceFromNexus(entry, rows, default_mask, mask_dict)
            for name, value in discriminators:
                _discriminator.apply(ws, name, value)
            workspaces.append(ws)
            peak_collections.append(collections)

        return workspaces, peak_collections

    @classmethod
    def _workspaceSelections(cls, indexed, scan_point_axis, discriminator_names: tuple):
        """Split one entry into per-workspace (rows, discriminators, collections).

        Each workspace's scan-point *set* is the union of its peak collections'
        scan points, and its rows of the scan-point family are selected by
        membership in that set -- not by position. That is what makes the read
        independent of the order `write` happened to concatenate the inputs in.

        Returns
        -------
        list[tuple]
            `(rows, discriminators, collections)` per workspace, where `rows` is
            a boolean mask over `scan_point_axis`, or `None` for a
            single-workspace entry (which needs no selection at all).
        """
        groups: dict = {}
        for item in indexed:
            groups.setdefault(item.discriminators, []).append(item.collection)

        if not discriminator_names or len(groups) <= 1:
            # One workspace: a pre-04b file, a single-workspace write, or a file
            # from the since-removed `merge_workspaces`, all of which read back whole -- no rows
            # need selecting. Its discriminators are still carried through, so a
            # single-workspace save round-trips its value like any other.
            discriminators, collections = next(iter(groups.items())) if groups else ((), [])
            return [(None, discriminators, collections)]

        selections = []
        claimed = np.zeros(len(scan_point_axis), dtype=bool)
        for discriminators, collections in groups.items():
            points = set()
            for collection in collections:
                points |= set(collection.sub_runs.raw_copy().tolist())
            rows = np.isin(scan_point_axis, sorted(points))
            overlap = claimed & rows
            if overlap.any():
                raise RuntimeError(
                    f"NXstress: scan points {sorted(set(np.asarray(scan_point_axis)[overlap].tolist()))} "
                    f"are claimed by more than one workspace in this entry "
                    f"(at {dict(discriminators)}).\n"
                    "  Input workspaces must cover disjoint scan points; this file cannot be split."
                )
            claimed |= rows
            selections.append((rows, discriminators, collections))

        if not claimed.all():
            unclaimed = sorted(set(np.asarray(scan_point_axis)[~claimed].tolist()))
            raise RuntimeError(
                f"NXstress: scan points {unclaimed} belong to no workspace in this entry.\n"
                "  Every scan point must be covered by some input workspace's `PeakCollection`s."
            )
        return selections

    @classmethod
    def _workspaceFromNexus(cls, entry, rows, default_mask, mask_dict) -> HidraWorkspace:
        """Reconstruct one `HidraWorkspace` from its rows of a (possibly shared) entry."""
        # Read sample logs. `rows` is applied inside, before any `SubRuns` is
        # built: the entry's concatenated scan-point axis need not be
        # monotonic, and `SubRuns` rejects a non-monotonic array outright.
        sample_logs = _Sample.sampleLogsFromNexus(entry[GROUP_NAME.SAMPLE_DESCRIPTION], rows)

        # Read instrument
        geometry, shift, wavelength = _Instrument.instrumentFromNexus(entry[GROUP_NAME.INSTRUMENT], rows)
        is_calibrated = shift is not None

        # Build workspace
        ws = HidraWorkspace()
        ws.set_sample_logs_from_object(sample_logs)
        # `set_wavelength` expects `dict[int, float]`
        ws.set_wavelength(
            {subrun_index: wavelength[n] for n, subrun_index in enumerate(ws.get_sub_runs())}, is_calibrated
        )
        ws.set_instrument_geometry(geometry)
        if shift is not None:
            ws.set_detector_shift(shift)
        ws.set_masks_from_dict(default_mask, mask_dict)

        # Read raw counts if present
        if GROUP_NAME.INPUT_DATA in entry:
            _InputData.readSubruns(ws, entry[GROUP_NAME.INPUT_DATA], rows)

        # Read reduced diffraction data from FIT group's DIFFRACTOGRAM subgroups
        if GROUP_NAME.FIT in entry:
            fit_group = entry[GROUP_NAME.FIT]
            diff_data = {}
            var_data = {}
            two_theta_matrix = None

            for child_name in fit_group:
                child = fit_group[child_name]
                if not isinstance(child, NXdata):
                    continue
                mask_name = suffix_from_group_name(child_name, GROUP_NAME.DIFFRACTOGRAM)
                scan_pts, two_theta, data, errors = _Diffractogram.diffractogramFromNexus(child, rows)

                # The two-theta axis is entry-wide, so take it from whichever group
                # comes first -- including one whose data is dropped below.
                if two_theta_matrix is None:
                    two_theta_matrix = two_theta

                # A mask that is entirely NaN for THIS workspace's rows is one it
                # never reduced. One DIFFRACTOGRAM group spans the whole entry, so
                # when one input reduced a mask and another did not,
                # `_concatenated_diffraction` NaN-fills the second's rows to keep the
                # scan-point axis aligned. Handing those rows back would give the
                # workspace a mask it never had: `reduction_masks` would gain an
                # entry, and that property is *counted* by
                # `texture_fitting_crtl` and `mantid_peakfit_calibration`.
                #
                # The cost is that a reduction which genuinely produced nothing but
                # NaN is indistinguishable from one that never ran, and is dropped
                # too. That trade is deliberate: the alternative silently changes a
                # workspace's mask set across a round trip.
                if data.size and np.isnan(data).all():
                    logger.info(
                        f"NXstress: mask '{mask_name}' is entirely NaN for this workspace's scan "
                        f"points -- it was reduced by another input in this entry, not by this one. "
                        f"Omitting it rather than reporting a mask this workspace does not have."
                    )
                    continue

                # Map DEFAULT_TAG to None for workspace dict keys
                ws_mask_key = None if mask_name == DEFAULT_TAG else mask_name
                diff_data[ws_mask_key] = data

                # NOTE: Despite the field name 'diffractogram_errors',
                #   variance values (not standard errors) are stored in this field.
                var_data[ws_mask_key] = errors

            if two_theta_matrix is not None:
                ws.set_reduced_diffraction_data_set(two_theta_matrix, diff_data, var_data)

        return ws

    ############################################
    # ALL non-context-manager related methods ##
    #   must be `classmethod`.                ##
    ############################################

    @classmethod
    @validate_call_
    def _validateWorkspaceAndPeaksData(
        cls, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]], indexed: list
    ):
        if len(peakss) != len(wss):
            raise ValueError(
                f"NXstress.write expects one list of `PeakCollection` per workspace: "
                f"got {len(wss)} workspace(s) and {len(peakss)} peak list(s)."
            )
        if not wss:
            raise ValueError("NXstress.write requires at least one `HidraWorkspace`.")

        for n, ws in enumerate(wss):
            # VERIFY that all required logs are present.
            logs = ws.sample_log_names
            for k in REQUIRED_LOGS:
                if k not in logs:
                    raise ValueError(f"NXstress requires log '{k}', which is not present in workspace [{n}]")

        if len(wss) > 1:
            cls._validateMultiWorkspace(wss, peakss)

        # VERIFY that no duplicate PeakCollections exist
        _Peaks.validateNoDuplicatePeaks(indexed)

        # VERIFY that any <scan point> or <mask> referenced by any `PeakCollection` is included in the workspace.
        for ws, peaks in zip(wss, peakss):
            _Fit.validateWorkspaceAndPeaksData(ws, peaks)

    @classmethod
    def _validateMultiWorkspace(cls, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]):
        """The invariants that only bite when an entry holds N > 1 workspaces.

        The first three are write-time gates rather than documented conventions,
        because violating any of them produces a file that is *readable* and
        wrong rather than one that fails: an unsplittable entry, or one whose
        rows are attributed to the wrong workspace. The fourth guards a file
        that would be written correctly and then could not be read back.
        """
        # 1. Without a discriminator, the boundary between inputs is not recorded
        #    anywhere. NXstress does not merge workspaces: that is PyRS's job,
        #    done before the write (Decisions Log row 45).
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

        # 3. The read side recovers each workspace by scan-point value membership,
        #    which cannot distinguish two workspaces that share a scan point.
        seen: dict = {}
        for n, ws in enumerate(wss):
            for point in ws.get_sub_runs().raw_copy().tolist():
                if point in seen:
                    raise ValueError(
                        f"NXstress.write: scan point {point} appears in both input workspace "
                        f"[{seen[point]}] and [{n}].\n"
                        "  Input workspaces sharing an entry must cover disjoint scan points: the "
                        "reader recovers each workspace's rows by scan-point value, so an overlap "
                        "cannot be attributed."
                    )
                seen[point] = n

        # 4. Each workspace read back must hold its scan points in increasing order.
        discriminator_names = cls._discriminatorNames(wss)
        points_by_key: dict = {}
        for ws in wss:
            points_by_key.setdefault(_discriminator.key(ws, discriminator_names), set()).update(
                ws.get_sub_runs().raw_copy().tolist()
            )
        axis = np.concatenate([ws.get_sub_runs().raw_copy() for ws in wss])
        cls._refuseUnreadableScanPointOrder(axis, points_by_key, discriminator_names, operation="write")

    @staticmethod
    def _refuseUnreadableScanPointOrder(
        axis: np.ndarray, points_by_key: dict, discriminator_names: tuple, *, operation: str
    ) -> None:
        """Refuse an entry whose read-back workspaces would hold out-of-order scan points.

        `read` gives each workspace the rows whose scan points it claims, in the
        order they are on disk, and builds a `SubRuns` from them -- which refuses
        anything not strictly increasing. Interleaved inputs are therefore fine
        when a discriminator separates them, but workspaces sharing one
        discriminator value are read back as one, and their concatenated axis
        must increase.

        Refused rather than sorted: the on-disk row order is the inputs' order,
        and re-sorting the rows would break the round trip -- there would
        then be no way to re-assign them to the individual workspaces on read.
        The requirement itself is arbitrary, and its removal is recommended in
        DD-2 of `plans/PR_review/deferred-defects.md`.

        Args:
            axis: The entry's whole scan-point axis, as it would be on disk.
            points_by_key: Each discriminator key's scan points, mirroring how
                `_workspaceSelections` groups the entry on read.
            discriminator_names: The entry's discriminator fields; empty for a
                single-workspace entry.
            operation: "write" or "append", for the message.

        Raises:
            NotImplementedError: If any read-back workspace's scan points would
                not be strictly increasing. Nothing has been written, and an
                appending instance stays usable.
        """
        if not discriminator_names or len(points_by_key) <= 1:
            selections = [axis]
        else:
            selections = [axis[np.isin(axis, sorted(points))] for points in points_by_key.values()]
        for points in selections:
            if np.any(points[1:] <= points[:-1]):
                raise NotImplementedError(
                    f"NXstress: cannot {operation} -- one workspace would be read back from this entry "
                    f"with scan points {points.tolist()}, in that order, and PyRS requires a "
                    "workspace's scan points to be strictly increasing.\n"
                    "  Pass the workspaces in increasing scan-point order, or give them distinct "
                    "discriminator values so that they are read back separately."
                )

    @classmethod
    def _entryTimes(cls, wss: list[HidraWorkspace]) -> tuple[list[str], list[str]]:
        """Per-scan-point ISO-8601 start and end times, concatenated across inputs.

        Shared by the fresh-write path and the append path, which must produce
        rows of exactly the same form -- an append writes into the array the
        first write created.

        Args:
            wss: Input workspaces, in the order their rows are written.

        Returns:
            `(start_times, end_times)`, each one entry per scan point.
        """
        n_scan_point = sum(len(ws._sample_logs.subruns) for ws in wss)
        try:
            start_times: list[str] = [
                datetime.fromisoformat(to_text(t)).astimezone().isoformat()
                for ws in wss
                for t in ws.get_sample_log_values("start_time")
            ]
            end_times: list[str] = [
                datetime.fromisoformat(to_text(t)).astimezone().isoformat()
                for ws in wss
                for t in ws.get_sample_log_values("end_time")
            ]
        except ValueError as e:
            if "Invalid isoformat string" not in str(e):
                raise
            logger.warning(
                f"Log entries for sub-run start and end times are not in ISO-8601 format:\n"
                f"  in order to continue writing, a value of '{NO_LOG}' will be used for all time entries!"
            )
            start_times = end_times = [NO_LOG] * n_scan_point
        return start_times, end_times

    @classmethod
    @validate_call_
    def _init(cls, wss: list[HidraWorkspace]) -> NXentry:
        # Create the NXentry and initialize any required attributes.

        """
        ├─ definition                             (dataset: "NXstress")
        ├─ start_time                             (dataset: ISO8601 string)
        ├─ end_time                               (dataset: ISO8601 string)
        ├─ processing_type                        (dataset: string)
        :: apart from 'definition', these fields may also be
             lists by subrun.
        """
        entry = NXentry()
        entry["definition"] = "NXstress"

        start_times, end_times = cls._entryTimes(wss)
        # `growable`: these two are per-scan-point, so an append has to extend them
        # in lockstep with the rest of the scan-point family.
        entry["start_time"] = NXfield(start_times, **growable(1))
        entry["end_time"] = NXfield(end_times, **growable(1))

        # the type of the primary strain calculation:
        #   this might also be 'two-theta', but 'd-spacing' seems more likely
        entry["processing_type"] = "d-spacing"

        return entry

    @classmethod
    def _discriminatorNames(cls, wss: list[HidraWorkspace]) -> tuple:
        """Discriminator fields to write this entry with.

        Whenever any field is configured, its column is written -- `N == 1`
        included. That keeps the read-side cross-check a plain equality test,
        and lets a single-workspace save round-trip its discriminator value
        the same way a multi-workspace one does.
        """
        return _discriminator.field_names()

    @classmethod
    @validate_call_
    def init_group(cls, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]) -> NXentry:
        # Create and initialize a single NXstress-compatible NXentry tree:
        #   _multiple_ NXentry can exist within an NXstress-compatible HDF5 file.
        #   For example, distinct entries might be added for each set of
        #   data-reduction or sample conditions.
        #
        #   One NXentry may also hold _multiple_ `HidraWorkspace`: their rows are
        #   concatenated in workspace order throughout, and told apart on read by
        #   the discriminator columns on the peak index.

        discriminator_names = cls._discriminatorNames(wss)

        # One flattened list of peak collections, each tagged with its own
        # workspace's discriminator values and sample logs. This is what keeps
        # the three position-aligned groups (PEAKS, peak parameters, background
        # parameters) in the same order as each other.
        indexed = [
            IndexedPeaks(_discriminator.key(ws, discriminator_names), collection, ws._sample_logs)
            for ws, collections in zip(wss, peakss)
            for collection in collections
        ]

        # Verify that all data required by NXstress are present.
        cls._validateWorkspaceAndPeaksData(wss, peakss, indexed)

        # Initialize this NXentry, and add required attributes.
        entry = cls._init(wss)

        # 'input_data' group
        entry[GROUP_NAME.INPUT_DATA] = _InputData.init_group(wss)

        # 'instrument' group
        entry[GROUP_NAME.INSTRUMENT] = _Instrument.init_group(wss)

        # 'SAMPLE_DESCRIPTION' group
        logss = [ws._sample_logs for ws in wss]
        entry[GROUP_NAME.SAMPLE_DESCRIPTION] = _Sample.init_group(logss)

        # 'FIT' group
        entry[GROUP_NAME.FIT] = _Fit.init_group(wss, indexed, logss)

        # 'PEAKS' group
        entry[GROUP_NAME.PEAKS] = _Peaks.init_group(indexed, logss[0], discriminator_names)

        return entry
