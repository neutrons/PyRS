"""
pyrs/utilities/NXstress/NXstress.py

Primary service class for NeXus NXstress-compatible I/O.
"""

from datetime import datetime
from nexusformat.nexus import NXentry, NXfield, NXFile, nxopen
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    apply_appends,
    deferred_appends,
    GROUP_NAME,
    group_naming_scheme,
    growable,
    suffix_from_group_name,
    tail_append,
    NO_LOG,
    logger,
)
from ._input_data import _InputData
from ._instrument import _Instrument
from ._sample import _Sample
from ._fit import _Fit
from ._peaks import _Peaks, IndexedPeaks
from . import _append, _read, _validation, _workspace_key
from ._workspace_key import WorkspaceKey

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
            _validation.validate_entry_number(entry_number)

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
                a missing required log, overlapping scan points between inputs,
                two inputs sharing a workspace key, or a workspace whose
                `PeakCollection`s miss some of its scan points), or if
                `entry_number` skips past the next free entry.
            RuntimeError: If called outside the context manager, where there is
                no open root to write to; if this instance was already
                invalidated; or, on an append, for a violated invariant -- a
                duplicate row, an unmet precondition (the `ValueError` cases
                above included), or a disagreement with the target entry. The last **invalidates this instance**: a later
                `write` on it raises rather than proceeding.
            NotImplementedError: On an append, for Case B -- extending a
                compound key the entry already holds with further scan points.
                That needs a mid-array insertion rather than a tail-append and
                is deliberately out of scope. Nothing is written, and the
                instance stays usable.

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
            _validation.validate_entry_number(entry_number)
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

        self._appendToEntry(self._root[entry_name], wss, peakss)

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

    def _appendToEntry(self, entry, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]):
        """Grow an existing `NXentry` with more scan points (spec 04c), as a tail-append.

        `_append.preflight` runs every check before anything is resized, so a
        refused append leaves the entry byte-for-byte unchanged; `_append.commit`
        then grows the groups.

        Args:
            entry: The target `NXentry`, file-backed and open for writing.
            wss: Workspaces to append.
            peakss: One list of `PeakCollection` per workspace, same order.

        Raises:
            RuntimeError: On a violated invariant -- a duplicate row, a missing
                precondition, or a disagreement with the entry -- and on any other
                refusal before the first write, such as a run number that is not
                whole. Invalidates this instance.
            NotImplementedError: For Case B, extending a compound key already on
                disk with further scan points. The instance stays usable.
        """
        # One policy for both stages before the first write. A refusal -- the
        # expected types, whether raised by a check or on the way to one (a run
        # number that is not whole, building the workspace keys) -- is reported as
        # `InvariantViolated`. Anything else is a defect, and propagates as it is.
        # Either way the instance is invalidated; only Case B's
        # `NotImplementedError` (unsupported, not wrong) leaves it usable.
        try:
            plan = _append.preflight(entry, wss, peakss)
        except NotImplementedError:
            raise  # caught first: it is a `RuntimeError` too
        except _append.InvariantViolated:
            self._invalid = True
            raise
        except _append.REFUSALS as error:
            self._invalid = True
            raise _append.refusal(error) from error
        except BaseException:
            self._invalid = True
            raise

        # Every group's rows are computed and checked before the first is written:
        # the groups grow through `tail_append`, which inside `deferred_appends`
        # only collects. Whatever fails while computing -- a missing sample
        # position, a wavelength of the wrong type, values a dataset cannot hold --
        # leaves the entry untouched, and is a refusal like any other.
        try:
            with deferred_appends() as pending:
                self._appendEntryTimes(entry, wss)
                _append.commit(entry, wss, plan)
        except _append.InvariantViolated:
            self._invalid = True
            raise
        except _append.REFUSALS as error:
            self._invalid = True
            raise _append.refusal(error) from error
        except BaseException:
            self._invalid = True
            raise

        # Past this point the entry is being written dataset by dataset, and there
        # is no way to undo a partial write. Anything that escapes here -- by now
        # only h5py itself failing -- has left the entry desynchronized, which reads
        # back without error and is worse than a crash. The instance is therefore
        # invalidated on ANY escape: a caller must not be able to append again onto
        # a damaged entry.
        try:
            apply_appends(pending)
        except BaseException:
            self._invalid = True
            raise

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
        _validation.validate_entry_number(entry_number)

        entry = self._root[group_naming_scheme(GROUP_NAME.ENTRY, entry_number)]
        return _read.read_entry(entry, path=self._path)

    ############################################
    # ALL non-context-manager related methods ##
    #   must be `classmethod`.                ##
    ############################################

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
    @validate_call_
    def init_group(cls, wss: list[HidraWorkspace], peakss: list[list[PeakCollection]]) -> NXentry:
        # Create and initialize a single NXstress-compatible NXentry tree:
        #   _multiple_ NXentry can exist within an NXstress-compatible HDF5 file.
        #   For example, distinct entries might be added for each set of
        #   data-reduction or sample conditions.
        #
        #   One NXentry may also hold _multiple_ `HidraWorkspace`: their rows are
        #   concatenated in workspace order throughout, and told apart on read by
        #   the workspace key on the peak index: the discriminator columns and,
        #   when it is part of the key, the run.

        discriminator_names = _workspace_key.entry_discriminator_names(wss)
        run_boundary = _workspace_key.entry_carries_run(wss)

        # One flattened list of peak collections, each tagged with its own
        # workspace's key and sample logs. This is what keeps the three
        # position-aligned groups (PEAKS, peak parameters, background
        # parameters) in the same order as each other.
        indexed = [
            IndexedPeaks(
                WorkspaceKey.of(ws, discriminator_names, carries_run=run_boundary), collection, ws._sample_logs
            )
            for ws, collections in zip(wss, peakss)
            for collection in collections
        ]

        # Verify that all data required by NXstress are present.
        _validation.validate_inputs(wss, peakss, indexed)

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
        entry[GROUP_NAME.PEAKS] = _Peaks.init_group(indexed, logss[0], discriminator_names, run_boundary=run_boundary)

        return entry
