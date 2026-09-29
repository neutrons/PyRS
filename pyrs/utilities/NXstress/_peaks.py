# ruff: noqa: E741 # use `l` for `l` in `(h, k, l)`!
"""
pyrs/utilities/NXstress/_peaks.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `peaks` `NXreflections` subgroup:
  this subgroup includes fitted peak data, as used in reduction.
"""

import logging

import h5py
import numpy as np
from nexusformat.nexus import NXreflections, NXfield
import re
from typing import NamedTuple

from pyrs.peaks.peak_collection import PeakCollection
from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import is_text_array, to_text, to_text_array

from ._definitions import CHUNK_SHAPE, FIELD_DTYPE, GROUP_NAME, row_aligned_fields
from . import _discriminator
from ._discriminator import DiscriminatorKey


_logger = logging.getLogger(__name__)


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ peaks                                  (NXreflections, group)
│   ├─ h                                   (dataset)
│   ├─ k                                   (dataset)
│   ├─ l                                   (dataset)
│   └─ phase_name                          (dataset)

`PeakCollection` to `peaks` (NXreflections), `FIT` (NXprocess) mapping:
-----------------------------------------------------------------------

1. `peaks` provides the `n_Peaks` index which identifies `FIT` entries, with the exception of the `diffractogram` which are indexed separately.

- A flattened index is used `(<phase>, h, k, l, <mask>, <scan point>)`: all <scan point> may not be present, and to support legacy code specifying the <mask> is optional,
  and it will default to the key '_DEFAULT_';  Note that <mask> was not retained as a `PeakCollection` field prior to this implementation, but it does seem to be required;

- This flattened index allows appending, however each index value must identify a *unique* entry (i.e. there can be no duplicates);

- Each combination of `(<phase>, h, k, l, <mask>, ...)` corresponds to *one* `PeakCollection` instance;

- The format guarantees exactly two things about row order, and *not* a global lexicographic sort:
  each compound key occupies one contiguous run, and `scan_point` increases within a run.  `peakCollectionRanges`
  -- the only reader-side splitter -- enforces both and checks nothing else, so a file may be "locally sorted,
  globally segmented".  A single-step write does still emit a fully sorted index, but that is a property of this
  writer, not a promise the reader relies on.  Robustness against duplicates comes from `validateNoDuplicatePeaks`
  and the contiguity check, independently of sort order.  See `plans/NXstress-prod/04b-multi-workspace-nxstress.md`
  (Decisions Log item 17), which relaxed the earlier lexicographic framing precisely so that append could be a
  plain tail-append;

- When more than one `HidraWorkspace` is written into one NXentry, the index additionally carries one
  *discriminator* column per configured field (see `_discriminator.py`).  Discriminator values are the most
  slowly varying coordinates of the sort key, so each input workspace's rows form one contiguous super-block;

2. `diffractogram` are stored as 'diffractogram_<mask key>', and indexed by <scan point>.  Any single <scan point> that does not have an entry will be filled in with `NaN`.

"""


# At module scope, not nested in `_Peaks`, because `@validate_call_` resolves
# annotations while the class body is still executing, when a nested name does
# not yet exist.
class IndexedPeaks(NamedTuple):
    """A `PeakCollection` together with the input workspace it came from.

    The workspace is identified by its discriminator values rather than by
    position, so a single flattened list can carry the peak collections of
    all N inputs while remaining splittable.

    Attributes:
        discriminators: Name-keyed discriminator values for the input
            workspace this collection came from. Empty for a
            single-workspace write with no field configured.
        collection: The peak collection itself.
        logs: Sample logs of that same input workspace, used to look up
            the sample position for this collection's scan points. `None`
            on the read side, where no workspace exists yet.
    """

    discriminators: DiscriminatorKey
    collection: PeakCollection
    logs: SampleLogs | None = None

    @classmethod
    def sort_key(cls, item: "IndexedPeaks") -> tuple:
        """Ordering with discriminators as the most slowly varying coordinates.

        Putting them first is what makes each input workspace's rows one
        contiguous super-block in every position-aligned group, which in
        turn makes the read-side split a `groupby` over ranges
        `peakCollectionRanges` already returns.

        The three groups that must stay positionally aligned -- this one,
        `_PeakParameters` and `_BackgroundParameters` -- all sort the same
        decorated list with this key, which is the only reason their rows
        correspond.
        """
        return (
            *_discriminator.sort_values(item.discriminators),
            *_Peaks.PeakIndex.sort_key(item.collection),
        )


class _Peaks:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    class PeakIndex(NamedTuple):
        # Corresponds to the `n_Peaks` index in the `NXstress` schema.
        # Each `PeakCollection` instance provides
        #   `(<phase>, h, k, l, <mask>, ...)`, i.e. multiple scan_point;
        #   `scan_point` are distinct, but are not required to be contiguous, nor complete.
        phase_name: str
        h: int
        k: int
        l: int  # noqa: E741
        mask: str
        scan_point: int

        @classmethod
        def sort_key(cls, peaks: PeakCollection) -> tuple[str, int, int, int, str]:
            # Define an ordering for `PeakCollection` instances
            phase_name, (h, k, l) = _Peaks._parse_peak_tag(peaks.peak_tag)
            mask = peaks.mask
            return (phase_name, h, k, l, mask)

    @classmethod
    def indexed(cls, peakss: list[PeakCollection], logs: SampleLogs | None = None) -> list[IndexedPeaks]:
        """Wrap a single workspace's peak collections with an empty discriminator key."""
        return [IndexedPeaks((), peaks, logs) for peaks in peakss]

    @classmethod
    def _parse_peak_tag(cls, tag: str) -> tuple[str, tuple[int, int, int]]:
        # Parse a peak-tag string into its <phase name> and Miller indices (h, k, l).
        match: re.Match[str] | None = max(
            re.finditer(r"\d+", tag),
            key=lambda m: len(m.group(0)),
            default=None,
        )
        if match is None or len(match.group(0)) % 3 != 0:
            raise RuntimeError(f"Unable to parse peak tag '{tag}' into its <phase name> and Miller indices (h, k, l).")
        # Extract <phase name> as the rest of the tag.
        i, j = match.span()
        phase = (tag[:i] + tag[j:]).strip()
        if not bool(phase):
            raise RuntimeError(f"Unable to parse <phase name> from peak tag '{tag}'.")

        # Extract (h, k, l)
        maybeHKL = match.group(0)
        N_d = len(maybeHKL) // 3
        h, k, l_ = int(maybeHKL[0:N_d]), int(maybeHKL[N_d : 2 * N_d]), int(maybeHKL[2 * N_d : 3 * N_d])

        return phase, (h, k, l_)

    @classmethod
    def _init(cls, logs: SampleLogs, discriminator_dtypes: dict | None = None) -> NXreflections:
        # Initialize the 'PEAKS' group
        peaks = NXreflections()

        # Discriminator columns, one per configured field, written FIRST so the
        # group's own ordering mirrors the sort key's. Each records the original
        # (un-encoded) field name in `local_name`, the convention `_sample.py`
        # already uses for retained logs -- the on-disk name is the NeXus-legal
        # encoding of it, which is not always the same string.
        for name, dtype in (discriminator_dtypes or {}).items():
            column = _discriminator.column_name(name)
            peaks[column] = NXfield(np.empty((0,), dtype=dtype), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="")
            peaks[column].attrs["local_name"] = name
        # Which fields discriminate, recorded rather than left to be inferred from
        # the columns: a reader cannot otherwise tell an unconfigured discriminator
        # from another producer's reflection field. Empty when there are no
        # discriminator columns, so "no workspace boundary" is recorded too.
        peaks.attrs[_discriminator.FIELDS_ATTRIBUTE] = np.array(
            sorted(discriminator_dtypes or {}), dtype=h5py.string_dtype()
        )

        peaks["scan_point"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1))

        peaks["h"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="")
        peaks["k"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="")
        peaks["l"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="")

        peaks["phase_name"] = NXfield(
            np.empty((0,), dtype=FIELD_DTYPE.STRING.value), maxshape=(None,), chunks=CHUNK_SHAPE(1)
        )

        peaks["mask"] = NXfield(
            np.empty((0,), dtype=FIELD_DTYPE.STRING.value), maxshape=(None,), chunks=CHUNK_SHAPE(1)
        )

        ## Components of the normalized scattering vector Q in the sample reference frame
        ##   'qx', 'qy', and 'qz' are *required* by NXstress, but it looks as if PyRS doesn't
        ##   use these -- initialize to `NaN`.
        peaks["qx"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), fillvalue=np.nan
        )
        peaks["qx"].attrs["units"] = "1"
        peaks["qy"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), fillvalue=np.nan
        )
        peaks["qy"].attrs["units"] = "1"
        peaks["qz"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), fillvalue=np.nan
        )
        peaks["qz"].attrs["units"] = "1"
        ##

        peaks["center"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="angstrom"
        )
        peaks["center_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="angstrom"
        )
        peaks["center_type"] = NXfield("d-spacing")

        # Sample position for each subrun -- initialize to `NaN`.
        #
        # NXstress requires `peaks/sx,sy,sz` and defines them as the *sample position
        # in the sample reference frame* (see the vendored application definition at
        # `docs/developer/source/design/nexus/NXstress.nxdl.xml`). In PyRS that
        # quantity is `PointList.(vx, vy, vz)` -- NOT the stage-position logs that
        # happen to be spelled `sx`/`sy`/`sz`, which the stress/strain workflow does
        # not use. Those remain available in their own right, under
        # SAMPLE_DESCRIPTION/logs. Each field records its true source in
        # `local_name`, the same convention `_sample.py` uses for retained logs,
        # because the field name alone is misleading here.
        ss_units = {
            ## work around: units may be an empty string
            axis: logs.units(axis) if bool(logs.units(axis)) else "mm"
            for axis in HidraConstants.SAMPLE_COORDINATE_NAMES
        }
        peaks["sx"] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,),
            chunks=CHUNK_SHAPE(1),
            fillvalue=np.nan,
            units=ss_units["vx"],
        )
        peaks["sy"] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,),
            chunks=CHUNK_SHAPE(1),
            fillvalue=np.nan,
            units=ss_units["vy"],
        )
        peaks["sz"] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,),
            chunks=CHUNK_SHAPE(1),
            fillvalue=np.nan,
            units=ss_units["vz"],
        )

        for peak_axis, coord_axis in zip(("sx", "sy", "sz"), HidraConstants.SAMPLE_COORDINATE_NAMES):
            peaks[peak_axis].attrs["local_name"] = coord_axis

        return peaks

    @classmethod
    def discriminator_dtypes(cls, indexed: list[IndexedPeaks], names: tuple[str, ...]) -> dict:
        """HDF5 dtype for each discriminator column, inferred from its values.

        Inferred rather than fixed, so a numeric discriminator (a run number,
        say) stays numeric on disk instead of being stringified. Strings use the
        same variable-length dtype as `phase_name` and `mask`, which is what
        makes the column resizable.

        Args:
            indexed: Every peak collection being written, with its workspace's
                discriminator values.
            names: Configured discriminator field names.

        Returns:
            Field name -> dtype, in `names` order. **Empty when `indexed` is
            empty**: an entry with no peak collections has no discriminator
            values, so there is nothing to type and no column to write.

        Note:
            That empty case is not a detail. Inferring from no values at all
            gives `np.asarray([]).dtype` -- `float64` -- so a string
            discriminator would have been written as a numeric column. It is
            reachable: `CombineRunsModel` exports with `write([ws], [[]])`.
            Emitting no column instead is also the more honest record, and
            `_discriminator.names_for_read` already reads a column-less entry
            back as a single workspace. Such an entry cannot be appended to --
            see `NXstress._rejectUnmetCaseAPreconditions` -- which is correct:
            there is nothing to tell an appended workspace apart from.
        """
        if not indexed:
            return {}

        dtypes = {}
        for name in names:
            values = [dict(item.discriminators)[name] for item in indexed]
            inferred = np.asarray(values).dtype
            dtypes[name] = FIELD_DTYPE.STRING.value if inferred.kind in ("U", "S", "O") else inferred
        return dtypes

    @classmethod
    def init_group(
        cls,
        indexed: list[IndexedPeaks],
        logs: SampleLogs,
        discriminator_names: tuple[str, ...] = (),
        data: NXreflections | None = None,
    ) -> NXreflections:
        # Initialize the PEAKS group:
        #   according to the NXstress schema, this group contains the canonical reduction data,
        #   in a form usable for stress / strain calculations.
        #
        #   `data` is an existing on-disk PEAKS group to tail-append to (spec 04c). The
        #   incoming batch is sorted among itself and added after the current end; nothing
        #   already on disk is read back, reordered or rewritten. That is sound because the
        #   reader requires only that each compound key occupies one contiguous run and that
        #   `scan_point` increases within it -- never a global sort. The discriminator
        #   columns are NOT re-derived here: on append they already exist, with the dtypes
        #   the first write inferred, and `_init` is skipped entirely.
        peaks = data if data is not None else cls._init(logs, cls.discriminator_dtypes(indexed, discriminator_names))

        for item in sorted(indexed, key=IndexedPeaks.sort_key):
            cls._append_peak(peaks, item, logs, discriminator_names)

        return peaks

    @classmethod
    def _append_peak(
        cls,
        peaks: NXreflections,
        item: IndexedPeaks,
        logs: SampleLogs,
        discriminator_names: tuple[str, ...] = (),
    ) -> NXreflections:
        # Append a `PeakCollection` to an initialized PEAKS group.
        peak_collection = item.collection
        scan_point = peak_collection.sub_runs.raw_copy()
        N_scan = len(scan_point)
        phase_name, (h, k, l_) = cls._parse_peak_tag(peak_collection.peak_tag)
        mask = peak_collection.mask

        # Each dataset has scan point as its first index.
        phase_name_arr = np.array((phase_name,) * N_scan)
        h_arr = np.array((h,) * N_scan)
        k_arr = np.array((k,) * N_scan)
        l_arr = np.array((l_,) * N_scan)
        mask_arr = np.array((mask,) * N_scan)

        d_reference_arr, d_reference_error_arr = peak_collection.get_d_reference()

        curr_len = peaks["h"].shape[0]
        new_len = curr_len + N_scan

        peaks["scan_point"].resize((new_len,))

        peaks["h"].resize((new_len,))
        peaks["k"].resize((new_len,))
        peaks["l"].resize((new_len,))
        peaks["phase_name"].resize((new_len,))
        peaks["mask"].resize((new_len,))

        # For `PEAKS` (NXreflections) group: 'center' means `d_reference`.
        peaks["center"].resize((new_len,))
        peaks["center_errors"].resize((new_len,))

        peaks["sx"].resize((new_len,))
        peaks["sy"].resize((new_len,))
        peaks["sz"].resize((new_len,))

        # Discriminator values are per input workspace, so every row this
        # collection contributes carries the same value.
        discriminators = dict(item.discriminators)
        for name in discriminator_names:
            column = _discriminator.column_name(name)
            peaks[column].resize((new_len,))
            peaks[column][curr_len:] = np.array((discriminators[name],) * N_scan)

        peaks["scan_point"][curr_len:] = scan_point
        peaks["h"][curr_len:] = h_arr
        peaks["k"][curr_len:] = k_arr
        peaks["l"][curr_len:] = l_arr
        peaks["phase_name"][curr_len:] = phase_name_arr
        peaks["mask"][curr_len:] = mask_arr

        peaks["center"][curr_len:] = d_reference_arr.ravel()
        peaks["center_errors"][curr_len:] = d_reference_error_arr.ravel()

        # The original form of this block read `logs['sx']` wholesale, which is why it
        # was commented out as not making sense: a log spans *every* scan point in the
        # workspace, while this slice needs only the scan points of THIS
        # `PeakCollection`. `get_pointlist` takes the subrun subset directly.
        # This collection's scan points index into its *own* workspace's logs:
        # in a multi-workspace entry the entry-level `logs` argument belongs to
        # the first input only, and would give the wrong sample positions -- or
        # none at all -- for every other input's collections.
        collection_logs = item.logs if item.logs is not None else logs
        for peak_axis, values in zip(("sx", "sy", "sz"), cls._sample_positions(collection_logs, scan_point, N_scan)):
            peaks[peak_axis][curr_len:] = values

        return peaks

    @classmethod
    def _sample_positions(cls, logs: SampleLogs, scan_point: np.ndarray, N_scan: int) -> tuple:
        """Sample position for a `PeakCollection`'s scan points, as (vx, vy, vz).

        Args:
            logs: Sample logs for the whole workspace.
            scan_point: The subrun numbers this `PeakCollection` covers.
            N_scan: Number of scan points, used to shape the `NaN` fallback.

        Returns:
            Three arrays of length `N_scan`, in millimetres. All `NaN` when the
            coordinate logs are absent or non-finite.
        """
        try:
            point_list = logs.get_pointlist(scan_point)
            return (point_list.vx, point_list.vy, point_list.vz)
        except (ValueError, AssertionError) as e:
            # Match `_sample.py`'s handling: a workspace with no usable sample
            # coordinates still writes a valid file, with `NaN` where the positions
            # would go, rather than failing an entire save.
            _logger.warning(
                f"NXstress._peaks: sample coordinates "
                f"{HidraConstants.SAMPLE_COORDINATE_NAMES} are unavailable ({e});\n"
                "  `peaks/sx,sy,sz` will be written as NaN."
            )
            nan = np.full((N_scan,), np.nan)
            return (nan, nan.copy(), nan.copy())

    @classmethod
    def _decoded(cls, values: np.ndarray) -> np.ndarray:
        """HDF5 string columns come back as bytes; index keys must be `str`.

        Guarded rather than unconditional: a discriminator column is whatever the
        configured sample log holds, so it may legitimately be numeric, and
        `to_text_array` raises on a non-string array instead of passing it
        through. `phase_name` and `mask` are always text and would be safe
        either way.
        """
        return to_text_array(values) if is_text_array(values) else values

    @classmethod
    def peakCollectionRanges(
        cls, peaks, discriminator_names: tuple[str, ...] = ()
    ) -> list[tuple[DiscriminatorKey, tuple[str, int, int, int, str], int, int]]:
        """Identify contiguous blocks of PeakCollection data in NXreflections group.

        Each PeakCollection corresponds to a unique 5-tuple (phase_name, h, k, l, mask)
        -- prefixed by its discriminator values, when the entry holds more than one
        input workspace -- with multiple scan-points written as a contiguous block in
        increasing order.

        Parameters
        ----------
        peaks : NXreflections
            The peaks group from which to read the flattened index
        discriminator_names : tuple[str, ...]
            Discriminator fields this entry was written with, already
            cross-checked against the file by `_discriminator.names_for_read`.
            Empty for an entry that records no workspace boundary.

        Returns
        -------
        list[tuple[DiscriminatorKey, tuple[str, int, int, int, str], int, int]]
            List of (discriminators, key, start, end) where:
            - discriminators is the name-keyed discriminator value tuple, empty
              when the entry has no discriminator columns
            - key is (phase_name, h, k, l, mask)
            - start is the first index (inclusive)
            - end is the last index (exclusive)

        Raises
        ------
        RuntimeError
            If scan_point values are not strictly increasing within a block
        RuntimeError
            If interleaved blocks are detected for the same sub-index key
        """
        cls._validateDiscriminatorNames(peaks, discriminator_names)

        # Read index arrays via .nxdata
        phase_name = cls._decoded(peaks["phase_name"].nxdata[:])
        h = peaks["h"].nxdata[:]
        k = peaks["k"].nxdata[:]
        l_ = peaks["l"].nxdata[:]
        mask = cls._decoded(peaks["mask"].nxdata[:])
        scan_point = peaks["scan_point"].nxdata[:]

        # Discriminator columns, read in name order so a key built here compares
        # equal to one built by `_discriminator.key` regardless of config order.
        ordered_names = tuple(sorted(discriminator_names))
        discriminators = [cls._decoded(peaks[_discriminator.column_name(n)].nxdata[:]) for n in ordered_names]

        if len(phase_name) == 0:
            return []

        def key_at(i: int) -> tuple[DiscriminatorKey, tuple[str, int, int, int, str]]:
            disc = tuple((n, column[i].item()) for n, column in zip(ordered_names, discriminators))
            return disc, (str(phase_name[i]), int(h[i]), int(k[i]), int(l_[i]), str(mask[i]))

        def described(key: tuple) -> str:
            # Name the block the way a caller thinks of it: the compound key,
            # and the discriminators only when the entry actually has any.
            disc, base = key
            return f"{base}" + (f" for {dict(disc)}" if disc else "")

        ranges = []
        seen_keys = set()

        # Track current block
        current_key = key_at(0)
        start_idx = 0

        for i in range(1, len(phase_name)):
            key = key_at(i)

            if key != current_key:
                # Block boundary - validate and record current block
                end_idx = i

                # Check for strictly increasing scan_point within block
                block_scan_points = scan_point[start_idx:end_idx]
                if not np.all(block_scan_points[1:] > block_scan_points[:-1]):
                    raise RuntimeError(
                        f"scan_point values are not strictly increasing within PeakCollection block "
                        f"at {described(current_key)}, indices [{start_idx}, {end_idx})"
                    )

                # Check for interleaved blocks
                if current_key in seen_keys:
                    raise RuntimeError(f"Interleaved blocks detected for sub-index {described(current_key)}")

                seen_keys.add(current_key)
                ranges.append((*current_key, start_idx, end_idx))

                # Start new block
                current_key = key
                start_idx = i

        # Handle last block
        end_idx = len(phase_name)
        block_scan_points = scan_point[start_idx:end_idx]
        if not np.all(block_scan_points[1:] > block_scan_points[:-1]):
            raise RuntimeError(
                f"scan_point values are not strictly increasing within PeakCollection block "
                f"at {described(current_key)}, indices [{start_idx}, {end_idx})"
            )

        if current_key in seen_keys:
            raise RuntimeError(f"Interleaved blocks detected for sub-index {described(current_key)}")

        seen_keys.add(current_key)
        ranges.append((*current_key, start_idx, end_idx))

        return ranges

    @classmethod
    def _validateDiscriminatorNames(cls, peaks, discriminator_names: tuple[str, ...]) -> None:
        """Check the caller's discriminator names against the columns actually present.

        Getting this wrong does not produce a wrong *error* -- it produces a wrong
        *answer*, silently. Splitting a two-workspace entry with no names collapses
        both workspaces into one range whenever their blocks are adjacent and share
        a compound key, and the monotonicity check passes because the concatenated
        scan points are still increasing. Measured: 1 range returned where 2 were
        correct, no exception. With differently-shaped data the same mistake instead
        surfaces as "Interleaved blocks detected", which is at least loud but names
        the wrong cause.

        So the argument is checked against the group: every name must have its
        column, and when the entry records the fields it was written with
        (`_discriminator.FIELDS_ATTRIBUTE`), the names must be exactly those. Any
        other column is not a discriminator and is ignored -- it may be a
        reflection field from another NXstress producer. When the entry records
        nothing, a dropped discriminator cannot be detected here; that is the
        price of configuration being the authority for such files (see
        `plans/PRs/phase-2.md`). This is a different check from
        `_discriminator.names_for_read`, which compares *configuration* against
        the file; this compares what the caller passed.

        Args:
            peaks: The `peaks` (NXreflections) group being read.
            discriminator_names: Names the caller says the entry was written with.
                Empty is legitimate, and means the entry records no workspace
                boundary: a pre-04b file, a single-workspace file written before
                any field was configured, or one written with the since-removed
                `nxstress.merge_workspaces`, which concatenated several
                workspaces indistinguishably (Decisions Log row 45).

        Raises:
            RuntimeError: If the two disagree.
        """
        missing = sorted(name for name in discriminator_names if _discriminator.column_name(name) not in peaks)
        if missing:
            raise RuntimeError(
                f"NXstress._peaks: this entry has no column for discriminator field(s) {missing} "
                f"(given {list(discriminator_names)}).\n"
                "  Pass the names `_discriminator.names_for_read` returns for this group."
            )
        recorded = _discriminator.recorded_names(peaks)
        if recorded is not None and set(recorded) != set(discriminator_names):
            raise RuntimeError(
                f"NXstress._peaks: the discriminator names given do not match the fields this "
                f"entry records.\n"
                f"  Recorded: {list(recorded)}\n"
                f"  Given:    {sorted(discriminator_names)}\n"
                "  Splitting with the wrong set does not fail loudly -- it silently merges or "
                "interleaves workspaces. Pass the names `_discriminator.names_for_read` returns "
                "for this group."
            )

    @classmethod
    def validateNoDuplicatePeaks(cls, indexed: list[IndexedPeaks]) -> None:
        """Validate that no duplicate PeakCollections exist in the list.

        Each PeakCollection must have a unique key. The discriminator values
        are part of that key, so the same `(phase_name, h, k, l, mask)`
        arriving from two different input workspaces is *not* a duplicate --
        that is the ordinary multi-workspace case, and the two are told apart
        on disk by their discriminator columns.

        Parameters
        ----------
        indexed : list[IndexedPeaks]
            Peak collections to validate, each tagged with the discriminator
            values of the input workspace it came from.

        Raises
        ------
        ValueError
            If any duplicate keys are found
        """
        seen_keys = {}
        for item in indexed:
            key = IndexedPeaks.sort_key(item)
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate PeakCollection detected in output list at {key} "
                    f"-- did you forget to initialize the `mask` key?"
                )
            seen_keys[key] = item

    @classmethod
    def appendableDatasets(cls, peaks) -> dict:
        """Every dataset in the peak index that an append grows.

        All of them: the index columns, the per-row fit results, the sample
        positions and the discriminator columns are one row per peak record, and
        an append adds rows to every one. `center_type` is the group's one
        entry-wide scalar and drops out for being scalar. The discriminator
        columns are *configured*, so enumerating the group rather than listing
        names is what keeps them covered.

        Args:
            peaks: The target entry's existing `peaks` group.

        Returns:
            Path -> `NXfield`, for `NXstress._validateAppendableShapes`.
        """
        return row_aligned_fields(peaks, GROUP_NAME.PEAKS)

    @classmethod
    @validate_call_
    def peakCollectionsFromNexus(cls, peaks, fit, discriminator_names: tuple[str, ...] = ()) -> list[IndexedPeaks]:
        """Read PeakCollections from NXreflections and NXprocess groups.

        Note: This implementation assumes positive Miller indices. Negative indices
        are not supported by the current _parse_peak_tag implementation.

        Parameters
        ----------
        peaks : NXreflections
            The peaks (NXreflections) group containing d-spacing and Miller indices
        fit : NXprocess
            The FIT (NXprocess) group containing peak_parameters and background_parameters
        discriminator_names : tuple[str, ...]
            Discriminator fields this entry was written with, already
            cross-checked against the file by `_discriminator.names_for_read`.

        Returns
        -------
        list[IndexedPeaks]
            Reconstructed peak collections, each tagged with the discriminator
            values of the input workspace it came from. Group by
            `.discriminators` to recover the per-workspace lists; that grouping
            is `NXstress.read`'s job, because it also needs the scan-point sets
            the groups imply.
        """
        from ._fit import _PeakParameters, _BackgroundParameters
        from ._definitions import GROUP_NAME, UNDEFINED_PEAK_TAG

        # Get the parameter groups
        pp = fit[GROUP_NAME.PEAK_PARAMETERS]
        bp = fit[GROUP_NAME.BACKGROUND_PARAMETERS]

        # A file written with peakss=[] (e.g. CombineRunsViewer's NXstress export,
        # which has no PeakCollection concept at all) has its title fields set to
        # the writer's own UNDEFINED_PEAK_TAG sentinel (see
        # _PeakParameters.init_group / _BackgroundParameters.init_group) -- there
        # is nothing to reconstruct. The reader previously didn't recognize its
        # own writer's sentinel and crashed trying to parse it as a real
        # PeakShape/BackgroundFunction name.
        pp_title = to_text(pp["title"].nxdata)
        if pp_title == UNDEFINED_PEAK_TAG:
            return []

        # Get peak profile and background function from titles
        from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction

        peak_profile = PeakShape.getShape(pp["title"].nxdata)
        background_function = BackgroundFunction.getFunction(bp["title"].nxdata)

        # Get ranges for each PeakCollection
        ranges = cls.peakCollectionRanges(peaks, discriminator_names)

        peak_collections = []
        for discriminators, (phase_name, h, k, l_, mask), start, end in ranges:
            # Extract scan points for this range
            sub_runs_array = peaks["scan_point"].nxdata[start:end]

            # Get peak parameters
            native_peak_values, native_peak_errors = _PeakParameters.peakParametersForRange(pp, start, end)

            # Get background parameters
            bg_values, bg_errors = _BackgroundParameters.backgroundParametersForRange(bp, start, end)

            # Merge background into native peak arrays
            # A0 and A1 are always present, A2 only for Quadratic background
            native_peak_values["A0"] = bg_values["A0"]
            native_peak_values["A1"] = bg_values["A1"]
            native_peak_errors["A0"] = bg_errors["A0"]
            native_peak_errors["A1"] = bg_errors["A1"]
            # A2 only exists if background is Quadratic
            if "A2" in native_peak_values.dtype.names:
                native_peak_values["A2"] = bg_values["A2"]
                native_peak_errors["A2"] = bg_errors["A2"]

            param_values = native_peak_values
            param_errors = native_peak_errors

            # Reconstruct peak_tag with zero-padded Miller indices
            # Use max absolute value to ensure all indices have the same digit count
            max_val = max(abs(h), abs(k), abs(l_))
            N_d = len(str(max_val))
            peak_tag = f"{phase_name}{str(h).zfill(N_d)}{str(k).zfill(N_d)}{str(l_).zfill(N_d)}"

            # Extract d_reference and errors
            d_reference = peaks["center"].nxdata[start]
            d_reference_error = peaks["center_errors"].nxdata[start]

            # Construct PeakCollection with mask keyword
            pc = PeakCollection(
                peak_tag=peak_tag,
                peak_profile=peak_profile,
                background_type=background_function,
                wavelength=np.nan,  # Will be set by workspace if needed
                projectfilename="",
                runnumber=0,
                d_reference=d_reference,
                d_reference_error=d_reference_error,
                mask=mask,
            )

            # Set peak fitting values
            N = len(sub_runs_array)
            fit_costs = np.full(N, np.nan)
            pc.set_peak_fitting_values(sub_runs_array, param_values, param_errors, fit_costs)

            peak_collections.append(IndexedPeaks(discriminators, pc))

        return peak_collections
