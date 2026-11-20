"""
pyrs/utilities/NXstress/_read.py

Read one NXstress entry back into `HidraWorkspace`s and their `PeakCollection`s.

Several workspaces may share one entry, so a read has to split it again:

1. `read_entry(entry, path=...)` reads the peak collections first -- in a
   multi-workspace entry they are the only record of where one workspace's rows
   end and the next begin.
2. `workspace_selections(...)` groups them by workspace key and turns each group
   into the rows of the scan-point family that workspace owns.
3. `workspace_from_nexus(...)` rebuilds one workspace from its rows.
"""

import numpy as np
from nexusformat.nexus import NXdata, NXgroup

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.restorable_property import restore
from pyrs.utilities.unset import Unset

from ._definitions import DEFAULT_TAG, GROUP_NAME, logger, suffix_from_group_name
from ._fit import _Diffractogram
from ._input_data import _InputData
from ._instrument import _Instrument, _Masks
from ._peaks import _Peaks, IndexedPeaks
from ._sample import _Sample
from . import _discriminator, _runs
from ._workspace_key import WorkspaceKey


def read_entry(entry: NXgroup, *, path: str) -> tuple[list[HidraWorkspace], list[list[PeakCollection]]]:
    """Read back the workspaces and peak collections of one NXentry.

    Args:
        entry: The `NXentry` to read.
        path: The file it is in, recorded as the source of everything read: each
            peak collection's `projectfilename` and each workspace's
            `hidra_project_file`.

    Returns:
        `(workspaces, peakss)`, parallel and in the same order -- the shape
        `NXstress.write` accepts. An entry that records no workspace boundary
        yields one of each.

    Raises:
        RuntimeError: If the entry's discriminator columns disagree with
            `nxstress.discriminator_fields`, or if the recovered scan-point sets
            do not partition the entry's scan-point axis.
    """
    # Both are required by NXstress (`minOccurs="1"`), and PyRS always writes both.
    missing = [name for name in (GROUP_NAME.FIT, GROUP_NAME.PEAKS) if name not in entry]
    if missing:
        raise RuntimeError(
            f"NXstress: entry '{entry.nxname}' has no {' or '.join(missing)} group, which NXstress requires.\n"
            "  This file does not conform to NXstress, so its workspaces cannot be recovered from it."
        )

    # Read peak collections FIRST: in a multi-workspace entry they are the
    # only record of where one workspace's rows end and the next begin, so
    # nothing else can be split until they have been read and grouped.
    scan_point_axis = entry[GROUP_NAME.SAMPLE_DESCRIPTION]["scan_point"].nxdata
    run_axis = _runs.axis(entry[GROUP_NAME.SAMPLE_DESCRIPTION])
    peaks_group = entry[GROUP_NAME.PEAKS]
    discriminator_names = _discriminator.names_for_read(peaks_group)
    # Each collection's wavelength, which it needs to turn its fitted peak
    # positions into d-spacing: per scan point in the instrument group,
    # aligned with the scan-point axis.
    _, _, wavelength_axis = _Instrument.instrumentFromNexus(entry[GROUP_NAME.INSTRUMENT])
    wavelengths = dict(zip(_runs.pairs(run_axis, scan_point_axis), np.asarray(wavelength_axis).tolist()))
    indexed = _Peaks.peakCollectionsFromNexus(
        peaks_group, entry[GROUP_NAME.FIT], discriminator_names, wavelengths=wavelengths
    )
    # Each collection records the file it was loaded from, as one loaded from a
    # `HidraProjectFile` does. It is not optional: `SummaryGeneratorStress` refuses
    # to export a strain whose collections name no file.
    for item in indexed:
        item.collection.projectfilename = path

    selections = workspace_selections(indexed, scan_point_axis=scan_point_axis, run_axis=run_axis)

    # Instrument geometry, detector shift and masks are entry-wide.
    default_mask, mask_dict = _Masks.masksFromNexus(entry[GROUP_NAME.INSTRUMENT][GROUP_NAME.MASKS])

    workspaces = []
    peak_collections = []
    for rows, workspace_key, collections in selections:
        ws = workspace_from_nexus(entry, rows=rows, default_mask=default_mask, mask_dict=mask_dict)
        # The workspace records its source file, as its collections do. It is the
        # file a viewer's saved state names, to load it again from.
        restore(ws, "hidra_project_file", path)
        # The run, when it is part of the key, is not put back: it is already the
        # workspace's own `run_number` log, if it had one, and the `runnumber` of
        # its peak collections.
        for name, value in workspace_key.discriminators:
            _discriminator.apply(ws, name, value)
        workspaces.append(ws)
        peak_collections.append(collections)

    return workspaces, peak_collections


def workspace_selections(
    indexed: list[IndexedPeaks], *, scan_point_axis: np.ndarray, run_axis: np.ndarray
) -> list[tuple]:
    """Split one entry into per-workspace (rows, workspace key, collections).

    Each workspace's scan-point *set* is the union of its peak collections'
    scan points, and its rows of the scan-point family are selected by
    membership in that set -- not by position. That is what makes the read
    independent of the order `write` happened to concatenate the inputs in.

    Membership is by `(run, scan point)` wherever the workspace key carries the
    run (`_runs.is_boundary`): every run numbers its scan points from 1, so a
    scan point alone is ambiguous between runs. An entry written before the
    run index carries no run in its keys, and is split by scan point alone, as
    it always was.

    Args:
        indexed: Every peak collection of the entry, with its workspace key.
        scan_point_axis: The entry's scan point of every row.
        run_axis: The entry's run of every row, aligned with `scan_point_axis`.

    Returns:
        `(rows, workspace key, collections)` per workspace, where `rows` is a
        boolean mask over `scan_point_axis`, or `None` for a single-workspace
        entry, which needs no selection at all.

    Raises:
        RuntimeError: If two workspaces claim one row, or a row belongs to none.
    """
    groups: dict = {}
    for item in indexed:
        groups.setdefault(item.key, []).append(item.collection)

    if len(groups) <= 1:
        # One workspace: a single-workspace write, or a file written before entries
        # held several, either of which reads back whole -- no rows need selecting.
        # Its discriminators are still carried through, so a single-workspace save
        # round-trips its value like any other.
        workspace_key, collections = next(iter(groups.items())) if groups else (WorkspaceKey(), [])
        return [(None, workspace_key, collections)]

    axis_pairs = _runs.pairs(run_axis, scan_point_axis)
    selections = []
    claimed = np.zeros(len(scan_point_axis), dtype=bool)
    for workspace_key, collections in groups.items():
        points = set()
        for collection in collections:
            points |= set(collection.sub_runs.raw_copy().tolist())
        if workspace_key.run is not Unset:
            claims = {(workspace_key.run, point) for point in points}
            rows = np.array([pair in claims for pair in axis_pairs], dtype=bool)
        else:
            rows = np.isin(scan_point_axis, sorted(points))
        overlap = claimed & rows
        if overlap.any():
            raise RuntimeError(
                f"NXstress: scan points {sorted(set(np.asarray(scan_point_axis)[overlap].tolist()))} "
                f"are claimed by more than one workspace in this entry "
                f"(at {dict(workspace_key.as_pairs())}).\n"
                "  Input workspaces must cover disjoint scan points; this file cannot be split."
            )
        claimed |= rows
        selections.append((rows, workspace_key, collections))

    if not claimed.all():
        unclaimed = sorted(set(np.asarray(scan_point_axis)[~claimed].tolist()))
        raise RuntimeError(
            f"NXstress: scan points {unclaimed} belong to no workspace in this entry.\n"
            "  Every scan point must be covered by some input workspace's `PeakCollection`s."
        )
    return selections


def workspace_from_nexus(
    entry: NXgroup, *, rows: np.ndarray | None, default_mask: np.ndarray | None, mask_dict: dict
) -> HidraWorkspace:
    """Reconstruct one `HidraWorkspace` from its rows of a (possibly shared) entry.

    Args:
        entry: The NXentry.
        rows: This workspace's rows of the scan-point family, as a boolean mask;
            `None` for a single-workspace entry, which is read whole.
        default_mask: The entry-wide default mask (`_Masks.masksFromNexus`).
        mask_dict: The entry-wide named masks, by name.

    Returns:
        The workspace, holding only its own scan points.
    """
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
    ws.set_wavelength({subrun_index: wavelength[n] for n, subrun_index in enumerate(ws.get_sub_runs())}, is_calibrated)
    ws.set_instrument_geometry(geometry)
    if shift is not None:
        ws.set_detector_shift(shift)
    ws.set_masks_from_dict(default_mask, mask_dict)

    # Read raw counts if present
    if GROUP_NAME.INPUT_DATA in entry:
        _InputData.readSubruns(ws, entry[GROUP_NAME.INPUT_DATA], rows)

    # Read reduced diffraction data from FIT group's DIFFRACTOGRAM subgroups
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
