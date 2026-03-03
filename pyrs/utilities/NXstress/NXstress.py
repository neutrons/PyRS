"""
pyrs/utilities/NXstress/NXstress.py

Primary service class for NeXus NXstress-compatible I/O.
"""

from datetime import datetime
from nexusformat.nexus import NXdata, NXentry, NXfield, NXFile, nxopen
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import (
    DEFAULT_TAG,
    GROUP_NAME,
    group_naming_scheme,
    NO_LOG,
    suffix_from_group_name,
    logger,
    REQUIRED_LOGS,
)
from ._input_data import _InputData
from ._instrument import _Instrument, _Masks
from ._sample import _Sample
from ._fit import _Fit, _Diffractogram
from ._peaks import _Peaks


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
    def __init__(self, file_path: Path, mode: str = "r"):
        self._path = str(file_path)
        self._mode = mode
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

    def write(self, ws: HidraWorkspace, peakss: list[PeakCollection]):
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

        ######################################################
        ## Recommended usage:                               ##
        ## -------------------------------------------------##
        ## from pyrs/utilities/NXstress import NXstress     ##
        ## ...                                              ##
        ## ws: HidraWorkspace                               ##
        ## peakss: list[PeakCollection]                     ##
        ## ...                                              ##
        ## # To write the first (, or only) entry:          ##
        ## with NXstress(<file name>.nxs, 'w') as nxS:      ##
        ##     nxS.write(f, ws, peaks)                      ##
        ## -------------------------------------------------##
        ## # To write an additional entry:                  ##
        ## # alternatively, this could have been done       ##
        ## # in the first `with` clause above.              ##
        ## with NXfile(<same file name>.nxs, 'a') as nxS:   ##
        ##     nxS.write(f, ws, peaks)                      ##
        ######################################################

        if self._root is None:
            raise RuntimeError("Usage error: only usage as context manager is supported!")
        entry_number = len(self._root.NXentry) + 1
        entry_name = group_naming_scheme(GROUP_NAME.ENTRY, entry_number)
        if entry_name in self._root:
            raise RuntimeError(f"Not implemented: overwriting existing `NXentry` '/{entry_name}'.")

        entry = self.init_group(ws, peakss)
        self._root[entry_name] = entry

    def read(self, entry_number: int = 1):
        """Read back a (HidraWorkspace, list[PeakCollection]) from the NXstress file.

        Parameters
        ----------
        entry_number : int
            Which NXentry to read (1-based). Default is 1.

        Returns
        -------
        tuple
            (HidraWorkspace, list[PeakCollection])
        """
        # Verify context manager is active
        if self._root is None:
            raise RuntimeError("Usage error: only usage as context manager is supported!")

        # Resolve entry name
        entry_name = group_naming_scheme(GROUP_NAME.ENTRY, entry_number)

        # Access entry
        entry = self._root[entry_name]

        # Read sample logs
        sample_logs = _Sample.sampleLogsFromNexus(entry[GROUP_NAME.SAMPLE_DESCRIPTION])

        # Read instrument
        geometry, shift, wavelength = _Instrument.instrumentFromNexus(entry[GROUP_NAME.INSTRUMENT])
        is_calibrated = shift is not None

        # Read masks
        default_mask, mask_dict = _Masks.masksFromNexus(entry[GROUP_NAME.INSTRUMENT][GROUP_NAME.MASKS])

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
            _InputData.readSubruns(ws, entry[GROUP_NAME.INPUT_DATA])

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
                scan_pts, two_theta, data, errors = _Diffractogram.diffractogramFromNexus(child)

                # Map DEFAULT_TAG to None for workspace dict keys
                ws_mask_key = None if mask_name == DEFAULT_TAG else mask_name
                diff_data[ws_mask_key] = data

                # NOTE: Despite the field name 'diffractogram_errors',
                #   variance values (not standard errors) are stored in this field.
                var_data[ws_mask_key] = errors

                if two_theta_matrix is None:
                    two_theta_matrix = two_theta

            if two_theta_matrix is not None:
                ws.set_reduced_diffraction_data_set(two_theta_matrix, diff_data, var_data)

        # Read peak collections
        peak_collections = []
        if GROUP_NAME.PEAKS in entry:
            peaks_group = entry[GROUP_NAME.PEAKS]
            if GROUP_NAME.FIT in entry:
                fit_group = entry[GROUP_NAME.FIT]
                peak_collections = _Peaks.peakCollectionsFromNexus(peaks_group, fit_group)

        return ws, peak_collections

    ############################################
    # ALL non-context-manager related methods ##
    #   must be `classmethod`.                ##
    ############################################

    @classmethod
    @validate_call_
    def _validateWorkspaceAndPeaksData(cls, ws: HidraWorkspace, peakss: list[PeakCollection]):
        # VERIFY that all required logs are present.
        logs = ws.sample_log_names
        for k in REQUIRED_LOGS:
            if k not in logs:
                raise ValueError(f"NXstress requires log '{k}', which is not present")

        # VERIFY that no duplicate PeakCollections exist
        _Peaks.validateNoDuplicatePeaks(peakss)

        # VERIFY that any <scan point> or <mask> referenced by any `PeakCollection` is included in the workspace.
        _Fit.validateWorkspaceAndPeaksData(ws, peakss)

    @classmethod
    @validate_call_
    def _init(cls, ws: HidraWorkspace) -> NXentry:
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

        # lists of 'start_time', 'end_time' for all subruns
        try:
            start_times: list[str] = [
                datetime.fromisoformat(t.decode("utf-8")).astimezone().isoformat()
                for t in ws.get_sample_log_values("start_time")
            ]
            end_times: list[str] = [
                datetime.fromisoformat(t.decode("utf-8")).astimezone().isoformat()
                for t in ws.get_sample_log_values("end_time")
            ]
        except ValueError as e:
            if "Invalid isoformat string" not in str(e):
                raise
            logger.warning(
                f"Log entries for sub-run start and end times are not in ISO-8601 format:\n"
                f"  in order to continue writing, a value of '{NO_LOG}' will be used for all time entries!"
            )
            start_times = end_times = [NO_LOG for n in ws._sample_logs.subruns]
        entry["start_time"] = NXfield(start_times)
        entry["end_time"] = NXfield(end_times)

        # the type of the primary strain calculation:
        #   this might also be 'two-theta', but 'd-spacing' seems more likely
        entry["processing_type"] = "d-spacing"

        return entry

    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace, peakss: list[PeakCollection]) -> NXentry:
        # Create and initialize a single NXstress-compatible NXentry tree:
        #   _multiple_ NXentry can exist within an NXstress-compatible HDF5 file.
        #   For example, distinct entries might be added for each set of
        #   data-reduction or sample conditions.

        # Verify that all data required by NXstress are present.
        cls._validateWorkspaceAndPeaksData(ws, peakss)

        # Initialize this NXentry, and add required attributes.
        entry = cls._init(ws)

        # 'input_data' group
        entry[GROUP_NAME.INPUT_DATA] = _InputData.init_group(ws)

        # 'instrument' group
        entry[GROUP_NAME.INSTRUMENT] = _Instrument.init_group(ws)

        # 'SAMPLE_DESCRIPTION' group
        entry[GROUP_NAME.SAMPLE_DESCRIPTION] = _Sample.init_group(ws._sample_logs)

        # 'FIT' group
        entry[GROUP_NAME.FIT] = _Fit.init_group(ws, peakss, ws._sample_logs)

        # 'PEAKS' group
        entry[GROUP_NAME.PEAKS] = _Peaks.init_group(peakss, ws._sample_logs)

        return entry
