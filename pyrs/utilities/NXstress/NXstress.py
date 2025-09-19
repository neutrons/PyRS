"""
pyrs/utilities/NXstress/NXstress.py

Primary service class for NeXus NXstress-compatible I/O.
"""
import h5py
from nexusformat.nexus import (
    NXentry, NXFile, nxopen, NXroot
)
import numpy as np
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import (
    DEFAULT_TAG,
    FIELD_DTYPE,
    GROUP_NAME,
    group_naming_scheme,
    REQUIRED_LOGS
)    
from ._input_data import _InputData
from ._instrument import _Instrument
from ._sample import _Sample
from ._fit import _Fit
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
         
    def __enter__(self) -> 'NXstress':
        self._root = nxopen(self._path, self._mode)
        self._root.__enter__()

        return self
        
    def __exit__(self, exc_type, exc, tb):
        if self._root:
            self._root.__exit__(exc_type, exc, tb)
            self._root = None

        # Do not suppress exceptions
        return False

    def write(self, ws: HidraWorkspace, peaks: PeakCollection):
        # Write the _next_ NXentry to the file:
        #
        # -- multiple NXentry are allowed by the NXstress schema.
        # -- each NXentry includes:
        #
        #   -- [optional] input_data: raw detector counts, indexed by 'scan_point' (aka: 'subrun');
        #   -- the `NXinstrument`, including its `NXdetector`, applicable `NXtransformations`
        #      and detector and solid-angle masks; 
        #   -- a single canonical PEAKS instance,
        #   -- reduced 'diffraction_data' sections: organized by mask name, each section includes:
        #
        #     -- peak-fit details;
        #     -- normalized and reduced data, indexed by 'scan_point';
        #     -- a calculated model spectrum: this section is still in progress. 
        #
        
        ######################################################
        ## Recommended usage:                               ##
        ## -------------------------------------------------##
        ## from pyrs/utilities/NXstress import NXstress     ##
        ## ...                                              ##
        ## ws: HidraWorkspace                               ##
        ## peaks: PeakCollection                            ##
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
        
        entry = self.init_group(ws, peaks)
        self._root[entry_name] = entry

    ############################################
    # ALL non-context-manager related methods ##
    #   must be `classmethod`.                ##
    ############################################
    
    @classmethod
    @validate_call_
    def _validateWorkspace(cls, ws: HidraWorkspace):
        logs = ws.sample_log_names
        for k in REQUIRED_LOGS:
            if k not in logs:
                raise ValueError(f"NXstress requires log '{k}', which is not present")
    
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
        entry['definition'] = 'NXstress'

        # lists of 'start_time', 'end_time' for all subruns
        entry['start_time'] = ws.get_sample_log_values('start_time')
        entry['end_time'] = ws.get_sample_log_values('end_time')

        # the type of the primary strain calculation:
        #   this might also be 'two-theta', but 'd-spacing' seems more likely
        entry['processing_type'] = 'd-spacing'

        return entry 
                     
    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace, peaks: PeakCollection) -> NXentry:
        # Create and initialize a single NXstress-compatible NXentry tree:
        #   _multiple_ NXentry can exist within an NXstress-compatible HDF5 file.
        #   For example, distinct entries might be added for each set of
        #   data-reduction or sample conditions.
       
        # Verify that all properties required by NXstress are present.
        cls._validateWorkspace(ws)
        
        # Initialize this NXentry, and add required attributes.
        entry = cls._init(ws)
        
        # 'input_data' group
        entry[GROUP_NAME.INPUT_DATA] = _InputData.init_group(ws)
        
        # 'instrument' group
        entry[GROUP_NAME.INSTRUMENT] = _Instrument.init_group(ws)
        
        # 'SAMPLE_DESCRIPTION' group
        entry[GROUP_NAME.SAMPLE_DESCRIPTION] = _Sample.init_group(ws._sample_logs)
        
        # 'FIT' group[s] (one for each detector mask):
        masks = set(ws._mask_dict.keys())
        masks.add(DEFAULT_TAG)
        for mask in masks:
            ## TODO: mask naming (and storage) is messed up.  They all need to be accessed the same way,
            ##   regardless of whether or not the "default" mask is being accessed.
            ##   Here we assume that this loop also accesses data for the _DEFAULT_ mask, and that the default
            ##   mask has the '_DEFAULT_' name, and not some other name, such as 'main'?!          
            dgram_name = group_naming_scheme(GROUP_NAME.FIT, mask)
            if dgram_name in entry.NXprocess:
                raise RuntimeError(
                    f"Usage error: FIT (NXprocess) group '{name}' already exists in the current NXentry '{entry_name}'."
                )
            entry[dgram_name] = _Fit.init_group(mask, ws, peaks, ws._sample_logs)
        
        # 'PEAKS' group
        entry[GROUP_NAME.PEAKS] = _Peaks.init_group(peaks, ws._sample_logs)
        
        return entry
