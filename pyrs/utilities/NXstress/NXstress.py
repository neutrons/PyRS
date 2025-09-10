"""
pyrs/utilities/NXstress/NXstress.py

Primary service class for NeXus NXstress-compatible I/O.
"""
import h5py
from nexusformat.nexus import (
    NXentry, NXFile
)
import numpy as np

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import REQUIRED_LOGS, FIELD_DTYPE
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
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################
    
    @classmethod
    @validate_call_
    def _validateWorkspace(cls, ws: HidraWorkspace):
        logs = ws.sample_log_names
        for k in REQUIRED_LOGS:
            if k not in logs:
                raise ValueError(f"NXstress requires log '{k}', which is not present")
    
    @classmethod
    @validate_call_
    def _init_group(cls, ws: HidraWorkspace) -> NXentry:
        # Create the NXentry and initialize any required attributes.
  
        ## TODO: support appending to an existing entry.
        
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
    def write(cls, nx: NXFile, ws: HidraWorkspace, peaks: PeakCollection, entry_number: int = 1):
        # Add an NXstress NXentry tree to a NeXus-format HDF5 file:
        #   this form allows _multiple_ NXentry to be added, each with its own <entry number>.
        #   For example, an entry could be added for each distint set of sample conditions.
        
        ## TODO: support appending additional subruns (, or masks) to an existing entry.
        
        ######################################################
        ## Recommended usage:                               ##
        ## -------------------------------------------------##
        ## from nexusformat import NXfile                   ##
        ## ...                                              ##
        ## ws: HidraWorkspace                               ##
        ## peaks: PeakCollection                            ##
        ## ...                                              ##
        ## # To write the first (, or only) entry:          ##
        ## with NXfile(<file name>.nxs, 'w') as f:          ##
        ##     NXstress.write(f, ws, peaks)                 ##
        ## -------------------------------------------------##
        ## # To write an additional entry:                  ##
        ## with NXfile(<file name>.nxs, 'a') as f:          ##
        ##     NXstress.write(f, ws, peaks, entry_number=2) ##
        ######################################################
        
        # Verify that all properties required by NXstress are present.
        cls._validateWorkspace(ws)

        # Do not assume that this is the only NXentry:
        #   start with the existing NXroot.
        root: NXroot = nx.root       
        entry_name = group_naming_scheme(GROUP_NAME.ENTRY, entry_number)
        if entry_name in root:
            raise RuntimeError("Not implemented: appending to existing entry '{entry_name}'.")
        
        # Initialize this NXentry, and add required attributes.
        entry = cls._init_group(ws)
        root[entry_name] = entry
        
        # 'input_data' group
        entry[GROUP_NAME.INPUT_DATA] = _InputData.init_group(nx, ws)
        
        # 'instrument' group
        entry[GROUP_NAME.INSTRUMENT] = _Instrument.init_group(ws)
        
        # 'SAMPLE_DESCRIPTION' group
        entry[GROUP_NAME.SAMPLE_DESCRIPTION] = _Sample.init_group(ws.sampleLogs)
        
        # 'FIT' group[s] (one for each detector mask):
        for mask in ws._diff_data_set:
            ## TODO: mask naming (and storage) is messed up.  They all need to be accessed the same way,
            ##   regardless of whether or not the "default" mask is being accessed.
            ##   Here we assume that this loop also accesses data for the _DEFAULT_ mask, and that the default
            ##   mask has the '_DEFAULT_' name, and not some other name, such as 'main'?!          
            name = group_naming_scheme(GROUP_NAME.FIT, mask)
            if name in entry:
                raise RuntimeError(f"Usage error: FIT (NXprocess) group '{name}' already exists in the NXstress file.")
            entry[group_naming_scheme(GROUP_NAME.FIT, mask)] = _Fit.init_group(mask, ws, peaks, ws.sampleLogs)
        
        # 'PEAKS' group
        entry[GROUP_NAME.PEAKS] = _Peaks.init_group(peaks, ws.sampleLogs)
        
