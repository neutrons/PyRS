"""
NXstress_IO

Primary service class for NeXus NXstress-compatible I/O.
"""
from nexusformat.nexus import NXEntry, NXFile
from pydantic import validate_call

from pyrs.core.workspaces import HidraWorkspace

from NXstress import required_logs
from .input_data_IO_ import InputData_IO
from .instrument_IO_ import Instrument_IO
from .sample_IO_ import Sample_IO
from .fit_IO_ import Fit_IO
from .peaks_IO_ import Peaks_IO


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

/<entryname>                               (NXentry, group)
│
├─ @definition                             (attribute: "NXstress")
├─ @start_time                             (attribute: ISO8601 string)
├─ @end_time                               (attribute: ISO8601 string)
├─ @processingtype                         (attribute: string)
│
├─ instrument                             (NXinstrument, group)
│   ├─ name                                (dataset)
│   ├─ source                              (NXsource, group)
│   ├─ detector                            (NXdetector, group)
│   └─ mask (optional)                     (NXcollection, group)
│
├─ sample                                 (NXsample, group)
│   ├─ name                                (dataset)
│   ├─ chemical_formula (optional)         (dataset)
│   ├─ temperature (optional)              (dataset)
│   ├─ stress_field (optional)             (dataset)
│   └─ gauge_volume (optional)             (NXparameters, group)
│
├─ fit                                    (NXprocess, group)
│   ├─ @date                               (attribute: ISO8601 string)
│   ├─ @program                            (attribute: string)
│   ├─ description                         (NXnote, group)
│   ├─ peakparameters                      (NXparameters, group)
│   └─ diffractogram                       (NXdata, group)
│        ├─ diffractogram                  (dataset)
│        ├─ diffractogram_errors           (dataset)
│        ├─ daxis/xaxis                    (dataset)
│        ├─ @axes                          (attribute: string)
│        └─ @signal                        (attribute: string)
│
├─ peaks                                  (NXreflections, group)
│   ├─ h                                   (dataset)
│   ├─ k                                   (dataset)
│   ├─ l                                   (dataset)
│   └─ phase_name                          (dataset)
"""

class NXstress_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################
    
    @classmethod
    @validate_call
    def _validateWorkspace(cls, ws: HidraWorkspace):
        logs = ws.sample_log_names
        for k in required_logs:
            if k not in logs:
                raise ValueError(f"NXstress requires log '{k}', which is not present")
    
    @classmethod
    @validate_call
    def _init_group(cls, nx: NXFile, ws: HidraWorkspace) -> NXEntry:
        # Write the attributes required at the outermost level.
        """
        ├─ @definition                             (attribute: "NXstress")
        ├─ @start_time                             (attribute: ISO8601 string)
        ├─ @end_time                               (attribute: ISO8601 string)
        ├─ @processingtype                         (attribute: string)
        :: apart from 'definition', these attributes may also be
             lists by subrun.
        """
        
        entry = None
        if 'entry' in nx.root:
            entry = nx.root['entry']
        else:
            entry = NXentry()
            nx.root['entry'] = NXentry()
            entry['definition'] = 'NXstress'
            
            # lists of 'start_time', 'end_time' for all subruns
            entry['start_time'] = ws.get_sample_log_values('start_time')
            entry['end_time'] = ws.get_sample_log_values('end_time')
            
            # the type of the primary strain calculation:
            #   this might also be 'two-theta', but 'd-spacing' seems more likely
            entry['processingtype'] = 'd-spacing'
        return entry 
             
        
    @classmethod
    @validate_call
    def write(cls, nx: NXFile, ws: HidraWorkspace):
        # Validate that all properties required by NXstress are present.
        cls._validateWorkspace(ws)
        
        # add required attributes, when initializing the file
        cls._init_group(nx, ws)
        
        # 'input_data' group
        InputData_IO.writeSubruns(nx, ws)
        
        # 'instrument' group
        # Q: 'calibrated' or not, shifted detectors or NOT -- how does 'HidraProjectFile' deal with this?
        Instrument_IO.writeInstrument(nx, ws.get_instrument_setup(), ws.get_detector_shift())
        
        # 'sample' group
        
