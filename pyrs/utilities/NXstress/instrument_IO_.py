"""
instrument_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `instrument` `NXinstrument` subgroup.
"""

from nexusformat.nexus import NXdata, Nxfile
import numpy as np
from pydantic import validate_call

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ instrument                             (NXinstrument, group)
│   ├─ name                                (dataset)
│   ├─ source                              (NXsource, group)
│   ├─ detector                            (NXdetector, group)
│   └─ mask (optional)                     (NXcollection, group)
"""

class Instrument_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    @validatecall
    def writeInstrument(cls, nx, DENEXDetectorGeometry?, DENEXDetectorShift?, calibrated: bool):
        pass
        

    @classmethod
    @validatecall
    def readInstrument(cls, ws: HidraWorkspace, nx: NXFile):
        pass
    
