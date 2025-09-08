"""
input_data_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `input_data` `NXdata` subgroup.
"""

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

NONE: 'input_data' (NXdata, group) is allowed by the NXstress schema, but it is optional.
"""


from nexusformat.nexus import NXdata, Nxfile
import numpy as np
from pydantic import validate_call

class _InputData:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    @validate_call
    def init_group(cls, ws: HidraWorkspace, data: NXdata = None):
        # Initialize the input-data group.
        scan_points = ws.get_sub_runs()
        scans = np.stack([ws.get_detector_counts(p) for p in scan_points])
        
        # TODO: append to the group, if it already exists.
        if data is not None:
            raise RuntimeError("not implemented: append detector_counts data to NXstress file")
        else:        
            data = NXdata()
        data['detector_counts'] = scans
        data['scan_point'] = scan_points

        # Set attributes for axes and signal
        data.attrs['signal'] = 'detector_counts'
        data.attrs['axes'] = ['scan_point', '.']
        
        return data

    @classmethod
    @validate_call
    def readSubruns(cls, ws: HidraWorkspace, nx: NXFile, data: NXdata):
        # Initialize `HidraWorkspace` detector_counts from input-data group.
        
        # TODO: append to the `HidraWorkspace`, if any detector_counts data already exists.
        if len(ws.get_sub_runs():
            raise RuntimeError("not implemented: append detector_counts data to workspace")
        
        scan_points = data['scan_point']
        scans = data['detector_counts']
        for n, p in enumerate(scan_points):
            ws.set_raw_counts(p, scans[n])
