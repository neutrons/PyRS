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

class InputData_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    @validate_call
    def writeSubruns(cls, nx: NXFile, ws: HidraWorkspace):
        # Write input data for all subruns in the workspace.
        scan_points = ws.get_sub_runs()
        scans = np.stack([ws.get_detector_counts(p) for p in scan_points])
        
        # TODO: append to the group, if it already exists.
        if 'input_data' in nx['entry']:
            raise RuntimeError("not implemented: append detector_counts data to NXstress file")
        
        data_group = NXdata()
        data_group['detector_counts'] = scans
        data_group['scan_point'] = scan_points

        # Set attributes for axes and signal
        data_group.attrs['signal'] = 'detector_counts'
        data_group.attrs['axes'] = ['scan_point', '.']
        nx['entry']['input_data'] = data_group

    @classmethod
    @validate_call
    def readSubruns(cls, ws: HidraWorkspace, nx: NXFile):
        # Initialize `HidraWorkspace` detector_counts data.
        
        # TODO: append to the `HidraWorkspace`, if any detector_counts data already exists.
        if len(ws.get_sub_runs():
            raise RuntimeError("not implemented: append detector_counts data to workspace")
        
        data_group = nx['entry']['input_data']
        scan_points = data_group['scan_point']
        scans = data_group['detector_counts']
        for n, p in enumerate(scan_points):
            ws.set_raw_counts(p, scans[n])
