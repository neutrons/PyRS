"""
pyrs/utilities/NXstress/_input_data.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `input_data` `NXdata` subgroup.
"""

from nexusformat.nexus import NXdata, NXfield
import numpy as np

from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import CHUNK_SHAPE, FIELD_DTYPE


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

NONE: 'input_data' (NXdata, group) is allowed by the NXstress schema, but it is optional.
"""


class _InputData:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace, data: NXdata = None):
        # Initialize the input-data group.

        # Raw data may not actually be loaded in the `HidraWorkspace`:
        #   in that case, just initialize an empty NXdata group.
        scan_points = ws._raw_counts.keys()
        scans = (
            np.stack([ws.get_detector_counts(p).astype(FIELD_DTYPE.FLOAT_DATA.value) for p in scan_points])
            if len(scan_points)
            else np.empty((0, 0), dtype=FIELD_DTYPE.FLOAT_DATA.value)
        )

        # TODO: append to the group, if it already exists.
        if data is not None:
            raise RuntimeError("not implemented: append detector_counts data to NXstress file")
        else:
            data = NXdata()
        data["detector_counts"] = NXfield(scans, maxshape=(None, None), chunks=CHUNK_SHAPE(2))
        data["scan_point"] = scan_points

        # Set attributes for axes and signal
        data.attrs["signal"] = "detector_counts"
        data.attrs["axes"] = ["scan_point", "."]

        return data

    @classmethod
    @validate_call_
    def readSubruns(cls, ws: HidraWorkspace, data: NXdata):
        # Initialize `HidraWorkspace` detector_counts from input-data group.

        # TODO: append to the `HidraWorkspace`, if any detector_counts data already exists.
        scan_points = data["scan_point"].nxdata

        # An empty group means raw counts were not available when the file was written — nothing to load.
        if len(scan_points) == 0:
            return

        # `HidraWorkspace` must already contain its `SampleLogs`, and scan-points must match.
        if ws.get_sub_runs() != scan_points:
            raise RuntimeError("not implemented: append or change detector_counts data on existing workspace")

        scans = data["detector_counts"].nxdata
        for n, p in enumerate(scan_points):
            ws.set_raw_counts(p, scans[n])
