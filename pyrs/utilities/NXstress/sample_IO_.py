"""
sample_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `sample` `NXsample` subgroup.
"""

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ sample                                 (NXsample, group)
│   ├─ name                                (dataset)
│   ├─ chemical_formula (optional)         (dataset)
│   ├─ temperature (optional)              (dataset)
│   ├─ stress_field (optional)             (dataset)
│   └─ gauge_volume (optional)             (NXparameters, group)
"""

class Sample_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################
    pass


# """
from typing import Optional
import numpy as np
from nexusformat.nexus import NXFile, NXsample, NXfield
from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.samplelogs import SampleLogs

def writeSampleDescription(nx: NXFile, sampleLogs: SampleLogs) -> NXsample:
    """
    Populate ENTRY/SAMPLE_DESCRIPTION (NXsample) following NXstress schema:
      - subrun[nP]: link to the scanpoint axis
      - vx[nP], vy[nP], vz[nP]: sample positions in mm (from SampleLogs, converted via PointList)
      - name: sample descriptive name if present in logs; otherwise 'unknown'
      - chemical_formula: sample formula if present in logs; otherwise 'unknown'
      - [optional fields, only if present in the logs]: 'temperature', 'stress_field'
    Assumptions:
      - nx contains an existing NXentry at path 'ENTRY'
      - this is the first write of 'SAMPLE_DESCRIPTION' for this file
    """
    # Fetch the existing NXentry group
    entry = nx["ENTRY"]

    # Create SAMPLE_DESCRIPTION as an NXsample
    sd = NXsample()
    entry["sample"] = sd  # creates ENTRY/SAMPLE_DESCRIPTION

    # 1) Sample name (required by NXstress). Try the expected log key; fall back to 'unknown'.
    # there's only one value for 'SampleName'
    sd["name"] = sampleLogs.get(HidraConstants.SAMPLE_NAME, ('unknown',))[0]

    # 2) Link scanpoints to subruns: subrun[nP] (unitless)
    # SampleLogs.subruns is a SubRuns object; use .rawcopy() to get a NumPy array
    subruns = sampleLogs.subruns.rawcopy()
    subrun_field = NXfield(subruns.astype(np.int64), name="subrun")
    subrun_field.attrs["units"] = "NX_UNITLESS"
    sd["subrun"] = subrun_field
    N_scan = len(subruns)

    # 3) Sample positions per scanpoint (mm). Use SampleLogs.getpointlist().
    # PointList returns vx, vy, vz arrays in millimeters by design.
    pl = sampleLogs.getpointlist()
    for axis_name, axis_values in zip(HidraConstants.SAMPLECOORDINATENAMES, (pl.vx, pl.vy, pl.vz)):
        vs = np.asarray(axis_values, dtype=np.float64)
        if vs.shape[0] != N_scan:
            raise RuntimeError(
                f"NXstress required log '{axis_name}' has unexpected shape.\n"
                f"  First axis should be <scan point> (== {N_scan}), not {vs.shape[0]}"
            ) 
        f = NXfield(vs, name=axis_name)
        f.attrs["units"] = "mm"
        sd[axis_name] = f

    # Optionally, add other NXstress SAMPLE_DESCRIPTION fields if available in logs:
    #   - `HidraConstants.CHEMICAL_FORMULA` (NXCHAR)
    #   - `HidraConstants.TEMPERATURE`[nTemp] (NXTEMPERATURE)
    #   - `HidraConstants.STRESS_FIELD`[nsField] (with `@direction` attr = 'x'|'y'|'z')
    # The lines below are safe no-ops if the corresponding logs are not present.
    sd["chemical_formula" = sampleLogs.get(HidraConstants.CHEMICAL_FORMULA, ('unknown',))[0]

    # Example of temperature if present (stored as numeric array and units carried separately)
    if HidraConstants.TEMPERATURE in sampleLogs:
        tvals = np.asarray(sampleLogs[tkey], dtype=np.float64)
        tf = NXfield(tvals, name="temperature")
        tf.attrs["units"] = sampleLogs.units(tkey) or "K"
        sd["temperature"] = tf
 
    # Example of stress_field if present (values + direction attribute)
    if HidraConstants.STRESS_FIELD in sampleLogs:
        # TODO: we don't have an example of these entries, so the dimensions may not be correct!
        # -- Assuming:
        #      <stress field> :: (<scan points>, ...)
        #      <stress field direction > :: {'x', 'y', 'z'}: scalar
        #              
        sf = np.asarray(sampleLogs[HidraConstants.STRESS_FIELD], dtype=np.float64)
        if sf.shape[0] != N_scan:
            raise RuntimeError(
                f"NXstress required log '{HidraConstants.STRESS_FIELD}' has unexpected shape.\n"
                f"  First axis should be <scan point> (== {N_scan}), not {sf.shape[0]}"
            )
        sff = NXfield(sf, name="stress_field")
        # If a direction log exists, attach it; otherwise default to 'x'
        direction_key = HidraConstants.STRESS_FIELD_DIRECTION
        direction = sampleLogs[direction_key] if direction_key in sampleLogs else 'x'
        sff.attrs["direction"] = direction
        sd["stress_field"] = sff

    return sd
