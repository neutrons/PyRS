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

"""
# BLURB from 'perplexity.ai':

from nexusformat.nexus import (
    NXroot, NXentry, NXinstrument, NXsource, NXmonochromator,
    NXdetector, NXtransformations, NXnote, NXfield, nxsave
)
import numpy as np
import json

def write_instrument_geometry_nexusformat(self, setup, filename):
    """
    Create a new NeXus file with a compliant /entry/instrument subtree using nexusformat.
    Conventions:
      - Arrays saved to file use explicit NumPy dtypes (np.int64 / np.float64).
      - Python native int/float are used for scalars.
      - DENEXDetectorGeometry.detectorsize -> (rows, cols)
      - DENEXDetectorGeometry.pixeldimension -> (px, py) (meters)
      - If present, setup.geometryshift is DENEXDetectorShift with attribute access.
      - self.readwavelength() returns Å and is converted to meters.
    """

    # Wavelength (Å -> m)
    try:
        wl_ang = float(self.readwavelength())  # Å
        wl_m = wl_ang * 1e-10                 # meters
    except Exception:
        wl_m = float("nan")

    # Geometry (prefer calibrated)
    try:
        geom = setup.getinstrumentgeometry(calibrated=True)
    except Exception:
        geom = setup.getinstrumentgeometry(True)

    # Detector size (in rows and columns) and pixel size (in meters)
    nrows, ncols = geom.detector_size
    px_m, py_m = geom.pixel_dimension  # meters

    # Sample-to-detector distance (meters): try ISD (mm) then geometry arm length
    L2_m = None
    try:
        isd_mm = self.projecth5["raw data"]["logs"]["ISD"][()]
        L2_m = float(np.ravel(isd_mm)) / 1000.0
    except Exception:
        arm = getattr(geom, "armlength", None)
        L2_m = float(arm) if arm is not None else None

    # Optional detector shift/rotation (DENEXDetectorShift)
    shift_obj = getattr(setup, "geometryshift", None)
    is_calibrated = shift_obj is not None
    if is_calibrated:
        tx = float(shift_obj.centershiftx)  # meters
        ty = float(shift_obj.centershifty)  # meters
        tz = float(shift_obj.centershiftz)  # meters
        rotx = float(shift_obj.rotationx)   # degrees
        roty = float(shift_obj.rotationy)   # degrees
        rotz = float(shift_obj.rotationz)   # degrees
    else:
        tx = ty = tz = 0.0
        rotx = roty = rotz = 0.0

    # Build NeXus groups
    src = NXsource()
    src["type"] = NXfield("neutron")

    mono = NXmonochromator()
    mono["wavelength"] = NXfield(wl_m, units="m")

    det = NXdetector()
    det["data_origin"] = NXfield(np.array([0, 0], dtype=np.int64), dtype=np.int64)
    det["data_size"]   = NXfield(np.array([nrows, ncols], dtype=np.int64), dtype=np.int64)
    det["x_pixel_size"] = NXfield(np.array(px_m, dtype=np.float64), dtype=np.float64, units="m")
    det["y_pixel_size"] = NXfield(np.array(py_m, dtype=np.float64), dtype=np.float64, units="m")
    if L2_m is not None:
        det["distance"] = NXfield(np.array(L2_m, dtype=np.float64), dtype=np.float64, units="m")

    # Transformations chain (values as native floats; axis vectors as float64 arrays)
    trans = NXtransformations()
    depends = "."

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for name, val, vec, units, trtype in [
        ("translation_x", tx,   ex, "m",   "translation"),
        ("translation_y", ty,   ey, "m",   "translation"),
        ("translation_z", tz,   ez, "m",   "translation"),
        ("rotation_x",    rotx, ex, "deg", "rotation"),
        ("rotation_y",    roty, ey, "deg", "rotation"),
        ("rotation_z",    rotz, ez, "deg", "rotation"),
    ]:
        f = NXfield(val, units=units)
        f.attrs["transformation_type"] = trtype
        f.attrs["vector"] = vec
        f.attrs["depends_on"] = depends
        trans[name] = f
        depends = f"./transformations/{name}"

    det["transformations"] = trans
    det.attrs["depends_on"] = depends

    # Calibrated flags as extra metadata
    det.attrs["calibrated"] = bool(is_calibrated)
    det["transformations"].attrs["calibrated"] = bool(is_calibrated)

    # Optional calibration provenance
    if is_calibrated:
        try:
            caldict = shift_obj.converttodict()
        except Exception:
            caldict = {
                "centershiftx": tx, "centershifty": ty, "centershiftz": tz,
                "rotationx": rotx, "rotationy": roty, "rotationz": rotz,
            }
        note = NXnote()
        note["type"] = NXfield("text/plain")
        note["data"] = NXfield(json.dumps(caldict, indent=2))
    else:
        note = None

    inst = NXinstrument()
    inst["source"] = src
    inst["monochromator"] = mono
    inst["detector"] = det
    if note is not None:
        inst["detector_calibration"] = note

    entry = NXentry()
    entry["instrument"] = inst

    root = NXroot()
    root["entry"] = entry

    nxsave(filename, root)

"""        

    @classmethod
    @validatecall
    def readInstrument(cls, ws: HidraWorkspace, nx: NXFile):
        pass
    
