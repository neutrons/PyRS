"""
instrument_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `instrument` `NXinstrument` subgroup.
"""

from nexusformat.nexus import (
    NXdetector, NXfield, NXinstrument, NXmonochomator,
    NXnote, NXsource, NXtransformations
)
import numpy as np
import json
from pydantic import validate_call

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ instrument                             (NXinstrument, group)
│   ├─ name                                (dataset)
│   ├─ source                              (NXsource, group)
│   ├─ detector                            (NXdetector, group)
│   └─ masks (optional)                     (NXcollection, group)
"""

class _Masks:
    # `INSTRUMENT/masks` (NXcollection) is allowed by the `NXstress` schema,
    #    but is not specified by the schema.

    # Masks are stored by name.
    # Mask names must be distinct over both <detector masks> and <solid angle masks>:
    #   this allows us to successfully use the mask name as a suffix tag on other groups,
    #   without requiring the same sub-categorization for those groups.

    def _init(cls) -> NXcollection:
        # initialize the `masks` (NXcollection) group
        masks = NXcollection()
        masks['names'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                 maxshape=(None,), chunks=chunk_shape)
        masks['detector'] = NXcollection()
        masks['solid_angle'] = NXcollection()

        return masks

    def init_group(cls, ws: HidraWorkspace, bool detectorMasks = True, masks: NXcollection = None):
        # write or append masks to the `INSTRUMENT/masks` group

        # Allow append: both 'detector' and 'solid_angle' masks may exist,
        #   and will need to be added in separate steps.
        masks = masks if masks is not None else cls._init()
        names = masks['names'].aslist()
        appending = len(names) > 0
        
        # Unify the `_mask_dict` to a standard Python `dict`.
        _masks = ws._mask_dict.copy()
        if not appending and bool(ws._default_mask):
            # There's only one default mask.
            _masks[DEFAULT_TAG] = ws._default_mask
        
        dest = masks['detector'] if detectorMasks else masks['solid_angle']
        for mask in _masks:
            if mask in names:
                raise RuntimeError(
                    f'Usage error: mask "{name}" has already been written:\n'
                    + '  names must be distinct over both detector and solid-angle masks'
                )
            names.append(mask)
            dest[mask] = NXfield(_masks[mask], units='')
        masks['names'].resize((len(names),))
        masks['names'] = names
        
        return masks
    
class _Instrument:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    @validatecall
    def init_group(cls, ws: HidraWorkspace):
        """
        Create a new NXinstrument group subtree.
        Conventions:
          - Array datasets use explicit NumPy dtypes (np.int64 / np.float64).
          - Python native int/float are used for scalars.
          - DENEXDetectorGeometry.detectorsize -> (rows, cols)
          - DENEXDetectorGeometry.pixeldimension -> (px, py) (meters)
          - If present, setup._geometryshift is DENEXDetectorShift.
        """
        setup: HidraSetup = ws.get_instrument_setup()

        # Wavelength (Angstrom -> m)
        wl_ang = setup.get_wavelength(None)  # maybe: Angstrom
        wl_m = float(wl_ang) * 1.0e-10 if wl_ang is not None else float("nan")

        # Geometry (prefer calibrated)
        geom: DENEXDetectorGeometry = setup.getinstrumentgeometry(calibrated=True)
        
        # Optional detector shift / rotation:
        #   use any detector shift attached to the setup,
        #   fallback to any shift attached to the workspace.
        shift_obj: DENEXDetectorShift = setup._geometry_shift\
            if bool(setup._geometry_shift) else ws.get_detector_shift()
        is_calibrated = shift_obj is not None

        # Construct required NeXus subgroups:
        #   NXsource, NXmonochromator, NXdetector, NXtransformations.
        src = NXsource()
        src['type'] = NXfield('Reactor neutron source')
        src['probe'] = NXfield('neutron')

        mono = NXmonochromator()
        mono['wavelength'] = NXfield(wl_m, units='m')

        det = NXdetector()
        det['type'] = 'He_3 PSD'
        # Detector size (in rows and columns) and pixel size (in meters)
        nrows, ncols = geom.detector_size
        px_m, py_m = geom.pixel_dimension  # meters

        # Sample-to-detector distance (meters).
        L2_m = geom.arm_length
        
        det["data_origin"] = NXfield(np.array([0, 0], dtype=np.int64), dtype=np.int64)
        det["data_size"]   = NXfield(np.array([nrows, ncols], dtype=np.int64), dtype=np.int64)
        det["x_pixel_size"] = NXfield(np.array(px_m, dtype=np.float64), dtype=np.float64, units="m")
        det["y_pixel_size"] = NXfield(np.array(py_m, dtype=np.float64), dtype=np.float64, units="m")
        
        # TODO: "L2" could *possibly* go into TRANSFORMATIONS,
        #   where the combined endpoint-transformation of the detector array
        #   could be stored.
        det["distance"] = NXfield(np.array(L2_m, dtype=np.float64), dtype=np.float64, units="m")

        # Beam intensity profile
        beam = NXbeam()
        
        # Transformations chain (values as native floats; axis vectors as float64 arrays)
        trans = NXtransformations()
        depends = "."

        if is_calibrated:
            tx = float(shift_obj.centershiftx)  # meters
            ty = float(shift_obj.centershifty)  # meters
            tz = float(shift_obj.centershiftz)  # meters
            rotx = float(shift_obj.rotationx)   # degrees
            roty = float(shift_obj.rotationy)   # degrees
            rotz = float(shift_obj.rotationz)   # degrees
            tth0 = float(shift_obj.two_theta_0) # degrees
        else:
            tx = ty = tz = 0.0
            rotx = roty = rotz = tth0 = 0.0
            
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for name, val, vec, units, trtype in [
            ("translation_x",  tx,   ex, "m",   "translation"),
            ("translation_y",  ty,   ey, "m",   "translation"),
            ("translation_z",  tz,   ez, "m",   "translation"),
            ("rotation_x",     rotx, ex, "deg", "rotation"),
            ("rotation_y",     roty, ey, "deg", "rotation"),
            ("rotation_z",     rotz, ez, "deg", "rotation"),
            ("two_theta_zero", tth0, ex, "deg", "rotation")
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
            note["file_name"] = setup._calibration_file
            note["data"] = NXfield(json.dumps(caldict, indent=2))
        else:
            note = None

        inst = NXinstrument(name='HB2B')
        inst[GROUP_NAME.SOURCE] = src
        inst[GROUP_NAME.MONOCHROMATOR] = mono
        inst[GROUP_NAME.DETECTOR] = det
        inst[GROUP_NAME.BEAM] = beam
        if note is not None:
            inst["detector_calibration"] = note
        
        # Add an optional 'masks' subgroup, to contain any detector or solid-angle masks.
        # For the moment, we only write detector masks -- the `HidraWorkspace` doesn't
        # yet seem to provide a way to distinguish between a detector and a solid-angle mask.
        inst[GROUP_NAME.MASKS] = _Masks.init_group(ws, detectorMasks=True)
        
        return inst
