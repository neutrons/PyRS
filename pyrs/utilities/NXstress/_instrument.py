"""
pyrs/utilities/NXstress/_instrument.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `instrument` `NXinstrument` subgroup.
"""

import logging
from nexusformat.nexus import (
    NXbeam,
    NXcollection,
    NXdetector,
    NXdetector_module,
    NXfield,
    NXinstrument,
    NXlink,
    NXmonochromator,
    NXnote,
    NXsource,
    NXtransformations,
)
import numpy as np
import json

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import CHUNK_SHAPE, DEFAULT_TAG, FIELD_DTYPE, GROUP_NAME


_logger = logging.getLogger(__name__)

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ instrument                             (NXinstrument, group)
│   ├─ name                                (dataset)
│   ├─ source                              (NXsource, group)
│   ├─ detector                            (NXdetector, group)
│   └─ masks (optional)                     (NXcollection, group)
"""


class _Instrument:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    def _init(cls, name: str, short_name: str) -> NXinstrument:
        inst = NXinstrument()  # WARNING: cannot assign 'name' field via kwarg!
        inst["name"] = name
        inst["name"].attrs["short_name"] = short_name
        return inst

    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace) -> NXinstrument:
        """
        Create a new NXinstrument group subtree.
        Conventions:
          - Array datasets use explicit NumPy dtypes (np.int64 / np.float64).
          - Python native int/float are used for scalars.
          - DENEXDetectorGeometry.detectorsize -> (rows, cols)
          - DENEXDetectorGeometry.pixeldimension -> (px, py) (meters)
          - If present, setup._geometryshift is DENEXDetectorShift.
        """
        inst = cls._init("HB2B", "HB2B")

        N_scan_point = len(ws.get_sub_runs())

        # Detector base geometry and transformations
        geom: DENEXDetectorGeometry = ws.get_instrument_setup()
        shift: DENEXDetectorShift | None = ws.get_detector_shift()
        is_calibrated = shift is not None

        # Wavelength (`get_wavelength` returns either a single `float` or a `dict` keyed by subrun)
        wavelength = ws.get_wavelength(is_calibrated, False)
        if isinstance(wavelength, dict):
            # `dict` order should be the same as the sorted subruns order
            wavelength = [l_ for l_ in wavelength.values()]
        elif isinstance(wavelength, float):
            wavelength = list((wavelength,) * N_scan_point)
        elif wavelength is None:
            wavelength = list((np.nan,) * N_scan_point)
        else:
            raise RuntimeError(f"unable to parse wavelength from `HidraWorkspace.get_wavelength`: {wavelength}")
        if len(wavelength) != N_scan_point:
            raise ValueError(
                "Workspace must have either a single wavelength value,\n"
                "  or one wavelength value for each of {N_scan_point} subruns."
            )

        # Construct required NeXus subgroups:
        #   NXsource, NXmonochromator, NXdetector, NXtransformations.
        src = NXsource()
        src["type"] = NXfield("Reactor Neutron Source")
        src["probe"] = NXfield("neutron")

        mono = NXmonochromator()
        # `wavelength` by <sub run>?
        mono["wavelength"] = NXfield(wavelength, units="angstrom", calibrated=is_calibrated)

        det = NXdetector()
        det["type"] = "He_3 PSD"
        # Detector size (in rows and columns) and pixel size (in meters)
        nrows, ncols = geom.detector_size
        px_m, py_m = geom.pixel_dimension  # meters

        # det['data_size']   = NXfield(np.array([nrows, ncols], dtype=np.int64), dtype=np.int64)
        # det['x_pixel_size'] = NXfield(np.array(px_m, dtype=np.float64), dtype=np.float64, units='m')
        # det['y_pixel_size'] = NXfield(np.array(py_m, dtype=np.float64), dtype=np.float64, units='m')

        # Note: moving these fields to a subgroup `NXdetector_module` allows us to use scalars here,
        #   otherwise, the strict-mode validators require that we enter one value for each pixel!
        det["detector_bank"] = NXdetector_module(
            data_size=NXfield(np.array([nrows, ncols], dtype=np.int64), dtype=np.int64),
            fast_pixel_direction=NXfield(np.array(px_m, dtype=np.float64), dtype=np.float64, units="m"),
            slow_pixel_direction=NXfield(np.array(py_m, dtype=np.float64), dtype=np.float64, units="m"),
            depends_on=".",
        )

        # Beam intensity profile
        beam = NXbeam()
        # TODO: fill in the beam-intensity profile.

        # Transformations chain (values as native floats; axis vectors as float64 arrays)
        trans = NXtransformations()

        if is_calibrated and shift is not None:
            tx = float(shift.center_shift_x)  # meters
            ty = float(shift.center_shift_y)  # meters
            tz = float(shift.center_shift_z)  # meters

            # Sample-to-detector distance:
            # TODO: RE `L2`: At present there seems no way to determine if the `DENEXDetectorGeometry`
            #   already has had the _arm_ shift applied to it -- this issue needs to be fixed!
            distance = float(geom.arm_length)  # meters

            rotx = float(shift.rotation_x)  # degrees
            roty = float(shift.rotation_y)  # degrees
            rotz = float(shift.rotation_z)  # degrees
            tth0 = float(shift.two_theta_0)  # degrees
        else:
            tx = ty = tz = 0.0
            # Always write the actual arm_length, not 0.0
            distance = float(geom.arm_length)  # meters
            rotx = roty = rotz = tth0 = 0.0

        ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        depends = "."
        for name, val, vec, units, trtype in [
            ("translation_x", tx, ex, "m", "translation"),
            ("translation_y", ty, ey, "m", "translation"),
            ("translation_z", tz, ez, "m", "translation"),
            ("distance", distance, ez, "m", "translation"),
            ("rotation_x", rotx, ex, "deg", "rotation"),
            ("rotation_y", roty, ey, "deg", "rotation"),
            ("rotation_z", rotz, ez, "deg", "rotation"),
            # TODO: check order of rotations here!!!
            ("two_theta_zero", tth0, ex, "deg", "rotation"),
        ]:
            f = NXfield(val, units=units)
            f.attrs["transformation_type"] = trtype
            f.attrs["vector"] = vec
            # each transformation depends on the previous one in the chain
            f.attrs["depends_on"] = depends
            trans[name] = f
            depends = f"./transformations/{name}"

        det["transformations"] = trans
        # detector depends on the first transformation in the chain
        det["depends_on"] = "./transformations/translation_x"

        # Add a calibrated flag as extra metadata
        det["transformations"].attrs["calibrated"] = bool(is_calibrated)

        # Optional calibration provenance
        if is_calibrated and shift is not None:
            try:
                caldict = shift.convert_to_dict()
            except Exception:
                caldict = {
                    "center_shift_x": tx,
                    "center_shift_y": ty,
                    "center_shift_z": tz,
                    "rotation_x": rotx,
                    "rotation_y": roty,
                    "rotation_z": rotz,
                }
            note = NXnote()
            note["type"] = NXfield("text/plain")
            # Note: calibration_file may not be available on all shift objects
            try:
                note["file_name"] = shift.calibration_file
            except AttributeError:
                note["file_name"] = ""
            note["data"] = NXfield(json.dumps(caldict, indent=2))
        else:
            note = None

        inst[GROUP_NAME.SOURCE] = src
        inst[GROUP_NAME.BEAM] = beam
        inst[GROUP_NAME.MONOCHROMATOR] = mono
        inst[GROUP_NAME.DETECTOR] = det
        if note is not None:
            inst["detector_calibration"] = note

        # Add an optional 'masks' subgroup, to contain any detector or solid-angle masks.
        # For the moment, we only write detector masks -- the `HidraWorkspace` doesn't
        # yet seem to provide a way to distinguish between a detector and a solid-angle mask.
        inst[GROUP_NAME.MASKS] = _Masks.init_group(ws)

        return inst

    @classmethod
    @validate_call_
    def instrumentFromNexus(cls, instrument):
        """Read instrument geometry, detector shift, and wavelength from NXinstrument group.

        Parameters
        ----------
        instrument : NXinstrument
            The NXinstrument group from the HDF5 file

        Returns
        -------
        tuple
            (DENEXDetectorGeometry, DENEXDetectorShift | None, wavelength: np.ndarray)
        """

        # Read detector geometry from detector/detector_bank
        detector = instrument[GROUP_NAME.DETECTOR]
        detector_bank = detector["detector_bank"]

        # data_size: (nrows, ncols)
        data_size = detector_bank["data_size"].nxdata
        nrows, ncols = int(data_size[0]), int(data_size[1])

        # Pixel sizes in meters
        px_m = float(detector_bank["fast_pixel_direction"].nxdata)
        py_m = float(detector_bank["slow_pixel_direction"].nxdata)

        # Read transformations
        trans = detector["transformations"]
        calibrated = bool(trans.attrs.get("calibrated", False))

        # Read distance (arm_length)
        distance = float(trans["distance"].nxdata) if "distance" in trans else 0.0
        arm_length = distance

        # Create geometry object
        geometry = DENEXDetectorGeometry(nrows, ncols, px_m, py_m, arm_length, calibrated)

        # If calibrated, read shift parameters
        shift = None
        if calibrated:
            tx = float(trans["translation_x"].nxdata) if "translation_x" in trans else 0.0
            ty = float(trans["translation_y"].nxdata) if "translation_y" in trans else 0.0
            tz = float(trans["translation_z"].nxdata) if "translation_z" in trans else 0.0
            rotx = float(trans["rotation_x"].nxdata) if "rotation_x" in trans else 0.0
            roty = float(trans["rotation_y"].nxdata) if "rotation_y" in trans else 0.0
            rotz = float(trans["rotation_z"].nxdata) if "rotation_z" in trans else 0.0
            tth0 = float(trans["two_theta_zero"].nxdata) if "two_theta_zero" in trans else 0.0

            shift = DENEXDetectorShift(tx, ty, tz, rotx, roty, rotz, tth0)

        # Read wavelength from monochromator:
        #   we don't have access to the scan-point indices at this level,
        #     so we just return an `np.ndarray` in scan-point order.
        wavelength = None
        if GROUP_NAME.MONOCHROMATOR in instrument:
            mono = instrument[GROUP_NAME.MONOCHROMATOR]
            if "wavelength" in mono:
                wavelength = mono["wavelength"].nxdata

        return geometry, shift, wavelength


class _Masks:
    # `INSTRUMENT/masks` (NXcollection) is allowed by the `NXstress` schema,
    #    but is not specified by the schema.

    #
    #  * Masks are stored by name.
    #
    #  * Mask names must be distinct over both <detector masks> and <solid angle masks>:
    #    this allows us to successfully use the mask name as a suffix tag on other groups,
    #    without requiring the same sub-categorization for those groups.
    #
    #  * Throughout the PyRS codebase `None` is used to indicate that the default mask is
    #    being used.  For the purposes of the NXstress-compliant output, `None` will be
    #    mapped to `_definitions.DEFAULT_TAG`.  For this key *only*, the mask-name suffix
    #    is _omitted_ from gener
    #

    @classmethod
    @validate_call_
    def _init(cls) -> NXcollection:
        # initialize the `masks` (NXcollection) group
        masks = NXcollection()
        masks["names"] = NXfield(
            np.empty((0,), dtype=FIELD_DTYPE.STRING.value), maxshape=(None,), chunks=CHUNK_SHAPE(1)
        )
        masks["detector"] = NXcollection()
        masks["solid_angle"] = NXcollection()

        return masks

    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace, *, masks: NXcollection = None):
        # Write or append masks to the `NXcollection`

        # Allow append: both 'detector' and 'solid_angle' masks may exist,
        #   and if so, the masks will need to be added in separate steps.
        masks = masks if masks is not None else cls._init()
        names = masks["names"].nxvalue

        appending = len(names) > 0
        detector_masks = masks["detector"]
        solid_angle_masks = masks["solid_angle"]

        # Unify the `_mask_dict` to a standard Python `dict`.
        _mask_dict = ws._mask_dict.copy()
        if not appending:
            # There is only *one* default detector-mask, and for output purposes,
            #   the default mask *must* be initialized.
            default_mask = ws.get_detector_mask(True)
            if default_mask is None:
                _logger.warning(
                    "NXstress._instrument: no default "
                    " detector-mask is defined;\n"
                    "  for output purposes, a default mask will be created."
                )
            _mask_dict[DEFAULT_TAG] = (
                default_mask if default_mask is not None else cls._generate_default_mask(ws, detector_mask=True)
            )

            # Write the default-mask *once* to the masks group:
            #   this must happen first, as we may re-use it below as an `NXlink`.
            detector_masks[DEFAULT_TAG] = NXfield(_mask_dict[DEFAULT_TAG], units="")
            names.append(DEFAULT_TAG)

        # Check key correspondance in order to generate warning messages:
        #   here we do NOT replace the `None` key with `_definitions.DEFAULT_TAG`!
        ws_data_keys = set(ws._diff_data_set.keys())
        ws_mask_keys = set(ws._mask_dict.keys())

        for mask in cls.mask_keys(ws):
            if mask == DEFAULT_TAG:
                # WARNING: the default-mask should have been written before this point.
                continue

            if mask in names:
                raise RuntimeError(
                    f'Usage error: mask "{mask}" has already been written;\n'
                    + "  names must be distinct over both detector and solid-angle masks."
                )
            if mask in ws_data_keys and mask not in ws_mask_keys:
                _logger.warning(
                    f"NXstress._instrument: no mask entry exists corresponding to diffraction data '{mask}';\n"
                    "  for output purposes, the *default* mask will be written for this mask."
                )

            # WARNING: this section assumes that `detector_masks[DEFAULT_TAG]` already exists:
            #   it should have been written above.
            mask_array = _mask_dict.get(mask)
            units = "degrees" if (mask_array is not None and cls._is_solid_angle_mask(mask_array)) else ""

            # If no specific mask is present corresponding to a reduced diffraction dataset,
            #   a link will be created to the default detector-mask.
            if mask_array is not None:
                ds = NXfield(mask_array, units=units)
            else:
                # WORKAROUND to create an `NXlink` within an *unattached* group:
                #   this bypasses `NXlink.__init__` attempt to dereferene the parent group.
                ds = NXlink(target=DEFAULT_TAG, name=f"link_to_{DEFAULT_TAG}")
                ds._group = detector_masks

            if cls._is_solid_angle_mask(ds.nxdata):
                solid_angle_masks[mask] = ds
            else:
                detector_masks[mask] = ds

            # append the mask's name to the `names` list
            names.append(mask)

        masks["names"].resize((len(names),))
        masks["names"] = names

        return masks

    @classmethod
    def mask_keys(cls, ws: HidraWorkspace):
        # The complete set of mask names to be used for the `NXstress`-format file:
        #
        #   * The default mask is a detector-mask and will use `_definitions.DEFAULT_TAG` as a key;
        #     for output purposes, a default-mask will be generated, if not present.
        #
        #   * mask entries may be either detector or solid-angle masks, but they must have distinct names;
        #
        #   * There may be more mask entries than entries in `ws._diff_data_set`.
        #     For example, if the reduction process may not have been completed for all mask entries.
        #
        #   * Each entry in `ws._diff_data_set` *must*  have a corresponding mask.
        #     When a corresponding entry is not present in `ws._mask_dict`,
        #     this will be logged (as a warning), and then such an entry will be *linked*
        #     to the default mask, if not present.
        #
        #   * Any entry in `ws._var_data_set` that does not have a corresponding
        #     entry in `ws._diff_data_set` will be logged (as a warning) and skipped.
        #
        #   * At present, there's no special name for any default solid-angle mask.
        #

        keys = set(ws._mask_dict.keys()).union(ws._diff_data_set.keys())
        keys.discard(None)
        # a key for the default detector-mask must always be present
        keys.add(DEFAULT_TAG)
        return keys

    @classmethod
    def _generate_default_mask(cls, ws: HidraWorkspace, *, detector_mask: bool) -> np.ndarray | list[float]:
        # Generate an unmasked default mask.
        if not detector_mask:
            _logger.warning(
                "NXstress._instrument: *generating* a default solid-angle mask as `[-180.0, 180.0]`;\n"
                "  if this is not correct for your usage, please contact the developers."
            )
            return [-180.0, 180.0]

        if not ws._instrument_setup:
            raise RuntimeError("`_Masks._generate_default_mask`: workspace must have an instrument")
        return np.ones(ws._instrument_setup.detector_size, dtype=np.int64)

    @classmethod
    def _is_solid_angle_mask(cls, mask: np.ndarray) -> bool:
        # Check if a mask is a solid-angle mask

        # Solid-angle masks are comprised of pairs of <start angle> <stop angle>
        #   azimuthal *inclusion* zones.
        return len(mask.shape) == 1 and mask.shape[0] % 2 == 0 and np.issubdtype(mask.dtype, np.floating)

    @classmethod
    @validate_call_
    def masksFromNexus(cls, masks):
        """Read masks from NXcollection group.

        Parameters
        ----------
        masks : NXcollection
            The masks NXcollection group from the HDF5 file

        Returns
        -------
        tuple
            (default_mask_or_None, {mask_name: np.ndarray})
        """
        # Read mask names
        mask_names = masks["names"].nxdata
        if isinstance(mask_names, np.ndarray):
            mask_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in mask_names]
        else:
            mask_names = [mask_names.decode("utf-8") if isinstance(mask_names, bytes) else mask_names]

        default_mask = None
        mask_dict = {}

        # Check both detector and solid_angle collections
        for collection_name in ["detector", "solid_angle"]:
            if collection_name in masks:
                collection = masks[collection_name]
                for name in mask_names:
                    if name in collection:
                        mask_array = collection[name].nxdata
                        if name == DEFAULT_TAG:
                            default_mask = mask_array
                        else:
                            mask_dict[name] = mask_array

        return default_mask, mask_dict
