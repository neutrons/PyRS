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
    NXmonochromator,
    NXnote,
    NXsource,
    NXtransformations,
)
import numpy as np
import json

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.config import Config
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    CHUNK_SHAPE,
    DEFAULT_TAG,
    FIELD_DTYPE,
    GROUP_NAME,
    growable,
    nxstress_mask_name,
    nxstress_mask_names,
    tail_append,
)


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
    def _instrument_names(cls) -> tuple[str, str]:
        """Instrument name and short name for the NXinstrument group.

        Drawn from configuration rather than from the workspace: the Hidra project
        format records no instrument name anywhere, so there is nothing to read.
        Overriding `nxstress.instrument_name` is therefore what makes this writer
        usable at another beamline without a code change.

        Returns:
            `(name, short_name)`, falling back to `("HB2B", "HB2B")` if either is
            unset or blank.
        """
        fallback = "HB2B"
        names = []
        for key in ("nxstress.instrument_name", "nxstress.instrument_short_name"):
            try:
                value = Config[key]
            except Exception:  # noqa: BLE001 - any config failure falls back, loudly
                value = None
            if not isinstance(value, str) or not value.strip():
                _logger.warning(
                    f"NXstress._instrument: config key '{key}' is unset or blank;\n"
                    f"  falling back to '{fallback}'. Set it to write files for another instrument."
                )
                value = fallback
            names.append(value.strip())
        return names[0], names[1]

    @classmethod
    def _entry_wide_geometry(cls, wss: list[HidraWorkspace]) -> tuple:
        """Geometry, detector shift and calibration state, which all inputs must share.

        Unlike wavelength, these are single entry-wide values: `NXinstrument`
        holds one detector with one transformation chain. A mixed-instrument or
        mixed-calibration merge is not supported by this pass, and is rejected
        rather than silently resolved to the first input's geometry.
        """
        geometries = [ws.get_instrument_setup() for ws in wss]
        shifts = [ws.get_detector_shift() for ws in wss]

        # Compared field by field, not by `str()`: neither class defines
        # `__eq__`, and `DENEXDetectorGeometry` has no `__str__` either, so the
        # default repr embeds the object id and two identical geometries would
        # always read as different. The keys below are exactly the values this
        # writer goes on to write, which is what has to agree.
        for label, values, key in (
            ("instrument geometry", geometries, cls._geometry_key),
            ("detector shift", shifts, cls._shift_key),
        ):
            keys = [key(v) for v in values]
            if any(k != keys[0] for k in keys):
                raise RuntimeError(
                    f"NXstress._instrument: input workspaces disagree on {label}:\n"
                    + "".join(f"    [{n}] {k}\n" for n, k in enumerate(keys))
                    + "  A single NXentry describes one instrument configuration;\n"
                    "  mixed-instrument and mixed-calibration merges are not supported."
                )
        return geometries[0], shifts[0]

    @staticmethod
    def _geometry_key(geometry: DENEXDetectorGeometry | None) -> tuple | None:
        if geometry is None:
            return None
        return (geometry.arm_length, geometry.detector_size, geometry.pixel_dimension)

    @staticmethod
    def _shift_key(shift: DENEXDetectorShift | None) -> tuple | None:
        if shift is None:
            return None
        return (
            shift.center_shift_x,
            shift.center_shift_y,
            shift.center_shift_z,
            shift.rotation_x,
            shift.rotation_y,
            shift.rotation_z,
            shift.two_theta_0,
        )

    @classmethod
    def _concatenated_wavelength(cls, wss: list[HidraWorkspace], is_calibrated: bool) -> list:
        """Wavelength for every input, concatenated in workspace order.

        Wavelength is stored per scan point and can already vary *within* one
        workspace under existing PyRS semantics (`get_wavelength` may return a
        per-subrun dict), so it belongs to the scan-point family's
        concatenation pattern -- not to the cross-workspace equality check
        `_entry_wide_geometry` applies. See the plan's Decisions Log item 18.
        """
        combined: list = []
        for ws in wss:
            n_scan_point = len(ws.get_sub_runs())
            wavelength = ws.get_wavelength(is_calibrated, False)
            if isinstance(wavelength, dict):
                # `dict` order should be the same as the sorted subruns order
                wavelength = [l_ for l_ in wavelength.values()]
            elif isinstance(wavelength, float):
                wavelength = list((wavelength,) * n_scan_point)
            elif wavelength is None:
                wavelength = list((np.nan,) * n_scan_point)
            else:
                raise RuntimeError(f"unable to parse wavelength from `HidraWorkspace.get_wavelength`: {wavelength}")
            if len(wavelength) != n_scan_point:
                raise ValueError(
                    "Workspace must have either a single wavelength value,\n"
                    f"  or one wavelength value for each of {n_scan_point} subruns."
                )
            combined.extend(wavelength)
        return combined

    @classmethod
    @validate_call_
    def init_group(cls, wss: list[HidraWorkspace], data: NXinstrument | None = None) -> NXinstrument:
        """
        Create a new NXinstrument group subtree.
        Conventions:
          - Array datasets use explicit NumPy dtypes (np.int64 / np.float64).
          - Python native int/float are used for scalars.
          - DENEXDetectorGeometry.detectorsize -> (rows, cols)
          - DENEXDetectorGeometry.pixeldimension -> (px, py) (meters)
          - If present, setup._geometryshift is DENEXDetectorShift.

        Accepts the full set of input workspaces. Geometry, shift and
        calibration state are validated for agreement across them; wavelength
        is concatenated per scan point.
        """
        # Detector base geometry and transformations
        geom: DENEXDetectorGeometry
        shift: DENEXDetectorShift | None
        geom, shift = cls._entry_wide_geometry(wss)
        is_calibrated = shift is not None

        # Wavelength (`get_wavelength` returns either a single `float` or a `dict` keyed by subrun)
        wavelength = cls._concatenated_wavelength(wss, is_calibrated)

        if data is not None:
            # Append (spec 04c): geometry, detector shift and the masks are entry-wide
            # and already written; `wavelength` is the one per-scan-point field here, so
            # it is the only thing that grows. That the incoming geometry agrees with the
            # entry's is checked by `validateAppend`, before any mutation anywhere.
            tail_append(
                data[GROUP_NAME.MONOCHROMATOR]["wavelength"],
                np.asarray(wavelength, dtype=np.float64),
            )
            return data

        inst = cls._init(*cls._instrument_names())

        # Construct required NeXus subgroups:
        #   NXsource, NXmonochromator, NXdetector, NXtransformations.
        src = NXsource()
        src["type"] = NXfield("Reactor Neutron Source")
        src["probe"] = NXfield("neutron")

        mono = NXmonochromator()
        # `wavelength` by <sub run>?
        # `growable`: wavelength is per-scan-point and is sliced by `rows` on read,
        # so it belongs to the scan-point family and an append must extend it.
        mono["wavelength"] = NXfield(wavelength, units="angstrom", calibrated=is_calibrated, **growable(1))

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
            # Rotation ORDER is settled -- see the `depends_on` note below. The axis
            # on the next line is NOT: `two_theta_zero` is written about `ex`, while
            # the reduction applies two-theta about Y
            # (`reduce_hb2b_pyrs.py::rotate_detector_2theta` -> `_cal_rotation_matrix_y`)
            # after folding `two_theta_0` into the arm angle as a scalar offset
            # (`reduce_hb2b_pyrs.py:139`) rather than as a transformation of its own.
            # `instrument_geometry.py:166` calls it "rotation of detector arm about X",
            # contradicting that code; this writer followed the comment. Unresolved and
            # carried forward -- README Section 2.1, and spec 08/09's detector-calibration
            # fidelity work, which owns `DENEXDetectorShift`.
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

        # The detector's `depends_on` is the chain's ENTRY POINT, and NeXus resolves a
        # chain by following each field's own `depends_on` from there until ".".
        # `depends` currently holds the LAST link written, whose chain runs back
        # through every other one -- so naming it here makes all eight transformations
        # part of the geometry.
        #
        # This previously named `translation_x`, the FIRST link, whose own `depends_on`
        # is "." -- so traversal terminated immediately and the composed transform was
        # a bare x-translation with identity rotation. The other seven, including all
        # three rotations and the two-theta zero, were written to the file but never
        # reached. `instrumentFromNexus` reads each field by name and ignores the
        # chain, which is why nothing in PyRS ever noticed.
        #
        # The written ORDER was already correct, and this is now settled rather than
        # assumed: traversed from the last link, the rotation sub-chain composes to
        # Rx @ Ry @ Rz, matching `reduce_hb2b_pyrs.py::generate_rotation_matrix`.
        # Verified by `plans/NXstress-prod/probes/a5_transformations_chain.py` and
        # pinned by `test_instrument.py`. The order question that stood open here is
        # closed; the separate question of `two_theta_zero`'s AXIS is not -- see the
        # note at its entry in the chain above.
        det["depends_on"] = depends

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
        #
        # Masks are name-keyed and entry-wide, like the geometry above: the FIT
        # group writes one DIFFRACTOGRAM per mask spanning every scan point, so
        # the inputs have to agree on which masks exist before that is meaningful.
        cls._validate_masks_agree(wss)
        inst[GROUP_NAME.MASKS] = _Masks.init_group(wss[0])

        return inst

    @classmethod
    def _validate_masks_agree(cls, wss: list[HidraWorkspace]) -> None:
        """Every input must define the same masks: the same names, with the same arrays.

        Masks are entry-wide and written from the first input alone, so an input
        whose mask of the same name differs would have it silently replaced by the
        first input's on read.
        """
        key_sets = [set(_Masks.mask_keys(ws)) for ws in wss]
        if any(keys != key_sets[0] for keys in key_sets):
            raise RuntimeError(
                "NXstress._instrument: input workspaces define different masks:\n"
                + "".join(f"    [{n}] {sorted(keys)}\n" for n, keys in enumerate(key_sets))
                + "  Every input must define the same mask names."
            )
        reference = _Masks.mask_arrays(wss[0])
        differing = [
            f"    [{n}] {name}"
            for n, ws in enumerate(wss[1:], start=1)
            for name, array in sorted(_Masks.mask_arrays(ws).items())
            if not np.array_equal(array, reference[name])
        ]
        if differing:
            raise RuntimeError(
                "NXstress._instrument: input workspaces define masks of the same name with different "
                "contents than input [0]:\n" + "\n".join(differing) + "\n"
                "  Masks are entry-wide; every input must use the same mask arrays."
            )

    @classmethod
    def appendableDatasets(cls, instrument) -> dict:
        """Every dataset in `instrument` that an append grows -- just the wavelength.

        Geometry, detector shift and the masks are entry-wide: `validateAppend`
        requires them to *match*, so they never grow. Enumerated explicitly rather
        than by walking the group, because unlike `SAMPLE_DESCRIPTION` this group's
        non-scalar fields are mostly entry-wide arrays -- which is exactly how the
        length sweep this replaced came to demand that `masks/names` be resizable.

        Args:
            instrument: The target entry's existing `instrument` group.

        Returns:
            Path -> `NXfield`, for `NXstress._validateAppendableShapes`.
        """
        path = f"{GROUP_NAME.INSTRUMENT}/{GROUP_NAME.MONOCHROMATOR}/wavelength"
        mono = instrument[GROUP_NAME.MONOCHROMATOR] if GROUP_NAME.MONOCHROMATOR in instrument else None
        return {path: mono["wavelength"]} if mono is not None and "wavelength" in mono else {}

    @classmethod
    def validateAppend(cls, wss: list[HidraWorkspace], instrument) -> None:
        """Check incoming workspaces against an existing NXinstrument, without mutating it.

        One `NXentry` describes one instrument configuration: geometry, detector
        shift and calibration state are written once and never grow. Appending a
        workspace that disagrees with them would attribute its scan points to a
        geometry that is not the one they were measured with -- readable, and
        wrong. Kept separate from `init_group`'s append branch so that it can run
        in the pre-flight pass, before anything has been resized.

        Args:
            wss: Workspaces being appended.
            instrument: The target entry's existing `NXinstrument` group.

        Raises:
            RuntimeError: If geometry, detector shift or calibration state differ.
        """
        incoming_geometry, incoming_shift = cls._entry_wide_geometry(wss)
        existing_geometry, existing_shift, _ = cls.instrumentFromNexus(instrument)

        # The mask arrays are entry-wide and were written at a fixed size, so the
        # append branch below grows only `wavelength` and leaves them alone. That is
        # correct only while the incoming masks are the ones already on disk: a
        # workspace carrying a different set would have its masks silently dropped,
        # and `masksFromNexus` would hand it the first write's masks on read.
        cls._validate_masks_agree(wss)
        on_disk = instrument[GROUP_NAME.MASKS]["names"].nxdata
        if not isinstance(on_disk, np.ndarray):
            on_disk = [on_disk]
        existing_masks = {to_text(name) for name in on_disk}
        # `mask_keys` already routes through `nxstress_mask_names`, so the default
        # key is present and no second normalisation is needed.
        incoming_masks = set(_Masks.mask_keys(wss[0]))
        if incoming_masks != existing_masks:
            raise RuntimeError(
                f"NXstress._instrument: cannot append -- the incoming workspaces' detector masks "
                f"do not match the target entry's.\n"
                f"    on disk:  {sorted(existing_masks)}\n"
                f"    incoming: {sorted(incoming_masks)}\n"
                "  Masks are entry-wide and are written once; an append grows the scan-point axis "
                "only, so a mask set that differs cannot be recorded."
            )
        default_on_disk, masks_on_disk = _Masks.masksFromNexus(instrument[GROUP_NAME.MASKS])
        masks_on_disk[DEFAULT_TAG] = default_on_disk
        differing = sorted(
            name
            for name, array in _Masks.mask_arrays(wss[0]).items()
            if not np.array_equal(array, masks_on_disk.get(name))
        )
        if differing:
            raise RuntimeError(
                f"NXstress._instrument: cannot append -- the incoming workspaces' detector masks "
                f"{differing} have different contents from the target entry's.\n"
                "  Masks are entry-wide and are written once, so a different array cannot be recorded."
            )

        for label, existing, incoming, key in (
            ("instrument geometry", existing_geometry, incoming_geometry, cls._geometry_key),
            ("detector shift", existing_shift, incoming_shift, cls._shift_key),
        ):
            if key(existing) != key(incoming):
                raise RuntimeError(
                    f"NXstress._instrument: cannot append -- the incoming workspaces disagree with "
                    f"the target entry on {label}:\n"
                    f"    on disk:  {key(existing)}\n"
                    f"    incoming: {key(incoming)}\n"
                    "  A single NXentry describes one instrument configuration."
                )

    @classmethod
    @validate_call_
    def instrumentFromNexus(cls, instrument, rows=None):
        """Read instrument geometry, detector shift, and wavelength from NXinstrument group.

        Parameters
        ----------
        instrument : NXinstrument
            The NXinstrument group from the HDF5 file
        rows : np.ndarray, optional
            Boolean mask selecting one input workspace's rows of the
            concatenated wavelength array. Geometry and detector shift are
            entry-wide and are returned whole regardless.

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
                # A multi-workspace entry concatenates every input's wavelength
                # into one per-scan-point array; `rows` selects this workspace's.
                if rows is not None:
                    wavelength = wavelength[rows]

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
        # `masks["solid_angle"]` is created by `_init` and deliberately left empty:
        # it is the placeholder for solid-angle support, and `_generate_default_mask`
        # and the loop below both raise rather than write into it. `masksFromNexus`
        # still reads it, so a future writer needs no reader change.

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

        for mask in cls.mask_keys(ws):
            if mask == DEFAULT_TAG:
                # WARNING: the default-mask should have been written before this point.
                continue

            if mask in names:
                raise RuntimeError(
                    f'Usage error: mask "{mask}" has already been written;\n'
                    + "  names must be distinct over both detector and solid-angle masks."
                )
            # Every name here is a real detector mask with a real array. The former
            # `NXlink` branch -- for a name with no array -- is gone with the
            # `_diff_data_set` union that created such names. It never worked
            # anyway (the hand-built link was not serialised, so `names` listed
            # masks that resolved to nothing), and making it work would have been
            # worse: `masksFromNexus` would then have returned the default array
            # under each composite key, and `set_masks_from_dict` would have added
            # entries to `_mask_dict` that the original workspace never had.
            mask_array = _mask_dict[mask]

            if cls._is_solid_angle_mask(mask_array):
                raise NotImplementedError(
                    f"NXstress._instrument: mask '{mask}' looks like a solid-angle mask "
                    f"(1-D, even length, floating point).\n"
                    "  Solid-angle masks are not implemented: nothing in PyRS loads one into a "
                    "`HidraWorkspace`, so writing one has never been exercised and the "
                    "`solid_angle` group below is a placeholder for that work.\n"
                    "  Raising rather than guessing -- the alternative is filing a detector mask "
                    "under `solid_angle/` because it happened to be float-typed with an even "
                    "pixel count, which is every square detector."
                )

            detector_masks[mask] = NXfield(mask_array, units="")
            names.append(mask)

        masks["names"].resize((len(names),))
        masks["names"] = names

        return masks

    @classmethod
    def mask_arrays(cls, ws: HidraWorkspace) -> dict[str, np.ndarray]:
        """The detector-mask arrays `init_group` writes for `ws`, keyed by NXstress name.

        Includes the default mask, generated when the workspace has none, exactly as
        `init_group` does -- so two workspaces compare equal here exactly when they
        would write the same masks.
        """
        arrays = {nxstress_mask_name(key): np.asarray(value) for key, value in ws._mask_dict.items()}
        default_mask = ws.get_detector_mask(True)
        arrays[DEFAULT_TAG] = np.asarray(
            default_mask if default_mask is not None else cls._generate_default_mask(ws, detector_mask=True)
        )
        return arrays

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

        # The MASK namespace only. `_diff_data_set` keys are *diffractogram* keys --
        # reduction identifiers that reference a mask and optionally intersect it
        # with an eta region -- and they used to be unioned in here, which is what
        # put composites like `eta_-5.0` into `masks/names` with no array behind
        # them. Every such key's detector component already resolves, because the
        # default mask is always written; see
        # `_definitions.diffractogram_detector_mask`, which checks exactly that.
        return nxstress_mask_names(ws._mask_dict.keys())

    @classmethod
    def _generate_default_mask(cls, ws: HidraWorkspace, *, detector_mask: bool) -> np.ndarray | list[float]:
        # Generate an unmasked default mask.
        if not detector_mask:
            # Unreachable today: the only call site asks for a detector mask. Kept as a
            # raise rather than a plausible-looking `[-180.0, 180.0]`, so that whoever
            # implements solid-angle masks has to decide what a default one means
            # instead of inheriting a guess.
            raise NotImplementedError(
                "NXstress._instrument: generating a default solid-angle mask is not implemented.\n"
                "  Nothing in PyRS loads a solid-angle mask into a `HidraWorkspace`; the "
                "`solid_angle` group is a placeholder for that work."
            )

        if not ws._instrument_setup:
            raise RuntimeError("`_Masks._generate_default_mask`: workspace must have an instrument")
        # 1-D, (n_pixels,), matching what `set_detector_mask` stores and what a real
        # default looks like on disk. `detector_size` is `(nrows, ncols)`, and
        # returning that shape wrote a 2-D default which `set_detector_mask` then
        # REFUSED on read -- "Mask array with shape (4, 4) is not acceptable" -- so a
        # workspace with no mask of its own wrote a file PyRS could not read back.
        nrows, ncols = ws._instrument_setup.detector_size
        return np.ones(nrows * ncols, dtype=np.int64)

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
        # `nxvalue`/`nxdata` yields a bare scalar for a length-1 array, not a
        # 1-element sequence, so normalise the shape before the text conversion.
        mask_names = [to_text(name) for name in np.atleast_1d(masks["names"].nxdata)]

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
