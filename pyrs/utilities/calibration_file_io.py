# Class providing a series of static methods to work with files
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import DENEXDetectorShift, DENEXDetectorGeometry
import json
import warnings


def check_calibration_dictionary(calib_dict):
    """Check calibration file input

    The 2theta offset is stored as ``TTH_0`` (as written by ``DENEXDetectorShift.convert_to_dict``);
    files using the legacy key ``two_theta_0`` are accepted and moved to ``TTH_0`` (#1021). Missing
    parameters are set to 0 with an error of -1.

    Parameters
    ----------
    calib_dict: dictionary
        dictionary with calibration

    Returns
    -------
    calib_dict: dictionary
        dictionary with calibration

    Raises
    ------
    ValueError
        If the file gives both ``TTH_0`` and ``two_theta_0`` (or their errors) with different values.

    """
    for key, legacy in (("TTH_0", "two_theta_0"), ("error_TTH_0", "error_two_theta_0")):
        if legacy in calib_dict:
            if key in calib_dict and float(calib_dict[key]) != float(calib_dict[legacy]):
                raise ValueError(
                    "Calibration gives both {} = {} and legacy {} = {}; they must agree".format(
                        key, calib_dict[key], legacy, calib_dict[legacy]
                    )
                )
            calib_dict[key] = calib_dict.pop(legacy)

    keys = ["Lambda", "Rot_x", "Rot_y", "Rot_z", "Shift_x", "Shift_y", "Shift_z", "TTH_0"]

    for key in keys:
        if key not in calib_dict:
            calib_dict[key] = 0
            calib_dict["error_{}".format(key)] = -1
    # TTH_0 used to be ignored, so a file giving it without its error loaded; keep accepting such files
    calib_dict.setdefault("error_TTH_0", -1)

    return calib_dict


def check_calibration_status(status):
    """Verify that a calibration's optimizer status indicates it was actually refined

    The ``Status`` field stored in a calibration JSON file is the exit code from the
    ``scipy.optimize.least_squares`` call used to refine the calibration
    (see ``pyrs.calibration.mantid_peakfit_calibration``). That field is initialized
    to ``-1`` before any refinement has run, and negative values in general indicate
    the optimizer never produced a valid result. Applying such a calibration to
    reduction would silently shift every reduced pattern with no diagnostic.

    Note that ``status == 0`` (scipy's "maximum number of function evaluations
    exceeded" code, meaning the optimizer ran but did not converge) is currently
    accepted rather than rejected; only negative codes are treated as invalid here.

    Parameters
    ----------
    status : int
        Value of the calibration's ``Status`` field, as returned by
        :func:`read_calibration_json_file`.

    Raises
    ------
    RuntimeError
        If ``status`` is negative, meaning the calibration was never successfully
        refined.
    """
    if status < 0:
        raise RuntimeError(
            "Calibration status is {}, indicating it was never successfully refined "
            "(negative status codes are only produced by an optimizer that never converged, "
            "or by the -1 sentinel set before any refinement has run). "
            "Refusing to apply this calibration to reduction.".format(status)
        )


# calibration file formats reduction reads, by extension (#1054)
CALIBRATION_FORMATS = {".json": "json", ".nxs": "nexus"}


def calibration_format(calibration_file_name):
    """The format of a calibration file from its extension: ``"json"`` or ``"nexus"`` (#1054)

    Raises
    ------
    ValueError
        If the extension is neither ``.json`` nor ``.nxs`` (the ASCII geometry format was never readable,
        #1051; ``.h5`` is a PyRS project file and ``.nxs.h5`` raw event data)
    """
    name = str(calibration_file_name).lower()
    if name.endswith(".nxs.h5"):
        raise ValueError(f"{calibration_file_name!r} is raw event data (.nxs.h5), not a calibration file")
    for extension, file_format in CALIBRATION_FORMATS.items():
        if name.endswith(extension):
            return file_format
    raise ValueError(f"Calibration file must be a .json or .nxs file, got {calibration_file_name!r}")


def read_calibration_file(calibration_file_name):
    """A calibration from a JSON or NeXus calibration file (#1054): the one reader reduction uses

    Returns
    -------
    ~tuple
        (DENEXDetectorShift, DENEXDetectorShift, float, float, int) as :func:`read_calibration_json_file`

    Raises
    ------
    ValueError
        If the extension is not ``.json`` or ``.nxs`` (:func:`calibration_format`)
    """
    if calibration_format(calibration_file_name) == "nexus":
        from pyrs.utilities.calibration_nexus import read_calibration_nexus

        return read_calibration_nexus(calibration_file_name)
    return read_calibration_json_file(calibration_file_name)


def read_calibration_json_file(calibration_file_name):
    """Import calibration file in json format

    Example:  input JSON
    {u'Lambda': 1.452,
    u'Rot_x': 0.0,
    u'Rot_y': 0.0,
    u'Rot_z': 0.0,
    u'Shift_x': 0.0,
    u'Shift_y': 0.0,
    u'Shift_z': 0.0,
    u'TTH_0': 0.0,
    u'Status': 3,
    u'error_Lambda': 1.0829782933282927e-07,
    u'error_Rot_x': -1.0,
    u'error_Rot_y': -1.0,
    u'error_Rot_z': -1.0,
    u'error_Shift_x': -1.0,
    u'error_Shift_y': -1.0,
    u'error_Shift_z': -1.0,
    u'error_TTH_0': -1.0}

    The 2theta offset may also be given under the legacy key ``two_theta_0`` (see
    :func:`check_calibration_dictionary`).

    Parameters
    ----------
    calibration_file_name

    Returns
    -------
    ~tuple
        (DENEXDetectorShift, DENEXDetectorShift, float, float, int)
        detector position shifts as the calibration result,detector position shifts error from fitting
        status

    """

    # Check input
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "Calibration JSON file")

    # Parse JSON file
    with open(calibration_file_name, "r") as calib_file:
        calib_dict = json.load(calib_file)
    if calib_dict is None:
        raise RuntimeError("Failed to load JSON calibration file {}".format(calibration_file_name))

    return calibration_from_dict(calib_dict, calibration_file_name)


def calibration_from_dict(calib_dict, calibration_file_name=""):
    """A calibration as read from its dictionary (the calibration JSON's keys)

    Parameters
    ----------
    calib_dict : dict
        ``Shift_x`` ... ``TTH_0``, ``Lambda``, their ``error_`` keys and ``Status``; the legacy
        ``two_theta_0`` is accepted (:func:`check_calibration_dictionary`)
    calibration_file_name : str
        where the dictionary came from, for the shift and for error messages

    Returns
    -------
    ~tuple
        (DENEXDetectorShift, DENEXDetectorShift, float, float, int) as :func:`read_calibration_json_file`
    """
    # check for properly formated dictionary
    calib_dict = check_calibration_dictionary(dict(calib_dict))

    # Convert dictionary to DENEXDetectorShift
    try:
        shift = DENEXDetectorShift(
            shift_x=float(calib_dict["Shift_x"]),
            shift_y=float(calib_dict["Shift_y"]),
            shift_z=float(calib_dict["Shift_z"]),
            rotation_x=float(calib_dict["Rot_x"]),
            rotation_y=float(calib_dict["Rot_y"]),
            rotation_z=float(calib_dict["Rot_z"]),
            tth_0=float(calib_dict["TTH_0"]),
            calibration_file=calibration_file_name,
        )

    except KeyError as key_error:
        raise RuntimeError("Missing key parameter from JSON file {}: {}".format(calibration_file_name, key_error))

    # shift error
    try:
        shift_error = DENEXDetectorShift(
            shift_x=float(calib_dict["error_Shift_x"]),
            shift_y=float(calib_dict["error_Shift_y"]),
            shift_z=float(calib_dict["error_Shift_z"]),
            rotation_x=float(calib_dict["error_Rot_x"]),
            rotation_y=float(calib_dict["error_Rot_y"]),
            rotation_z=float(calib_dict["error_Rot_z"]),
            tth_0=float(calib_dict["error_TTH_0"]),
        )

    except KeyError as key_error:
        raise RuntimeError("Missing key parameter from JSON file {}: {}".format(calibration_file_name, key_error))

    # Wave length
    try:
        wave_length = calib_dict["Lambda"]
        wave_length_error = calib_dict["error_Lambda"]
    except KeyError as key_error:
        raise RuntimeError(
            "Missing wave length related parameter from JSON file {}: {}".format(calibration_file_name, key_error)
        )

    # Calibration status
    try:
        status = calib_dict["Status"]
    except KeyError as key_error:
        raise RuntimeError("Missing status parameter from JSON file {}: {}".format(calibration_file_name, key_error))

    return shift, shift_error, wave_length, wave_length_error, status


def import_instrument_setup(instrument_ascii_file):
    """Import instrument file in ASCII format

    Example:
      # comment
      arm = xxx  meter
      rows = 2048
      columns = 2048
      pixel_size_x = 0.00
      pixel_size_y = 0.00

    Parameters
    ----------
    instrument_ascii_file : str
        instrument file in plain ASCII format

    Returns
    -------
    DENEXDetectorGeometry
        Instrument geometry setup for HB2B

    """
    checkdatatypes.check_file_name(instrument_ascii_file, False, True, False, "Instrument definition ASCII file")

    instr_file = open(instrument_ascii_file, "r")
    setup_lines = instr_file.readlines()
    instr_file.close()

    # Init
    arm_length = detector_rows = detector_columns = pixel_size_x = pixel_size_y = None

    # Parse each line
    for line in setup_lines:
        line = line.strip()

        # skip empty and comment
        if line == "" or line.startswith("#"):
            continue

        terms = line.replace("=", " ").split()
        arg_name = terms[0].strip().lower()
        arg_value = terms[1]

        if arg_name == "arm":
            arm_length = float(arg_value)
        elif arg_name == "rows":
            detector_rows = int(arg_value)
        elif arg_name == "columns":
            detector_columns = int(arg_value)
        elif arg_name == "pixel_size_x":
            pixel_size_x = float(arg_value)
        elif arg_name == "pixel_size_y":
            pixel_size_y = float(arg_value)
        else:
            raise RuntimeError("Argument {} is not recognized".format(arg_name))
    # END-FOR

    instrument = DENEXDetectorGeometry(
        num_rows=detector_rows,
        num_columns=detector_columns,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        arm_length=arm_length,
        calibrated=False,
    )

    return instrument


def geometry_to_dict(geometry):
    """A detector geometry as the calibration's ``"geometry"`` block: ``detector_size`` [rows, columns],
    ``pixel_dimension`` [x, y] (m) and ``arm_length`` (m)."""
    assert isinstance(geometry, DENEXDetectorGeometry)
    rows, columns = geometry.detector_size
    pixel_x, pixel_y = geometry.pixel_dimension
    return {
        "detector_size": [int(rows), int(columns)],
        "pixel_dimension": [float(pixel_x), float(pixel_y)],
        "arm_length": float(geometry.arm_length),
    }


def geometry_from_dict(geometry_dict):
    """The DENEXDetectorGeometry of a calibration's ``"geometry"`` block (:func:`geometry_to_dict`)."""
    rows, columns = geometry_dict["detector_size"]
    pixel_x, pixel_y = geometry_dict["pixel_dimension"]
    return DENEXDetectorGeometry(
        int(rows), int(columns), float(pixel_x), float(pixel_y), float(geometry_dict["arm_length"]), False
    )


def read_calibration_geometry(calibration_file_name):
    """The detector geometry a calibration file is valid for, or None if it has none (a JSON file written
    before calibrations carried their geometry, #1053); a NeXus file always has one (#1054)."""
    if calibration_format(calibration_file_name) == "nexus":
        from pyrs.utilities.calibration_nexus import read_calibration_nexus_geometry

        return read_calibration_nexus_geometry(calibration_file_name)
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "Calibration JSON file")
    with open(calibration_file_name, "r") as calib_file:
        calib_dict = json.load(calib_file)
    geometry = calib_dict.get("geometry") if isinstance(calib_dict, dict) else None
    return geometry_from_dict(geometry) if isinstance(geometry, dict) else None


def calibration_to_dict(shifts, shifts_error, wave_length, wave_length_error, calibration_status):
    """The calibration JSON's dictionary of a calibration: ``Shift_x`` ... ``TTH_0`` and ``Lambda``, their
    ``error_`` keys and ``Status``, every value a float — what :func:`write_calibration_to_json` writes
    and :func:`calibration_from_dict` reads.
    """
    assert isinstance(shifts, DENEXDetectorShift)
    assert isinstance(shifts_error, DENEXDetectorShift)

    calibration_dict = shifts.convert_to_dict()
    calibration_dict["Lambda"] = wave_length

    calibration_dict.update(shifts_error.convert_error_to_dict())
    calibration_dict["error_Lambda"] = wave_length_error

    calibration_dict.update({"Status": calibration_status})

    for item in list(calibration_dict.keys()):
        calibration_dict[item] = float(calibration_dict[item])
    return calibration_dict


def write_calibration_to_json(
    shifts,
    shifts_error,
    wave_length,
    wave_lenngth_error,
    calibration_status,
    file_name=None,
    provenance=None,
    geometry=None,
):
    """Write geometry and wave length calibration to a JSON file

    Parameters
    ----------
    provenance : dict or None
        How the calibration was obtained (``pyrs.calibration.provenance``, #1053), written under the
        key ``"provenance"`` beside the values; None writes none. Readers of the values ignore it.
    geometry : DENEXDetectorGeometry or None
        The detector geometry the calibration is valid for (detector size, pixel size, L2), written
        under ``"geometry"`` (:func:`geometry_to_dict`); the calibration carries its geometry (#1053).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``provenance`` is not a dict or cannot be written as JSON (nothing is written then).
    """
    # Check inputs
    checkdatatypes.check_file_name(file_name, False, True, False, "Output JSON calibration file")

    calibration_dict = calibration_to_dict(shifts, shifts_error, wave_length, wave_lenngth_error, calibration_status)
    print("DICTIONARY:\n{}".format(calibration_dict))

    if geometry is not None:
        calibration_dict["geometry"] = geometry_to_dict(geometry)
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise TypeError(f"provenance must be a dict, got {type(provenance).__name__}")
        calibration_dict["provenance"] = provenance

    # serialized before the file is opened, so a provenance that is not valid JSON writes nothing
    text = json.dumps(calibration_dict)
    with open(file_name, "w") as outfile:
        outfile.write(text)
    print("[INFO] Calibration file is written to {}".format(file_name))


def read_calibration_provenance(calibration_file_name):
    """The provenance of a calibration file (#1053): the JSON's ``"provenance"`` block, or the NeXus file's
    (#1054); None if it has none

    Parameters
    ----------
    calibration_file_name : str
        calibration JSON or NeXus file

    Returns
    -------
    dict or None
    """
    if calibration_format(calibration_file_name) == "nexus":
        from pyrs.utilities.calibration_nexus import read_calibration_nexus_provenance

        provenance = read_calibration_nexus_provenance(calibration_file_name)
    else:
        checkdatatypes.check_file_name(calibration_file_name, True, False, False, "Calibration JSON file")
        with open(calibration_file_name, "r") as calib_file:
            calib_dict = json.load(calib_file)
        provenance = calib_dict.get("provenance") if isinstance(calib_dict, dict) else None
    if not isinstance(provenance, dict):
        return None
    from pyrs.calibration.provenance import SCHEMA_VERSION  # pyrs.calibration imports this module

    if provenance.get("schema") != SCHEMA_VERSION:
        warnings.warn(
            f"{calibration_file_name}: provenance schema {provenance.get('schema')!r}, this PyRS reads "
            f"schema {SCHEMA_VERSION}; returned as stored",
            stacklevel=2,
        )
    return provenance
