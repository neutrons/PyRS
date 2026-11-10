"""
NeXus (HDF5) calibration files for HB2B (#1054).

A NeXus calibration file is a complete calibration: the same values, errors and status as the calibration
JSON, the detector geometry it is valid for, and its provenance (#1053), described with NeXus base
classes. Reduction reads it interchangeably with the JSON through
:func:`pyrs.utilities.calibration_file_io.read_calibration_file`.

Layout (``.nxs``)::

    /entry                                  NXentry
      /instrument                           NXinstrument
        /monochromator                      NXmonochromator: wavelength, wavelength_errors (angstrom)
        /detector                           NXdetector
          x_pixel_size, y_pixel_size        pixel pitch (m)
          x_pixel_offset [n_columns]        uncalibrated pixel offsets from the detector centre (m), along
          y_pixel_offset [n_rows]             index_0 and index_1 (DetectorModel.local_positions)
          calibration_date                  ISO 8601, when the calibration was made (if known)
          depends_on                        "transformations/shift_x"
          /calibration_method               NXnote: status (int), data = the provenance as JSON
          /transformations                  NXtransformations: DetectorModel's placement, innermost first
                                            (pyrs.core.detector_model.CALIBRATION_TRANSFORMATIONS); each a
                                            value with <name>_errors beside it, @transformation_type,
                                            @vector, @units and @depends_on
      /calibration                          NXprocess (optional, #1054 phase 2): program, version, date, method,
                                            the final covariance and its parameter names, one NXdata table
                                            per diagnostic (e.g. ring_points, strip_fits, peaks) and, on
                                            request, the reduced patterns under /patterns

Written and read with ``nexusformat``, as PyRS's NXstress output is. The frame is PyRS's: +Z along the incident
beam, +Y up. ``arm_rotation`` is stored as 0: the arm angle
belongs to each sub-run (``-get_detector_2theta()``), not to the calibration. Error codes are the JSON's:
-1 not refined (or held), -2 refined but undetermined.
"""

import datetime
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from nexusformat.nexus import (
    NeXusError,
    NXcollection,
    NXdata,
    NXdetector,
    NXentry,
    NXfield,
    NXgroup,
    NXinstrument,
    NXmonochromator,
    NXnote,
    NXprocess,
    NXtransformations,
    nxopen,
)

from pyrs import __version__

from pyrs.core.detector_model import CALIBRATION_TRANSFORMATIONS, DetectorModel, transformation_chain
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.utilities import checkdatatypes

__all__ = [
    "write_calibration_nexus",
    "read_calibration_nexus",
    "read_calibration_nexus_provenance",
    "read_calibration_nexus_geometry",
    "read_calibration_nexus_diagnostics",
    "table_columns",
]

_DETECTOR = "entry/instrument/detector"
_TRANSFORMATIONS = _DETECTOR + "/transformations"
_NOTE = _DETECTOR + "/calibration_method"
_MONOCHROMATOR = "entry/instrument/monochromator"
_PROCESS = "entry/calibration"


def _value(value: float, units: Optional[str] = None) -> NXfield:
    """A float64 field (bit for bit what was given), with its units."""
    field = NXfield(np.float64(value), dtype="float64")
    if units is not None:
        field.attrs["units"] = units
    return field


def write_calibration_nexus(
    file_name: str,
    shifts: DENEXDetectorShift,
    shifts_error: DENEXDetectorShift,
    wave_length: float,
    wave_length_error: float,
    calibration_status: int,
    geometry: DENEXDetectorGeometry,
    provenance: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write a calibration as a NeXus file (see the module docstring for the layout).

    Args:
        file_name: Output ``.nxs`` file; replaced if it exists.
        shifts: Calibrated detector shift, rotations and 2theta_0.
        shifts_error: Their uncertainties (or the -1 / -2 codes).
        wave_length: Calibrated wavelength (angstrom).
        wave_length_error: Its uncertainty (or a code).
        calibration_status: The refinement's status (reduction refuses < 0).
        geometry: The detector geometry the calibration is valid for. Unlike the JSON's, it is required:
            the transformation chain and pixel offsets are built from it.
        provenance: How the calibration was obtained (``pyrs.calibration.provenance``, #1053), or None.
        diagnostics: What the calibration was fitted to (#1054 phase 2), written to ``/entry/calibration``,
            or None: ``method`` (str); ``covariance``, None or ``(parameter names, matrix)``; ``tables``,
            ``{name: {column: 1D array}}`` with columns of one length per table (strings allowed); and
            ``patterns``, ``{name: {"two_theta", "intensity", "errors"}}`` (optional). Built by the
            calibrations' ``diagnostics()``.

    Raises:
        TypeError: If an argument has the wrong type, or the provenance cannot be written as JSON (nothing
            is written then).
    """
    checkdatatypes.check_file_name(file_name, False, True, False, "Output NeXus calibration file")
    for name, value, kind in (
        ("shifts", shifts, DENEXDetectorShift),
        ("shifts_error", shifts_error, DENEXDetectorShift),
        ("geometry", geometry, DENEXDetectorGeometry),
    ):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}, got {type(value).__name__}")
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError(f"provenance must be a dict, got {type(provenance).__name__}")
    # serialized before the file is opened, so a provenance that is not valid JSON writes nothing
    provenance_text = None if provenance is None else json.dumps(provenance)

    rows, columns = geometry.detector_size
    pixel_x, pixel_y = geometry.pixel_dimension
    local = DetectorModel(geometry).local_positions

    detector = NXdetector()
    detector["x_pixel_size"] = _value(pixel_x, "m")
    detector["y_pixel_size"] = _value(pixel_y, "m")
    detector["x_pixel_offset"] = NXfield(local(np.arange(columns), 0.0)[:, 0], dtype="float64", units="m")
    detector["y_pixel_offset"] = NXfield(local(0.0, np.arange(rows))[:, 1], dtype="float64", units="m")
    created = (provenance or {}).get("created")
    if created:
        detector["calibration_date"] = NXfield(str(created))
    detector["depends_on"] = NXfield("transformations/" + CALIBRATION_TRANSFORMATIONS[0][0])

    note = NXnote()
    note["status"] = NXfield(np.int32(calibration_status), dtype="int32")
    note["type"] = NXfield("application/json")
    note["description"] = NXfield("how this calibration was obtained (PyRS provenance, #1053)")
    if provenance_text is not None:
        note["data"] = NXfield(provenance_text)
    detector["calibration_method"] = note

    transformations = NXtransformations()
    chain = transformation_chain(shifts, geometry.arm_length)
    errors = transformation_chain(shifts_error, geometry.arm_length)  # the parameters' errors, by name
    for index, (step, error) in enumerate(zip(chain, errors)):
        field = _value(step.value, step.units)
        field.attrs["transformation_type"] = step.transformation_type
        field.attrs["vector"] = np.array(step.vector, dtype=np.float64)
        field.attrs["depends_on"] = chain[index + 1].name if index + 1 < len(chain) else "."
        if step.name == "arm_rotation":
            field.attrs["note"] = "set per sub-run to -(DAS arm 2theta); not a calibration parameter"
        transformations[step.name] = field
        if step.parameter is not None:
            transformations[step.name + "_errors"] = _value(error.value, step.units)
    detector["transformations"] = transformations

    monochromator = NXmonochromator()
    monochromator["wavelength"] = _value(wave_length, "angstrom")
    monochromator["wavelength_errors"] = _value(wave_length_error, "angstrom")

    instrument = NXinstrument()
    instrument["monochromator"] = monochromator
    instrument["detector"] = detector
    entry = NXentry()
    entry["program_name"] = NXfield("pyrs")
    entry["instrument"] = instrument
    if diagnostics is not None:
        entry["calibration"] = _diagnostics_group(diagnostics)
    with nxopen(file_name, "w") as root:
        root["entry"] = entry


def _column(values: Any) -> NXfield:
    """A table column: numbers as given, strings as UTF-8 bytes (HDF5 has no numpy unicode type)."""
    array = np.asarray(values)
    if array.dtype.kind in ("U", "O"):
        array = np.array([str(value).encode("utf-8") for value in array.reshape(-1)], dtype="S").reshape(array.shape)
    return NXfield(array)


def _diagnostics_group(diagnostics: Mapping[str, Any]) -> NXprocess:
    """The ``/entry/calibration`` NXprocess of a calibration's diagnostics (see write_calibration_nexus)."""
    process = NXprocess()
    process["program"] = NXfield("pyrs")
    process["version"] = NXfield(str(__version__))
    process["date"] = NXfield(datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    process["method"] = NXfield(str(diagnostics.get("method", "")))
    covariance = diagnostics.get("covariance")
    if covariance is not None:
        names, matrix = covariance
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (len(names), len(names)):
            raise TypeError(f"covariance of shape {matrix.shape} for {len(names)} parameters")
        process["covariance"] = NXfield(matrix, dtype="float64")
        process["covariance"].attrs["units"] = "products of the parameters' units (m, deg, angstrom)"
        process["covariance_parameters"] = _column(list(names))
    for name, columns in (diagnostics.get("tables") or {}).items():
        lengths = {column: np.size(values) for column, values in columns.items()}
        if len(set(lengths.values())) > 1:
            raise TypeError(f"diagnostics table {name!r} has columns of different lengths: {lengths}")
        table = NXdata()
        for column, values in columns.items():
            table[column] = _column(values)
        if "residual" in columns:
            table.attrs["signal"] = "residual"
        process[name] = table
    patterns = diagnostics.get("patterns")
    if patterns:
        collection = NXcollection()
        for name, pattern in patterns.items():
            data = NXdata()
            data["two_theta"] = NXfield(np.asarray(pattern["two_theta"], dtype=np.float64), units="deg")
            data["intensity"] = NXfield(np.asarray(pattern["intensity"], dtype=np.float64))
            data["intensity_errors"] = NXfield(np.asarray(pattern["errors"], dtype=np.float64))
            data.attrs["signal"] = "intensity"
            data.attrs["axes"] = "two_theta"
            collection[name] = data
        process["patterns"] = collection
    return process


def _open(file_name: str):
    """The file's NeXus tree, read-only (OSError if it is not an HDF5 file, as h5py raises)."""
    try:
        return nxopen(file_name, "r")
    except NeXusError as error:
        raise OSError(f"{file_name} cannot be read as a NeXus (HDF5) file: {error}") from None


def _child(group: NXgroup, path: str) -> Any:
    """``group[path]``, or None if it is missing (nexusformat raises NeXusError for a missing path)."""
    return group[path] if path in group else None


def _text(value: Any) -> Any:
    """A string as ``str``: bytes decoded, a one-element array (as other tools write) unwrapped."""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else value


def _scalar(field: NXfield, file_name: str) -> Any:
    """The single value of a field (a scalar, or of shape (1,) as other tools may write), else ValueError."""
    value = np.asarray(field.nxdata)
    if value.size != 1:
        raise ValueError(f"{file_name}: {field.nxpath} has shape {value.shape}, expected one value")
    return value.reshape(-1)[0].item()


def _check_chain(root: NXgroup, file_name: str) -> NXgroup:
    """The transformations group, after checking it is exactly the chain this PyRS applies (else ValueError):
    every transformation's type, vector, units and ``depends_on``, the detector's ``depends_on``, and the
    per-sub-run ``arm_rotation`` stored as 0."""
    detector, transformations = _child(root, _DETECTOR), _child(root, _TRANSFORMATIONS)
    if detector is None or transformations is None:
        raise RuntimeError(f"{file_name}: no {_TRANSFORMATIONS}; not a PyRS NeXus calibration file")
    first = "transformations/" + CALIBRATION_TRANSFORMATIONS[0][0]
    depends_on = _text(detector["depends_on"].nxdata) if "depends_on" in detector else None
    if depends_on != first:
        raise ValueError(f"{file_name}: the detector depends_on {depends_on!r}, expected {first!r}")
    for index, (name, transformation_type, vector, units, _) in enumerate(CALIBRATION_TRANSFORMATIONS):
        field = _child(transformations, name)
        if field is None:
            raise RuntimeError(f"{file_name}: transformation {name!r} is missing")
        expected_next = (
            CALIBRATION_TRANSFORMATIONS[index + 1][0] if index + 1 < len(CALIBRATION_TRANSFORMATIONS) else "."
        )
        attrs = field.attrs
        found = (
            _text(attrs.get("transformation_type")),
            tuple(float(v) for v in np.asarray(attrs.get("vector", ()), dtype=float).reshape(-1)),
            _text(attrs.get("units")),
            _text(attrs.get("depends_on")),
        )
        if found != (transformation_type, vector, units, expected_next):
            raise ValueError(
                f"{file_name}: transformation {name!r} is {found}, expected "
                f"{(transformation_type, vector, units, expected_next)}; it would be applied differently"
            )
    arm = _scalar(transformations["arm_rotation"], file_name)
    if arm != 0.0:
        raise ValueError(f"{file_name}: arm_rotation is {arm}, expected 0 (the arm angle belongs to each sub-run)")
    return transformations


def _read_geometry(root: NXgroup, file_name: str) -> DENEXDetectorGeometry:
    """The detector geometry, after checking its pixel offsets are DetectorModel's layout (else ValueError)."""
    detector, transformations = _child(root, _DETECTOR), _child(root, _TRANSFORMATIONS)
    fields = {}
    for name, group in (
        ("x_pixel_offset", detector),
        ("y_pixel_offset", detector),
        ("x_pixel_size", detector),
        ("y_pixel_size", detector),
        ("arm_length", transformations),
    ):
        field = None if group is None else _child(group, name)
        if field is None:
            raise ValueError(f"{file_name}: missing {name}")
        fields[name] = field
    x_offsets = np.asarray(fields["x_pixel_offset"].nxdata, dtype=float)
    y_offsets = np.asarray(fields["y_pixel_offset"].nxdata, dtype=float)
    pixel_x = float(_scalar(fields["x_pixel_size"], file_name))
    pixel_y = float(_scalar(fields["y_pixel_size"], file_name))
    arm_length = float(_scalar(fields["arm_length"], file_name))
    if x_offsets.ndim != 1 or y_offsets.ndim != 1 or x_offsets.size < 1 or y_offsets.size < 1:
        raise ValueError(
            f"{file_name}: pixel offsets of shapes {x_offsets.shape}, {y_offsets.shape}; expected one value per "
            "column (x) and per row (y)"
        )
    geometry = DENEXDetectorGeometry(y_offsets.size, x_offsets.size, pixel_x, pixel_y, arm_length, False)
    # the offsets must be PyRS's pixel layout (index_0 across columns with X decreasing, index_1 along rows
    # with Y increasing): a file numbering pixels otherwise would be applied to the wrong pixels
    local = DetectorModel(geometry).local_positions
    expected_x = local(np.arange(x_offsets.size), 0.0)[:, 0]
    expected_y = local(0.0, np.arange(y_offsets.size))[:, 1]
    if not (
        np.allclose(x_offsets, expected_x, rtol=0, atol=1e-12)
        and np.allclose(y_offsets, expected_y, rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{file_name}: the pixel offsets are not PyRS's pixel layout (DetectorModel.local_positions)")
    return geometry


def read_calibration_nexus(
    calibration_file_name: str,
) -> Tuple[DENEXDetectorShift, DENEXDetectorShift, float, float, int]:
    """A calibration from a NeXus calibration file, as :func:`read_calibration_json_file` returns it.

    Returns:
        ``(shift, shift_error, wavelength, wavelength_error, status)``, all values float64.

    Raises:
        RuntimeError: If the file is not a PyRS NeXus calibration file (a group or value is missing).
        ValueError: If it would be applied differently from how this PyRS writes it: another transformation
            chain, a non-zero ``arm_rotation``, or another pixel layout.
        OSError: If it is not an HDF5 file.
    """
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "NeXus calibration file")
    with _open(calibration_file_name) as root:
        transformations = _check_chain(root, calibration_file_name)
        _read_geometry(root, calibration_file_name)
        values: Dict[str, float] = {}
        errors: Dict[str, float] = {}
        for name, _, _, _, parameter in CALIBRATION_TRANSFORMATIONS:
            if parameter is None:
                continue
            error = _child(transformations, name + "_errors")
            if error is None:
                raise RuntimeError(f"{calibration_file_name}: {name}_errors is missing")
            values[parameter] = float(_scalar(transformations[name], calibration_file_name))
            errors[parameter] = float(_scalar(error, calibration_file_name))
        fields = {}
        for path in (_MONOCHROMATOR + "/wavelength", _MONOCHROMATOR + "/wavelength_errors", _NOTE + "/status"):
            field = _child(root, path)
            if field is None:
                raise RuntimeError(f"{calibration_file_name}: missing {path}")
            fields[path] = field
        wave_length = float(_scalar(fields[_MONOCHROMATOR + "/wavelength"], calibration_file_name))
        wave_length_error = float(_scalar(fields[_MONOCHROMATOR + "/wavelength_errors"], calibration_file_name))
        status = int(_scalar(fields[_NOTE + "/status"], calibration_file_name))

    def shift(parameters: Dict[str, float], file_name: str = "") -> DENEXDetectorShift:
        return DENEXDetectorShift(
            shift_x=parameters["Shift_x"],
            shift_y=parameters["Shift_y"],
            shift_z=parameters["Shift_z"],
            rotation_x=parameters["Rot_x"],
            rotation_y=parameters["Rot_y"],
            rotation_z=parameters["Rot_z"],
            tth_0=parameters["TTH_0"],
            calibration_file=file_name,
        )

    return shift(values, calibration_file_name), shift(errors), wave_length, wave_length_error, status


def read_calibration_nexus_provenance(calibration_file_name: str) -> Optional[Dict[str, Any]]:
    """The provenance of a NeXus calibration file (#1053), or None if it has none.

    Raises:
        ValueError: If the stored provenance is not valid JSON.
    """
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "NeXus calibration file")
    with _open(calibration_file_name) as root:
        data = _child(root, _NOTE + "/data")
        if data is None:
            return None
        provenance = json.loads(_text(data.nxdata))
    return provenance if isinstance(provenance, dict) else None


def read_calibration_nexus_geometry(calibration_file_name: str) -> DENEXDetectorGeometry:
    """The detector geometry a NeXus calibration file is valid for.

    Raises:
        ValueError: If the detector's pixel layout or arm length is missing, malformed, or not PyRS's pixel
            layout (as the JSON geometry reader's errors, so ``HidraSetup`` notes it and reduction goes on).
    """
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "NeXus calibration file")
    with _open(calibration_file_name) as root:
        return _read_geometry(root, calibration_file_name)


def _plain(field: NXfield) -> Any:
    """A stored column as numbers, or as a list of str for byte strings."""
    value = np.asarray(field.nxdata)
    if value.dtype.kind == "S":
        return [item.decode("utf-8") for item in value.reshape(-1)]
    return value


def read_calibration_nexus_diagnostics(calibration_file_name: str) -> Optional[Dict[str, Any]]:
    """The diagnostics of a NeXus calibration file (#1054 phase 2), or None if it has none.

    Returns:
        ``method``, ``version``, ``date``; ``covariance``, None or ``(parameter names, matrix)``; ``tables``,
        ``{name: {column: array or list of str}}``; ``patterns``, ``{name: {"two_theta", "intensity",
        "errors"}}``, as :func:`write_calibration_nexus` takes them.
    """
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "NeXus calibration file")
    with _open(calibration_file_name) as root:
        process = _child(root, _PROCESS)
        if process is None:
            return None
        diagnostics: Dict[str, Any] = {
            key: _text(process[key].nxdata) if key in process else None for key in ("method", "version", "date")
        }
        diagnostics["covariance"] = None
        if "covariance" in process:
            names: Sequence[str] = _plain(process["covariance_parameters"])
            diagnostics["covariance"] = (tuple(names), np.asarray(process["covariance"].nxdata, dtype=float))
        tables: Dict[str, Dict[str, Any]] = {}
        patterns: Dict[str, Dict[str, Any]] = {}
        for name in process:
            group = process[name]
            if name == "patterns":
                for pattern_name in group:
                    pattern = group[pattern_name]
                    patterns[pattern_name] = {
                        "two_theta": np.asarray(pattern["two_theta"].nxdata, dtype=float),
                        "intensity": np.asarray(pattern["intensity"].nxdata, dtype=float),
                        "errors": np.asarray(pattern["intensity_errors"].nxdata, dtype=float),
                    }
            elif group.nxclass == "NXdata":
                tables[name] = {column: _plain(group[column]) for column in group}
        diagnostics["tables"] = tables
        diagnostics["patterns"] = patterns
    return diagnostics


def table_columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Dict[str, List[Any]]:
    """Rows (dicts) as the columns of a diagnostics table, in the given column order."""
    return {column: [row[column] for row in rows] for column in columns}
