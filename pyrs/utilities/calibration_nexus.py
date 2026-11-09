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

The frame is PyRS's: +Z along the incident beam, +Y up. ``arm_rotation`` is stored as 0: the arm angle
belongs to each sub-run (``-get_detector_2theta()``), not to the calibration. Error codes are the JSON's:
-1 not refined (or held), -2 refined but undetermined.
"""

import json
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from pyrs.core.detector_model import CALIBRATION_TRANSFORMATIONS, DetectorModel, transformation_chain
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.utilities import checkdatatypes

__all__ = [
    "write_calibration_nexus",
    "read_calibration_nexus",
    "read_calibration_nexus_provenance",
    "read_calibration_nexus_geometry",
]

_DETECTOR = "entry/instrument/detector"
_TRANSFORMATIONS = _DETECTOR + "/transformations"
_NOTE = _DETECTOR + "/calibration_method"
_MONOCHROMATOR = "entry/instrument/monochromator"


def _group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    group = parent.create_group(name)
    group.attrs["NX_class"] = nx_class
    return group


def _value(group: h5py.Group, name: str, value: float, units: Optional[str] = None) -> h5py.Dataset:
    dataset = group.create_dataset(name, data=np.float64(value))
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def write_calibration_nexus(
    file_name: str,
    shifts: DENEXDetectorShift,
    shifts_error: DENEXDetectorShift,
    wave_length: float,
    wave_length_error: float,
    calibration_status: int,
    geometry: DENEXDetectorGeometry,
    provenance: Optional[Dict[str, Any]] = None,
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
    errors = transformation_chain(shifts_error, -1.0)  # same order and names; only the parameters' errors are used

    with h5py.File(file_name, "w") as nexus:
        entry = _group(nexus, "entry", "NXentry")
        entry.create_dataset("program_name", data="pyrs")
        instrument = _group(entry, "instrument", "NXinstrument")

        monochromator = _group(instrument, "monochromator", "NXmonochromator")
        _value(monochromator, "wavelength", wave_length, "angstrom")
        _value(monochromator, "wavelength_errors", wave_length_error, "angstrom")

        detector = _group(instrument, "detector", "NXdetector")
        _value(detector, "x_pixel_size", pixel_x, "m")
        _value(detector, "y_pixel_size", pixel_y, "m")
        detector.create_dataset("x_pixel_offset", data=local(np.arange(columns), 0.0)[:, 0]).attrs["units"] = "m"
        detector.create_dataset("y_pixel_offset", data=local(0.0, np.arange(rows))[:, 1]).attrs["units"] = "m"
        created = (provenance or {}).get("created")
        if created:
            detector.create_dataset("calibration_date", data=str(created))
        detector.create_dataset("depends_on", data="transformations/" + CALIBRATION_TRANSFORMATIONS[0][0])

        note = _group(detector, "calibration_method", "NXnote")
        note.create_dataset("status", data=np.int32(calibration_status))
        note.create_dataset("type", data="application/json")
        note.create_dataset("description", data="how this calibration was obtained (PyRS provenance, #1053)")
        if provenance_text is not None:
            note.create_dataset("data", data=provenance_text)

        transformations = _group(detector, "transformations", "NXtransformations")
        chain = transformation_chain(shifts, geometry.arm_length)
        for index, (step, error) in enumerate(zip(chain, errors)):
            dataset = _value(transformations, step.name, step.value, step.units)
            dataset.attrs["transformation_type"] = step.transformation_type
            dataset.attrs["vector"] = np.array(step.vector, dtype=np.float64)
            dataset.attrs["depends_on"] = chain[index + 1].name if index + 1 < len(chain) else "."
            if step.parameter is not None:
                _value(transformations, step.name + "_errors", error.value, step.units)
            elif step.name == "arm_rotation":
                dataset.attrs["note"] = "set per sub-run to -(DAS arm 2theta); not a calibration parameter"


def _text(value: Any) -> Any:
    """An HDF5 string as ``str``: bytes decoded, a one-element array (as other tools write) unwrapped."""
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else value


def _scalar(dataset: h5py.Dataset, file_name: str) -> Any:
    """The single value of a dataset (a scalar, or of shape (1,) as other tools may write), else ValueError."""
    value = np.asarray(dataset[()])
    if value.size != 1:
        raise ValueError(f"{file_name}: {dataset.name} has shape {value.shape}, expected one value")
    return value.reshape(-1)[0].item()


def _check_chain(nexus: h5py.File, file_name: str) -> h5py.Group:
    """The transformations group, after checking it is exactly the chain this PyRS applies (else ValueError):
    every transformation's type, vector, units and ``depends_on``, the detector's ``depends_on``, and the
    per-sub-run ``arm_rotation`` stored as 0."""
    try:
        detector, transformations = nexus[_DETECTOR], nexus[_TRANSFORMATIONS]
    except KeyError:
        raise RuntimeError(f"{file_name}: no {_TRANSFORMATIONS}; not a PyRS NeXus calibration file") from None
    first = "transformations/" + CALIBRATION_TRANSFORMATIONS[0][0]
    depends_on = _text(detector["depends_on"][()]) if "depends_on" in detector else None
    if depends_on != first:
        raise ValueError(f"{file_name}: the detector depends_on {depends_on!r}, expected {first!r}")
    for index, (name, transformation_type, vector, units, _) in enumerate(CALIBRATION_TRANSFORMATIONS):
        if name not in transformations:
            raise RuntimeError(f"{file_name}: transformation {name!r} is missing")
        dataset = transformations[name]
        expected_next = (
            CALIBRATION_TRANSFORMATIONS[index + 1][0] if index + 1 < len(CALIBRATION_TRANSFORMATIONS) else "."
        )
        found = (
            _text(dataset.attrs.get("transformation_type")),
            tuple(float(v) for v in np.asarray(dataset.attrs.get("vector", ()), dtype=float).reshape(-1)),
            _text(dataset.attrs.get("units")),
            _text(dataset.attrs.get("depends_on")),
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


def _read_geometry(nexus: h5py.File, file_name: str) -> DENEXDetectorGeometry:
    """The detector geometry, after checking its pixel offsets are DetectorModel's layout (else ValueError)."""
    try:
        detector = nexus[_DETECTOR]
        x_offsets = np.asarray(detector["x_pixel_offset"][()], dtype=float)
        y_offsets = np.asarray(detector["y_pixel_offset"][()], dtype=float)
        pixel_x = float(_scalar(detector["x_pixel_size"], file_name))
        pixel_y = float(_scalar(detector["y_pixel_size"], file_name))
        arm_length = float(_scalar(nexus[_TRANSFORMATIONS]["arm_length"], file_name))
    except KeyError as missing:
        raise ValueError(f"{file_name}: missing {missing}") from None
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
    with h5py.File(calibration_file_name, "r") as nexus:
        transformations = _check_chain(nexus, calibration_file_name)
        _read_geometry(nexus, calibration_file_name)
        values: Dict[str, float] = {}
        errors: Dict[str, float] = {}
        for name, _, _, _, parameter in CALIBRATION_TRANSFORMATIONS:
            if parameter is None:
                continue
            if name + "_errors" not in transformations:
                raise RuntimeError(f"{calibration_file_name}: {name}_errors is missing")
            values[parameter] = float(_scalar(transformations[name], calibration_file_name))
            errors[parameter] = float(_scalar(transformations[name + "_errors"], calibration_file_name))
        try:
            wave_length = float(_scalar(nexus[_MONOCHROMATOR]["wavelength"], calibration_file_name))
            wave_length_error = float(_scalar(nexus[_MONOCHROMATOR]["wavelength_errors"], calibration_file_name))
            status = int(_scalar(nexus[_NOTE]["status"], calibration_file_name))
        except KeyError as missing:
            raise RuntimeError(f"{calibration_file_name}: missing {missing}") from None

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
    with h5py.File(calibration_file_name, "r") as nexus:
        note = nexus.get(_NOTE)
        if note is None or "data" not in note:
            return None
        provenance = json.loads(_text(note["data"][()]))
    return provenance if isinstance(provenance, dict) else None


def read_calibration_nexus_geometry(calibration_file_name: str) -> DENEXDetectorGeometry:
    """The detector geometry a NeXus calibration file is valid for.

    Raises:
        ValueError: If the detector's pixel layout or arm length is missing, malformed, or not PyRS's pixel
            layout (as the JSON geometry reader's errors, so ``HidraSetup`` notes it and reduction goes on).
    """
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, "NeXus calibration file")
    with h5py.File(calibration_file_name, "r") as nexus:
        return _read_geometry(nexus, calibration_file_name)
