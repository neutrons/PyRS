"""NeXus calibration files (#1054): the same calibration as the JSON, and a transformation chain that places
the detector exactly as reduction does"""

import json

import h5py
import numpy as np
import pytest

from pyrs.core.detector_model import DetectorModel, from_shift
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.utilities import calibration_file_io
from pyrs.utilities.calibration_nexus import (
    read_calibration_nexus,
    read_calibration_nexus_geometry,
    read_calibration_nexus_provenance,
    write_calibration_nexus,
)

GEOMETRY = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)
NON_SQUARE = DENEXDetectorGeometry(4, 6, 0.01, 0.02, 0.95, False)  # 4 rows, 6 columns: catches swapped axes
SHIFT = DENEXDetectorShift(-0.0154027, -0.00255874, -0.0445212, 0.0642667, -0.254056, 0.16946, 0.85294)
ERROR = DENEXDetectorShift(9.889e-05, 0.001167, 0.00026, 0.188275, 0.322219, 0.010681, -1.0)
WAVELENGTH, WAVELENGTH_ERROR, STATUS = 1.531347247129065, 8.25e-05, 1
PROVENANCE = {"schema": 1, "method": "ring", "created": "2026-11-09T12:00:00-05:00", "steps": [{"step": 1}]}


def _write(path, geometry=GEOMETRY, status=STATUS, provenance=PROVENANCE):
    write_calibration_nexus(str(path), SHIFT, ERROR, WAVELENGTH, WAVELENGTH_ERROR, status, geometry, provenance)
    return str(path)


def _write_json(path):
    calibration_file_io.write_calibration_to_json(
        SHIFT, ERROR, WAVELENGTH, WAVELENGTH_ERROR, STATUS, str(path), provenance=PROVENANCE, geometry=GEOMETRY
    )
    return str(path)


def test_nexus_reads_back_the_json_values_exactly(tmp_path):
    """The same calibration in both formats reads back bit for bit (float64)"""
    # Arrange
    nexus, json_file = _write(tmp_path / "cal.nxs"), _write_json(tmp_path / "cal.json")

    # Act
    from_nexus = calibration_file_io.read_calibration_file(nexus)
    from_json = calibration_file_io.read_calibration_file(json_file)

    # Assert
    np.testing.assert_array_equal(from_shift(from_nexus[0]), from_shift(from_json[0]))
    np.testing.assert_array_equal(from_shift(from_nexus[1]), from_shift(from_json[1]))
    assert (from_nexus[2], from_nexus[3]) == (from_json[2], from_json[3])
    assert from_nexus[4] == int(from_json[4]) == STATUS
    assert from_nexus[0].calibration_file == nexus


def test_provenance_and_geometry_round_trip(tmp_path):
    # Arrange
    path = _write(tmp_path / "cal.nxs", geometry=NON_SQUARE)

    # Act
    provenance = calibration_file_io.read_calibration_provenance(path)
    geometry = calibration_file_io.read_calibration_geometry(path)

    # Assert
    assert provenance == PROVENANCE
    assert geometry.detector_size == (4, 6) and geometry.pixel_dimension == (0.01, 0.02)
    assert geometry.arm_length == 0.95


def test_file_without_provenance_reads_none(tmp_path):
    # Act
    path = _write(tmp_path / "cal.nxs", provenance=None)

    # Assert
    assert read_calibration_nexus_provenance(path) is None
    with h5py.File(path, "r") as nexus:
        assert "calibration_date" not in nexus["entry/instrument/detector"]


def test_unwritable_provenance_raises_and_writes_nothing(tmp_path):
    # Act / Assert
    with pytest.raises(TypeError):
        _write(tmp_path / "cal.nxs", provenance={"bad": {1, 2}})
    assert not (tmp_path / "cal.nxs").exists()


def test_layout_uses_nexus_base_classes(tmp_path):
    # Arrange
    path = _write(tmp_path / "cal.nxs")

    # Act
    with h5py.File(path, "r") as nexus:
        classes = {
            name: nexus[name].attrs["NX_class"]
            for name in (
                "entry",
                "entry/instrument",
                "entry/instrument/monochromator",
                "entry/instrument/detector",
                "entry/instrument/detector/calibration_method",
                "entry/instrument/detector/transformations",
            )
        }
        depends_on = nexus["entry/instrument/detector/depends_on"][()]
        date = nexus["entry/instrument/detector/calibration_date"][()]

    # Assert
    assert [_text(c) for c in classes.values()] == [
        "NXentry",
        "NXinstrument",
        "NXmonochromator",
        "NXdetector",
        "NXnote",
        "NXtransformations",
    ]
    assert _text(depends_on) == "transformations/shift_x" and _text(date) == PROVENANCE["created"]


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def _matrix(transformation_type, vector, value, units):
    """4x4 homogeneous matrix of one NeXus transformation, built from its attributes only."""
    matrix = np.eye(4)
    axis = np.asarray(vector, dtype=float)
    if transformation_type == "translation":
        assert units == "m"
        matrix[:3, 3] = value * axis
        return matrix
    assert transformation_type == "rotation" and units == "deg"
    angle = np.deg2rad(value)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    matrix[:3, :3] = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross  # Rodrigues
    return matrix


def _placed_by_chain(path, arm):
    """Pixel lab positions from the file alone: its pixel offsets moved through its transformations."""
    with h5py.File(path, "r") as nexus:
        detector = nexus["entry/instrument/detector"]
        x, y = detector["x_pixel_offset"][()], detector["y_pixel_offset"][()]
        transformations = detector["transformations"]
        name = _text(detector["depends_on"][()]).split("/")[-1]
        total = np.eye(4)
        while name != ".":
            dataset = transformations[name]
            value = arm if name == "arm_rotation" else float(dataset[()])
            step = _matrix(
                _text(dataset.attrs["transformation_type"]),
                dataset.attrs["vector"],
                value,
                _text(dataset.attrs["units"]),
            )
            total = step @ total  # innermost first: each later step acts on the result so far
            name = _text(dataset.attrs["depends_on"])
    index_0, index_1 = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing="ij")
    local = np.stack([x[index_0], y[index_1], np.zeros(index_0.shape), np.ones(index_0.shape)], axis=-1)
    return (local @ total.T)[..., :3], index_0, index_1


@pytest.mark.parametrize("geometry", [GEOMETRY, NON_SQUARE], ids=["hb2b", "non-square"])
@pytest.mark.parametrize("arm", [-76.0, -90.0, -103.0, 0.0])
def test_chain_places_the_detector_as_reduction_does(tmp_path, geometry, arm):
    """Applied without DetectorModel, the file's chain reproduces DetectorModel.lab_positions (#1054)"""
    # Arrange
    path = _write(tmp_path / "cal.nxs", geometry=geometry)

    # Act
    positions, index_0, index_1 = _placed_by_chain(path, arm)

    # Assert - pixel positions (m) and 2theta (deg) to the reference-value tolerance of test_detector_model
    model = DetectorModel(geometry)
    np.testing.assert_allclose(positions, model.lab_positions(index_0, index_1, arm, None, SHIFT), rtol=0, atol=1e-12)
    two_theta = np.rad2deg(np.arccos(positions[..., 2] / np.linalg.norm(positions, axis=-1)))
    np.testing.assert_allclose(two_theta, model.two_theta(index_0, index_1, arm, None, SHIFT), rtol=1e-12, atol=1e-9)


def test_chain_with_other_conventions_is_refused(tmp_path):
    """A file whose chain would be applied differently is not misread"""
    # Arrange - rotation_x about the wrong axis (as the old NXstress chain did for 2theta_0)
    path = _write(tmp_path / "cal.nxs")
    with h5py.File(path, "a") as nexus:
        nexus["entry/instrument/detector/transformations/two_theta_0"].attrs["vector"] = [1.0, 0.0, 0.0]

    # Act / Assert
    with pytest.raises(ValueError, match="two_theta_0"):
        read_calibration_nexus(path)


def test_file_without_transformations_raises(tmp_path):
    # Arrange
    path = tmp_path / "other.nxs"
    with h5py.File(path, "w") as nexus:
        nexus.create_group("entry")

    # Act / Assert
    with pytest.raises(RuntimeError, match="not a PyRS NeXus calibration file"):
        read_calibration_nexus(str(path))


def test_missing_error_raises(tmp_path):
    # Arrange
    path = _write(tmp_path / "cal.nxs")
    with h5py.File(path, "a") as nexus:
        del nexus["entry/instrument/detector/transformations/shift_x_errors"]

    # Act / Assert
    with pytest.raises(RuntimeError, match="shift_x_errors"):
        read_calibration_nexus(path)


def test_unrefined_calibration_is_refused(tmp_path):
    """Status < 0 reads back, and reduction's status check refuses it, as for the JSON"""
    # Arrange
    path = _write(tmp_path / "cal.nxs", status=-1)

    # Act
    status = calibration_file_io.read_calibration_file(path)[4]

    # Assert
    assert status == -1
    with pytest.raises(RuntimeError, match="never successfully refined"):
        calibration_file_io.check_calibration_status(status)


def test_geometry_without_detector_raises(tmp_path):
    """A ValueError, as the JSON geometry reader's errors (review)"""
    # Arrange
    path = tmp_path / "other.nxs"
    with h5py.File(path, "w") as nexus:
        nexus.create_group("entry")

    # Act / Assert
    with pytest.raises(ValueError, match="missing"):
        read_calibration_nexus_geometry(str(path))


def _edit(path, change):
    with h5py.File(path, "a") as nexus:
        change(nexus["entry/instrument/detector"])
    return path


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d["transformations/arm_rotation"].write_direct(np.array(45.0)), "arm_rotation"),
        (
            lambda d: (d.__delitem__("depends_on"), d.create_dataset("depends_on", data="transformations/rotation_z")),
            "depends_on",
        ),
        (lambda d: d["x_pixel_offset"].write_direct(d["x_pixel_offset"][()][::-1].copy()), "pixel layout"),
        (lambda d: d.__delitem__("x_pixel_offset"), "missing"),
    ],
    ids=["arm-rotation", "detector-depends-on", "reversed-pixels", "no-offsets"],
)
def test_file_applied_differently_is_refused(tmp_path, change, match):
    """Every part of the file that decides where pixels go is checked, not only the chain (review)"""
    # Arrange
    path = _edit(_write(tmp_path / "cal.nxs"), change)

    # Act / Assert
    with pytest.raises(ValueError, match=match):
        read_calibration_nexus(path)


def test_other_tools_string_and_scalar_shapes_read(tmp_path):
    """A one-element string array and shape-(1,) values, as other tools write them, read as ours (review)"""
    # Arrange
    path = _write(tmp_path / "cal.nxs")
    with h5py.File(path, "a") as nexus:
        detector = nexus["entry/instrument/detector"]
        del detector["depends_on"]
        detector.create_dataset("depends_on", data=np.array([b"transformations/shift_x"]))
        wavelength = nexus["entry/instrument/monochromator/wavelength"]
        value = wavelength[()]
        del nexus["entry/instrument/monochromator/wavelength"]
        nexus["entry/instrument/monochromator"].create_dataset("wavelength", data=np.array([value]))

    # Act
    shift, _, wavelength_read, _, _ = read_calibration_nexus(path)

    # Assert
    assert wavelength_read == WAVELENGTH and shift.two_theta_0 == SHIFT.two_theta_0


@pytest.mark.parametrize(
    "name, expected", [("a.json", "json"), ("A.JSON", "json"), ("a.nxs", "nexus"), ("a.NXS", "nexus")]
)
def test_calibration_format_by_extension(name, expected):
    # Act / Assert
    assert calibration_file_io.calibration_format(name) == expected


@pytest.mark.parametrize("name", ["a.txt", "HB2B_1327.h5", "HB2B_3510.nxs.h5", "a.json.bak"])
def test_unknown_calibration_format_raises(name):
    # Act / Assert
    with pytest.raises(ValueError, match="(?i)calibration file must be|raw event data"):
        calibration_file_io.read_calibration_file(name)


def test_old_json_still_reads_through_the_dispatch(tmp_path):
    """A legacy file (two_theta_0, no provenance or geometry) reads as before"""
    # Arrange
    old = {"Shift_x": 0.01, "Shift_y": 0, "Shift_z": 0, "Rot_x": 0, "Rot_y": 0, "Rot_z": 0, "two_theta_0": 0.5}
    old.update({f"error_{key}": -1.0 for key in ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z")})
    old.update({"error_two_theta_0": -1.0, "Lambda": 1.54, "error_Lambda": 1e-5, "Status": 3})
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))

    # Act
    shift, _, wavelength, _, status = calibration_file_io.read_calibration_file(str(path))

    # Assert
    assert (shift.center_shift_x, shift.two_theta_0, wavelength, status) == (0.01, 0.5, 1.54, 3)
    assert calibration_file_io.read_calibration_provenance(str(path)) is None
    assert calibration_file_io.read_calibration_geometry(str(path)) is None


def test_nxs_that_is_not_hdf5_raises_oserror(tmp_path):
    """As a malformed .json raises its JSON error: not wrapped, not silently ignored"""
    # Arrange
    path = tmp_path / "HB2B_CAL.nxs"
    path.write_text("not an HDF5 file")

    # Act / Assert
    with pytest.raises(OSError, match="cannot be read as a NeXus"):
        calibration_file_io.read_calibration_file(str(path))


def test_file_is_written_with_nexusformat(tmp_path):
    """Written with nexusformat, as PyRS's NXstress output is"""
    # Act
    with h5py.File(_write(tmp_path / "cal.nxs"), "r") as nexus:
        creator = nexus.attrs.get("creator")

    # Assert
    assert _text(creator) == "nexusformat"
