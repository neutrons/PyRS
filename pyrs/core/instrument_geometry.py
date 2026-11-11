"""
Containing classes serving for
1. instrument geometry
2. instrument geometry calibration
"""

import datetime
import hashlib
import json
import math
import os
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int


class HidraSetup:
    """A class to work with instrument geometry calculation

    Handle anything to do with HB2B (HiDRA) including geometry, wavelength and calibration

    """

    def __init__(self, detector_setup):
        """
        initialization
        :param detector_setup: DENEXDetectorGeometry
        """

        # check inputs
        checkdatatypes.check_type("Detector geometry setup", detector_setup, DENEXDetectorGeometry)

        # set for original instrument setup defined by engineering
        self._geometry_setup = detector_setup
        self._single_wave_length = None

        self._geometry_shift = None
        self._calibrated_wave_length = None

        # calibration state
        self._calibration_applied = False

        # the calibration a reduction applies, and where it came from (#1053): set by
        # load_calibration_file or use_workspace_calibration, written by HidraProjectFile
        self._geometry_shift_error = None
        self._wave_length_error = None
        self._calibration_status = None
        self._calibration_source = "none"
        self._calibration_file = None
        self._calibration_file_sha256 = None
        self._calibration_created = None
        self._calibration_created_source = None
        self._lambda_source = None
        self._calibration_provenance = None
        self._calibration_note = None
        # the detector geometry the calibration is valid for (it lives within the calibration, #1053),
        # where it came from, and a note if it differs from the run's (which reduction uses)
        self._calibration_geometry = None
        self._geometry_source = None
        self._geometry_note = None
        # where the detector mask came from (#1068): set by ReductionApp from the workspace
        self._mask_record = {"source": "not recorded"}

    @property
    def mask_record(self):
        """Where the applied detector mask came from (#1068): ``{"source": "file", "file": ...,
        "sha256": ...}``, ``{"source": "none"}``, or ``{"source": "not recorded"}``."""
        return dict(self._mask_record)

    def set_mask_record(self, record):
        """Record where the applied detector mask came from (see :attr:`mask_record`)."""
        if not isinstance(record, dict) or "source" not in record:
            raise ValueError(f"mask record must be a dict with a 'source' key, got {record!r}")
        self._mask_record = dict(record)

    def get_instrument_geometry(self, calibrated):
        """Get instrument geometry parameters
        Get HB2B geometry setup, raw or calibrated optionally
        :param calibrated: Bool
        :return GeometrySetup: Geometry setup parameters
        """
        # TODO: this next `if calibrated...` clause has some serious issues:
        #   (1) `apply_shift` returns `None`.
        #   (2) Potentially, shift will be applied multiple times.
        #   (3) Non-calibrated _geometry_setup is modified:
        #       what happens if the next call has `calibrated = False`?
        if calibrated and self._geometry_shift is not None:
            return self._geometry_setup.apply_shift(self._geometry_shift)

        return self._geometry_setup

    def get_wavelength(self, wave_length_tag):
        """Get wave length
        Get wave length for only calibrated

        :param wave_length_tag: str
         user tag (as 111, 222) for wave length. None for single wave length

        :return float: wave length in A
        """
        if wave_length_tag is not None:
            raise NotImplementedError("Need use case to re-define the method")

        return self._calibrated_wave_length

    def get_wavelength_shift(self):
        return self._calibrated_wave_length

    @property
    def name(self):
        return "HB2B"

    # ------------------------------------------------------------ calibration (#1053)

    @property
    def geometry_shift(self):
        """The applied detector shift (DENEXDetectorShift), or None."""
        return self._geometry_shift

    @property
    def geometry_shift_error(self):
        """Its one-sigma errors (DENEXDetectorShift; -1 not refined, -2 undetermined), or None if unknown."""
        return self._geometry_shift_error

    @property
    def wavelength_error(self):
        return self._wave_length_error

    @property
    def calibration_status(self):
        """The calibration's refinement status, or None if unknown."""
        return self._calibration_status

    @property
    def calibration_source(self):
        """Where the applied calibration came from: ``"file"``, ``"workspace"`` or ``"none"``."""
        return self._calibration_source

    @property
    def calibration_geometry(self):
        """The detector geometry the calibration is valid for (DENEXDetectorGeometry), or None."""
        return self._calibration_geometry

    @property
    def geometry_note(self):
        """Why the calibration's geometry needs attention (it differs from the run's), or None."""
        return self._geometry_note

    def _take_geometry(self, geometry, source):
        """Record the calibration's geometry, flagging it if it is not the run's (reduction uses the run's)."""
        from pyrs.utilities import calibration_file_io

        self._calibration_geometry, self._geometry_source, self._geometry_note = geometry, source, None
        run, calibration = (calibration_file_io.geometry_to_dict(g) for g in (self._geometry_setup, geometry))
        if run != calibration:
            self._geometry_note = (
                f"the calibration's detector geometry {calibration} differs from the run's {run}; "
                "the reduction used the run's"
            )

    @property
    def calibration_provenance(self):
        """The calibration's provenance block (#1053 part A), or None."""
        return self._calibration_provenance

    def load_calibration_file(self, calibration_file):
        """Take the calibration from a calibration file (JSON or NeXus, #1054): shift, errors, wavelength, status, its
        provenance and creation time (the provenance's ``created``, else the file's modification time,
        labelled as such — a copy resets it).

        Raises
        ------
        RuntimeError
            if the file cannot be read as a calibration (as read_calibration_file)
        ValueError
            if it is neither a .json nor a .nxs file, or a NeXus file would be applied differently
        OSError
            if a .nxs file is not HDF5 (as a malformed .json raises its JSON error)
        """
        from pyrs.utilities import calibration_file_io  # calibration_file_io imports this module

        shift, shift_error, wave_length, wave_length_error, status = calibration_file_io.read_calibration_file(
            calibration_file
        )
        self._geometry_shift, self._geometry_shift_error = shift, shift_error
        self._calibrated_wave_length, self._wave_length_error = float(wave_length), float(wave_length_error)
        self._calibration_status = int(status)
        self._calibration_source = "file"
        self._calibration_file = str(calibration_file)
        self._calibration_file_sha256 = _file_sha256(calibration_file)
        self._lambda_source = "calibration file"
        # the record of where it came from must not stop a reduction
        try:
            self._calibration_provenance = calibration_file_io.read_calibration_provenance(calibration_file)
        except (OSError, ValueError) as read_err:
            self._calibration_provenance = None
            self._calibration_note = f"provenance could not be read: {type(read_err).__name__}: {read_err}"
        try:
            geometry = calibration_file_io.read_calibration_geometry(calibration_file)
        except (OSError, ValueError, KeyError, TypeError) as read_err:
            geometry = None
            self._calibration_note = f"geometry could not be read: {type(read_err).__name__}: {read_err}"
        if geometry is not None:
            self._take_geometry(geometry, "calibration file")
        else:  # written before calibrations carried their geometry
            self._take_geometry(self._geometry_setup, "run (the calibration file has no geometry)")
        created = (self._calibration_provenance or {}).get("created")
        if created:
            self._calibration_created, self._calibration_created_source = str(created), "calibration provenance"
        else:
            try:
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(calibration_file)).astimezone()
                self._calibration_created = modified.isoformat(timespec="seconds")
                self._calibration_created_source = "file modification time"
            except OSError:
                pass

    def use_workspace_calibration(self, workspace, sub_runs=None):
        """Take the detector shift already in ``workspace`` (e.g. read from an NXstress file). Its errors,
        status and provenance are unknown; the wavelength is taken only if the workspace marks it
        calibrated and every sub-run has the same value (never the nominal monochromator value).
        """
        self._geometry_shift = workspace.get_detector_shift()
        self._geometry_shift_error = None
        self._calibration_status = None
        self._calibration_source = "workspace"
        self._calibrated_wave_length, self._lambda_source = _calibrated_wavelength(workspace, sub_runs)
        self._wave_length_error = None
        self._take_geometry(self._geometry_setup, "run")

    def calibration_dict(self):
        """The applied calibration in the calibration JSON's keys (``Shift_x`` ... ``Lambda``, ``error_``
        keys, ``Status``), or None without a calibration; unknown errors and status are left out."""
        if self._geometry_shift is None:
            return None
        from pyrs.utilities import calibration_file_io

        if self._geometry_shift_error is not None and self._calibration_status is not None:
            calibration = calibration_file_io.calibration_to_dict(
                self._geometry_shift,
                self._geometry_shift_error,
                self._calibrated_wave_length,
                self._wave_length_error,
                self._calibration_status,
            )
        else:
            values = self._geometry_shift.convert_to_dict()
            values["Lambda"] = self._calibrated_wave_length if self._calibrated_wave_length is not None else math.nan
            calibration = {key: float(value) for key, value in values.items()}
        if self._calibration_geometry is not None:
            calibration["geometry"] = calibration_file_io.geometry_to_dict(self._calibration_geometry)
        return calibration

    def calibration_record(self):
        """Everything about the applied calibration, for the project file (HidraProjectFile)."""
        record = {
            "source": self._calibration_source,
            "calibration": self.calibration_dict(),
            "calibration_file": self._calibration_file,
            "calibration_file_sha256": self._calibration_file_sha256,
            "created": self._calibration_created,
            "created_source": self._calibration_created_source,
            "lambda_source": self._lambda_source,
            "provenance": self._calibration_provenance,
            "error": self._calibration_note,
            "geometry_source": self._geometry_source,
            "geometry_note": self._geometry_note,
        }
        if self._geometry_note is not None:  # what reduction used, as it is not the calibration's
            from pyrs.utilities import calibration_file_io

            record["run_geometry"] = calibration_file_io.geometry_to_dict(self._geometry_setup)
        return {key: value for key, value in record.items() if value is not None}

    def set_calibration_record(self, record):
        """Restore the applied calibration from a :meth:`calibration_record` (e.g. read from a project file)."""
        from pyrs.utilities import calibration_file_io

        calibration = record.get("calibration")
        self._calibration_source = record.get("source", "none")
        self._calibration_file = record.get("calibration_file")
        self._calibration_file_sha256 = record.get("calibration_file_sha256")
        self._calibration_created = record.get("created")
        self._calibration_created_source = record.get("created_source")
        self._lambda_source = record.get("lambda_source")
        self._calibration_provenance = record.get("provenance")
        self._calibration_note = record.get("error")
        self._geometry_source = record.get("geometry_source")
        self._geometry_note = record.get("geometry_note")
        geometry = (calibration or {}).get("geometry")
        self._calibration_geometry = None if geometry is None else calibration_file_io.geometry_from_dict(geometry)
        if calibration is None:
            self._geometry_shift = self._geometry_shift_error = None
            self._calibrated_wave_length = self._wave_length_error = self._calibration_status = None
            return
        if "Status" in calibration:
            shift, shift_error, wave_length, wave_length_error, status = calibration_file_io.calibration_from_dict(
                calibration, self._calibration_file or ""
            )
            self._geometry_shift, self._geometry_shift_error = shift, shift_error
            self._calibrated_wave_length, self._wave_length_error = float(wave_length), float(wave_length_error)
            self._calibration_status = int(status)
        else:  # a workspace shift: values only
            self._geometry_shift = DENEXDetectorShift(
                calibration["Shift_x"],
                calibration["Shift_y"],
                calibration["Shift_z"],
                calibration["Rot_x"],
                calibration["Rot_y"],
                calibration["Rot_z"],
                calibration["TTH_0"],
            )
            self._geometry_shift_error = self._wave_length_error = self._calibration_status = None
            wave_length = calibration.get("Lambda")
            self._calibrated_wave_length = None if wave_length is None or math.isnan(wave_length) else wave_length


def _file_sha256(path):
    """SHA-256 of a file, or None if it cannot be read."""
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        return sha.hexdigest()
    except OSError:
        return None


def _calibrated_wavelength(workspace, sub_runs=None):
    """The workspace's calibrated wavelength if every sub-run has the same one, and where it came from."""
    sub_runs = list(sub_runs) if sub_runs is not None else list(workspace.get_sub_runs())
    try:
        values = {float(workspace.get_wavelength(True, True, sub_run=int(s))) for s in sub_runs}
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None, "workspace has no calibrated wavelength: not recorded"
    if len(values) != 1:
        return None, "workspace calibrated wavelength differs between sub-runs: not recorded"
    return values.pop(), "workspace calibrated wavelength"


class DENEXDetectorGeometry:
    """
    A class to handle and save instrument geometry setup
    """

    def __init__(
        self,
        num_rows: int,
        num_columns: int,
        pixel_size_x: float,
        pixel_size_y: float,
        arm_length: float,
        calibrated: bool,
    ) -> None:
        """
        Initialization of instrument geometry setup for 1 denex detector
        :param num_rows: number of rows of pixels in detector (number of pixels per column)
        :param num_columns: number of columns of pixels in detector (number of pixels per row)
        :param pixel_size_x: pixel size at X direction (along a row)
        :param pixel_size_y: pixel size at Y direction (along a column)
        :param arm_length: arm length
        :param calibrated: flag whether these values are calibrated
        """
        # check inputs
        self._arm_length = to_float("Arm length", arm_length, min_value=1e-5)
        self._pixel_size_x = to_float("Pixel size (x)", pixel_size_x, min_value=1e-7)
        self._pixel_size_y = to_float("Pixel size (y)", pixel_size_y, min_value=1e-7)
        self._detector_rows = to_int("Number of rows in detector", num_rows, min_value=1)
        self._detector_columns = to_int("Number of columns in detector", num_columns, min_value=1)
        self._calibration_file = ""

        checkdatatypes.check_bool_variable("Flag indicating instrument setup been calibrated", calibrated)

    def apply_shift(self, geometry_shift):
        """Apply detector Calibration
        :param geometry_shift: DENEXDetectorShift
        :return:
        """
        checkdatatypes.check_type("Detector geometry shift", geometry_shift, DENEXDetectorShift)
        self._calibration_file = DENEXDetectorShift.calibration_file
        self._arm_length += geometry_shift.center_shift_z

    @property
    def arm_length(self):
        """
        :return float: sample to detector distance
        """
        return self._arm_length

    @property
    def detector_size(self):
        """Detector size (number of pixels)
        :return: number of rows, number of columns
        """
        return self._detector_rows, self._detector_columns

    @property
    def pixel_dimension(self):
        """Each pixel's linear size at X (across columns) and Y (across rows) direction
        :return: size along X-axis, size along Y-axis in unit of meter
        """
        return self._pixel_size_x, self._pixel_size_y


class DENEXDetectorShift:
    """
    A class to handle and save instrument geometry calibration information
    """

    def __init__(self, shift_x, shift_y, shift_z, rotation_x, rotation_y, rotation_z, tth_0, calibration_file=""):
        """
        Initialization of instrument geometry setup for 1 denex detector
        :param shift_x: detector shift along x
        :param shift_y: detector shift along y
        :param shift_z: detector shift along z
        :param rotation_x: detector rotation about x
        :param rotation_y: detector rotation about y
        :param rotation_z: detector rotation about z
        :param tth_0: shift of detector tth reference frame
        :param calibration_file: name of calibration file

        Every value is stored as a Python float (float64) whatever type it is given in, so a numpy
        float32 never makes the geometry float32 (#1052). Unlike the setters there is no range check:
        this class also holds a calibration's errors, and a rotation's error may exceed 360 degrees.
        """

        self._center_shift_x = to_float("Center shift along X direction", shift_x)
        self._center_shift_y = to_float("Center shift along Y direction", shift_y)
        self._center_shift_z = to_float("Center shift along Z direction", shift_z)  # along detector arm

        self._rotation_x = to_float("Rotation along X direction", rotation_x)  # in Y-Z plane, i.e., flip
        self._rotation_y = to_float("Rotation along Y direction", rotation_y)  # in X-Z plane, i.e., rotate
        self._rotation_z = to_float("Rotation along Z direction", rotation_z)  # in X-Y plane, i.e., spin

        self._two_theta_0 = to_float("offset in two_theta arm for ideal 0", tth_0)  # arm rotation offset

        self.calibration_file = calibration_file

        # Need data from client to finish this
        self.calibrated_wave_length = {"Si001": 1.00}

        self.calibration_file = calibration_file

    def __str__(self):
        nice = "[Calibration]\nShift:    {},  {},  {}\nRotation: {}, {}, {}\n".format(
            self.center_shift_x,
            self.center_shift_y,
            self.center_shift_z,
            self.rotation_x,
            self.rotation_y,
            self.rotation_z,
        )

        return nice

    @property
    def center_shift_x(self):
        """
        :return float: detector shift along x
        """
        return self._center_shift_x

    @center_shift_x.setter
    def center_shift_x(self, value: float) -> None:
        self._center_shift_x = to_float("Center shift along X direction", value)

    @property
    def center_shift_y(self):
        """
        :return float: detector shift along y
        """
        return self._center_shift_y

    @center_shift_y.setter
    def center_shift_y(self, value: float) -> None:
        self._center_shift_y = to_float("Center shift along Y direction", value)

    @property
    def center_shift_z(self):
        """
        :return float: detector shift along z
        """
        return self._center_shift_z

    @center_shift_z.setter
    def center_shift_z(self, value: float) -> None:
        self._center_shift_z = to_float("Center shift along Z direction", value)

    @property
    def rotation_x(self):
        """
        :return float: detector rotation about z
        """
        return self._rotation_x

    @rotation_x.setter
    def rotation_x(self, value: float) -> None:
        self._rotation_x = to_float("Rotation along X direction", value, -360, 360)

    @property
    def rotation_y(self):
        """
        :return float: detector rotation about y
        """
        return self._rotation_y

    @rotation_y.setter
    def rotation_y(self, value: float) -> None:
        self._rotation_y = to_float("Rotation along Y direction", value, -360, 360)

    @property
    def rotation_z(self):
        """
        :return float: detector rotation about z
        """
        return self._rotation_z

    @rotation_z.setter
    def rotation_z(self, value: float) -> None:
        self._rotation_z = to_float("Rotation along Z direction", value, -360, 360)

    @property
    def two_theta_0(self):
        """
        :return float: Shift of the 2theta zero point
        """
        return self._two_theta_0

    @two_theta_0.setter
    def two_theta_0(self, value: float) -> None:
        self._two_theta_0 = to_float("offset in two_theta arm for ideal 0", value, -360, 360)

    def convert_to_dict(self):
        """
        Convert instrument geometry calibration to a dictionary
        :return:
        """
        geometry_shift_dict = dict()

        geometry_shift_dict["Shift_x"] = self._center_shift_x
        geometry_shift_dict["Shift_y"] = self._center_shift_y
        geometry_shift_dict["Shift_z"] = self._center_shift_z
        geometry_shift_dict["Rot_x"] = self._rotation_x
        geometry_shift_dict["Rot_y"] = self._rotation_y
        geometry_shift_dict["Rot_z"] = self._rotation_z
        geometry_shift_dict["TTH_0"] = self._two_theta_0

        return geometry_shift_dict

    def convert_error_to_dict(self):
        """
        Convert instrument geometry calibration to a dictionary if this shift is shift error
        :return:
        """
        geometry_shift_dict = dict()

        geometry_shift_dict["error_Shift_x"] = self._center_shift_x
        geometry_shift_dict["error_Shift_y"] = self._center_shift_y
        geometry_shift_dict["error_Shift_z"] = self._center_shift_z
        geometry_shift_dict["error_Rot_x"] = self._rotation_x
        geometry_shift_dict["error_Rot_y"] = self._rotation_y
        geometry_shift_dict["error_Rot_z"] = self._rotation_z
        geometry_shift_dict["error_TTH_0"] = self._two_theta_0

        return geometry_shift_dict

    def set_from_dict(self, geometry_shift_dict):
        """Set geometry shift parameters from a dictionary, which may miss some parameters
        :param geometry_shift_dict:
        :return:
        """
        checkdatatypes.check_dict("Geometry shift parameters", geometry_shift_dict)

        if "shift x" in geometry_shift_dict:
            self._center_shift_x = to_float("Center shift along X direction", geometry_shift_dict["shift x"])
        if "shift y" in geometry_shift_dict:
            self._center_shift_y = to_float("Center shift along Y direction", geometry_shift_dict["shift y"])
        if "shift z" in geometry_shift_dict:
            self._center_shift_z = to_float("Center shift along Z direction", geometry_shift_dict["shift z"])

        if "rotation x" in geometry_shift_dict:
            self._rotation_x = to_float("Rotation along X direction", geometry_shift_dict["rotation x"])
        if "rotation y" in geometry_shift_dict:
            self._rotation_y = to_float("Rotation along Y direction", geometry_shift_dict["rotation y"])
        if "rotation z" in geometry_shift_dict:
            self._rotation_z = to_float("Rotation along Z direction", geometry_shift_dict["rotation z"])

    # TODO - #86 - Synchronize with convert_to_dict and implement
    def set_from_dict_error(self):
        return

    def to_json(self, file_name):
        """Convert to a dictionary and convert to Json string
        :return:
        """
        checkdatatypes.check_file_name(file_name, False, True, False, "Json file name to export instrument setup")

        # construct dictionary
        instrument_dict = self.convert_to_dict()

        # create file
        jfile = open(file_name, "w")
        json.dump(instrument_dict, jfile)
        jfile.close()

    def from_json(self, file_name):
        """Convert from a Json string (dictionary) and set to parameters
        :param file_name: json file name
        :return:
        """
        checkdatatypes.check_file_name(file_name, True, False, False, "Json file name to import instrument setup")

        # read file
        json_file = open(file_name, "r")
        lines = json_file.readlines()
        json_string = ""
        for line in lines:
            json_string += line.strip()

        instrument_dict = json.loads(json_string)

        self.set_from_dict(instrument_dict)


# if __name__ == '__main__':
#     # Test main
#     shift = DENEXDetectorShift(0., 0., 0., 0., 0., 0.)
#     shift.to_json('geometry_shift_template.json')
