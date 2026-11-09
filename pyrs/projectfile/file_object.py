# This is rs_scan_io.DiffractionFile's 2.0 version
from enum import Enum
import h5py
import hashlib
import json
import os
import traceback
from mantid.kernel import Logger
import numpy
from pathlib import Path
from pyrs import __version__
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int
from pyrs.utilities.file_util import to_filepath
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, HidraSetup
from pyrs.peaks import PeakCollection  # type: ignore
from pyrs.dataobjects import HidraConstants, SampleLogs  # type: ignore
from pyrs.projectfile import HidraProjectFileMode  # type: ignore
from typing import Union
import warnings

__all__ = ["HidraProjectFile"]


# the record of the calibration a reduction applied, in the CALIBRATION group (#1053); a reduction
# writing a record replaces every entry of the previous one, so a re-reduced file never mixes two
_APPLIED_DATASETS = ("calibration_json", "provenance_json", "sources_json")
# file attribute -> key of the record dict (the dict ReductionApp passes, and read_reduction_provenance returns)
_RECORD_ATTRIBUTES = {
    "calibration_file": "calibration_file",
    "calibration_file_sha256": "calibration_file_sha256",
    "vanadium_run": "vanadium_run",
    "calibration_source": "source",
    "calibration_created": "created",
    "calibration_created_source": "created_source",
    "calibration_lambda_source": "lambda_source",
    "calibration_record_error": "error",
    "calibration_geometry_source": "geometry_source",
    "calibration_geometry_note": "geometry_note",
    "calibration_run_geometry": "run_geometry",
    "calibration_record_warning": "warning",
}
# the mask and binning a reduction used (#1068), as JSON text in the REDUCTION group beside CALIBRATION
_REDUCTION_DATASET = "reduction_json"
_REDUCTION_ERROR = "reduction_record_error"
# on calibration_json: what an error of -1 or -2 means
_ERROR_CODES = "error_*: -1 not refined (or held), -2 refined but undetermined"


def _plain(value):
    """An HDF5 attribute or dataset value as a plain Python value (bytes as text, arrays as lists)."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, numpy.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _report_record_failure(message):
    """A record failure is warned about and logged with its traceback, never raised (#1053)."""
    warnings.warn(message, stacklevel=3)
    Logger("HidraProjectFile").error(f"{message}\n{traceback.format_exc()}")


def _clear_record(group):
    """Remove every entry of an earlier calibration record from the CALIBRATION group."""
    for name in _APPLIED_DATASETS:
        if name in group:
            del group[name]
    for name in _RECORD_ATTRIBUTES:
        if name in group.attrs:
            del group.attrs[name]


def _instrument_record(instrument):
    """The calibration record of a HidraSetup; never raises (the save must go ahead, #1053)."""
    try:
        return instrument.calibration_record()
    except Exception as record_err:  # noqa: BLE001 - recorded in the file, never stops the save
        message = f"applied calibration could not be recorded: {type(record_err).__name__}: {record_err}"
        _report_record_failure(message)
        return {"source": "unknown", "error": message}


def _reduction_record(instrument, reduction_settings):
    """The mask (from the instrument) and binning a reduction used, for the REDUCTION group (#1068)."""
    record = {"mask": instrument.mask_record}
    if reduction_settings is not None:
        record["settings"] = dict(reduction_settings)
    return record


def _write_reduction_record(instrument_group, reduction):
    """Write the REDUCTION group (the caller removed the old one). Never raises, as the calibration record."""
    # imported here: pyrs.calibration imports the project file module
    from pyrs.calibration.provenance import jsonable

    try:
        text = json.dumps(jsonable(reduction), allow_nan=False)
        group = instrument_group.create_group(HidraConstants.REDUCTION)
        group.create_dataset(_REDUCTION_DATASET, data=text, dtype=h5py.string_dtype(encoding="utf-8"))
    except Exception as write_err:  # noqa: BLE001 - recorded in the file, never stops the save
        message = f"mask and binning could not be recorded: {type(write_err).__name__}: {write_err}"
        _report_record_failure(message)
        try:
            if HidraConstants.REDUCTION in instrument_group:
                del instrument_group[HidraConstants.REDUCTION]
            instrument_group.create_group(HidraConstants.REDUCTION).attrs[_REDUCTION_ERROR] = message
        except Exception:  # noqa: BLE001 - the file itself is failing; the reduced data are still saved
            Logger("HidraProjectFile").error(f"could not record the failure either:\n{traceback.format_exc()}")


def _prepare_record(applied):
    """Validate and convert the record before anything is written, so a bad record writes nothing of itself."""
    # imported here: pyrs.calibration imports the project file module
    from pyrs.calibration.provenance import jsonable

    prepared = {"source": str(applied.get("source", "unknown"))}
    for key in (
        "calibration_file",
        "calibration_file_sha256",
        "created",
        "created_source",
        "lambda_source",
        "error",
        "geometry_source",
        "geometry_note",
        "warning",
        "vanadium_run",
    ):
        if applied.get(key) is not None:
            prepared[key] = str(applied[key])
    if applied.get("run_geometry") is not None:
        prepared["run_geometry"] = json.dumps(jsonable(applied["run_geometry"]), allow_nan=False)
    if applied.get("calibration") is not None:
        calibration = applied["calibration"]
        if not isinstance(calibration, dict) or "Shift_x" not in calibration:
            raise ValueError(f"calibration must be the calibration JSON's dictionary, got {calibration!r}")
        prepared["calibration_json"] = json.dumps(jsonable(calibration), allow_nan=False)
    if applied.get("provenance") is not None:
        prepared["provenance_json"] = json.dumps(jsonable(applied["provenance"]), allow_nan=False)
    if applied.get("sources") is not None:  # a combined file's sources (#1063)
        prepared["sources_json"] = json.dumps(jsonable(applied["sources"]), allow_nan=False)
    return prepared


def _write_applied_calibration(group, applied):
    """Write what the reduction applied into the CALIBRATION group (the caller cleared the old record).

    Never raises: the record must not stop the reduced data from being saved (#1053). A record that
    cannot be prepared or written leaves ``calibration_source = "unknown"`` and the reason(s) in
    ``calibration_record_error``, with nothing half-written beside them.
    """
    try:
        prepared = _prepare_record(applied)
    except Exception as prepare_err:  # noqa: BLE001 - recorded in the file, never stops the save
        message = f"applied calibration could not be recorded: {type(prepare_err).__name__}: {prepare_err}"
        _report_record_failure(message)
        prepared = {"source": "unknown", "error": "; ".join(filter(None, [applied.get("error"), message]))}
    try:
        for attribute, key in _RECORD_ATTRIBUTES.items():
            if key in prepared:
                group.attrs[attribute] = prepared[key]
        if "calibration_json" in prepared:
            dataset = group.create_dataset(
                "calibration_json", data=prepared["calibration_json"], dtype=h5py.string_dtype(encoding="utf-8")
            )
            dataset.attrs["codes"] = _ERROR_CODES
        for name in ("provenance_json", "sources_json"):
            if name in prepared:
                group.create_dataset(name, data=prepared[name], dtype=h5py.string_dtype(encoding="utf-8"))
    except Exception as write_err:  # noqa: BLE001 - recorded in the file, never stops the save
        message = f"applied calibration could not be written: {type(write_err).__name__}: {write_err}"
        _report_record_failure(message)
        try:  # nothing half-written: the record is then only "unknown" and why
            for name in _APPLIED_DATASETS:
                if name in group:
                    del group[name]
            group.attrs["calibration_source"] = "unknown"
            group.attrs["calibration_record_error"] = "; ".join(filter(None, [prepared.get("error"), message]))
        except Exception:  # noqa: BLE001 - the file itself is failing; the reduced data are still saved
            Logger("HidraProjectFile").error(f"could not record the failure either:\n{traceback.format_exc()}")


class CalibrationMismatchError(ValueError):
    """Project files reduced with different calibrations cannot be combined (#1063)."""


class ReductionMismatchError(ValueError):
    """Project files reduced with the same calibration but different vanadium runs, masks or binning:
    combining them needs the user's approval (#1063, #1068); pass ``accept_reduction_differences=True``
    once given."""


def _calibration_identity(record):
    """What makes two reductions' calibrations the same: the applied calibration (its values, errors,
    status and geometry: ``calibration``) and the calibration file's fingerprint (#1063, user decision)."""
    return json.dumps([record.get("calibration"), record.get("calibration_file_sha256")], sort_keys=True)


def _describe_calibration(record):
    """A short description of a record's calibration for the user, e.g. "HB2B_CAL_x.json, λ 1.53061 Å"."""
    parts = []
    if record.get("calibration_file"):
        parts.append(os.path.basename(str(record["calibration_file"])))
    elif record.get("source"):
        parts.append(f"{record['source']} calibration")
    calibration = record.get("calibration") or {}
    if calibration.get("Lambda") is not None:
        parts.append(f"λ {calibration['Lambda']:.5f} Å")
    if all(calibration.get(key) is not None for key in ("Shift_x", "Shift_y", "Shift_z")):
        shift = ", ".join(f"{calibration[key] * 1e3:.2f}" for key in ("Shift_x", "Shift_y", "Shift_z"))
        parts.append(f"shift ({shift}) mm")
    return ", ".join(parts) or "no calibration"


def _describe_mask(reduction):
    """The mask a reduction applied, for the user: the file and its SHA-256 prefix, or "no mask" (#1068)."""
    mask = reduction.get("mask") or {}
    if mask.get("source") == "file":
        described = f"{os.path.basename(str(mask.get('file')))} (SHA-256 {str(mask.get('sha256'))[:12]})"
    elif mask.get("source") == "none":
        described = "no mask"
    else:
        described = f"mask {mask.get('source', 'not recorded')}"
    extra = (reduction.get("settings") or {}).get("extra_mask")
    if extra is not None:
        described += f" + extra mask {json.dumps(extra, sort_keys=True)}"
    return described


def _describe_binning(reduction):
    """The binning a reduction used, for the user, e.g. "num_bins=720.0, eta_step=3.0, ..." (#1068); numbers
    as floats, so that 3 and 3.0 are the same binning."""

    def number(value):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value

    settings = {key: number(value) for key, value in (reduction.get("settings") or {}).items() if key != "extra_mask"}
    return ", ".join(f"{key}={value}" for key, value in sorted(settings.items())) or "not recorded"


def _reduction(record):
    """A record's mask and binning record (#1068), or {}."""
    reduction = record.get("reduction")
    return reduction if isinstance(reduction, dict) else {}


def _mask_known(record):
    """Whether a record names its mask: a mask file, or none (not "not recorded", e.g. a re-reduced older file)."""
    return (_reduction(record).get("mask") or {}).get("source") not in (None, "not recorded")


# what must match, beyond the calibration, for files to be combined without asking: the label shown to the
# user, and how a record describes it (None: not recorded, so it cannot be compared)
_REDUCTION_CHOICES = (
    ("vanadium runs", lambda record: record.get("vanadium_run") or "no vanadium"),
    ("masks", lambda record: _describe_mask(_reduction(record)) if _mask_known(record) else None),
    ("binning", lambda record: _describe_binning(_reduction(record)) if _reduction(record).get("settings") else None),
)


def combine_reduction_records(sources, accept_reduction_differences=False):
    """The calibration record of a file combined from several reduced project files (#1063, #1068)

    Runs reduced with different calibrations must not be combined (user decision): if two source files
    have records that differ (:func:`_calibration_identity`), :class:`CalibrationMismatchError` is
    raised, naming the files of each calibration. The same calibration with different vanadium runs,
    masks or binning needs the user's approval: :class:`ReductionMismatchError` lists them unless
    ``accept_reduction_differences``. Historic files (reduced before #1053, so without a record) may be
    combined, but cannot be checked: the result carries a ``warning`` naming them; so are files whose
    mask and binning were not recorded (reduced before #1068).

    Parameters
    ----------
    sources : list of dict
        one per source file: ``file``, ``sha256`` and ``record`` (``read_reduction_provenance()`` of it)
    accept_reduction_differences : bool
        the user approved combining files with different vanadium runs, masks or binning

    Returns
    -------
    dict
        the shared record of the recorded sources (``source = "not recorded"`` if none has one), with
        ``sources`` listing every source file and its record, and ``warning`` naming files without a
        record and any accepted differences; an accepted difference leaves out the shared value
        (``vanadium_run``, or ``reduction`` for masks or binning): each source keeps its own

    Raises
    ------
    CalibrationMismatchError
        if the recorded sources were reduced with different calibrations
    ReductionMismatchError
        if they share a calibration but not a vanadium run, mask or binning, and the differences were
        not accepted
    """
    listed = [
        {"file": str(entry["file"]), "sha256": entry.get("sha256"), "record": dict(entry.get("record") or {})}
        for entry in sources
    ]
    recorded = [entry for entry in listed if entry["record"].get("source") not in (None, "not recorded")]
    historic = [entry["file"] for entry in listed if entry not in recorded]

    groups = {}
    for entry in recorded:
        groups.setdefault(_calibration_identity(entry["record"]), []).append(entry)
    if len(groups) > 1:
        lines = [
            f"  {', '.join(os.path.basename(e['file']) for e in group)}: {_describe_calibration(group[0]['record'])}"
            for group in groups.values()
        ]
        raise CalibrationMismatchError(
            "These project files were reduced with different calibrations and cannot be combined:\n"
            + "\n".join(lines)
            + "\nReduce them again with one calibration, or combine each group separately."
        )

    differences = {}  # label -> {description: [file names]}, for each choice that differs
    for label, describe in _REDUCTION_CHOICES:
        by_choice = {}
        for entry in recorded:
            described = describe(entry["record"])
            if described is not None:
                by_choice.setdefault(described, []).append(os.path.basename(entry["file"]))
        if len(by_choice) > 1:
            differences[label] = by_choice
    difference_note = None
    if differences:
        lines = [
            f"{label} — " + "; ".join(f"{choice}: {', '.join(files)}" for choice, files in by_choice.items())
            for label, by_choice in differences.items()
        ]
        if not accept_reduction_differences:
            raise ReductionMismatchError(
                f"These project files share a calibration but were reduced with different "
                f"{', '.join(differences)}:\n" + "\n".join(f"  {line}" for line in lines) + "\nCombine them anyway?"
            )
        difference_note = f"combined with different {', '.join(differences)}, approved by the user: " + "; ".join(
            lines
        )

    combined = dict(recorded[0]["record"]) if recorded else {"source": "not recorded"}
    for key in ("sources", "warning", "error", "reduction"):
        combined.pop(key, None)
    shared = {}  # the mask and binning the sources agree on, where recorded
    known_mask = [entry for entry in recorded if _mask_known(entry["record"])]
    if known_mask and "masks" not in differences:
        shared["mask"] = _reduction(known_mask[0]["record"])["mask"]
    with_settings = [entry for entry in recorded if _reduction(entry["record"]).get("settings")]
    if with_settings and "masks" not in differences and "binning" not in differences:
        shared["settings"] = _reduction(with_settings[0]["record"])["settings"]  # it carries the extra mask
    if shared:
        combined["reduction"] = shared
    if "vanadium runs" in differences:
        combined.pop("vanadium_run", None)  # no single vanadium run: each source's is under sources
    combined["sources"] = listed
    notes = []
    if historic:
        notes.append(
            f"{len(historic)} of {len(listed)} files have no calibration record (reduced before calibration "
            f"records were kept), so their calibration could not be checked: "
            + ", ".join(os.path.basename(name) for name in historic)
        )
    failed = [entry["file"] for entry in recorded if _reduction(entry["record"]).get("error")]
    if failed:
        notes.append(
            f"{len(failed)} of {len(listed)} files' mask and binning record could not be written when they were "
            f"reduced, so their mask and binning could not be checked: "
            + ", ".join(os.path.basename(name) for name in failed)
        )
    unrecorded = [
        entry["file"] for entry in recorded if entry["file"] not in failed and not _mask_known(entry["record"])
    ]
    if unrecorded:
        notes.append(
            f"{len(unrecorded)} of {len(listed)} files do not record which mask they were reduced with (reduced, or "
            f"converted, before masks were recorded), so their mask could not be checked: "
            + ", ".join(os.path.basename(name) for name in unrecorded)
        )
    if difference_note is not None:
        notes.append(difference_note)
    if notes:
        combined["warning"] = "; ".join(notes)
    return combined


def _compute_file_sha256(file_path):
    """Compute the SHA-256 hex digest of a file's contents

    :param str file_path: path to the file to hash
    :return: hex digest, or None if the file cannot be read (e.g. it no longer exists)
    :rtype: str or None
    """
    try:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


class DiffractionUnit(Enum):
    """Enumeration for diffraction data's unit (2theta or d-spacing)"""

    TwoTheta = "2theta"
    DSpacing = "dSpacing"

    def __str__(self):
        return self[()]


class HidraProjectFile:
    """Read and/or write an HB2B project to an HDF5 with entries for detector counts, sample logs, reduced data,
    fitted peaks and etc.
    All the import/export information will be buffered in order to avoid exception during operation

    File structure:

    - experiment
        - scans (raw counts)
        - logs

    - instrument
        - calibration

    - reduced diffraction data
        - main
          - sub-run
          - ...
        - mask_A
          - sub-run
          - ...
        - mask_B
          - sub-run
          - ...
    """

    def __init__(
        self, project_file_name: Union[str, Path], mode: HidraProjectFileMode = HidraProjectFileMode.READONLY
    ):
        """
        Initialization
        :param project_file_name: project file name
        :param mode: I/O mode
        """
        # configure logging for this class
        self._log = Logger(__name__)

        # convert the mode to the enum
        self._io_mode = HidraProjectFileMode.getMode(mode)

        # check the file
        if not project_file_name:
            raise RuntimeError("Must supply a filename")
        # force the file to be a Path
        self._file_name = to_filepath(
            project_file_name, check_exists=bool(self._io_mode != HidraProjectFileMode.OVERWRITE)
        )
        self._checkFileAccess()

        # open the file using h5py
        self._project_h5 = h5py.File(self._file_name, mode=str(self._io_mode))
        if self._io_mode == HidraProjectFileMode.OVERWRITE:
            self._init_project()

    ######################################
    ## Context-manager support methods: ##
    ######################################
    def __enter__(self) -> "HidraProjectFile":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # do not suppress exceptions

    ######################################

    def _checkFileAccess(self):
        """Verify the file has the correct acces permissions and set the value of ``self._is_writable``"""
        # prepare the call to check the file permissions
        check_exist = (self._io_mode == HidraProjectFileMode.READONLY) or (
            self._io_mode == HidraProjectFileMode.READWRITE
        )
        self._is_writable = not self._io_mode == HidraProjectFileMode.READONLY

        # create a custom message based on requested access mode
        if self._io_mode == HidraProjectFileMode.READONLY:
            description = "Read-only project file"
        elif self._io_mode == HidraProjectFileMode.OVERWRITE:
            description = "Write-only project file"
        elif self._io_mode == HidraProjectFileMode.READWRITE:
            description = "Append-mode project file"
        else:  # this should never happen
            raise RuntimeError("Hidra project file I/O mode {} is not supported".format(HidraProjectFileMode))

        # convert the filename to an absolute path so error messages are clearer
        self._file_name = self._file_name.absolute()

        # do the check
        checkdatatypes.check_file_name(
            self._file_name, check_exist, self._is_writable, is_dir=False, description=description
        )

    def _init_project(self):
        """
        initialize the current opened project file from scratch by opening it
        """
        assert self._project_h5 is not None, "cannot be None"
        assert self._is_writable, "must be writable"

        # data
        exp_entry = self._project_h5.create_group(HidraConstants.RAW_DATA)
        exp_entry.create_group(HidraConstants.SUB_RUNS)
        exp_entry.create_group(HidraConstants.SAMPLE_LOGS)

        # instrument
        instrument = self._project_h5.create_group(HidraConstants.INSTRUMENT)
        instrument.create_group(HidraConstants.CALIBRATION)
        # geometry
        geometry_group = instrument.create_group("geometry setup")
        geometry_group.create_group("detector")
        geometry_group.create_group("wave length")
        # detector (pixel) efficiency
        instrument.create_group(HidraConstants.DETECTOR_EFF)

        # mask entry and 2 sub entries
        mask_entry = self._project_h5.create_group(HidraConstants.MASK)
        mask_entry.create_group(HidraConstants.DETECTOR_MASK)
        mask_entry.create_group(HidraConstants.SOLID_ANGLE_MASK)

        # peaks
        self._project_h5.create_group("peaks")

        # reduced data
        self._project_h5.create_group(HidraConstants.REDUCED_DATA)

    def __del__(self):
        self.close()

    @property
    def name(self):
        """
        File name on HDD
        """
        return self._project_h5.filename

    def append_raw_counts(self, sub_run_number: int, counts_array: numpy.ndarray) -> None:
        """Add raw detector counts collected in a single scan/Pt

        Parameters
        ----------
        sub_run_number : int
            sub run number
        counts_array : ~numpy.ndarray
            detector counts
        """
        # check
        assert self._project_h5 is not None, "cannot be None"
        assert self._is_writable, "must be writable"
        sub_run_number = to_int("Sub-run index", sub_run_number, min_value=0)

        # create group
        scan_i_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS].create_group(
            "{:04}".format(sub_run_number)
        )
        scan_i_group.create_dataset("counts", data=counts_array.reshape(-1))

    def append_experiment_log(self, log_name, log_value_array, units=""):
        r"""
        Insert information about the experiment including scan indexes, sample logs, 2theta, etc

        Parameters
        ----------
        log_name: str
            Name of the sample log
        log_value_array: np.ndarray
            Values of the log, one value for each subrun
        units: str
            Units of the sample log

        Raises
        ------
        RuntimeError
            Unable to write this log to the project file
        TypeError
            Unable to write this type of log value to the project file
        """
        # check
        assert self._project_h5 is not None, "cannot be None"
        assert self._is_writable, "must be writable"
        checkdatatypes.check_string_variable("Log name", log_name)

        self._log.debug("Add sample log: {}".format(log_name))
        node_logs = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        try:
            data_set = node_logs.create_dataset(log_name, data=log_value_array)
        except RuntimeError as run_err:
            raise RuntimeError("Unable to add log {} due to {}".format(log_name, run_err))
        except TypeError as type_err:
            raise RuntimeError(
                "Failed to add log {} with value {} of type {}: {}".format(
                    log_name, log_value_array, type(log_value_array), type_err
                )
            )

        if units:
            data_set.attrs["units"] = units

    def read_default_masks(self):
        """Read default mask, i.e., for pixels at the edges

        Returns
        -------
        numpy.ndarray
            array for mask.  None for no mask

        """
        try:
            mask_name = HidraConstants.DEFAULT_MASK
            default_mask = self.read_mask_detector_array(mask_name)
        except RuntimeError:
            default_mask = None

        return default_mask

    def read_user_masks(self, mask_dict):
        """Read user-specified masks

        Parameters
        ----------
        mask_dict : dict
            dictionary to store masks (array)

        Returns
        -------
        None.

        """
        # Get mask names except default mask
        try:
            mask_names = sorted(self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK].keys())
        except KeyError:
            # return if the file has an old format
            return
        if HidraConstants.DEFAULT_MASK in mask_names:
            mask_names.remove(HidraConstants.DEFAULT_MASK)

        # Read mask one by one
        for mask_name in mask_names:
            mask_dict[mask_name] = self.read_mask_detector_array(mask_name)

    def read_mask_detector_array(self, mask_name):
        """Get the mask from hidra project file (.h5) in the form of numpy array

        Location
          root
            - mask
                - detector
                     - mask_name

        Parameters
        ----------
        mask_name : str
            name of mask

        Returns
        -------
        numpy.ndarray
            mask array

        """
        try:
            mask_array = self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK][mask_name][()]
        except KeyError as key_err:
            if HidraConstants.MASK not in self._project_h5.keys():
                err_msg = 'Project file {} does not have "{}" entry.  Its format is not up-to-date.'.format(
                    self._file_name, HidraConstants.MASK
                )
            elif HidraConstants.DETECTOR_MASK not in self._project_h5[HidraConstants.MASK]:
                err_msg = 'Project file {} does not have "{}" entry under {}. Its format is not up-to-date.'.format(
                    self._file_name, HidraConstants.DETECTOR_MASK, HidraConstants.MASK
                )
            else:
                err_msg = "Detector mask {} does not exist.  Available masks are {}.".format(
                    mask_name, self._project_h5[HidraConstants.MASK].keys()
                )
            raise RuntimeError("{}\nFYI: {}".format(err_msg, key_err))

        return mask_array

    def write_mask_detector_array(self, mask_name, mask_array):
        """Write detector mask

        Structure:
          root
            - mask
                - detector
                     - default/universal
                     - mask_name


        Parameters
        ----------
        mask_name : str or None
            mask name.  None for default/universal detector mask
        mask_array : numpy.ndarray
            (N, ), masks, 0 for masking, 1 for ROI

        Returns
        -------
        None

        """
        # Set the default case
        if mask_name is None:
            mask_name = HidraConstants.DEFAULT_MASK

        if mask_name in self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK]:
            # delete the existing mask
            del self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK][mask_name]

        # add new detector mask (array)
        self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK].create_dataset(mask_name, data=mask_array)

    def write_mask_solid_angle(self, mask_name, solid_angle_bin_edges):
        """
        Add mask in the form of solid angle
        Location: ..../main entry/mask/solid angle/
        data will be a range of solid angles and number of patterns to generate.
        example solid angle range = -8, 8, number of pattern = 3

        :param solid_angle_bin_edges: numpy 1D array as s0, s1, s2, ...
        """
        # Clean previously set if name exists
        if mask_name in self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK]:
            del self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK][mask_name]

        # Add new mask in
        solid_angle_entry = self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK]
        solid_angle_entry.create_dataset(mask_name, data=solid_angle_bin_edges)

    def read_mask_solid_angle(self, mask_name):
        """Get the masks in the form of solid angle bin edges"""
        try:
            mask_array = self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK][mask_name]
        except KeyError as key_err:
            raise RuntimeError(
                "Detector mask {} does not exist.  Available masks are {}. FYI: {}".format(
                    mask_name, self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK].keys(), key_err
                )
            )

        return mask_array

    def close(self):
        """
        Close the file without checking whether the file can be written or not. This can
        be called multiple times without issue.
        """
        if self._project_h5 is not None:
            self._project_h5.close()
            self._project_h5 = None  #
            self._log.information("File {} is closed".format(self._file_name))

    def save(self, verbose=False):
        """
        convert all the information about project to HDF file.
        As the data has been written to h5.File instance already, the only thing left is to close the file
        """
        self._validate_write_operation()

        if verbose:
            self._log.information("Changes are saved to {0}. File is now closed.".format(self._project_h5.filename))

        self.close()

    def read_diffraction_2theta_array(self):
        """Get the (reduced) diffraction data's 2-theta vector

        Returns
        -------
        numpy.ndarray
            1D vector for unified 2theta vector for all sub runs
            2D array for possibly various 2theta vector for each

        Raises
        ------
        KeyError
            If this project file's "REDUCED_DATA" group is empty -- a legitimate,
            common case (no reduction step was ever run/saved; the group itself is
            always created up front, regardless of whether reduction ever happens).
        RuntimeError
            If "REDUCED_DATA" is non-empty (e.g. has intensity datasets) but has no
            "TWO_THETA" coordinate dataset -- the file was written with an older,
            unsupported schema (e.g. a legacy capitalized "2Theta" key) and must be
            re-reduced/migrated.
        """
        try:
            two_theta_vec = self._project_h5[HidraConstants.REDUCED_DATA][HidraConstants.TWO_THETA][()]
        except KeyError as key_err:
            if len(self._project_h5[HidraConstants.REDUCED_DATA].keys()) == 0:
                # REDUCED_DATA group exists (always created up front) but is empty:
                # legitimately no reduced data has ever been written to this file.
                raise
            err_msg = (
                'Project file {} has a non-empty "{}" entry but no "{}" coordinate dataset within it. '
                "Its format is not up-to-date and must be re-reduced before it can be read "
                "by this version of PyRS."
            ).format(self._file_name, HidraConstants.REDUCED_DATA, HidraConstants.TWO_THETA)
            raise RuntimeError("{}\nFYI: {}".format(err_msg, key_err))

        return two_theta_vec

    def read_diffraction_intensity_vector(self, mask_id, sub_run):
        """Get the (reduced) diffraction data's intensity
        :param mask_id:
        :param sub_run: If sub run = None: ...
        :return: 1D array or 2D array depending on sub ru
        """
        # Get default for mask/main
        if mask_id is None:
            mask_id = HidraConstants.REDUCED_MAIN

        checkdatatypes.check_string_variable(
            "Mask ID", mask_id, list(self._project_h5[HidraConstants.REDUCED_DATA].keys())
        )

        # Get data to return
        if sub_run is None:
            # all the sub runs
            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id][()]
        else:
            # specific one sub run
            sub_run_list = self.read_sub_runs()
            sub_run_index = sub_run_list.index(sub_run)

            if mask_id is None:
                mask_id = HidraConstants.REDUCED_MAIN

            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id][()][sub_run_index]
        # END-IF-ELSE

        return reduced_diff_hist

    def read_diffraction_variance_vector(self, mask_id, sub_run):
        """Get the (reduced) diffraction data's intensity
        :param mask_id:
        :param sub_run: If sub run = None: ...
        :return: 1D array or 2D array depending on sub ru
        """
        # Get default for mask/main
        if mask_id is None:
            mask_id = HidraConstants.REDUCED_MAIN

        if "_var" not in mask_id:
            mask_id += "_var"

        try:
            checkdatatypes.check_string_variable(
                "Mask ID", mask_id, list(self._project_h5[HidraConstants.REDUCED_DATA].keys())
            )

            # Get data to return
            if sub_run is None:
                # all the sub runs
                reduced_variance_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id][()]
            else:
                # specific one sub run
                sub_run_list = self.read_sub_runs()
                sub_run_index = sub_run_list.index(sub_run)

                if mask_id is None:
                    mask_id = HidraConstants.REDUCED_MAIN

                if "_var" not in mask_id:
                    mask_id += "_var"

                reduced_variance_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id][()][sub_run_index]
            # END-IF-ELSE
        except ValueError:
            reduced_variance_hist = None

        return reduced_variance_hist

    def read_diffraction_masks(self):
        """
        Get the list of masks
        """
        masks = list(self._project_h5[HidraConstants.REDUCED_DATA].keys())

        # Clean up TWO_THETA from the data entry
        if HidraConstants.TWO_THETA in masks:
            masks.remove(HidraConstants.TWO_THETA)

        # Variance datasets are stored alongside intensity datasets with a '_var' suffix
        # (e.g. 'main_var' alongside 'main').  They are not masks and must be excluded so
        # that _load_reduced_diffraction_data does not treat them as intensity entries.
        masks = [m for m in masks if not m.endswith("_var")]

        return masks

    def read_instrument_geometry(self):
        """
        Get instrument geometry parameters
        :return: an instance of instrument_geometry.InstrumentSetup
            *** TODO: actually this returns an instance of `DENEXDetectorGeometry`! ***
        """
        # Get group
        geometry_group = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.GEOMETRY_SETUP]
        detector_group = geometry_group[HidraConstants.DETECTOR_PARAMS]

        # Get value
        num_rows, num_cols = detector_group["detector size"][()]
        pixel_size_x, pixel_size_y = detector_group["pixel dimension"][()]
        arm_length = detector_group["L2"][()]

        # Initialize
        instrument_setup = DENEXDetectorGeometry(
            num_rows=num_rows,
            num_columns=num_cols,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            arm_length=arm_length,
            calibrated=False,
        )

        return instrument_setup

    def read_sample_logs(self):
        """Get sample logs

        Retrieve all the (sample) logs from Hidra project file.
        Raw information retrieved from rs project file is numpy arrays

        Returns
        -------
        ndarray, dict
            ndarray : 1D array for sub runs
            dict : dict[sample log name] for sample logs in ndarray
        """
        # Get the group
        logs_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]

        if HidraConstants.SUB_RUNS not in logs_group.keys():
            raise RuntimeError(
                "Failed to find {} in {} group of the file".format(HidraConstants.SUB_RUNS, HidraConstants.SAMPLE_LOGS)
            )

        # Get 2theta and others
        samplelogs = SampleLogs()
        # first set subruns
        samplelogs[HidraConstants.SUB_RUNS] = logs_group[HidraConstants.SUB_RUNS][()]
        for log_name in logs_group.keys():
            data_set = logs_group[log_name]  # an instance of HDF5::DataSet
            try:
                samplelogs[log_name, data_set.attrs["units"]] = data_set[()]
            except KeyError:  # this log entry has no units. True for old project files
                samplelogs[log_name] = data_set[()]

        return samplelogs

    def read_run_number(self) -> int:
        """The file's run number: -1 if it is missing or the file combines several runs.

        -1 for combined runs matches what the peak-fit engine records for appended runs, so peaks
        saved into a combined file can be read back (#1080).
        """
        try:
            values = self.read_log_value("run_number")
            values_set = set(values[()].astype(int))
            if len(values_set) > 1:
                return -1
            return values_set.pop()
        except KeyError:
            # field is missing
            return -1

    def read_log_value(self, log_name):
        """Get a log's value

        Parameters
        ----------
        log_name

        Returns
        -------
        ndarray or single value
        """
        assert self._project_h5 is not None, "Project HDF5 is not loaded yet"

        log_value = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS][log_name]

        return log_value

    def read_log_units(self, log_name):
        r"""
        Read the units of a sample log

        Parameters
        ----------
        log_name: str

        Returns
        -------
        str
            Empty string if the sample log has no units associated
        """
        # Validation
        assert self._project_h5 is not None, "Project HDF5 is not loaded yet"
        group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        assert log_name in group.keys(), f"Missing sample log: {log_name}"

        data_set = group[log_name]
        try:
            return data_set.attrs["units"]
        except KeyError:
            return ""

    def read_raw_counts(self, sub_run: int) -> numpy.ndarray:
        """
        get the raw detector counts
        """
        assert self._project_h5 is not None, "blabla"
        sub_run = to_int("sun run", sub_run, min_value=0)

        sub_run_str = "{:04}".format(sub_run)
        try:
            counts = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS][sub_run_str]["counts"][()]
        except KeyError as key_error:
            err_msg = "Unable to access sub run {} with key {}: {}\nAvailable runs are: {}".format(
                sub_run,
                sub_run_str,
                key_error,
                self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS].keys(),
            )
            raise KeyError(err_msg)

        return counts

    def read_sub_runs(self):
        """
        get list of the sub runs
        """
        self._log.debug(str(self._project_h5.keys()))
        self._log.debug(str(self._file_name))
        # coded a little wacky to be less than 120 characters across
        sub_runs_str_list = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        if HidraConstants.SUB_RUNS in sub_runs_str_list:
            sub_runs_str_list = sub_runs_str_list[HidraConstants.SUB_RUNS][()]
        else:
            sub_runs_str_list = []

        self._log.debug(".... Sub runs: {}".format(sub_runs_str_list))

        sub_run_list = [None] * len(sub_runs_str_list)
        for index, sub_run_str in enumerate(sub_runs_str_list):
            sub_run_list[index] = int(sub_run_str)

        self._log.debug(".... Sub runs: {}".format(sub_run_list))

        return sub_run_list

    def write_instrument_geometry(self, instrument_setup):
        """
        Add instrument geometry and wave length information to project file
        """
        # check inputs
        self._validate_write_operation()
        checkdatatypes.check_type("Instrument geometry setup", instrument_setup, HidraSetup)

        # PyRS version producing this file: recorded at the root level so it survives
        # regardless of which writer touches the file afterwards
        self._project_h5.attrs["pyrs_version"] = __version__

        # write value to instrument
        instrument_group = self._project_h5[HidraConstants.INSTRUMENT]

        # write attributes
        instrument_group.attrs["name"] = instrument_setup.name

        # get the entry for raw instrument setup
        detector_group = instrument_group["geometry setup"]["detector"]
        raw_geometry = instrument_setup.get_instrument_geometry(False)
        detector_group.create_dataset("L2", data=numpy.array(raw_geometry.arm_length))
        det_size = numpy.array(instrument_setup.get_instrument_geometry(False).detector_size)
        detector_group.create_dataset("detector size", data=det_size)
        pixel_dimension = list(instrument_setup.get_instrument_geometry(False).pixel_dimension)
        detector_group.create_dataset("pixel dimension", data=numpy.array(pixel_dimension))

        # wave length
        wavelength_group = instrument_group[HidraConstants.GEOMETRY_SETUP][HidraConstants.WAVELENGTH]
        try:
            wl = instrument_setup.get_wavelength(None)
        except (NotImplementedError, RuntimeError) as run_err:
            # No wave length from workspace: do nothing
            self._log.error(str(run_err))
            wl = None

        # Set wave length
        if wl is not None:
            wavelength_group.create_dataset("Calibrated", data=numpy.array([wl]))

    def write_reduction_provenance(
        self,
        calibration_file=None,
        vanadium_run=None,
        applied_calibration=None,
        instrument=None,
        reduction_settings=None,
    ):
        """Record which calibration and vanadium run (if any) produced this file's reduced data

        Populates the CALIBRATION group (previously created but never populated) with enough
        information for a scientist to determine, from the output file alone, which calibration
        was applied and which vanadium run was used for normalization. Also (re-)records the
        PyRS version, in case this file was never touched by write_instrument_geometry.

        Parameters
        ----------
        calibration_file : str or None
            path to the calibration JSON/ASCII file applied during reduction, if any
        vanadium_run : str or None
            path or run identifier of the vanadium project/NeXus file used for normalization,
            if any
        instrument : HidraSetup or None
            the instrument the reduction used: the calibration record is read from it
            (``HidraSetup.calibration_record()``, #1053) — the usual way to call this
        applied_calibration : dict or None
            a calibration record without an instrument (keys as ``HidraSetup.calibration_record()``).
            Written as datasets ``calibration_json`` (the calibration JSON's dictionary) and
            ``provenance_json`` (JSON text, no size limit) and attributes; a re-reduction replaces them. A failure to write it is recorded and warned
            about, never raised: the reduced data are saved regardless. Its ``reduction`` entry, if any, is
            written to the REDUCTION group (as a combined file's, #1068).
        reduction_settings : dict or None
            with ``instrument``: the binning and extra mask the reduction used (``ReductionApp``). Written,
            with the instrument's ``mask_record``, as ``reduction_json`` in the REDUCTION group beside
            CALIBRATION (#1068); replaced by a re-reduction, and never raised on, as the calibration record.
        """
        self._validate_write_operation()

        self._project_h5.attrs["pyrs_version"] = __version__

        if instrument is not None:
            applied_calibration = dict(_instrument_record(instrument))
            applied_calibration["reduction"] = _reduction_record(instrument, reduction_settings)
        if calibration_file is None and vanadium_run is None and applied_calibration is None:
            return

        # created if missing: an older or externally written file may lack it, and recording the
        # provenance must not stop the reduced data from being saved (#1053)
        instrument_group = self._project_h5.require_group(HidraConstants.INSTRUMENT)
        calibration_group = instrument_group.require_group(HidraConstants.CALIBRATION)
        if applied_calibration is not None:
            _clear_record(calibration_group)  # a reduction's record replaces the previous one entirely
            if HidraConstants.REDUCTION in instrument_group:
                del instrument_group[HidraConstants.REDUCTION]

        if calibration_file is not None:
            calibration_group.attrs["calibration_file"] = str(calibration_file)
            sha256 = _compute_file_sha256(calibration_file)
            if sha256 is not None:
                calibration_group.attrs["calibration_file_sha256"] = sha256

        if vanadium_run is not None:
            calibration_group.attrs["vanadium_run"] = str(vanadium_run)

        if applied_calibration is not None:
            _write_applied_calibration(calibration_group, applied_calibration)
            if applied_calibration.get("reduction") is not None:
                _write_reduction_record(instrument_group, applied_calibration["reduction"])

    def write_mask_source(self, mask_source):
        """Record where the default mask came from in a converted (not yet reduced) file, as the REDUCTION
        group's ``mask`` (#1068); a reduction of the file replaces the group. Never raises.

        Parameters
        ----------
        mask_source : dict
            ``HidraWorkspace.get_detector_mask_source()``
        """
        self._validate_write_operation()
        instrument_group = self._project_h5.require_group(HidraConstants.INSTRUMENT)
        if HidraConstants.REDUCTION in instrument_group:
            del instrument_group[HidraConstants.REDUCTION]
        _write_reduction_record(instrument_group, {"mask": mask_source})

    def read_reduction_provenance(self):
        """What the reduction recorded about its calibration and vanadium (#1053), as a dict

        Returns
        -------
        dict
            with the keys of ``HidraSetup.calibration_record()``: ``source``, ``calibration`` (the
            calibration JSON's dictionary), ``calibration_file``, ``calibration_file_sha256``, ``created``,
            ``created_source``, ``lambda_source``, ``provenance``, ``error``, and ``vanadium_run`` — each
            where present (file attributes are prefixed ``calibration_``); ``reduction``: the mask and
            binning the reduction used (``mask``, ``settings``; ``error`` if they could not be recorded),
            where recorded (#1068); empty without either group
        """
        instrument_group = self._project_h5.get(HidraConstants.INSTRUMENT)
        if instrument_group is None:
            return {}
        record = {}
        group = instrument_group.get(HidraConstants.CALIBRATION)
        if group is not None:
            for attribute, value in group.attrs.items():
                record[_RECORD_ATTRIBUTES.get(attribute, attribute)] = _plain(value)
            if "run_geometry" in record:
                record["run_geometry"] = json.loads(record["run_geometry"])
            for name, key in (
                ("calibration_json", "calibration"),
                ("provenance_json", "provenance"),
                ("sources_json", "sources"),
            ):
                if name in group:
                    record[key] = json.loads(_plain(group[name][()]))
        reduction_group = instrument_group.get(HidraConstants.REDUCTION)
        if reduction_group is not None:
            reduction = {}
            if _REDUCTION_DATASET in reduction_group:
                reduction = json.loads(_plain(reduction_group[_REDUCTION_DATASET][()]))
            if _REDUCTION_ERROR in reduction_group.attrs:
                reduction["error"] = _plain(reduction_group.attrs[_REDUCTION_ERROR])
            record["reduction"] = reduction
        return record

    def read_reduction_instrument(self, geometry=None):
        """The instrument (HidraSetup) a reduction recorded, with the calibration it applied, so the data
        can be reduced again with exactly that calibration (#1053)

        Parameters
        ----------
        geometry : DENEXDetectorGeometry or None
            the detector geometry; None: the one the reduction used, as recorded with the calibration
            (it lives within the calibration, #1053), else the file's own ``geometry setup``.

        Returns
        -------
        HidraSetup
            with ``calibration_source`` ``"none"`` if nothing was recorded

        Raises
        ------
        RuntimeError
            if no geometry is given and the file has none
        """
        from pyrs.utilities.calibration_file_io import geometry_from_dict

        record = self.read_reduction_provenance()
        if geometry is None:
            recorded = record.get("run_geometry") or (record.get("calibration") or {}).get("geometry")
            if recorded is not None:
                geometry = geometry_from_dict(recorded)
        if geometry is None:
            try:
                geometry = self.read_instrument_geometry()
            except KeyError as missing:
                raise RuntimeError(
                    f"{self._project_h5.filename} stores no detector geometry ({missing}); pass the run's geometry"
                ) from None
        instrument = HidraSetup(geometry)
        if "source" in record:
            instrument.set_calibration_record(record)
        mask = (record.get("reduction") or {}).get("mask")
        if mask is not None:
            instrument.set_mask_record(mask)
        return instrument

    def read_peak_tags(self):
        """Get all the tags of peaks with parameters stored in HiDRA project

        Returns
        -------
        ~list
            list of string for all the peak tags

        """
        # Get main group
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        return list(peak_main_group.keys())

    def read_peak_parameters(self, peak_tag):
        """Get the parameters related to a peak

        The parameters including
        (1) peak profile (2) sub runs (3) chi2 (4) parameter names (5) parameter values

        Returns
        -------
        ~pyrs.peaks.peak_collection.PeakCollection
            All of the information from fitting a peak across subruns
        """
        # Get main group
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        # Get peak entry
        if peak_tag not in peak_main_group.keys():
            raise RuntimeError("Peak tag {} cannot be found".format(peak_tag))
        peak_entry = peak_main_group[peak_tag]

        # Get all the attribute and data
        profile = peak_entry.attrs[HidraConstants.PEAK_PROFILE]
        background = peak_entry.attrs[HidraConstants.BACKGROUND_TYPE]
        sub_run_array = peak_entry[HidraConstants.SUB_RUNS][()]
        chi2_array = peak_entry[HidraConstants.PEAK_FIT_CHI2][()]
        param_values = peak_entry[HidraConstants.PEAK_PARAMS][()]
        error_values = peak_entry[HidraConstants.PEAK_PARAMS_ERROR][()]

        try:
            exclude_list = peak_entry[HidraConstants.EXCLUDE_PEAKS][()]
        except KeyError:
            exclude_list = [False] * sub_run_array.size

        # validate the information makes sense
        if param_values.shape != error_values.shape:
            raise RuntimeError(
                "Parameters[{}] and Errors[{}] have different shape".format(param_values.shape, error_values.shape)
            )
        peak_collection = PeakCollection(
            peak_tag,
            profile,
            background,
            self.read_wavelengths(),
            projectfilename=self._file_name,
            runnumber=self.read_run_number(),
        )

        peak_collection.set_peak_fitting_values(
            subruns=sub_run_array,
            parameter_values=param_values,
            parameter_errors=error_values,
            fit_costs=chi2_array,
            exclude_list=exclude_list,
        )

        # Optionally for strain: reference peak center in dSpacing: (strain)
        if HidraConstants.D_REFERENCE in list(peak_entry.keys()):
            # If reference position D is ever written to this project
            ref_d_array = peak_entry[HidraConstants.D_REFERENCE][()]
            if HidraConstants.D_REFERENCE_ERROR in list(peak_entry.keys()):
                ref_d_error = peak_entry[HidraConstants.D_REFERENCE_ERROR][()]
                peak_collection.set_d_reference(ref_d_array, ref_d_error)
            else:
                peak_collection.set_d_reference(ref_d_array)

        return peak_collection

    def write_peak_parameters(self, fitted_peaks):
        """Set the peak fitting results to project file.

        The tree structure for fitted peak in all sub runs is defined as

          - peaks

              - [peak-tag]

                - attr/'peak profile'
                - sub runs
                - parameter values
                - parameter fitting error

        Parameters
        ----------
        fitted_peaks : pyrs.core.peak_collection.PeakCollection

        Returns
        -------

        """
        # Check inputs and file status
        self._validate_write_operation()

        # Get value from peak collection
        peak_tag = fitted_peaks.peak_tag
        peak_profile = str(fitted_peaks.peak_profile)
        background_type = str(fitted_peaks.background_type)

        checkdatatypes.check_string_variable("Peak tag", peak_tag)
        checkdatatypes.check_string_variable("Peak profile", peak_profile)
        checkdatatypes.check_string_variable("Background type", background_type)

        # access or create node for peak with given tag
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        if peak_tag not in peak_main_group:
            # create peak-tag entry if it does not exist
            single_peak_entry = peak_main_group.create_group(peak_tag)
        else:
            # if peak-tag entry, get the reference to the entry
            single_peak_entry = peak_main_group[peak_tag]

        # Attributes
        self.set_attributes(single_peak_entry, HidraConstants.PEAK_PROFILE, peak_profile)
        self.set_attributes(single_peak_entry, HidraConstants.BACKGROUND_TYPE, background_type)

        # Create or replace each dataset as appropriate
        if HidraConstants.SUB_RUNS not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.SUB_RUNS, data=fitted_peaks.sub_runs)
        else:
            single_peak_entry[HidraConstants.SUB_RUNS][...] = fitted_peaks.sub_runs

        if HidraConstants.PEAK_FIT_CHI2 not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.PEAK_FIT_CHI2, data=fitted_peaks.fitting_costs)
        else:
            single_peak_entry[HidraConstants.PEAK_FIT_CHI2][...] = fitted_peaks.fitting_costs

        if HidraConstants.EXCLUDE_PEAKS not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.EXCLUDE_PEAKS, data=fitted_peaks.exclude)
        else:
            single_peak_entry[HidraConstants.EXCLUDE_PEAKS][...] = fitted_peaks.exclude

        peak_values, peak_errors = fitted_peaks.get_native_params()
        if HidraConstants.PEAK_PARAMS not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.PEAK_PARAMS, data=peak_values)
        else:
            single_peak_entry[HidraConstants.PEAK_PARAMS][...] = peak_values

        if HidraConstants.PEAK_PARAMS_ERROR not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.PEAK_PARAMS_ERROR, data=peak_errors)
        else:
            single_peak_entry[HidraConstants.PEAK_PARAMS_ERROR][...] = peak_errors

        effective_peak_values, effective_peak_errors = fitted_peaks.get_effective_params()
        if HidraConstants.EFFECTIVE_PEAK_PARAMS not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.EFFECTIVE_PEAK_PARAMS, data=effective_peak_values)
        else:
            single_peak_entry[HidraConstants.EFFECTIVE_PEAK_PARAMS][...] = effective_peak_values

        if HidraConstants.EFFECTIVE_PEAK_PARAMS_ERROR not in single_peak_entry:
            single_peak_entry.create_dataset(HidraConstants.EFFECTIVE_PEAK_PARAMS_ERROR, data=effective_peak_errors)
        else:
            single_peak_entry[HidraConstants.EFFECTIVE_PEAK_PARAMS_ERROR][...] = effective_peak_errors

        # Reference peak center in dSpacing: (strain)
        ref_d_array, ref_d_errors = fitted_peaks.get_d_reference()
        if isinstance(ref_d_array, numpy.ndarray):
            # if reference peak position in D is set
            if HidraConstants.D_REFERENCE not in single_peak_entry:
                single_peak_entry.create_dataset(HidraConstants.D_REFERENCE, data=ref_d_array)
            else:
                single_peak_entry[HidraConstants.D_REFERENCE][...] = ref_d_array

            if HidraConstants.D_REFERENCE_ERROR not in single_peak_entry:
                single_peak_entry.create_dataset(HidraConstants.D_REFERENCE_ERROR, data=ref_d_errors)
            else:
                single_peak_entry[HidraConstants.D_REFERENCE_ERROR][...] = ref_d_errors
        else:
            raise RuntimeError("Unexpected type for reference d-values")

    def read_wavelengths(self):
        """Get calibrated wave length

        Returns
        -------
        Float
            Calibrated wave length.  NaN for wave length is not ever set
        """
        # Init wave length
        wl = numpy.nan

        # Get the node
        try:
            mono_node = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO]
            if HidraConstants.WAVELENGTH in mono_node:
                wl = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO][HidraConstants.WAVELENGTH][()]
                if wl.shape[0] == 0:
                    # empty numpy array: no data. keep as nan
                    pass
                elif wl.shape[0] == 1:
                    # 1 calibrated wave length
                    wl = wl[0]
                else:
                    # not supported
                    raise RuntimeError("There are more than 1 wave length registered")
                    # END-IF
        except KeyError:
            # monochromator node does not exist
            self._log.error(
                "Node {} does not exist in HiDRA project file {}".format(HidraConstants.MONO, self._file_name)
            )
        # END

        return wl

    def write_wavelength(self, wave_length: float):
        """Set the calibrated wave length

        Location:

          .../instrument/monochromator setting/ ... .../

        Note:

          - same wave length to all sub runs
          - only calibrated wave length in project file
          - raw wave length comes from a table with setting

        :param wave_length: wave length in A

        :return: None
        """
        wave_length = to_float("Wave length", wave_length, min_value=0, max_value=1000)

        # Create 'monochromator setting' node if it does not exist
        if HidraConstants.MONO not in list(self._project_h5[HidraConstants.INSTRUMENT].keys()):
            self._project_h5[HidraConstants.INSTRUMENT].create_group(HidraConstants.MONO)

        # Get node and write value
        wl_entry = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO]
        # delete the dataset if it does exist to replace
        if HidraConstants.WAVELENGTH in list(wl_entry.keys()):
            del wl_entry[HidraConstants.WAVELENGTH]
        wl_entry.create_dataset(HidraConstants.WAVELENGTH, data=numpy.array([wave_length]))

    def read_efficiency_correction(self):
        """
        Set detector efficiency correction measured from vanadium (efficiency correction)
        Returns
        -------
        numpy ndarray
            Efficiency array
        """
        calib_run_number = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].attrs[
            HidraConstants.RUN
        ]

        det_eff_array = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF][
            "{}".format(calib_run_number)
        ]

        return det_eff_array

    def write_efficiency_correction(self, calib_run_number, eff_array):
        """Set detector efficiency correction measured from vanadium (efficiency correction)
        Location: ... /main entry/calibration/efficiency:
        Data: numpy array with 1024**2...
        Attribute: add the run number created from to the attribute
        Parameters
        ----------
        calib_run_number : integer
            Run number where the efficiency calibration comes from
        eff_array : numpy ndarray (1D)
            Detector (pixel) efficiency
        """
        # Add attribute
        self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].attrs[HidraConstants.RUN] = (
            calib_run_number
        )

        # Set data
        self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].create_dataset(
            "{}".format(calib_run_number), data=eff_array
        )

    def write_information(self, info_dict):
        """
        set project information to attributes
        """
        # check and validate
        checkdatatypes.check_dict("Project file general information", info_dict)
        self._validate_write_operation()

        for info_name in info_dict:
            self._project_h5.attrs[info_name] = info_dict[info_name]

    def write_reduced_diffraction_data_set(self, two_theta_array, diff_data_set, var_data_set):
        """Set the reduced diffraction data (set)

        Parameters
        ----------
        two_theta_array : numppy.ndarray
            2D array for 2-theta vector, which could be various to each other among sub runs
        diff_data_set : dict
            dictionary of 2D arrays for reduced diffraction patterns' intensities
        var_data_set : dict
            dictionary of 2D arrays for reduced diffraction patterns' variances
        """

        # Check input
        checkdatatypes.check_numpy_arrays("Two theta vector", [two_theta_array], 2, False)
        checkdatatypes.check_dict("Diffraction data set", diff_data_set)

        # Retrieve diffraction group
        diff_group = self._project_h5[HidraConstants.REDUCED_DATA]

        # Add 2theta vector
        if HidraConstants.TWO_THETA in diff_group.keys():
            # the file keeps one 2theta axis for all masks: replacing it would leave masks already in the
            # file, and not rewritten now, on the wrong axis
            stored = diff_group[HidraConstants.TWO_THETA][()]
            written = {HidraConstants.REDUCED_MAIN if mask_id is None else mask_id for mask_id in diff_data_set}
            left_behind = sorted(
                name
                for name in diff_group.keys()
                if name != HidraConstants.TWO_THETA and not name.endswith("_var") and name not in written
            )
            same_axis = stored.shape == two_theta_array.shape and numpy.allclose(
                stored, two_theta_array, rtol=0, atol=1e-9
            )
            if left_behind and not same_axis:
                raise RuntimeError(
                    "Cannot write reduced data with a different 2theta axis: masks {} already in {} would be "
                    "relabelled. Reduce all masks with the same 2theta range, or write to a new project "
                    "file.".format(left_behind, self._file_name)
                )
            # over write data
            try:
                diff_group[HidraConstants.TWO_THETA][...] = two_theta_array
            except TypeError:
                # usually two theta vector size changed
                del diff_group[HidraConstants.TWO_THETA]
                diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_array)
        else:
            # new data
            diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_array)

        # Add Diffraction data
        for mask_id in diff_data_set:
            # Get data
            diff_data_matrix_i = diff_data_set[mask_id]
            self._log.information("Mask {} data set shape: {}".format(mask_id, diff_data_matrix_i.shape))
            # Check
            checkdatatypes.check_numpy_arrays("Diffraction data (matrix)", [diff_data_matrix_i], None, False)
            if two_theta_array.shape != diff_data_matrix_i.shape:
                raise RuntimeError(
                    "Length of 2theta vector ({}) is different from intensities ({})".format(
                        two_theta_array.shape, diff_data_matrix_i.shape
                    )
                )
            # Set name for default mask
            if mask_id is None:
                data_name = HidraConstants.REDUCED_MAIN
            else:
                data_name = mask_id

            # Write
            if data_name in diff_group.keys():
                # overwrite
                diff_h5_data = diff_group[data_name]
                try:
                    diff_h5_data[...] = diff_data_matrix_i
                except TypeError:
                    # usually two theta vector size changed
                    del diff_group[data_name]
                    diff_group.create_dataset(data_name, data=diff_data_matrix_i)
            else:
                # new
                diff_group.create_dataset(data_name, data=diff_data_matrix_i)

        # Add Variances data
        if var_data_set is None:
            var_data_set = diff_data_set
            for mask_id in var_data_set:
                var_data_set[mask_id] = numpy.sqrt(var_data_set[mask_id])

        for mask_id in var_data_set:
            # Get data
            var_data_matrix_i = var_data_set[mask_id]
            self._log.information("Mask {} data set shape: {}".format(mask_id, var_data_matrix_i.shape))
            # Check
            checkdatatypes.check_numpy_arrays("Diffraction data (matrix)", [var_data_matrix_i], None, False)
            if two_theta_array.shape != var_data_matrix_i.shape:
                raise RuntimeError(
                    "Length of 2theta vector ({}) is different from intensities ({})".format(
                        two_theta_array.shape, var_data_matrix_i.shape
                    )
                )
            # Set name for default mask
            if mask_id is None:
                data_name = HidraConstants.REDUCED_MAIN + "_var"
            else:
                data_name = mask_id + "_var"

            # Write
            if data_name in diff_group.keys():
                # overwrite
                diff_h5_data = diff_group[data_name]
                try:
                    diff_h5_data[...] = var_data_matrix_i
                except TypeError:
                    # usually two theta vector size changed
                    del diff_group[data_name]
                    diff_group.create_dataset(data_name, data=var_data_matrix_i)
            else:
                # new
                diff_group.create_dataset(data_name, data=var_data_matrix_i)

    def write_sub_runs(self, sub_runs):
        """Set sub runs to sample log entry"""
        if isinstance(sub_runs, list):
            sub_runs = numpy.array(sub_runs)
        else:
            checkdatatypes.check_numpy_arrays("Sub run numbers", [sub_runs], 1, False)

        sample_log_entry = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        sample_log_entry.create_dataset(HidraConstants.SUB_RUNS, data=sub_runs)

    def _create_diffraction_node(self, sub_run_number):
        """Create a node to record diffraction data
        It will check if such node already exists
        :exception: RuntimeError is raised if such 'sub run' node exists but not correct
        """
        # create a new node if it does not exist
        sub_run_group_name = "{0:04}".format(sub_run_number)

        self._log.debug("sub group entry name in hdf: {}".format(sub_run_group_name))

        # check existing node or create a new node
        self._log.debug(
            "Diffraction node sub group/entries: {}".format(self._project_h5[HidraConstants.REDUCED_DATA].keys())
        )
        if sub_run_group_name in self._project_h5[HidraConstants.REDUCED_DATA]:
            # sub-run node exist and check
            self._log("sub-group: {}".format(sub_run_group_name))
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA][sub_run_group_name]
            if not (DiffractionUnit.TwoTheta in diff_group and DiffractionUnit.DSpacing in diff_group):
                raise RuntimeError("Diffraction node for sub run {} exists but is not complete".format(sub_run_number))
        else:
            # create new node: parent, child-2theta, child-dspacing
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA].create_group(sub_run_group_name)
            diff_group.create_group(str(DiffractionUnit.TwoTheta))
            diff_group.create_group(str(DiffractionUnit.DSpacing))

        return diff_group

    def _validate_write_operation(self):
        """
        Validate whether a writing operation is allowed for this file
        :exception: RuntimeError
        """
        if not self._is_writable:
            raise RuntimeError("Project file {} is set to read-only by user".format(self._project_h5.name))

    @staticmethod
    def set_attributes(h5_group, attribute_name, attribute_value):
        """
        Set attribute to a group
        """
        checkdatatypes.check_string_variable("Attribute name", attribute_name)

        h5_group.attrs[attribute_name] = attribute_value
