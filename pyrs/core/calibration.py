"""
The calibration a reduction applied, as one immutable record with an identity (#1141).

A calibration is the detector shift, rotations and 2theta_0 (`DENEXDetectorShift`), their errors, the
wavelength and its error, and the sample-to-detector distance L2 reduction used. Two calibrations are the same
**if and only if** their content hashes are equal, and the record is the one place that decides it: callers
never compare individual values (#1141, user decision, 2026-11-24).

Error codes are the calibration files': -1 is the default (not refined, or held), -2 an unknown error (refined,
but the error could not be determined or was lost).

Decisions and their reasons: docs/ground_truths.md, "The applied calibration has an identity".

Typical workflow:
    1. A reduction builds the record of what it applied: from a calibration file, from the workspace's own
       shift, or :func:`nominal_calibration` when no calibration file is given.
    2. Files store the record with its source and its hash.
    3. Anything that combines data asks :meth:`AppliedCalibration.same_calibration_as`.

Example:
    >>> from pyrs.core import MonoSetting
    >>> nominal = nominal_calibration(MonoSetting.Si331, l2=0.985)
    >>> nominal.source
    <CalibrationSource.NONE: 'none'>
    >>> nominal.same_calibration_as(nominal_calibration(MonoSetting.Si331, l2=0.985))
    True
"""

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple

from pyrs.core.calibration_source import CalibrationSource
from pyrs.core.detector_model import from_shift, to_shift
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.monosetting import MonoSetting
from pyrs.utilities.calibration_file_io import calibration_from_dict

if TYPE_CHECKING:  # pyrs.core.workspaces imports this module
    from pyrs.core.workspaces import HidraWorkspace

__all__ = [
    "CalibrationSource",
    "AppliedCalibration",
    "nominal_calibration",
    "reduction_l2",
    "run_monosetting",
    "stated_l2",
    "DEFAULT_ERROR",
    "UNKNOWN_ERROR",
]

DEFAULT_ERROR = -1.0  # not refined (or held): the nominal calibration's errors
UNKNOWN_ERROR = -2.0  # refined, but the error could not be determined or was lost
_PARAMETER_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0")  # the calibration JSON's
_N_PARAMETERS = len(_PARAMETER_KEYS)


def _check_error(name: str, error: float) -> None:
    if not math.isfinite(error) or (error < 0 and error not in (DEFAULT_ERROR, UNKNOWN_ERROR)):
        raise ValueError(f"{name} error must be >= 0, -1 (not refined) or -2 (unknown), got {error}")


@dataclass(frozen=True)
class AppliedCalibration:
    """The calibration a reduction applied: what it is (its values, hashed) and where it came from.

    Build it with :meth:`from_shift` or :func:`nominal_calibration`. It cannot change once built, so its hash
    cannot go stale.

    Attributes:
        values: Shift_x, Shift_y, Shift_z, Rot_x, Rot_y, Rot_z, TTH_0 (m and degrees).
        errors: Their errors: >= 0, or -1 (not refined) / -2 (unknown).
        wavelength: The wavelength reduction used (angstrom).
        wavelength_error: Its error: >= 0, -1 or -2.
        l2: The sample-to-detector distance reduction used (m).
        source: Where the calibration came from.
        status: The refinement's optimizer exit code; None when nothing was refined.
    """

    values: Tuple[float, ...]
    errors: Tuple[float, ...]
    wavelength: float
    wavelength_error: float
    l2: float
    source: CalibrationSource
    status: Optional[int]

    def __post_init__(self) -> None:
        if not isinstance(self.source, CalibrationSource):
            raise TypeError(f"source must be a CalibrationSource, got {type(self.source).__name__}")
        for name, sequence in (("values", self.values), ("errors", self.errors)):
            if len(sequence) != _N_PARAMETERS:
                raise ValueError(f"{name} must hold {_N_PARAMETERS} numbers, got {len(sequence)}")
        for value in self.values:
            if not math.isfinite(value):
                raise ValueError(f"calibration values must be finite, got {self.values}")
        for error in self.errors:
            _check_error("calibration", error)
        if not (math.isfinite(self.wavelength) and self.wavelength > 0):
            raise ValueError(f"wavelength must be finite and > 0, got {self.wavelength}")
        _check_error("wavelength", self.wavelength_error)
        if not (math.isfinite(self.l2) and self.l2 > 0):
            raise ValueError(f"L2 must be finite and > 0, got {self.l2}")
        if self.status is not None and not isinstance(self.status, int):
            raise TypeError(f"status must be an int or None, got {type(self.status).__name__}")
        if self.source is CalibrationSource.NONE and (
            any(value != 0.0 for value in self.values)
            or any(error != DEFAULT_ERROR for error in self.errors + (self.wavelength_error,))
        ):
            raise ValueError(
                "a calibration with source 'none' must be the nominal one (values 0, errors -1), got "
                f"values {self.values}, errors {self.errors}, wavelength error {self.wavelength_error}"
            )

    @classmethod
    def from_shift(
        cls,
        shift: DENEXDetectorShift,
        shift_error: DENEXDetectorShift,
        *,
        wavelength: float,
        wavelength_error: float,
        l2: float,
        source: CalibrationSource,
        status: Optional[int],
    ) -> "AppliedCalibration":
        """A record from a detector shift and its errors.

        Args:
            shift: Detector shift, rotations and 2theta_0.
            shift_error: Their errors (>= 0, -1 or -2).
            wavelength: Wavelength (angstrom).
            wavelength_error: Its error.
            l2: Sample-to-detector distance reduction used (m).
            source: Where the calibration came from.
            status: The refinement's optimizer exit code, or None when nothing was refined.

        Returns:
            The record.

        Raises:
            TypeError: If `shift` or `shift_error` is not a DENEXDetectorShift.
            ValueError: If a value breaks the record's rules (see the class).
        """
        for name, value in (("shift", shift), ("shift_error", shift_error)):
            if not isinstance(value, DENEXDetectorShift):
                raise TypeError(f"{name} must be a DENEXDetectorShift, got {type(value).__name__}")
        return cls(
            values=tuple(float(v) for v in from_shift(shift)),
            errors=tuple(float(e) for e in from_shift(shift_error)),
            wavelength=float(wavelength),
            wavelength_error=float(wavelength_error),
            l2=float(l2),
            source=source,
            status=None if status is None else int(status),
        )

    @property
    def shift(self) -> DENEXDetectorShift:
        """The detector shift, rotations and 2theta_0, as a new object each time."""
        return to_shift(self.values)

    @property
    def shift_error(self) -> DENEXDetectorShift:
        """Their errors, as a new object each time."""
        return to_shift(self.errors)

    @property
    def content_hash(self) -> str:
        """The calibration's identity: sha256 of its 17 numbers as little-endian float64, in a fixed order.

        The order is the 7 values, their 7 errors, the wavelength, its error and L2. The source, the status and
        the file it came from are not hashed: they say where a calibration came from, not what it is.
        """
        numbers = self.values + self.errors + (self.wavelength, self.wavelength_error, self.l2)
        return hashlib.sha256(struct.pack(f"<{len(numbers)}d", *numbers)).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """The record as plain JSON-ready values, with its content hash, for a file to store.

        Returns:
            ``values``, ``errors``, ``wavelength``, ``wavelength_error``, ``l2``, ``source`` (its stored string),
            ``status`` and ``content_hash``.
        """
        return {
            "values": list(self.values),
            "errors": list(self.errors),
            "wavelength": self.wavelength,
            "wavelength_error": self.wavelength_error,
            "l2": self.l2,
            "source": self.source.value,
            "status": self.status,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, stored: Mapping[str, Any], *, where: str) -> "AppliedCalibration":
        """The record a file stored with :meth:`to_dict`, checked against its stored hash.

        Args:
            stored: The stored values.
            where: The file it was read from, for the error message.

        Returns:
            The record.

        Raises:
            ValueError: If a field is missing or invalid, the source is not a member, or the hash the file stored
                is not the record's: the record is corrupt.
        """
        try:
            calibration = cls(
                values=tuple(float(v) for v in stored["values"]),
                errors=tuple(float(e) for e in stored["errors"]),
                wavelength=float(stored["wavelength"]),
                wavelength_error=float(stored["wavelength_error"]),
                l2=float(stored["l2"]),
                source=CalibrationSource.from_stored(stored["source"], where=where),
                status=None if stored["status"] is None else int(stored["status"]),
            )
            stored_hash = stored["content_hash"]
        except (KeyError, TypeError) as missing:
            raise ValueError(f"{where}: applied calibration record is incomplete: {missing!r}") from None
        except ValueError as invalid:
            raise ValueError(f"{where}: applied calibration record is invalid: {invalid}") from None
        if stored_hash != calibration.content_hash:
            raise ValueError(
                f"{where}: applied calibration record does not match its stored hash ({stored_hash} stored, "
                f"{calibration.content_hash} computed); the record is corrupt"
            )
        return calibration

    @classmethod
    def from_reduction_record(
        cls, record: Mapping[str, Any], *, workspace: "HidraWorkspace", where: str
    ) -> "AppliedCalibration":
        """Rebuild the record from a project file's calibration record of #1053, written before #1141 stored it.

        Only what the file states is used (user decisions, 2026-11-25). A record of no calibration file is not
        rebuilt: its distance is often stated nowhere, so the data must be reduced again.

        Args:
            record: ``HidraProjectFile.read_reduction_provenance()``.
            workspace: The workspace the file was loaded into: its sub-runs' `L2` and `mrot` logs, and geometry.
            where: The file, for the error message.

        Returns:
            Source ``file``: the stored values, errors, wavelength and status. ``workspace``: the stored values with
            errors -2 (unknown), and the stored wavelength with error -2, else the nominal one with error -1. Both
            at the distance the file states (:func:`stated_l2`) and the monochromator of its `mrot` log.

        Raises:
            ValueError: If the source is missing, not a member (e.g. ``"unknown"``) or ``"none"``, the record holds
                no usable calibration, or the file states no distance: the calibration cannot be stated.
            RuntimeError: As :func:`run_monosetting`.
        """
        source = CalibrationSource.from_stored(record.get("source"), where=where)
        if source is CalibrationSource.NONE:
            raise ValueError(
                f"{where}: reduced without a calibration file before #1141; its calibration is not rebuilt (reduce it "
                "again)"
            )
        calibration = record.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError(f"{where}: calibration record with source {source.value!r} holds no calibration")
        sub_runs = workspace.get_sub_runs()
        l2 = stated_l2(workspace, sub_runs, record=record, where=where)
        mono = run_monosetting(workspace, sub_runs)
        if source is CalibrationSource.FILE:
            try:
                shift, shift_error, wavelength, wavelength_error, status = calibration_from_dict(calibration, where)
            except RuntimeError as unusable:  # e.g. no Status
                raise ValueError(f"{where}: recorded calibration cannot be read: {unusable}") from None
            return cls.from_shift(
                shift,
                shift_error,
                wavelength=wavelength,
                wavelength_error=wavelength_error,
                l2=l2,
                source=source,
                status=int(status),
            )
        try:
            values = tuple(float(calibration[key]) for key in _PARAMETER_KEYS)
        except (KeyError, TypeError, ValueError) as unusable:
            raise ValueError(f"{where}: recorded workspace shift cannot be read: {unusable!r}") from None
        stored_wavelength = calibration.get("Lambda")
        if stored_wavelength is not None and math.isfinite(float(stored_wavelength)):
            wavelength, wavelength_error = float(stored_wavelength), UNKNOWN_ERROR
        else:
            wavelength, wavelength_error = float(mono), DEFAULT_ERROR
        return cls(
            values=values,
            errors=(UNKNOWN_ERROR,) * _N_PARAMETERS,
            wavelength=wavelength,
            wavelength_error=wavelength_error,
            l2=float(l2),
            source=source,
            status=None,
        )

    def same_calibration_as(self, other: "AppliedCalibration") -> bool:
        """Whether `other` is the same calibration: their content hashes are equal.

        Raises:
            TypeError: If `other` is not an AppliedCalibration.
        """
        if not isinstance(other, AppliedCalibration):
            raise TypeError(f"other must be an AppliedCalibration, got {type(other).__name__}")
        return self.content_hash == other.content_hash


def nominal_calibration(mono: MonoSetting, *, l2: float) -> AppliedCalibration:
    """The calibration reduction applies when no calibration file is given.

    Args:
        mono: The run's monochromator setting; its nominal wavelength is the calibration's.
        l2: The sample-to-detector distance reduction used (m).

    Returns:
        Values 0, every error -1 (not refined), the nominal wavelength with error -1, source NONE, status None.

    Raises:
        TypeError: If `mono` is not a MonoSetting.
    """
    if not isinstance(mono, MonoSetting):
        raise TypeError(f"mono must be a MonoSetting, got {type(mono).__name__}")
    return AppliedCalibration(
        values=(0.0,) * _N_PARAMETERS,
        errors=(DEFAULT_ERROR,) * _N_PARAMETERS,
        wavelength=float(mono),
        wavelength_error=DEFAULT_ERROR,
        l2=float(l2),
        source=CalibrationSource.NONE,
        status=None,
    )


def reduction_l2(workspace: "HidraWorkspace", sub_runs: Iterable[int]) -> float:
    """The one sample-to-detector distance a reduction of `sub_runs` uses (#1141): their `L2` log, else the
    instrument's arm length, as the reduction engine takes it.

    Args:
        workspace: The HidraWorkspace.
        sub_runs: The sub-runs reduced.

    Returns:
        The distance (m).

    Raises:
        RuntimeError: If the sub-runs' L2 differ (a reduction applies one calibration, at one distance), or a
            sub-run has no L2 log and the workspace no instrument geometry.
    """
    values = {workspace.get_l2(sub_run) for sub_run in sub_runs}
    if None in values:
        values.discard(None)
        geometry = workspace.get_instrument_setup()
        if geometry is None:
            raise RuntimeError(
                f"workspace {workspace.name!r} has no L2 log and no instrument geometry: its distance is not known"
            )
        values.add(geometry.arm_length)
    distances = sorted(float(value) for value in values)
    if len(distances) != 1:
        raise RuntimeError(
            f"workspace {workspace.name!r}: the sample-to-detector distance (L2) differs between the sub-runs being "
            f"reduced: {distances}; one reduction applies one calibration at one distance"
        )
    return distances[0]


def run_monosetting(workspace: "HidraWorkspace", sub_runs: Iterable[int]) -> MonoSetting:
    """The monochromator of `sub_runs`, from their `mrot` log (#1141): never from a wavelength, which may be the
    calibrated one and whose nominal value no neutron has.

    Args:
        workspace: The HidraWorkspace.
        sub_runs: The sub-runs reduced.

    Returns:
        The monochromator setting.

    Raises:
        RuntimeError: If there is no `mrot` log, or the sub-runs' rotations give different monochromators.
        ValueError: If a rotation matches no monochromator (as ``MonoSetting.getFromRotation``).
    """
    if not workspace.has_sample_log("mrot"):
        raise RuntimeError(f"workspace {workspace.name!r} has no 'mrot' log: its monochromator cannot be determined")
    settings = {
        MonoSetting.getFromRotation(float(workspace.get_sample_log_value("mrot", sub_run))) for sub_run in sub_runs
    }
    if len(settings) != 1:
        raise RuntimeError(
            f"workspace {workspace.name!r}: the sub-runs being reduced use different monochromators "
            f"{sorted(str(setting) for setting in settings)}; one run holds one monochromator"
        )
    return settings.pop()


def stated_l2(workspace: "HidraWorkspace", sub_runs: Iterable[int], *, record: Mapping[str, Any], where: str) -> float:
    """The distance a reduction used, from what a project file states: the sub-runs' `L2` log, else the file's
    instrument geometry, else the geometry in its calibration record (the run's, else the calibration's). Never a
    default (user decision, 2026-11-25).

    Args:
        workspace: The workspace the file was loaded into.
        sub_runs: Its sub-runs.
        record: The file's calibration record (``read_reduction_provenance()``).
        where: The file, for the error message.

    Returns:
        The distance (m).

    Raises:
        ValueError: If the file states no distance.
        RuntimeError: As :func:`reduction_l2`, when the distances it states differ.
    """
    logged = {workspace.get_l2(sub_run) for sub_run in sub_runs}
    if None not in logged or workspace.get_instrument_setup() is not None:
        return reduction_l2(workspace, sub_runs)
    geometry = record.get("run_geometry") or (record.get("calibration") or {}).get("geometry")
    if not isinstance(geometry, dict) or "arm_length" not in geometry:
        raise ValueError(
            f"{where}: the distance its reduction used is not stated (no L2 log, no instrument geometry, no geometry "
            "in its calibration record)"
        )
    distances = sorted({float(value) for value in logged if value is not None} | {float(geometry["arm_length"])})
    if len(distances) != 1:
        raise RuntimeError(f"{where}: the distances it states differ: {distances}")
    return distances[0]
