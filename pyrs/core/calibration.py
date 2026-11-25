"""
The calibration a reduction applied, as one immutable record with an identity (#1141).

A calibration is the detector shift, rotations and 2theta_0 (`DENEXDetectorShift`), their errors, the
wavelength and its error, and the sample-to-detector distance L2 reduction used. Two calibrations are the same
**if and only if** their content hashes are equal, and the record is the one place that decides it: callers
never compare individual values (#1141, user decision, 2026-11-24).

Error codes are the calibration files': -1 is the default (not refined, or held), -2 an unknown error (refined,
but the error could not be determined or was lost).

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
from typing import Optional, Tuple

from pyrs.core.calibration_source import CalibrationSource
from pyrs.core.detector_model import from_shift, to_shift
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.monosetting import MonoSetting

__all__ = ["CalibrationSource", "AppliedCalibration", "nominal_calibration", "DEFAULT_ERROR", "UNKNOWN_ERROR"]

DEFAULT_ERROR = -1.0  # not refined (or held): the nominal calibration's errors
UNKNOWN_ERROR = -2.0  # refined, but the error could not be determined or was lost
_N_PARAMETERS = 7  # Shift_x, Shift_y, Shift_z, Rot_x, Rot_y, Rot_z, TTH_0


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
