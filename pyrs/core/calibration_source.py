"""
Where an applied calibration came from (#1141): the one definition every writer and reader imports.

A leaf module, so both `pyrs.core.instrument_geometry` and `pyrs.core.calibration` can import it at the top:
`pyrs.core.calibration` itself imports `instrument_geometry`.

Example:
    >>> CalibrationSource.from_stored("file", where="run.h5")
    <CalibrationSource.FILE: 'file'>
"""

from enum import Enum

__all__ = ["CalibrationSource"]


class CalibrationSource(str, Enum):
    """Where an applied calibration came from. There is no "unknown": a calibration that cannot be stated is
    rejected, never recorded (#1141)."""

    NONE = "none"  # nominal calibration: no calibration file applied
    FILE = "file"  # a calibration file
    WORKSPACE = "workspace"  # the workspace's own shift (FitCalibration in the session, NXstress read-back)

    @classmethod
    def from_stored(cls, value: str, *, where: str) -> "CalibrationSource":
        """The member a file stores.

        Args:
            value: The stored string.
            where: The file (or other origin) it was read from, for the error message.

        Returns:
            The member.

        Raises:
            ValueError: If `value` is not a member's value: it is never passed through as a new state.
        """
        try:
            return cls(value)
        except ValueError:
            members = sorted(member.value for member in cls)
            raise ValueError(
                f"{where}: calibration source {value!r} is not one of {members}; the calibration cannot be stated"
            ) from None
