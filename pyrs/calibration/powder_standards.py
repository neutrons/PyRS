"""
Reference powders used to calibrate the HB2B detector.

Each calibration sub-run measures one cubic powder standard, selected by the sample stage
position ``sy``. This module holds the standards' lattice constants and allowed reflections so
that every calibration uses the same reference values. Typical use:

1. Find the powder measured in a sub-run with :func:`standard_for_sy`.
2. Get its expected d-spacings with :meth:`PowderStandard.d_spacings`, or its expected peak
   positions for a wavelength with :meth:`PowderStandard.two_theta`.

Lattice constants are the 2020 set (commit d6cd5e04, ``scripts/pyrs_calibration.py``), which
appear to be standard room-temperature reference values (Ni 3.5238, alpha-Fe 2.8664, Mo 3.1472 A).
They differ from the values ``FitCalibration`` uses since 2023 (3.526314 / 2.865579 / 3.147664,
fitted from HB2B data), whose Fe/Ni ratio is ~0.1 % off the ratio HB2B_3510 shows; see
docs/ground_truths.md. The FCC 400 reflection, missing in ``FitCalibration``, is included.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["PowderStandard", "REFLECTIONS", "HB2B_STANDARDS", "standard_for_sy"]

# h^2 + k^2 + l^2 of the allowed reflections (FCC: h, k, l all odd or all even; BCC: h + k + l even)
REFLECTIONS = {
    "FCC": (3, 4, 8, 11, 12, 16, 19),
    "BCC": (2, 4, 6, 8, 10, 12),
}


@dataclass(frozen=True)
class PowderStandard:
    """A cubic powder standard mounted at a fixed sample position.

    Attributes:
        name: Element name, e.g. ``"Ni"``.
        structure: ``"FCC"`` or ``"BCC"``.
        lattice_a: Cubic lattice constant in Angstrom.
        sy: Sample stage ``sy`` position (mm) at which this powder is in the beam.

    Example:
        >>> ni = PowderStandard("Ni", "FCC", 3.523799438, 62.0)
        >>> round(float(ni.d_spacings()[0]), 4)  # (111)
        2.0345
    """

    name: str
    structure: str
    lattice_a: float
    sy: float

    def __post_init__(self) -> None:
        if self.structure not in REFLECTIONS:
            raise ValueError(f"structure must be one of {sorted(REFLECTIONS)}, got {self.structure!r}")
        if not self.lattice_a > 0:
            raise ValueError(f"lattice_a must be positive, got {self.lattice_a}")

    @property
    def hkl_squared(self) -> np.ndarray:
        """``h^2 + k^2 + l^2`` of the allowed reflections, in order of decreasing d-spacing."""
        return np.array(REFLECTIONS[self.structure])

    def d_spacings(self) -> np.ndarray:
        """d-spacings (Angstrom) of the allowed reflections, largest first."""
        return self.lattice_a / np.sqrt(self.hkl_squared)

    def two_theta(self, wavelength: float) -> np.ndarray:
        """Expected peak positions 2theta (degrees) at ``wavelength``.

        Args:
            wavelength: Neutron wavelength in Angstrom.

        Returns:
            2theta of every reflection reachable at this wavelength (``lambda <= 2d``), in the
            same order as :meth:`d_spacings`.

        Raises:
            ValueError: If ``wavelength`` is not positive.
        """
        if not wavelength > 0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        sin_theta = wavelength / (2 * self.d_spacings())
        return 2 * np.rad2deg(np.arcsin(sin_theta[sin_theta <= 1]))


HB2B_STANDARDS = (
    PowderStandard("Ni", "FCC", 3.523799438, 62.0),
    PowderStandard("Fe", "BCC", 2.8663982, 12.0),
    PowderStandard("Mo", "BCC", 3.14719963, -13.0),
)


def standard_for_sy(
    sy: float,
    standards: Sequence[PowderStandard] = HB2B_STANDARDS,
    tolerance: float = 2.0,
) -> Optional[PowderStandard]:
    """Powder standard in the beam at stage position ``sy``.

    Args:
        sy: Sample stage ``sy`` position (mm) of a sub-run.
        standards: Candidate standards (default: the HB2B calibration powders).
        tolerance: Maximum ``|sy - standard.sy|`` (mm), exclusive, as in ``FitCalibration``.

    Returns:
        The closest matching standard, or None if no standard is within ``tolerance``.
    """
    distances = [abs(sy - standard.sy) for standard in standards]
    if not distances or min(distances) >= tolerance:
        return None
    return standards[int(np.argmin(distances))]
