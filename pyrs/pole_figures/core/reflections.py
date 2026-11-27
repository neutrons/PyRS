"""
Allowed reflection families of a crystal: the hkl search, the neutron structure factor and the Bragg angle.

A family is listed when it is not systematically absent in the space group and its neutron structure factor (coherent
scattering lengths, all sites in the cell) is above 1e-6 of the strongest possible. Symmetry-equivalent hkl are
grouped into one family with its multiplicity (Friedel pairs included). ``phase.Phase.reflections`` is the usual way
in.

Example:
    >>> hkl_label((3, 1, 1)), hkl_label((1, -1, 0))
    ('311', '1 -1 0')
    >>> round(bragg_two_theta(gemmi.UnitCell(3.6, 3.6, 3.6, 90, 90, 90), (1, 1, 1), 1.54), 3)
    43.489
"""

from collections.abc import Sequence
from dataclasses import dataclass

import gemmi
import numpy as np

#: the hkl search runs over -n..n in h, k and l with n = max(a, b, c) / d_min + 1. Above this n the search (n^3
#: Python iterations) takes minutes, so it is refused: lower two_theta_max or check the cell.
MAX_HKL_INDEX = 60
#: (coherent scattering length b (fm), occupancy, fractional positions of the site in the cell) of one site
ScatteringSite = tuple[complex, float, list[list[float]]]


@dataclass(frozen=True, kw_only=True)
class Reflection:
    """
    One family of symmetry-equivalent reflections.

    Attributes:
        hkl: Representative indices (fewest negative indices, then the largest h, k, l).
        d_A: d-spacing (Å).
        two_theta: 2theta at the wavelength used (deg), NaN beyond 180.
        multiplicity: Number of equivalent hkl, Friedel pairs included.
        f2: Neutron structure factor squared, ``|F|^2`` per cell (fm^2).
    """

    hkl: tuple[int, int, int]
    d_A: float
    two_theta: float
    multiplicity: int
    f2: float  # |F|^2 per cell, fm^2

    @property
    def label(self) -> str:
        """The hkl label (see :func:`hkl_label`)."""
        return hkl_label(self.hkl)


def hkl_label(hkl: tuple[int, int, int]) -> str:
    """
    Miller indices as a label.

    Args:
        hkl: Miller indices.

    Returns:
        Compact ``"311"`` when every index is a single non-negative digit, else ``"1 -1 0"``.
    """
    if all(0 <= i <= 9 for i in hkl):
        return "".join(str(i) for i in hkl)
    return " ".join(str(i) for i in hkl)


def bragg_two_theta(cell: gemmi.UnitCell, hkl: tuple[int, int, int], wavelength_A: float) -> float:
    """
    Bragg angle 2theta of a reflection.

    Args:
        cell: The unit cell.
        hkl: Miller indices.
        wavelength_A: Neutron wavelength (Å).

    Returns:
        2theta (deg), NaN when the reflection is beyond 2theta = 180.
    """
    s = wavelength_A / (2 * float(cell.calculate_d(list(hkl))))
    return float(2 * np.degrees(np.arcsin(s))) if s < 1 else float("nan")


def hkl_search_limit(*, name: str, cell_lengths: Sequence[float], wavelength_A: float,
                     two_theta_max: float) -> tuple[float, int]:  # fmt: skip
    """
    Smallest d-spacing to list and the largest hkl index the search needs.

    Args:
        name: Phase name, for the messages.
        cell_lengths: a, b, c (Å).
        wavelength_A: Neutron wavelength (Å).
        two_theta_max: Largest 2theta to list (deg).

    Returns:
        d_min (Å) and n: the search runs over -n..n in h, k and l.

    Raises:
        ValueError: If the wavelength is not positive, or n is above ``MAX_HKL_INDEX``.
    """
    if not (np.isfinite(wavelength_A) and wavelength_A > 0):
        raise ValueError(f"{name}: the wavelength must be positive (A), got {wavelength_A!r}")
    dmin = wavelength_A / (2 * np.sin(np.radians(two_theta_max / 2)))
    n = int(np.ceil(max(cell_lengths) / dmin)) + 1
    if n > MAX_HKL_INDEX:
        raise ValueError(f"{name}: listing reflections down to d {dmin:.3g} A in a cell of "
                         f"{max(cell_lengths):g} A needs hkl up to {n}, more than {MAX_HKL_INDEX}. Lower "
                         "two_theta_max or check the cell.")  # fmt: skip
    return float(dmin), n


def reflection_families(*, name: str, cell: gemmi.UnitCell, ops: gemmi.GroupOps, sites: Sequence[ScatteringSite],
                        dmin: float, n: int, wavelength_A: float) -> list[Reflection]:  # fmt: skip
    """
    Allowed reflection families with d >= dmin, in order of decreasing d.

    Args:
        name: Phase name, for the messages.
        cell: The unit cell.
        ops: The space group's operations.
        sites: Every site with its scattering length, occupancy and positions in the cell.
        dmin: Smallest d-spacing to list (Å).
        n: Largest hkl index searched (from :func:`hkl_search_limit`).
        wavelength_A: Neutron wavelength (Å), for each family's 2theta.

    Returns:
        The reflection families.

    Raises:
        ValueError: If no site scatters.
    """
    rots = list(ops.sym_ops)
    b_sum = sum(abs(b) * occ * len(p) for b, occ, p in sites)
    if not b_sum > 0:
        raise ValueError(f"{name}: the sites scatter nothing (all coherent scattering lengths are 0)")
    seen: set[tuple[int, int, int]] = set()
    out: list[Reflection] = []
    for h in range(-n, n + 1):
        for k in range(-n, n + 1):
            for l in range(-n, n + 1):  # noqa: E741
                hkl = (h, k, l)
                if hkl == (0, 0, 0) or hkl in seen:
                    continue
                d = float(cell.calculate_d([h, k, l]))
                if d < dmin - 1e-9:
                    continue
                eq = {tuple(op.apply_to_hkl([h, k, l])) for op in rots}
                eq |= {(-a, -b, -c) for a, b, c in eq}
                seen |= eq  # type: ignore[arg-type]
                if ops.is_systematically_absent([h, k, l]):
                    continue
                hv = np.array(hkl, float)
                F = sum(b * occ * np.exp(2j * np.pi * (np.asarray(ps) @ hv)).sum() for b, occ, ps in sites)
                f2 = float(abs(F) ** 2)
                if f2 < 1e-6 * b_sum**2:
                    continue
                v = min(eq, key=lambda v: (sum(i < 0 for i in v), -v[0], -v[1], -v[2]))
                rep = (v[0], v[1], v[2])
                out.append(Reflection(hkl=rep, d_A=d, two_theta=bragg_two_theta(cell, rep, wavelength_A),
                                      multiplicity=len(eq), f2=f2))  # fmt: skip
    out.sort(key=lambda r: (-r.d_A, r.hkl))
    return out
