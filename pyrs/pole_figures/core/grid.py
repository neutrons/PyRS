"""
The chi/phi grid of a predicted pole figure and the solid-angle weight of each point.

The reference script's grid is chi 0 to below 90 in 10 deg rings, phi 0 to below 360 in 5 deg steps, plus one point
at chi 90. With equal spacing the high-chi rings take coarser phi steps, so the points stay about as far apart on the
sphere as at chi 0.

Example:
    >>> Grid().describe()
    '10 x 5 deg'
    >>> len(Grid().points()[0])
    649
"""

from dataclasses import KW_ONLY, dataclass

import numpy as np

from .types import FloatArray

#: whole-degree phi steps that divide 360, for the equal-spacing grid (motor positions stay round numbers)
PHI_DIVISORS = tuple(d for d in range(1, 361) if 360 % d == 0)


@dataclass(frozen=True)
class Grid:
    """
    chi/phi grid: chi 0 to below 90 in ``chi_step``, phi 0 to below 360 in ``phi_step``, plus one point at chi 90.

    ``equal_spacing`` (off by default, the reference grid): a chi ring is a circle on the sphere whose size goes as
    cos(chi), so the same phi step crowds the points as chi grows. With equal spacing each ring takes the whole-degree
    divisor of 360 nearest to ``phi_step / cos(chi)`` in log ratio (never finer than ``phi_step``), which keeps the
    spacing on the sphere close to the chi 0 spacing. 10 x 5 deg: 454 points instead of 649.

    Attributes:
        chi_step: Step between chi rings (deg, default: 10).
        phi_step: Step along each ring, and on every ring unless equal_spacing (deg, default: 5).
        equal_spacing: Coarser phi steps on the high-chi rings (default: False).

    Example:
        >>> Grid(equal_spacing=True).ring_phi_step(60.0)
        10.0
    """

    chi_step: float = 10.0
    phi_step: float = 5.0
    _: KW_ONLY
    equal_spacing: bool = False

    def rings(self) -> FloatArray:
        """
        The chi of each ring below 90.

        Returns:
            Ring chi values (deg), from 0 in ``chi_step``.
        """
        return np.arange(0, 90, self.chi_step).astype(float)

    def ring_phi_step(self, chi: float) -> float:
        """
        The phi step on the ring at ``chi``.

        Args:
            chi: Ring chi (deg).

        Returns:
            The phi step (deg): ``phi_step``, or with equal spacing a coarser whole-degree divisor of 360.
        """
        ideal = self.phi_step / max(float(np.cos(np.radians(chi))), 1e-9)
        if not self.equal_spacing or ideal <= self.phi_step * (1 + 1e-9):
            return float(self.phi_step)
        coarser = [d for d in PHI_DIVISORS if d >= self.phi_step]
        if not coarser:
            return float(self.phi_step)
        return float(min(coarser, key=lambda d: abs(np.log(d / ideal))))

    def points(self) -> tuple[FloatArray, FloatArray]:
        """
        Cradle chi and phi of every grid point, in order (ring by ring, phi increasing, chi 90 last).

        Returns:
            Chi and phi (deg).
        """
        chi_parts, phi_parts = [], []
        for c in self.rings():
            phis = np.arange(0, 360, self.ring_phi_step(float(c)))
            chi_parts.append(np.full(len(phis), c))
            phi_parts.append(phis)
        chi = np.r_[np.concatenate(chi_parts), 90.0]
        phi = np.r_[np.concatenate(phi_parts), 0.0]
        return chi.astype(float), phi.astype(float)

    def weights(self, chi: FloatArray) -> FloatArray:
        """
        Solid-angle weight of each grid point (see :func:`solid_angle_weights`).

        Args:
            chi: Chi of each grid point (deg).

        Returns:
            Weights in units of a chi_step x phi_step cell.
        """
        c = np.asarray(chi, float)
        steps = np.array([self.ring_phi_step(float(x)) if x < 90 - 1e-6 else self.phi_step for x in c])
        return solid_angle_weights(c, chi_step=self.chi_step, phi_step=self.phi_step, ring_steps=steps)

    def describe(self) -> str:
        """
        Short text for captions.

        Returns:
            E.g. '10 x 5 deg', or with equal spacing the range of phi steps from chi 0 to the last ring.
        """
        if not self.equal_spacing:
            return f"{self.chi_step:g} x {self.phi_step:g} deg"
        last = float(self.rings().max())
        return (f"{self.chi_step:g} deg chi, equal spacing (phi {self.phi_step:g} deg at chi 0 to "
                f"{self.ring_phi_step(last):g} deg at chi {last:g})")  # fmt: skip


#: the reference script's grid, 10 x 5 deg (Grid is frozen, so one shared default is safe)
DEFAULT_GRID = Grid()


def solid_angle_weights(chi: FloatArray, *, chi_step: float, phi_step: float,
                        ring_steps: FloatArray | None = None) -> FloatArray:  # fmt: skip
    """
    Solid-angle weight of each grid point: cos chi, half at chi 0, and the chi 90 cap shared by its points.

    Args:
        chi: Chi of each point (deg).
        chi_step: Ring step (deg). Points within chi_step / 2 of chi 90 share the polar cap.
        phi_step: Phi step (deg).
        ring_steps: The phi step of each point's ring, which scales a point whose ring is stepped more coarsely than
            ``phi_step``. None is the uniform grid (default).

    Returns:
        Weights in units of a chi_step x phi_step cell.
    """
    w = np.cos(np.radians(chi)) * np.where(np.abs(chi) < 0.05, 0.5, 1.0)
    if ring_steps is not None:
        w = w * np.asarray(ring_steps, float) / phi_step
    cap = chi > 90 - chi_step / 2 - 1e-6
    w[cap] = (
        2 * np.pi * (1 - np.cos(np.radians(chi_step / 2))) / (np.radians(phi_step) * np.radians(chi_step))
        / max(int(cap.sum()), 1)
    )  # fmt: skip
    return np.asarray(w, float)
