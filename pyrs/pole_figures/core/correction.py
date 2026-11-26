"""
The path-length correction step shared by predictions and measured runs: the gauge Monte Carlo at every orientation.

The gauge points are drawn once for the mount and beams (``gauge.gauge_points``). F is computed at every cradle chi,
phi and at chi 0, phi 0, the orientation every correction is relative to. The correction is F(0, 0) / F, inf where F
is 0 (the gauge misses the sample). Points whose correction is above the cutoff are ignored by the callers.

Typical workflow:

1. Set the beam and slit sizes with :class:`Beams`.
2. Call :func:`gauge_corrections` with the mount, the orientations, the sample mesh and mu.
3. Use the :class:`GaugeCorrections` fields by name, or :meth:`GaugeCorrections.take` to keep some orientations.
4. Split the correction into its gauge-volume and attenuation factors with :func:`correction_parts`.

Example:
    >>> Beams(incident_width=2.0, incident_height=2.0, diffracted_width=2.0)
    Beams(incident_width=2.0, incident_height=2.0, diffracted_width=2.0)
    >>> g = gauge_corrections(m, beams=Beams(), chi=chi, phi=phi, mesh=mesh, center=(0.0, 0.0, 0.0), mu=0.1,
    ...                       n_points=DEFAULT_MC_POINTS, progress=no_count_progress, workers="auto")  # doctest: +SKIP
    >>> g.correction[0]  # doctest: +SKIP
    1.0
"""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .gauge import gauge_points, intensity_factor
from .geometry import Mount
from .parallel import Workers, intensity_factors
from .stl_mesh import SampleMesh
from .types import CountProgress, FloatArray

DEFAULT_CUTOFF = 3.0  # points with a correction above this are ignored, as the reference script
#: a factor of the correction: the whole correction, its gauge-volume part or its attenuation part
CorrectionPart = Literal["total", "volume", "attenuation"]
#: every correction part, in the order total, volume, attenuation
PARTS: tuple[CorrectionPart, ...] = typing.get_args(CorrectionPart)
#: chi and phi within this of 0 (deg) are the reference orientation of ``correction_parts``
ZERO_TOL_DEG = 1e-9


@dataclass(frozen=True, kw_only=True)
class Beams:
    """
    Beam and slit sizes (mm).

    Attributes:
        incident_width: Incident beam width in the scattering plane (default: 4). HB-2B logs it as ISR:X:Gap.
        incident_height: Incident beam height, vertical (default: 4). HB-2B logs it as ISR:Y:Gap.
        diffracted_width: Receiving slit width (default: 3).

    Raises:
        ValueError: If a size is not positive and finite (a NaN slit log would make every point ignored).
    """

    incident_width: float = 4.0
    incident_height: float = 4.0
    diffracted_width: float = 3.0

    def __post_init__(self) -> None:
        """Refuse sizes that are not positive and finite (a NaN slit log would make every point ignored)."""
        for name in ("incident_width", "incident_height", "diffracted_width"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name.replace('_', ' ')} must be a positive size in mm, not {v}")


def check_reference(f_ref: float, center: npt.ArrayLike) -> None:
    """
    Check that the gauge sees sample at chi 0, phi 0, the orientation every correction is relative to.

    F(0, 0) of 0 (or NaN) would make every correction 0, which looks like a perfect prediction.

    Args:
        f_ref: F at chi 0, phi 0 (mm3).
        center: Gauge center in the sample frame (mm), for the message.

    Raises:
        ValueError: If f_ref is not positive and finite.
    """
    if not (np.isfinite(f_ref) and f_ref > 0):
        c = ", ".join(f"{float(x):g}" for x in np.asarray(center, float).ravel())
        raise ValueError(f"the gauge volume at ({c}) mm holds no sample at chi 0, phi 0 (F(0,0) = {f_ref}). "
                         "Move the gauge center inside the sample.")  # fmt: skip


@dataclass(frozen=True, kw_only=True)
class GaugeCorrections:
    """
    Gauge Monte Carlo results at each orientation (one entry per orientation in every array).

    Attributes:
        F: Attenuation-weighted sample volume in the gauge (mm3).
        F_ref: F at chi 0, phi 0 (mm3).
        volume_in_sample: Sample volume in the gauge (mm3).
        mean_path: Mean path in + out (mm), NaN where the gauge misses the sample.
        vy_offset: Intensity-weighted vy centroid of the gauge (mm), NaN where the gauge misses the sample.
        vy_rms: Intensity-weighted vy spread of the gauge (mm), NaN where the gauge misses the sample.
        correction: F_ref / F, inf where F is 0.
    """

    F: FloatArray
    F_ref: float
    volume_in_sample: FloatArray
    mean_path: FloatArray
    vy_offset: FloatArray
    vy_rms: FloatArray
    correction: FloatArray

    def take(self, index: npt.ArrayLike | slice) -> "GaugeCorrections":
        """
        The results at some orientations.

        Args:
            index: Indices, a boolean mask or a slice of the orientations.

        Returns:
            The selected results, with the same ``F_ref``.
        """
        i = index if isinstance(index, slice) else np.asarray(index)
        return GaugeCorrections(F=self.F[i], F_ref=self.F_ref, volume_in_sample=self.volume_in_sample[i],
                                mean_path=self.mean_path[i], vy_offset=self.vy_offset[i], vy_rms=self.vy_rms[i],
                                correction=self.correction[i])  # fmt: skip


def concatenate(parts: Sequence[GaugeCorrections]) -> GaugeCorrections:
    """
    The results of several pole figures one after another (for a combined pole figure).

    Each part's correction stays relative to its own F(0, 0). ``F_ref`` of the result is the first part's, kept as a
    record.

    Args:
        parts: One or more results.

    Returns:
        The results joined in order.

    Raises:
        ValueError: If there are no parts.
    """
    if not parts:
        raise ValueError(f"nothing to concatenate: give at least one result, got {list(parts)!r}")

    def cat(name: str) -> FloatArray:
        return np.concatenate([np.asarray(getattr(g, name), float) for g in parts])

    return GaugeCorrections(F=cat("F"), F_ref=parts[0].F_ref, volume_in_sample=cat("volume_in_sample"),
                            mean_path=cat("mean_path"), vy_offset=cat("vy_offset"), vy_rms=cat("vy_rms"),
                            correction=cat("correction"))  # fmt: skip


def gauge_corrections(
    m: Mount,
    *,
    beams: Beams,
    chi: npt.ArrayLike,
    phi: npt.ArrayLike,
    mesh: SampleMesh,
    center: npt.ArrayLike,
    mu: float,
    n_points: int,
    progress: CountProgress,
    workers: Workers,
) -> GaugeCorrections:
    """
    Gauge Monte Carlo at each cradle chi, phi, and the correction F(0, 0) / F.

    Args:
        m: Beams and rotation conventions for the plane's omega and 2theta.
        beams: Beam and slit sizes (mm).
        chi: Cradle chi per orientation (deg).
        phi: Phi per orientation (deg), same length as chi.
        mesh: Sample in its own frame (mm).
        center: Gauge center in the sample frame (mm).
        mu: Linear attenuation coefficient (1/mm).
        n_points: Monte Carlo gauge points (``gauge.DEFAULT_MC_POINTS`` for runs, ``gauge.PREDICTION_MC_POINTS`` for
            predictions).
        progress: ``progress(n_done)`` as orientations finish (``types.no_count_progress`` for none).
        workers: Number of processes, or "auto" to choose from the job size (``parallel.n_workers``).

    Returns:
        The results at every orientation, in the order of chi and phi.

    Raises:
        ValueError: If the gauge holds no sample at chi 0, phi 0, or workers is neither "auto" nor a positive whole
            number.
    """
    pts, vol = gauge_points(m, incident_width=beams.incident_width, incident_height=beams.incident_height,
                            diffracted_width=beams.diffracted_width, n_points=n_points)  # fmt: skip
    f_ref = intensity_factor(m, chi=0.0, phi=0.0, points=pts, volume=vol, mesh=mesh, center=center, mu=mu).F
    check_reference(f_ref, center)
    rows = intensity_factors(m, chi=chi, phi=phi, points=pts, volume=vol, mesh=mesh, center=center, mu=mu,
                             progress=progress, workers=workers)  # fmt: skip

    def column(name: str) -> FloatArray:
        return np.array([getattr(r, name) for r in rows], float)

    F = column("F")
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(F > 0, f_ref / np.where(F > 0, F, 1), np.inf)
    return GaugeCorrections(F=F, F_ref=float(f_ref), volume_in_sample=column("volume_in_sample"),
                            mean_path=column("mean_path"), vy_offset=column("vy_offset"), vy_rms=column("vy_rms"),
                            correction=corr)  # fmt: skip


def correction_parts(
    gauge: GaugeCorrections, *, chi: npt.ArrayLike, phi: npt.ArrayLike
) -> dict[CorrectionPart, FloatArray]:
    """
    The correction and its two factors (correction = volume x attenuation, exactly).

    - volume: V(0,0) / V, the sample volume in the gauge relative to chi 0, phi 0
    - attenuation: A(0,0) / A, with A = F / V the mean of exp(-mu path) over the sample part of the gauge

    Args:
        gauge: Gauge Monte Carlo results, one entry per orientation.
        chi: Cradle chi of each orientation (deg). The reference is the first orientation at chi 0, phi 0.
        phi: Phi of each orientation (deg).

    Returns:
        Arrays "total", "volume" and "attenuation", one value per orientation. The volume part is inf and the
        attenuation part NaN where the gauge misses the sample (V = 0).

    Raises:
        ValueError: If no orientation is at chi 0, phi 0.

    Example:
        >>> parts = correction_parts(plane.gauge, chi=plane.chi, phi=plane.phi)  # doctest: +SKIP
        >>> np.allclose(parts["volume"] * parts["attenuation"], parts["total"])  # doctest: +SKIP
        True
    """
    chi, phi = np.asarray(chi, float), np.asarray(phi, float)
    V, F = gauge.volume_in_sample, gauge.F
    at_zero = np.flatnonzero((np.abs(chi) < ZERO_TOL_DEG) & (np.abs(phi) < ZERO_TOL_DEG))
    if at_zero.size == 0:
        raise ValueError(f"no grid point at chi 0, phi 0 among the {len(chi)} orientations")
    i0 = int(at_zero[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(V > 0, F / np.where(V > 0, V, 1), np.nan)
        vol = np.where(V > 0, V[i0] / np.where(V > 0, V, 1), np.inf)
        att = A[i0] / A
    return {"total": gauge.correction, "volume": vol, "attenuation": att}
