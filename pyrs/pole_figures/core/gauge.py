"""
Gauge volume Monte Carlo and the attenuation-weighted intensity factor F.

Ported from the reference script (``gauge_points``, ``set_slit_offset``, ``intensity_factor``, ``beam_value``)
with the same numerics.

The gauge is the incident beam (width x height) crossed with the diffracted-slit band (width), sampled by uniform
Monte Carlo points in the cradle frame. For each point inside the sample, the path in (along -ki) and out (along kf)
is traced through the mesh. F = vol * mean(inside * exp(-mu (L_in + L_out))) holds both the attenuation and the gauge
volume inside the sample. The correction is F(0, 0) / F.

Typical workflow:

1. Build a :class:`~.geometry.Mount` for the plane's omega and 2theta.
2. Optionally shift the diffracted band with :func:`set_slit_offset`.
3. Draw the gauge points once with :func:`gauge_points`.
4. Call :func:`intensity_factor` at each cradle chi, phi (or ``parallel.intensity_factors`` for many).

:func:`path_at_gauge` gives the beam path along the centerlines from the gauge center alone (the mount check).

Example:
    >>> mesh = SampleMesh.from_stl("sample.stl")  # doctest: +SKIP
    >>> m = Mount(MountConventions(), omega=45.0, tth=90.0)
    >>> pts, vol = gauge_points(m, incident_width=2.0, incident_height=2.0, diffracted_width=2.0, n_points=40_000)
    >>> f = intensity_factor(m, chi=30.0, phi=90.0, points=pts, volume=vol, mesh=mesh,
    ...                      center=(0.0, 0.0, 0.0), mu=0.1)  # doctest: +SKIP
    >>> ref = intensity_factor(m, chi=0.0, phi=0.0, points=pts, volume=vol, mesh=mesh,
    ...                        center=(0.0, 0.0, 0.0), mu=0.1)  # doctest: +SKIP
    >>> correction = ref.F / f.F  # doctest: +SKIP
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .geometry import Mount, MountConventions
from .stl_mesh import SampleMesh
from .types import FloatArray

MC_SEED = 3
DEFAULT_MC_POINTS = 40_000  # Monte Carlo gauge points per orientation for a measured run, as the reference script
#: the prediction uses half of them, as the reference script does
PREDICTION_MC_POINTS = DEFAULT_MC_POINTS // 2
#: a beam size given as this prefix and a log name is the median of that run log (``beam_value``)
LOG_PREFIX: Final = "log:"


def gauge_points(
    m: Mount,
    *,
    incident_width: float,
    incident_height: float,
    diffracted_width: float,
    n_points: int,
    seed: int = MC_SEED,
) -> tuple[FloatArray, float]:
    """
    Uniform points in the incident beam x diffracted band (cradle frame, mm, about the gauge center).

    The gauge is the parallelepiped bounded by the incident beam (width across ki, height along e_x) and the
    diffracted band (width across kf). The points are uniform in it.

    Args:
        m: Beams for this omega and 2theta. ``m.band_offset`` shifts the diffracted band (see ``set_slit_offset``).
        incident_width: Incident beam width in the scattering plane (mm).
        incident_height: Incident beam height along e_x (mm).
        diffracted_width: Diffracted (receiving) slit width (mm).
        n_points: Number of points.
        seed: Random seed (default: ``MC_SEED`` = 3, as the reference script).

    Returns:
        The points, shape (n_points, 3), in the cradle frame relative to the gauge center (mm), and the gauge volume
        (mm3).
    """
    ex = np.array([1.0, 0, 0])
    ni = np.cross(ex, m.ki)
    nf = np.cross(ex, m.kf)
    A = np.vstack([ni / np.linalg.norm(ni), nf / np.linalg.norm(nf), ex])
    wi, hi, wd = incident_width, incident_height, diffracted_width
    rng = np.random.default_rng(seed)
    s = rng.uniform(-0.5, 0.5, (n_points, 3)) * [wi, wd, hi]
    if m.band_offset:
        s[:, 1] += m.band_offset  # off-center peak: the band passes through the slit center, not the gauge center
    return np.linalg.solve(A, s.T).T, float(wi * wd * hi / abs(np.linalg.det(A)))


def set_slit_offset(m: Mount, *, conv: MountConventions, omega: float, arm_tth: float, slit_mm: float) -> Mount:
    """
    Shift the diffracted band for a peak away from the arm angle.

    A peak away from the arm angle is seen through the receiving slit along its own diffracted direction, so its band
    is centered on the line through the slit center (``slit_mm`` along the arm direction from the rotation point).

    Args:
        m: Mount of the peak (its kf is the peak's diffracted direction). Changed in place.
        conv: Mount conventions (beam side).
        omega: Omega (deg).
        arm_tth: Detector arm 2theta (deg).
        slit_mm: Distance from the rotation point to the receiving slit (mm). 0 leaves ``m`` unchanged.

    Returns:
        ``m``, with ``band_offset`` (mm, along e_x x kf) set by ``Mount.set_band_offset``.
    """
    if not slit_mm:
        return m
    w, ta = np.radians(omega), np.radians(arm_tth)
    k_arm = np.array([0.0, conv.sb * np.cos(ta - w), np.sin(ta - w)])
    nf = np.cross([1.0, 0, 0], m.kf)
    m.set_band_offset(float(np.dot(nf / np.linalg.norm(nf), slit_mm * k_arm)))
    return m


@dataclass(frozen=True, kw_only=True)
class IntensityFactor:
    """
    Result of ``intensity_factor`` for one orientation.

    Attributes:
        F: Attenuation-weighted sample volume in the gauge (mm3).
        volume_in_sample: Sample volume in the gauge (mm3).
        mean_path: Mean path in + out over the sample part of the gauge (mm). NaN when the gauge misses the sample.
        vy_offset: Intensity-weighted centroid along the sample vy axis, relative to the gauge center (mm). NaN when
            the gauge misses the sample.
        vy_rms: Intensity-weighted rms spread along the sample vy axis about that centroid (mm). NaN when the gauge
            misses the sample.
    """

    F: float
    volume_in_sample: float
    mean_path: float
    vy_offset: float
    vy_rms: float


def intensity_factor(
    m: Mount,
    *,
    chi: float,
    phi: float,
    points: FloatArray,
    volume: float,
    mesh: SampleMesh,
    center: npt.ArrayLike,
    mu: float,
) -> IntensityFactor:
    """
    F and the path statistics for one orientation.

    The gauge points are rotated into the sample frame (``R^T p + center``). From each point inside the mesh the path
    in is traced along -ki and the path out along kf, and the point is weighted by ``exp(-mu (L_in + L_out))``.

    Args:
        m: Beams and rotation conventions.
        chi: Cradle chi (deg).
        phi: Phi (deg).
        points: Gauge points from ``gauge_points`` (cradle frame, mm).
        volume: Gauge volume from ``gauge_points`` (mm3).
        mesh: Sample in its own frame (mm).
        center: Gauge center = rotation point in the sample frame (vx, vy, vz in mm).
        mu: Linear attenuation coefficient (1/mm).

    Returns:
        F, the sample volume in the gauge, the mean path and the vy centroid and spread. All zero or NaN when no
        gauge point is inside the sample.
    """
    pts, vol = points, volume
    c = np.asarray(center, float)
    R = m.R(chi, phi)
    p = pts @ R + c  # rows: R^T p
    ki, kf = R.T @ m.ki, R.T @ m.kf
    wind, L_in = mesh.trace(p, -ki)
    inside = wind > 0.5
    if not inside.any():
        return IntensityFactor(F=0.0, volume_in_sample=0.0, mean_path=np.nan, vy_offset=np.nan, vy_rms=np.nan)
    L = L_in[inside] + mesh.trace(p[inside], kf)[1]
    w = np.exp(-mu * L)
    n = len(p)
    vy = p[inside, 1] - c[1]
    cy = float((w * vy).sum() / w.sum())
    sy = float(np.sqrt((w * (vy - cy) ** 2).sum() / w.sum()))
    return IntensityFactor(F=float(vol * w.sum() / n), volume_in_sample=float(vol * inside.sum() / n),
                           mean_path=float(L.mean()), vy_offset=cy, vy_rms=sy)  # fmt: skip


def beam_value(spec: float | str, logs: Mapping[str, Any]) -> float:
    """
    A beam size from a config value: a number, or ``LOG_PREFIX + "<name>"`` for the median of a run log.

    Args:
        spec: A size (mm) or ``"log:<name>"``, e.g. ``"log:ISR:X:Gap"``.
        logs: Run logs by name.

    Returns:
        The size (mm).

    Raises:
        ValueError: If spec is text that is neither a number nor ``"log:<name>"``, or names a log not in ``logs``.
    """
    if isinstance(spec, str) and spec.startswith(LOG_PREFIX):
        name = spec.removeprefix(LOG_PREFIX)
        if name not in logs:
            raise ValueError(f"beam size {spec!r} names a log the run does not have ({name!r})")
        return float(np.median(logs[name]))
    try:
        return float(spec)
    except ValueError as exc:
        raise ValueError(f"beam size must be a number (mm) or {LOG_PREFIX}<log name>, got {spec!r}") from exc


def path_at_gauge(
    mesh: SampleMesh, center: npt.ArrayLike, *, mount: Mount, chi: float, phi: float
) -> tuple[float, float]:
    """
    Beam path in and out from the gauge center at cradle chi, phi, along the beam centerlines.

    Args:
        mesh: Sample.
        center: Gauge center (rotation point) in the sample frame (mm), shape (3,).
        mount: Beams and conventions for one omega and 2theta.
        chi: Cradle chi (deg).
        phi: Phi (deg).

    Returns:
        Path in and path out (mm). Both are NaN when the center is outside the sample.

    Example:
        >>> path_at_gauge(mesh, (0.0, 0.0, 0.0), mount=m, chi=0.0, phi=0.0)  # doctest: +SKIP
        (2.0, 2.0)
    """
    c = np.asarray(center, float)
    R = mount.R(chi, phi)
    ki, kf = R.T @ mount.ki, R.T @ mount.kf
    wind, L_in = mesh.trace(c[None], -ki)
    if wind[0] < 0.5:
        return float("nan"), float("nan")
    return float(L_in[0]), float(mesh.trace(c[None], kf)[1][0])
