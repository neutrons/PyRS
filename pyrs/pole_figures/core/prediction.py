"""
Predicted beam path and correction pole figures from the sample model alone (no data).

Ported from ``predicted_corrections`` of the reference script. Each plane is measured at its own 2theta and omega
on a chi/phi grid. For each grid point the gauge Monte Carlo gives F, and the correction is F(0, 0) / F. Points whose
correction is above the cutoff are flagged as ignored.

The reference script takes the beam sizes and the wavelength from the first run. Here they are inputs, so the
prediction works before any data exist.

Free phi: where the phi axis lies along Q (chi 90 with omega = theta), every phi measures the same pole. With
``best_phi`` on, such points are evaluated over a phi sweep and the phi with the shortest mean path is kept
(paths within 1 % tie and the lowest phi wins, so symmetric shapes always give the same choice), so
the table scan measures that pole where the beams travel least through the sample. The reference script keeps phi 0
there, so the option is off by default (parity).

Typical workflow:

1. Choose the planes (hkl, 2theta, omega), e.g. with :meth:`Plane.symmetric`.
2. Set the beam and slit sizes (``correction.Beams``) and the grid (``grid.Grid``).
3. Call :func:`predict` with the sample mesh, gauge center, mount conventions and mu.
4. Plot each :class:`PlanePrediction` or plan its table scan (``table_scan.plan_scan``).

Example:
    >>> pred = predict(mesh, center=(0.0, 0.0, 0.0), conv=MountConventions(),
    ...                planes=[Plane.symmetric("311", 104.0)], beams=Beams(), mu=0.1)  # doctest: +SKIP
    >>> pred.planes[0].sphere_fraction_ignored  # doctest: +SKIP
    0.12
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .correction import DEFAULT_CUTOFF, Beams, GaugeCorrections, gauge_corrections
from .gauge import PREDICTION_MC_POINTS
from .geometry import Mount, MountConventions
from .grid import DEFAULT_GRID, Grid
from .parallel import Workers
from .stl_mesh import SampleMesh
from .types import BoolArray, CountProgress, FloatArray, Progress, no_count_progress, no_progress, part_progress


@dataclass(frozen=True)
class Plane:
    """
    One planned pole figure.

    Attributes:
        hkl: hkl label.
        two_theta: 2theta of the reflection (deg).
        omega: Omega (deg).

    Example:
        >>> Plane.symmetric("311", 104.0)
        Plane(hkl='311', two_theta=104.0, omega=52.0)
    """

    hkl: str
    two_theta: float
    omega: float

    @classmethod
    def symmetric(cls, hkl: str, two_theta: float) -> "Plane":
        """
        Plane with omega = theta, which puts Q along vz at chi 0, phi 0 (for the default mount).

        Args:
            hkl: hkl label.
            two_theta: 2theta of the reflection (deg).

        Returns:
            The plane with omega = two_theta / 2.
        """
        return cls(hkl, float(two_theta), float(two_theta) / 2)


@dataclass(kw_only=True)
class PlanePrediction:
    """
    Prediction for one plane on the grid.

    Attributes:
        plane: hkl, 2theta and omega.
        mount: Beams and rotation for the plane.
        chi: Cradle chi per point (deg).
        phi: Phi per point (deg).
        gauge: Gauge Monte Carlo per point: F, F(0, 0), sample volume, mean path (NaN where the gauge misses the
            sample), vy centroid and spread, correction F(0, 0) / F (inf where F is 0).
        ignored: correction > cutoff.
        sphere_fraction_ignored: Solid-angle-weighted fraction of the grid that is ignored.
        phi_chosen: True where phi was free (phi axis along Q) and set to the phi with the shortest path.
        phi_sweeps: For those points: index -> (phi values tried, mean path at each), for reporting.
    """

    plane: Plane
    mount: Mount
    chi: FloatArray
    phi: FloatArray
    gauge: GaugeCorrections
    ignored: BoolArray
    sphere_fraction_ignored: float
    phi_chosen: BoolArray
    phi_sweeps: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

    @property
    def hkl(self) -> str:
        """Hkl label of the plane."""
        return self.plane.hkl

    @property
    def correction(self) -> FloatArray:
        """Path-length correction F(0, 0) / F per point, inf where F is 0 (``gauge.correction``)."""
        return self.gauge.correction

    def poles(self) -> FloatArray:
        """Pole vectors in the sample frame, shape (n, 3)."""
        return np.array([self.mount.pole(c, f) for c, f in zip(self.chi, self.phi, strict=True)])


@dataclass(kw_only=True)
class Prediction:
    """
    Predictions for all planes plus the inputs that made them.

    Attributes:
        planes: One prediction per plane.
        center: Gauge center in the sample frame (mm).
        beams: Beam and slit sizes (mm).
        mu: Linear attenuation coefficient (1/mm).
        cutoff: Points with a correction above this are ignored.
        grid: The chi/phi grid.
        n_points: Monte Carlo points per orientation.
    """

    planes: list[PlanePrediction]
    center: FloatArray
    beams: Beams
    mu: float
    cutoff: float
    grid: Grid
    n_points: int


FREE_PHI_TOL_DEG = 0.1  # phi axis within this of Q counts as along Q (omega and 2theta are rounded to 0.01 deg)
TIE_REL = 0.01  # paths within 1 % of the shortest are a tie (Monte Carlo noise); the lowest phi wins


def phi_is_free(m: Mount, chi: float, *, tol_deg: float = FREE_PHI_TOL_DEG) -> bool:
    """
    True when the phi axis at cradle chi lies along Q, so phi does not change the pole.

    Args:
        m: Beams and rotation conventions.
        chi: Cradle chi (deg).
        tol_deg: Largest angle between the phi axis and +-Q (deg, default: 0.1).

    Returns:
        Whether phi is free at this chi.
    """
    return bool(abs(float(m.phi_axis(chi) @ m.q)) >= np.cos(np.radians(tol_deg)))


def shortest_path_index(paths: FloatArray, *, rel: float = TIE_REL) -> int:
    """
    Index of the shortest path. Paths within rel of the shortest tie, and the first (lowest phi) wins.

    Args:
        paths: Mean path at each trial phi (mm), NaN where the gauge misses the sample. At least one must be finite.
        rel: Relative tie tolerance (default: 0.01, Monte Carlo noise).

    Returns:
        The index of the chosen path.

    Example:
        >>> shortest_path_index(np.array([10.05, 10.0, 12.0]))
        0
    """
    best = float(np.nanmin(paths))
    return int(np.flatnonzero(np.nan_to_num(paths, nan=np.inf) <= best * (1 + rel))[0])


def predict_plane(
    mesh: SampleMesh,
    *,
    center: npt.ArrayLike,
    conv: MountConventions,
    plane: Plane,
    beams: Beams,
    mu: float,
    chi: FloatArray,
    phi: FloatArray,
    weights: FloatArray,
    cutoff: float = DEFAULT_CUTOFF,
    n_points: int = PREDICTION_MC_POINTS,
    progress: CountProgress = no_count_progress,
    best_phi: bool = False,
    best_phi_step: float,
    workers: Workers = "auto",
) -> PlanePrediction:
    """
    Correction for one plane at the given cradle chi, phi points.

    Args:
        mesh: Sample model in the sample frame (mm). Not checked for closure here (``predict`` does that).
        center: Gauge center in the sample frame (mm).
        conv: Mount conventions.
        plane: hkl, 2theta and omega.
        beams: Beam and slit sizes (mm).
        mu: Linear attenuation coefficient (1/mm).
        chi: Cradle chi per point (deg).
        phi: Phi per point (deg).
        weights: Solid-angle weight per point, for ``sphere_fraction_ignored``.
        cutoff: Points with a correction above this are ignored (default: 3).
        n_points: Monte Carlo points per orientation (default: 20,000, half the 40,000 used for runs).
        progress: ``progress(n_done)`` in points of the grid (a phi sweep counts as its one point).
        best_phi: Where phi is free (phi axis along Q), try phi 0 to 360 in ``best_phi_step`` and keep the phi with
            the shortest mean path (default: False, which keeps the given phi).
        best_phi_step: Step of that sweep (deg). Required so the sweep step is always stated (``predict`` passes the
            grid's phi step).
        workers: Number of processes, or "auto" (default) to choose from the job size (``parallel.n_workers``).

    Returns:
        The prediction for the plane.

    Raises:
        ValueError: If best_phi is on and best_phi_step is not a positive step, or the gauge holds no sample at
            chi 0, phi 0.
    """
    if best_phi and not (np.isfinite(best_phi_step) and best_phi_step > 0):
        raise ValueError(f"best_phi_step must be a positive step in degrees, got {best_phi_step!r}")
    m = Mount(conv, omega=plane.omega, tth=plane.two_theta)
    chi = np.asarray(chi, float)
    phi = np.array(phi, float)
    # every orientation to compute: the point itself, or where phi is free the phi sweep and then the given phi
    spans: list[tuple[int, FloatArray | None]] = []  # (first orientation, trial phis) per point
    all_chi: list[float] = []
    all_phi: list[float] = []
    for c, f in zip(chi, phi, strict=True):
        trial = np.arange(0.0, 360.0, float(best_phi_step)) if best_phi and phi_is_free(m, float(c)) else None
        ph = [*trial, f] if trial is not None else [f]
        spans.append((len(all_chi), trial))
        all_chi += [float(c)] * len(ph)
        all_phi += [float(x) for x in ph]
    n_all = len(all_chi)

    def tick(done: int) -> None:  # progress counts grid points, not the orientations of a phi sweep
        progress(int(done * len(chi) / max(n_all, 1)))

    g = gauge_corrections(m, beams=beams, chi=all_chi, phi=all_phi, mesh=mesh, center=center, mu=mu,
                          n_points=n_points, progress=tick, workers=workers)  # fmt: skip
    rows = np.array([first for first, _ in spans], int)
    chosen = np.zeros(len(chi), bool)
    sweeps: dict[int, tuple[FloatArray, FloatArray]] = {}
    for i, (first, trial) in enumerate(spans):
        if trial is None:
            continue
        paths = g.mean_path[first : first + len(trial)].copy()
        if np.isfinite(paths).any():
            best = shortest_path_index(paths)
            phi[i], chosen[i], rows[i] = trial[best], True, first + best
            sweeps[i] = (trial, paths)
        else:  # the gauge misses the sample at every phi: keep the given phi
            rows[i] = first + len(trial)
    q = g.take(rows)
    ign = q.correction > cutoff
    frac = float((weights * ign).sum() / weights.sum())
    return PlanePrediction(plane=plane, mount=m, chi=chi, phi=phi, gauge=q, ignored=ign, sphere_fraction_ignored=frac,
                           phi_chosen=chosen, phi_sweeps=sweeps)  # fmt: skip


def predict(
    mesh: SampleMesh,
    *,
    center: npt.ArrayLike,
    conv: MountConventions,
    planes: Sequence[Plane],
    beams: Beams,
    mu: float,
    grid: Grid = DEFAULT_GRID,
    cutoff: float = DEFAULT_CUTOFF,
    n_points: int = PREDICTION_MC_POINTS,
    progress: Progress = no_progress,
    best_phi: bool = False,
    allow_open_mesh: bool = False,
    workers: Workers = "auto",
) -> Prediction:
    """
    Predicted path and correction for each plane on a chi/phi grid.

    Args:
        mesh: Sample model in the sample frame (mm).
        center: Gauge center in the sample frame (mm).
        conv: Mount conventions.
        planes: hkl, 2theta and omega of each planned pole figure.
        beams: Beam and slit sizes (mm).
        mu: Linear attenuation coefficient (1/mm).
        grid: chi/phi grid (default: 10 x 5 deg).
        cutoff: Points with correction above this are ignored (default: 3).
        n_points: Monte Carlo points per orientation (default: 20,000, half the 40,000 used for runs).
        progress: ``progress(fraction, message)``. The default ignores progress.
        best_phi: Where phi is free (phi axis along Q), use the phi with the shortest mean path, tried in the grid's
            phi step (default: False, which keeps phi 0 there).
        allow_open_mesh: Trace a mesh that is not closed or not consistently wound (default: False, which raises
            ValueError because its path lengths can be wrong).
        workers: Number of processes, or "auto" (default) to choose from the job size (``parallel.n_workers``).

    Returns:
        The predictions and the inputs that made them.

    Raises:
        ValueError: If the mesh is open or inconsistently wound (unless allowed), there are no planes, or the gauge
            holds no sample at chi 0, phi 0.
    """
    if not allow_open_mesh:
        mesh.check()
    if not planes:
        raise ValueError(f"no planes to predict: select at least one reflection, got {list(planes)!r}")
    chi, phi = grid.points()
    w = grid.weights(chi)
    out = []
    for k, pl in enumerate(planes):
        tick = part_progress(
            progress, part=k, parts=len(planes), n=len(chi), label=f"{{{pl.hkl}}}", unit="orientations"
        )
        out.append(predict_plane(mesh, center=center, conv=conv, plane=pl, beams=beams, mu=mu, chi=chi, phi=phi,
                                 weights=w, cutoff=cutoff, n_points=n_points, progress=tick, best_phi=best_phi,
                                 best_phi_step=grid.phi_step, workers=workers))  # fmt: skip
    return Prediction(planes=out, center=np.asarray(center, float), beams=beams, mu=float(mu), cutoff=float(cutoff),
                      grid=grid, n_points=int(n_points))  # fmt: skip
