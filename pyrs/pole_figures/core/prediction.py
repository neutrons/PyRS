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
2. Set the beam and slit sizes (:class:`Beams`) and the grid (:class:`Grid`).
3. Call :func:`predict` with the sample mesh, gauge center, mount conventions and mu.
4. Plot each :class:`PlanePrediction` or plan its table scan (``table_scan.plan_scan``).

Example:
    >>> Grid().describe()
    '10 x 5 deg'
    >>> len(Grid().points()[0])
    649
    >>> pred = predict(mesh, center=(0.0, 0.0, 0.0), conv=MountConventions(),
    ...                planes=[Plane.symmetric("311", 104.0)], beams=Beams(), mu=0.1)  # doctest: +SKIP
    >>> pred.planes[0].sphere_fraction_ignored  # doctest: +SKIP
    0.12
"""

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import numpy.typing as npt

from .gauge import gauge_points, intensity_factor
from .geometry import Mount, MountConventions
from .parallel import intensity_factors, no_count_progress
from .stl_mesh import SampleMesh

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Progress = Callable[[float, str], None]

DEFAULT_CUTOFF = 3.0
DEFAULT_MC_POINTS = 40_000  # as the reference script. The prediction uses half of it, as the script does.
#: the GUI's default, 4 to 8 times faster than 20,000. On a thin, strongly absorbing IN718 slab (the test fixture) the
#: predicted correction at 5,000 points differs from that at 20,000 by 0.5 % (median), 1.5 % (95th percentile) and
#: 2.8 % at most; 3 of 2596 points cross the 3.0 cutoff. On the 50 mm Al cylinder 500 points were
#: within 0.04 %. 40,000 stays the core default for parity with the reference script.
GUI_MC_POINTS = 5_000


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


def no_progress(fraction: float, message: str) -> None:
    """
    Progress callback that ignores every report (the default where no progress is wanted).

    Args:
        fraction: Fraction of the job done, 0 to 1.
        message: What is being done.
    """


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


@dataclass(kw_only=True)
class PlanePrediction:
    """
    Prediction for one plane on the grid.

    Attributes:
        plane: hkl, 2theta and omega.
        mount: Beams and rotation for the plane.
        chi: Cradle chi per point (deg).
        phi: Phi per point (deg).
        F: Intensity factor, the attenuation-weighted sample volume in the gauge (mm3).
        volume_in_sample: Sample volume in the gauge (mm3).
        mean_path: Mean path in + out (mm), NaN where the gauge misses the sample.
        correction: F(0, 0) / F, inf where F is 0.
        ignored: correction > cutoff.
        sphere_fraction_ignored: Solid-angle-weighted fraction of the grid that is ignored.
        phi_chosen: True where phi was free (phi axis along Q) and set to the phi with the shortest path.
        phi_sweeps: For those points: index -> (phi values tried, mean path at each), for reporting.
    """

    plane: Plane
    mount: Mount
    chi: FloatArray
    phi: FloatArray
    F: FloatArray
    volume_in_sample: FloatArray
    mean_path: FloatArray
    correction: FloatArray
    ignored: BoolArray
    sphere_fraction_ignored: float
    phi_chosen: BoolArray = field(default_factory=lambda: np.zeros(0, bool))
    phi_sweeps: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

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
    n_points: int = DEFAULT_MC_POINTS // 2,
    progress: Callable[[int], None] = no_count_progress,
    best_phi: bool = False,
    best_phi_step: float,
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

    Returns:
        The prediction for the plane.

    Raises:
        ValueError: If best_phi is on and best_phi_step is not a positive step, or the gauge holds no sample at
            chi 0, phi 0.
    """
    if best_phi and not (np.isfinite(best_phi_step) and best_phi_step > 0):
        raise ValueError(f"best_phi_step must be a positive step in degrees, got {best_phi_step!r}")
    m = Mount(conv, omega=plane.omega, tth=plane.two_theta)
    phi = np.array(phi, float)
    pts, vol = gauge_points(m, incident_width=beams.incident_width, incident_height=beams.incident_height,
                            diffracted_width=beams.diffracted_width, n_points=n_points)  # fmt: skip
    F0 = intensity_factor(m, chi=0.0, phi=0.0, points=pts, volume=vol, mesh=mesh, center=center, mu=mu).F
    check_reference(F0, center)
    chi = np.asarray(chi, float)
    chosen = np.zeros(len(chi), bool)
    sweeps: dict[int, tuple[FloatArray, FloatArray]] = {}
    # every orientation to compute: the point itself, or the phi sweep where phi is free
    spans: list[tuple[int, int, FloatArray | None]] = []  # (first, count, trial phis) per point
    all_chi: list[float] = []
    all_phi: list[float] = []
    for c, f in zip(chi, phi, strict=True):
        trial = np.arange(0.0, 360.0, float(best_phi_step)) if best_phi and phi_is_free(m, float(c)) else None
        ph = list(trial) if trial is not None else [float(f)]
        spans.append((len(all_chi), len(ph), trial))
        all_chi += [float(c)] * len(ph)
        all_phi += [float(x) for x in ph]
    n_all = len(all_chi)

    def tick(done: int) -> None:  # progress counts grid points, not the orientations of a phi sweep
        progress(int(done * len(chi) / max(n_all, 1)))

    res = intensity_factors(m, chi=all_chi, phi=all_phi, points=pts, volume=vol, mesh=mesh, center=center, mu=mu,
                            progress=tick)  # fmt: skip
    rows = []
    for i, (first, count, trial) in enumerate(spans):
        block = [r[:3] for r in res[first : first + count]]
        if trial is not None:
            paths = np.array([r[2] for r in block], float)
            if np.isfinite(paths).any():
                best = shortest_path_index(paths)
                phi[i], chosen[i] = trial[best], True
                sweeps[i] = (trial, paths)
                rows.append(block[best])
            else:  # the gauge misses the sample at every phi: keep the given phi
                rows.append(intensity_factor(m, chi=float(chi[i]), phi=float(phi[i]), points=pts, volume=vol, mesh=mesh,
                                             center=center, mu=mu)[:3])  # fmt: skip
        else:
            rows.append(block[0])
    F, Vin, L = np.array(rows, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(F > 0, F0 / np.where(F > 0, F, 1), np.inf)
    ign = corr > cutoff
    frac = float((weights * ign).sum() / weights.sum())
    return PlanePrediction(plane=plane, mount=m, chi=np.asarray(chi, float), phi=phi, F=F, volume_in_sample=Vin,
                           mean_path=L, correction=corr, ignored=ign, sphere_fraction_ignored=frac, phi_chosen=chosen,
                           phi_sweeps=sweeps)  # fmt: skip


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
    n_points: int = DEFAULT_MC_POINTS // 2,
    progress: Progress = no_progress,
    best_phi: bool = False,
    allow_open_mesh: bool = False,
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
    total = len(planes) * len(chi)
    out = []
    for k, pl in enumerate(planes):

        def tick(i: int, k: int = k, pl: Plane = pl) -> None:
            if i % 10 == 0 or i == len(chi):
                progress((k * len(chi) + i) / total, f"{{{pl.hkl}}}: {i} of {len(chi)} orientations")

        out.append(predict_plane(mesh, center=center, conv=conv, plane=pl, beams=beams, mu=mu, chi=chi, phi=phi,
                                 weights=w, cutoff=cutoff, n_points=n_points, progress=tick, best_phi=best_phi,
                                 best_phi_step=grid.phi_step))  # fmt: skip
    return Prediction(planes=out, center=np.asarray(center, float), beams=beams, mu=float(mu), cutoff=float(cutoff),
                      grid=grid, n_points=int(n_points))  # fmt: skip
