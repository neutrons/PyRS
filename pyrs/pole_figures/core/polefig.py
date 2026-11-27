"""
Measured pole figures: per run and per central peak, the raw rate, the path-length correction and the ignored points.

Ported from the main loop of the reference script. For each run the peaks near the detector center are found on the
summed spectrum, and each sub-run's net rate is the window (or fit) intensity per second. The gauge Monte Carlo
(incident slits from the run's logs, receiving slit from the user) is traced through the sample at every measured chi,
phi. The correction is F(0, 0) / F and points above the cutoff are ignored. Measured data need not match a prediction:
any chi, phi set, count times and number of sub-runs work.

Typical workflow:

1. Read the run with ``runs.read_run``.
2. Call :func:`analyze_run` with the phase's reflections, the sample mesh and the mount. It measures the peaks
   with ``peaks.measure_peaks``.
3. Use each :class:`PolePeak`: ``corrected`` and ``ignored`` for plots and CSVs, :meth:`PolePeak.with_cutoff` to try
   another cutoff without a new Monte Carlo.

Example:
    >>> run = read_run("HB2B_1234.h5")  # doctest: +SKIP
    >>> lam = run.wavelength_A  # doctest: +SKIP
    >>> refl = [(r.label, r.two_theta) for r in phase.reflections(lam)]  # doctest: +SKIP
    >>> pfs = analyze_run(run, refl, mesh=mesh, center=(0.0, 0.0, 0.0), conv=MountConventions(),
    ...                   receiving_slit_mm=2.0, mu=phase.mu_per_mm(lam))  # doctest: +SKIP
    >>> [pf.label for pf in pfs]  # doctest: +SKIP
    ['1234 {311}']
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .correction import DEFAULT_CUTOFF, Beams, GaugeCorrections, gauge_corrections
from .gauge import DEFAULT_MC_POINTS
from .geometry import Mount, MountConventions
from .parallel import Workers
from .peaks import CENTRAL_HALF_WIDTH_DEG, Method, measure_peaks
from .runs import RunData
from .stl_mesh import SampleMesh
from .types import BoolArray, FloatArray, Progress, no_progress, part_progress


@dataclass(kw_only=True)
class PolePeak:
    """
    One measured pole figure: a run and one of its central peaks.

    chi is the cradle chi (logged chi + offset), rounded to 0.01 deg. phi is in 0 to 360. ``h`` holds the pole
    vector (sample frame) of every point: ``analyze_run`` fills it from the mount, and a combined pole figure keeps
    each part's own poles. Per-point arrays all have one entry per sub-run. Omega and the chi offset are those of
    ``mount`` (``combine.combine`` refuses parts that differ in either), so they are not stored twice.

    Attributes:
        run: Run number ("<run>+<run>" when combined).
        hkl: hkl label.
        title: Run title.
        two_theta: Fitted center of the summed peak (deg).
        fwhm: FWHM of the summed peak (deg).
        wavelength_A: Wavelength (Å).
        beams: Incident beam and receiving slit sizes (mm).
        mount: Beams and rotation used for the Monte Carlo (and to draw cells). Its omega is the median of the log.
        chi: Cradle chi per point (deg).
        phi: Phi per point, 0 to 360 (deg).
        count_s: Count time per point (s).
        rate: Raw net rate per point (counts/s).
        sigma: Sigma of the rate (counts/s).
        gauge: Gauge Monte Carlo per point: F, F_ref, sample volume, mean path, vy centroid and spread, correction.
        cutoff: Points with a correction above this are ignored.
        mu: Linear attenuation coefficient (1/mm).
        method: "window" or "fit".
        h: Pole vectors in the sample frame, shape (n, 3).
        live: The file was reduced while the run was still counting.
        source_runs: Runs merged into this pole figure (combined only).
        notes: Warnings about the peak or the fit.

    Raises:
        ValueError: If ``h`` does not have one (vx, vy, vz) row per point, or ``gauge`` not one entry per point.
    """

    run: str
    hkl: str
    title: str
    two_theta: float
    fwhm: float
    wavelength_A: float
    beams: Beams
    mount: Mount
    chi: FloatArray
    phi: FloatArray
    count_s: FloatArray
    rate: FloatArray
    sigma: FloatArray
    gauge: GaugeCorrections
    cutoff: float
    mu: float
    method: Method
    h: FloatArray  # pole vectors (n, 3), sample frame
    live: bool = False
    source_runs: tuple[str, ...] = ()  # runs merged into this pole figure (combined only)
    notes: str = ""

    def __post_init__(self) -> None:
        """Refuse pole vectors or gauge results that do not match the points (one per chi, phi): ValueError."""
        if np.shape(self.h) != (len(self.chi), 3):
            raise ValueError(
                f"h must hold one pole vector per point, shape ({len(self.chi)}, 3), got {np.shape(self.h)}"
            )
        if len(self.gauge.correction) != len(self.chi):
            raise ValueError(f"gauge must hold one result per point ({len(self.chi)}), got "
                             f"{len(self.gauge.correction)}")  # fmt: skip

    @property
    def omega(self) -> float:
        """Omega of the mount (deg): the median of the run's omega log."""
        return self.mount.omega

    @property
    def chi_offset_deg(self) -> float:
        """Cradle chi = logged chi + this offset (deg), from the mount conventions."""
        return self.mount.conv.chi_offset_deg

    @property
    def correction(self) -> FloatArray:
        """Path-length correction F_ref / F per point, inf where F is 0 (``gauge.correction``)."""
        return self.gauge.correction

    @property
    def n_points(self) -> int:
        """Number of points."""
        return len(self.chi)

    @property
    def label(self) -> str:
        """Run and hkl, e.g. '1234 {311}'."""
        return f"{self.run} {{{self.hkl}}}"

    @property
    def ignored(self) -> BoolArray:
        """Points whose correction is above the cutoff (or infinite), or whose rate is not finite (a failed fit)."""
        return np.asarray(~(self.correction <= self.cutoff) | ~np.isfinite(self.rate), bool)

    @property
    def corrected(self) -> FloatArray:
        """Corrected rate, rate x correction (counts/s), NaN where the correction is not finite."""
        return np.where(np.isfinite(self.correction), self.rate * self.correction, np.nan)

    @property
    def corrected_sigma(self) -> FloatArray:
        """Sigma of the corrected rate, sigma x correction (counts/s), NaN where the correction is not finite."""
        return np.where(np.isfinite(self.correction), self.sigma * self.correction, np.nan)

    def poles(self) -> FloatArray:
        """Pole vectors in the sample frame, shape (n, 3)."""
        return self.h

    def with_cutoff(self, cutoff: float) -> "PolePeak":
        """
        Same result with another cutoff (no new Monte Carlo).

        Args:
            cutoff: The new cutoff.

        Returns:
            A copy with the new cutoff, so ``ignored`` changes.
        """
        return replace(self, cutoff=float(cutoff))


def cradle_angles(run: RunData, conv: MountConventions) -> tuple[FloatArray, FloatArray]:
    """
    Cradle chi and phi of each sub-run, rounded to 0.01 deg.

    Args:
        run: The run.
        conv: Mount conventions. Cradle chi = logged chi + ``conv.chi_offset_deg``.

    Returns:
        Cradle chi and phi (wrapped to 0 to 360), both rounded to 0.01 deg.
    """
    chi = np.round(np.asarray(conv.cradle_chi(run.chi), float), 2) if conv.chi_offset_deg else np.round(run.chi, 2)
    phi = np.round(np.mod(run.phi, 360), 2)
    return chi.astype(float), phi.astype(float)


def analyze_run(
    run: RunData,
    candidates: Iterable[tuple[str, float]],
    *,
    mesh: SampleMesh,
    center: npt.ArrayLike,
    conv: MountConventions,
    receiving_slit_mm: float,
    mu: float,
    cutoff: float = DEFAULT_CUTOFF,
    n_points: int = DEFAULT_MC_POINTS,
    central_hw: float = CENTRAL_HALF_WIDTH_DEG,
    method: Method = "window",
    progress: Progress = no_progress,
    allow_open_mesh: bool = False,
    workers: Workers = "auto",
) -> list[PolePeak]:
    """
    Raw rates, corrections and ignored points for every central peak of one run.

    Args:
        run: From ``read_run``.
        candidates: (hkl label, expected 2theta) pairs at the run's wavelength (from the phase).
        mesh: Sample model in the sample frame (mm).
        center: Gauge center in the sample frame (mm).
        conv: Mount conventions.
        receiving_slit_mm: Diffracted (receiving) slit width (mm). Not in the file. The incident slits come from the
            run's logs.
        mu: Linear attenuation coefficient at the run's wavelength (1/mm).
        cutoff: Points with a correction above this are ignored (default: 3).
        n_points: Monte Carlo gauge points (default: 40,000).
        central_hw: Use peaks within this many degrees of the arm 2theta (default: 5).
        method: "window" integration (default) or pseudo-Voigt "fit".
        progress: ``progress(fraction, message)`` within this run. The default ignores progress.
        allow_open_mesh: Trace a mesh that is not closed or not consistently wound (default: False, which raises
            ValueError).
        workers: Number of processes, or "auto" (default) to choose from the job size (``parallel.n_workers``).

    Returns:
        One pole figure per central peak, empty when the run has none.

    Raises:
        ValueError: If the mesh is open or inconsistently wound (unless allowed), method is unknown, a central peak
            has no counts above the background, the slit sizes are not positive, or the gauge holds no sample at
            chi 0, phi 0.
    """
    if not allow_open_mesh:
        mesh.check()
    peaks = measure_peaks(run, candidates, central_hw=central_hw, method=method)
    if not peaks:
        return []
    beams = Beams(incident_width=run.incident_width, incident_height=run.incident_height,
                  diffracted_width=float(receiving_slit_mm))  # fmt: skip
    chi, phi = cradle_angles(run, conv)
    out = []
    for k, pk in enumerate(peaks):
        m = Mount(conv, omega=run.omega_median, tth=pk.two_theta)
        tick = part_progress(progress, part=k, parts=len(peaks), n=len(chi), label=f"run {run.run} {{{pk.hkl}}}",
                             unit="points")  # fmt: skip
        g = gauge_corrections(m, beams=beams, chi=chi, phi=phi, mesh=mesh, center=center, mu=mu, n_points=n_points,
                              progress=tick, workers=workers)  # fmt: skip
        out.append(PolePeak(
            run=run.run, hkl=pk.hkl, title=run.title, two_theta=pk.two_theta, fwhm=pk.fwhm,
            wavelength_A=run.wavelength_A, beams=beams, mount=m, chi=chi, phi=phi, count_s=run.count_s,
            rate=pk.rate, sigma=pk.sigma, gauge=g, cutoff=float(cutoff), mu=float(mu), method=method, live=run.live,
            notes="; ".join(pk.notes), h=np.array([m.pole(c, f) for c, f in zip(chi, phi, strict=True)]),
        ))  # fmt: skip
    return out
