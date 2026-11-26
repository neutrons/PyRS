"""
Measured pole figures: per run and per central peak, the raw rate, the path-length correction and the ignored points.

Ported from the main loop of the reference script. For each run the peaks near the detector center are found on the
summed spectrum, and each sub-run's net rate is the window (or fit) intensity per second. The gauge Monte Carlo
(incident slits from the run's logs, receiving slit from the user) is traced through the sample at every measured chi,
phi. The correction is F(0, 0) / F and points above the cutoff are ignored. Measured data need not match a prediction:
any chi, phi set, count times and number of sub-runs work.

Typical workflow:

1. Read the run with ``runs.read_run``.
2. Call :func:`analyze_run` with the phase's reflections, the sample mesh and the mount.
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

import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from .gauge import gauge_points, intensity_factor
from .geometry import Mount, MountConventions
from .parallel import intensity_factors, no_count_progress
from .peaks import (
    CENTRAL_HALF_WIDTH_DEG,
    WINDOW_FWHM,
    PeakCandidate,
    find_peak,
    fit_peak,
    integrate,
    peak_list,
    peak_quality,
)
from .prediction import DEFAULT_CUTOFF, DEFAULT_MC_POINTS, Beams, check_reference, no_progress
from .runs import RunData
from .stl_mesh import SampleMesh

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Progress = Callable[[float, str], None]
Method = Literal["window", "fit"]


@dataclass(kw_only=True)
class PolePeak:
    """
    One measured pole figure: a run and one of its central peaks.

    chi is the cradle chi (logged chi + offset), rounded to 0.01 deg. phi is in 0 to 360. ``h`` holds the pole
    vector (sample frame) of every point: ``analyze_run`` fills it from the mount, and a combined pole figure keeps
    each part's own poles (several mounts). Per-point arrays all have one entry per sub-run.

    Attributes:
        run: Run number ("<run>+<run>" when combined).
        hkl: hkl label.
        title: Run title.
        two_theta: Fitted center of the summed peak (deg).
        fwhm: FWHM of the summed peak (deg).
        omega: Omega (deg, median of the log).
        wavelength_A: Wavelength (Å).
        beams: Incident beam and receiving slit sizes (mm).
        mount: Beams and rotation used for the Monte Carlo (and to draw cells).
        chi: Cradle chi per point (deg).
        phi: Phi per point, 0 to 360 (deg).
        count_s: Count time per point (s).
        rate: Raw net rate per point (counts/s).
        sigma: Sigma of the rate (counts/s).
        F: Attenuation-weighted sample volume in the gauge per point (mm3).
        F_ref: F at chi 0, phi 0 (mm3).
        volume_in_sample: Sample volume in the gauge per point (mm3).
        mean_path: Mean path in + out per point (mm).
        vy_offset: Intensity-weighted vy centroid of the gauge per point (mm).
        vy_rms: Intensity-weighted vy spread of the gauge per point (mm).
        correction: F_ref / F per point, inf where F is 0.
        cutoff: Points with a correction above this are ignored.
        mu: Linear attenuation coefficient (1/mm).
        method: "window" or "fit".
        chi_offset_deg: Cradle chi = logged chi + offset (deg).
        h: Pole vectors in the sample frame, shape (n, 3).
        live: The file was reduced while the run was still counting.
        source_runs: Runs merged into this pole figure (combined only).
        notes: Warnings about the peak or the fit.

    Raises:
        ValueError: If ``h`` does not have one (vx, vy, vz) row per point.
    """

    run: str
    hkl: str
    title: str
    two_theta: float
    fwhm: float
    omega: float
    wavelength_A: float
    beams: Beams
    mount: Mount
    chi: FloatArray
    phi: FloatArray
    count_s: FloatArray
    rate: FloatArray
    sigma: FloatArray
    F: FloatArray
    F_ref: float
    volume_in_sample: FloatArray
    mean_path: FloatArray
    vy_offset: FloatArray
    vy_rms: FloatArray
    correction: FloatArray
    cutoff: float
    mu: float
    method: Method
    chi_offset_deg: float
    h: FloatArray  # pole vectors (n, 3), sample frame
    live: bool = False
    source_runs: tuple[str, ...] = ()  # runs merged into this pole figure (combined only)
    notes: str = ""

    def __post_init__(self) -> None:
        """Refuse pole vectors that do not match the points (one (vx, vy, vz) row per chi, phi): ValueError."""
        if np.shape(self.h) != (len(self.chi), 3):
            raise ValueError(
                f"h must hold one pole vector per point, shape ({len(self.chi)}, 3), got {np.shape(self.h)}"
            )

    @property
    def n(self) -> int:
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


def find_peaks(run: RunData, candidates: Iterable[tuple[str, float]], *,
               central_hw: float = CENTRAL_HALF_WIDTH_DEG) -> list[PeakCandidate]:  # fmt: skip
    """
    Reflections on this run's detector (central and edge).

    Args:
        run: The run.
        candidates: (hkl label, expected 2theta) pairs at the run's wavelength.
        central_hw: A reflection within this of the arm is central (deg, default: 5).

    Returns:
        The reflections on the detector (see ``peaks.peak_list``).
    """
    return peak_list(run.x, arm=run.arm, candidates=candidates, central_hw=central_hw)


@dataclass(frozen=True, kw_only=True)
class PeakIntensity:
    """
    A central peak of one run.

    Attributes:
        hkl: hkl label.
        two_theta: Center of the summed peak (deg).
        fwhm: FWHM of the summed peak (deg).
        rate: Raw net rate per sub-run (counts/s). NaN where the fit failed.
        sigma: Sigma of the rate (counts/s).
        notes: Warnings about the peak (see ``peaks.peak_quality``) or the fit.
    """

    hkl: str
    two_theta: float
    fwhm: float
    rate: FloatArray
    sigma: FloatArray
    notes: list[str] = field(default_factory=list)


def measure_peaks(run: RunData, candidates: Iterable[tuple[str, float]], *, central_hw: float = CENTRAL_HALF_WIDTH_DEG,
                  method: Method = "window") -> list[PeakIntensity]:  # fmt: skip
    """
    Net rate and sigma per sub-run for each central peak.

    Args:
        run: The run.
        candidates: (hkl label, expected 2theta) pairs at the run's wavelength.
        central_hw: Peaks within this of the arm are used (deg, default: 5).
        method: "window" integration (default) or pseudo-Voigt "fit".

    Returns:
        One entry per central peak.

    Raises:
        ValueError: If method is unknown, or a central peak has no counts above the background.
    """
    if method not in typing.get_args(Method):
        raise ValueError(f"method must be one of {', '.join(typing.get_args(Method))}, got {method!r}")
    x, y, t = run.x, run.y, run.count_s
    ysum = y.sum(axis=0)
    out = []
    central = [pk for pk in find_peaks(run, candidates, central_hw=central_hw) if pk.central]
    for pk in central:
        c, fw = find_peak(x, ysum, pk.two_theta)
        half = WINDOW_FWHM * fw
        if method == "fit":
            net = np.array([fit_peak(x, y[i], center=c, fwhm=fw, half_width=half) for i in range(len(t))])
        else:
            net = np.array([integrate(x, y[i], center=c, half_width=half) for i in range(len(t))])
        notes = peak_quality(x, ysum, center=c, fwhm=fw, hkl=pk.hkl, others=[q for q in central if q is not pk])
        failed = int((~np.isfinite(net[:, 0])).sum())
        if failed:
            notes.append(f"the fit failed in {failed} of {len(t)} sub-runs (those points are ignored)")
        out.append(
            PeakIntensity(hkl=pk.hkl, two_theta=c, fwhm=fw, rate=net[:, 0] / t, sigma=net[:, 1] / t, notes=notes)
        )
    return out


def corrections(
    m: Mount,
    *,
    chi: FloatArray,
    phi: FloatArray,
    beams: Beams,
    mesh: SampleMesh,
    center: npt.ArrayLike,
    mu: float,
    n_points: int = DEFAULT_MC_POINTS,
    progress: Callable[[int], None] = no_count_progress,
) -> tuple[FloatArray, float, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Gauge Monte Carlo at each cradle chi, phi.

    Args:
        m: Beams and rotation conventions.
        chi: Cradle chi per point (deg).
        phi: Phi per point (deg).
        beams: Beam and slit sizes (mm).
        mesh: Sample in its own frame (mm).
        center: Gauge center in the sample frame (mm).
        mu: Linear attenuation coefficient (1/mm).
        n_points: Monte Carlo gauge points (default: 40,000).
        progress: ``progress(n_done)`` as orientations finish.

    Returns:
        F, F(0, 0), sample volume, mean path, vy offset, vy rms and correction F(0, 0) / F (inf where F is 0).

    Raises:
        ValueError: If the gauge holds no sample at chi 0, phi 0.
    """
    pts, vol = gauge_points(m, incident_width=beams.incident_width, incident_height=beams.incident_height,
                            diffracted_width=beams.diffracted_width, n_points=n_points)  # fmt: skip
    rows = intensity_factors(m, chi=chi, phi=phi, points=pts, volume=vol, mesh=mesh, center=center, mu=mu,
                             progress=progress)  # fmt: skip
    F, Vin, L, cy, sy = np.array(rows, float).reshape(-1, 5).T
    F_ref = intensity_factor(m, chi=0.0, phi=0.0, points=pts, volume=vol, mesh=mesh, center=center, mu=mu).F
    check_reference(F_ref, center)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(F > 0, F_ref / np.where(F > 0, F, 1), np.inf)
    return F, float(F_ref), Vin, L, cy, sy, corr


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
    omega = run.omega_median
    out = []
    for k, pk in enumerate(peaks):
        m = Mount(conv, omega=omega, tth=pk.two_theta)

        def tick(i: int, k: int = k, hkl: str = pk.hkl) -> None:
            if i % 10 == 0 or i == len(chi):
                progress((k + i / len(chi)) / len(peaks), f"run {run.run} {{{hkl}}}: {i} of {len(chi)} points")

        F, F_ref, Vin, L, cy, sy, corr = corrections(m, chi=chi, phi=phi, beams=beams, mesh=mesh, center=center, mu=mu,
                                                     n_points=n_points, progress=tick)  # fmt: skip
        out.append(PolePeak(
            run=run.run, hkl=pk.hkl, title=run.title, two_theta=pk.two_theta, fwhm=pk.fwhm, omega=omega,
            wavelength_A=run.wavelength_A, beams=beams, mount=m, chi=chi, phi=phi, count_s=run.count_s,
            rate=pk.rate, sigma=pk.sigma, F=F, F_ref=F_ref, volume_in_sample=Vin, mean_path=L,
            vy_offset=cy, vy_rms=sy, correction=corr, cutoff=float(cutoff), mu=float(mu), method=method,
            live=run.live, chi_offset_deg=conv.chi_offset_deg, notes="; ".join(pk.notes),
            h=np.array([m.pole(c, f) for c, f in zip(chi, phi, strict=True)]),
        ))  # fmt: skip
    return out
