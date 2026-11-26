"""
Peak intensity per sub-run: window integration or a pseudo-Voigt fit, ported from the reference script.

``peak_list``, ``find_peak``, ``integrate`` and ``fit_peak`` keep its numerics.

Intensity = net counts in a fixed 2theta window (+- 1.5 FWHM of the summed peak) above a straight background from
both sides, or (option) the area of a pseudo-Voigt plus linear background fit. Rates are per second of count time.
All 2theta values and widths are in degrees.

Typical workflow:

1. List the reflections on the detector with :func:`peak_list`.
2. Find each central peak's center and FWHM on the spectrum summed over sub-runs with :func:`find_peak`.
3. Measure each sub-run with :func:`integrate` (or :func:`fit_peak`) and flag doubtful peaks with
   :func:`peak_quality`. :func:`measure_peaks` does all three for every central peak of a run.

Example:
    >>> peaks = peak_list(x, arm=90.0, candidates=[("111", 70.1), ("200", 82.3)])  # doctest: +SKIP
    >>> center, fwhm = find_peak(x, y.sum(axis=0), peaks[0].two_theta)  # doctest: +SKIP
    >>> net, sigma = integrate(x, y[0], center=center, half_width=WINDOW_FWHM * fwhm)  # doctest: +SKIP
"""

import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

from .runs import RunData
from .types import FloatArray

CENTRAL_HALF_WIDTH_DEG = 5.0  # only peaks this close to the arm 2theta are used (edge pixels see another volume)
WINDOW_FWHM = 1.5  # integration half-window in FWHM of the summed peak
WEAK_PEAK_SIGMA = 5.0  # a summed peak below this net/sigma is flagged as weak (it is still used)
MIN_FWHM_BINS = 2  # a summed-peak FWHM narrower than this many bins is flagged
Method = Literal["window", "fit"]


@dataclass(frozen=True)
class PeakCandidate:
    """
    A reflection on the detector.

    Attributes:
        hkl: hkl label, e.g. "111".
        two_theta: Expected 2theta at the run's wavelength (deg).
        central: Within ``central_hw`` of the arm 2theta, so it is used for a pole figure.
    """

    hkl: str
    two_theta: float
    central: bool


def peak_list(x: FloatArray, *, arm: float, candidates: Iterable[tuple[str, float]],
              central_hw: float = CENTRAL_HALF_WIDTH_DEG) -> list[PeakCandidate]:  # fmt: skip
    """
    Reflections whose expected 2theta is on the detector (0.5 deg inside its edges), flagged central or not.

    Args:
        x: 2theta of the spectrum bins (deg).
        arm: Detector arm 2theta (deg).
        candidates: (hkl label, expected 2theta) pairs at the run's wavelength. Non-finite 2theta are skipped.
        central_hw: A reflection within this of the arm is central (deg, default: 5).

    Returns:
        The reflections on the detector, in the order given.
    """
    out = []
    for hkl, tt in candidates:
        if np.isfinite(tt) and x.min() + 0.5 < tt < x.max() - 0.5:
            out.append(PeakCandidate(hkl, float(tt), abs(tt - arm) <= central_hw))
    return out


def find_peak(x: FloatArray, ysum: FloatArray, guess: float) -> tuple[float, float]:
    """
    Center and FWHM of the summed peak near ``guess``.

    The highest bin within 1 deg of the guess starts the search. A straight background fitted 1.2 to 3 deg from it
    is subtracted. The center is the centroid of the positive net counts within 1.2 deg, and the FWHM is the span of
    the bins above half the net maximum.

    Args:
        x: 2theta of the bins (deg).
        ysum: Counts summed over sub-runs.
        guess: Expected 2theta (deg).

    Returns:
        The center and the FWHM (deg).

    Raises:
        ValueError: If there are no net counts above the background near the guess.
    """
    near = np.abs(x - guess) < 1.0
    c0 = x[near][np.argmax(ysum[near])]
    bgm = (np.abs(x - c0) > 1.2) & (np.abs(x - c0) < 3.0)
    net = ysum - np.polyval(np.polyfit(x[bgm], ysum[bgm], 1), x)
    near = np.abs(x - c0) < 1.2
    w = np.clip(net[near], 0, None)
    if not w.sum() > 0:
        raise ValueError(f"no peak above the background near 2theta {guess:.2f} deg")
    c = float((x[near] * w).sum() / w.sum())
    above = x[near][net[near] > net[near].max() / 2]
    return c, float(above.max() - above.min())


def integrate(x: FloatArray, y: FloatArray, *, center: float, half_width: float) -> tuple[float, float]:
    """
    Net counts in a 2theta window above a straight background, and its sigma.

    The background line is fitted to the bins from half_width + 0.3 to half_width + 2.5 deg on both sides. The sigma
    holds the Poisson error of the net counts and of the background under the window.

    Args:
        x: 2theta of the bins (deg).
        y: Counts per bin.
        center: Window center (deg).
        half_width: Window half-width (deg): bins with ``|x - center| <= half_width`` are summed.

    Returns:
        The net counts and their sigma (counts).
    """
    c, half = center, half_width
    bgm = (np.abs(x - c) > half + 0.3) & (np.abs(x - c) < half + 2.5)
    s = np.abs(x - c) <= half
    coef = np.polyfit(x[bgm], y[bgm], 1)
    bg = np.polyval(coef, x[s]).sum()
    net = float(y[s].sum() - bg)
    bgv = float(s.sum() ** 2 / bgm.sum() * np.mean(y[bgm]) + bg)
    return net, float(np.sqrt(max(net, 0) + bgv))


def _pv(t: FloatArray, a: float, x0: float, f: float, eta: float, b0: float, b1: float, c: float) -> FloatArray:
    g = np.exp(-4 * np.log(2) * ((t - x0) / f) ** 2) * 2 * np.sqrt(np.log(2) / np.pi) / f
    lo = 2 / (np.pi * f) / (1 + 4 * ((t - x0) / f) ** 2)
    return np.asarray(a * (eta * lo + (1 - eta) * g) + b0 + b1 * (t - c))


def fit_peak(x: FloatArray, y: FloatArray, *, center: float, fwhm: float, half_width: float) -> tuple[float, float]:
    """
    Peak area from a pseudo-Voigt plus linear background fit, with Poisson weights.

    The fit range is ``|x - center| < half_width + 2.5`` deg. The center is bounded to center +- 1 deg, the FWHM to
    0.3 to 3 times ``fwhm`` and the Lorentzian fraction to 0 to 1.

    Args:
        x: 2theta of the bins (deg).
        y: Counts per bin. Non-finite bins are left out.
        center: Center of the summed peak (deg).
        fwhm: FWHM of the summed peak (deg).
        half_width: Integration half-width (deg).

    Returns:
        The area and its sigma in counts (area over the bin width), or (nan, nan) when the fit fails.
    """
    c, fw, half = center, fwhm, half_width
    s = np.abs(x - c) < half + 2.5
    t, v = x[s], y[s]
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    bw = float(np.median(np.diff(t)))
    edge = np.abs(t - c) > half + 0.3
    b0 = float(np.median(v[edge])) if edge.any() else float(np.min(v))
    a0 = max(float((v[~edge] - b0).sum()) * bw, 1.0)
    p0 = [a0, c, fw, 0.5, b0, 0.0]
    lb = [-np.inf, c - 1.0, 0.3 * fw, 0.0, -np.inf, -np.inf]
    ub = [np.inf, c + 1.0, 3.0 * fw, 1.0, np.inf, np.inf]
    try:

        def model(tt: FloatArray, a: float, x0: float, f: float, eta: float, b0_: float, b1: float) -> FloatArray:
            return _pv(tt, a, x0, f, eta, b0_, b1, c)

        p, cov = curve_fit(model, t, v, p0=p0, sigma=np.sqrt(np.maximum(v, 1.0)), absolute_sigma=False,
                           bounds=(lb, ub), maxfev=20000)  # fmt: skip
        return float(p[0] / bw), float(np.sqrt(cov[0, 0]) / bw)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):  # the fit did not converge: NaN, so the point is ignored
        return np.nan, np.nan


def peak_quality(x: FloatArray, ysum: FloatArray, *, center: float, fwhm: float, hkl: str,
                 others: Iterable[PeakCandidate] = ()) -> list[str]:  # fmt: skip
    """
    Warnings about the summed peak (the values are still used): weak, too narrow, or overlapping another.

    The pole figure then says why it may be unreliable, instead of looking like any other.

    Args:
        x: 2theta of the bins (deg).
        ysum: Counts summed over sub-runs.
        center: Center of the summed peak (deg).
        fwhm: FWHM of the summed peak (deg).
        hkl: hkl label of the peak.
        others: The other central reflections. One within 1.5 FWHM (or 1.5 bins) of the center overlaps.

    Returns:
        Warning texts, empty when the peak looks fine.
    """
    c, fw = center, fwhm
    notes = []
    net, sig = integrate(x, ysum, center=c, half_width=WINDOW_FWHM * fw)
    if not (sig > 0 and net / sig >= WEAK_PEAK_SIGMA):
        notes.append(f"weak peak: summed net/sigma {net / sig if sig > 0 else 0:.1f} (below {WEAK_PEAK_SIGMA:g})")
    bin_w = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    if bin_w > 0 and fw < MIN_FWHM_BINS * bin_w:
        notes.append(f"peak FWHM {fw:.3f} deg is under {MIN_FWHM_BINS} bins")
    close = [o.hkl for o in others if abs(o.two_theta - c) < WINDOW_FWHM * max(fw, bin_w)]
    if close:
        notes.append(f"{{{hkl}}} overlaps {', '.join('{' + h + '}' for h in close)} on the detector")
    return notes


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
    return peak_list(run.x, arm=run.arm_two_theta, candidates=candidates, central_hw=central_hw)


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
