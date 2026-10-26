"""
Detector position response from a slit scan.

An absorbing sheet with one narrow open slit, placed just in front of the detector window and lit by
a diffuse scatterer at the sample position (e.g. vanadium), casts a bright line on the detector. Moving
the slit across the face in known steps maps detector channel to physical position, row by row, like a
tube "bar scan" (Mantid ``Calibration/tube.py``) with a bright line instead of a shadow. The result is
the pixel pitch, how uniform it is across the face, and where the active area falls in channel space,
independent of the diffraction geometry (docs/ground_truths.md).

The line is a projection from the source: a slit at stage position ``x`` lands on the conversion plane
at ``x0 + M * x`` with ``M = (L + z) / (L - s)`` (``L`` source to window, ``s`` slit in front of the
window, ``z`` mean conversion depth behind it), see :class:`SlitScanGeometry`.

Typical workflow (images ``[position, index_0, index_1]`` as ``get_detector_counts(sub_run)`` reshaped to
``(num_columns, num_rows)``; the slit runs along ``index_1`` and moves along ``index_0``):

1. Locate the line in every band of rows with :func:`find_line_centres`.
2. Fit channel against physical position per band with :func:`fit_position_response`.
3. Optionally find the active-area edges with :func:`active_edges`, and the source distance from two
   scans at different slit gaps with :func:`source_distance_from_two_gaps`.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "SlitScanGeometry",
    "LineCentres",
    "PositionResponse",
    "find_line_centres",
    "fit_position_response",
    "active_edges",
    "source_distance_from_two_gaps",
]


@dataclass(frozen=True)
class SlitScanGeometry:
    """Projection of the slit onto the detector's conversion plane.

    Attributes:
        source_to_window: Distance from the scattering source (sample position) to the detector window (m).
        slit_gap: Distance of the slit in front of the window (m).
        conversion_depth: Mean depth behind the window at which neutrons are detected (m).

    Example:
        >>> round(SlitScanGeometry(0.985, 0.005, 0.005).magnification, 6)
        1.010152
    """

    source_to_window: float
    slit_gap: float
    conversion_depth: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_to_window > 0:
            raise ValueError(f"source_to_window must be positive, got {self.source_to_window}")
        if not 0 <= self.slit_gap < self.source_to_window:
            raise ValueError(f"slit_gap must be in [0, source_to_window), got {self.slit_gap}")
        if self.conversion_depth < 0:
            raise ValueError(f"conversion_depth must be non-negative, got {self.conversion_depth}")

    @property
    def magnification(self) -> float:
        """Slit displacement on the conversion plane per unit stage displacement."""
        return (self.source_to_window + self.conversion_depth) / (self.source_to_window - self.slit_gap)


@dataclass
class LineCentres:
    """Fitted bright-line centres, one per slit position and band of rows.

    Attributes:
        centre: Line centre in ``index_0`` channels (pixel ``c`` spans ``[c - 0.5, c + 0.5]``), shape
            ``(positions, bands)``; NaN where no line was found.
        error: Uncertainty of ``centre`` (channels).
        amplitude: Fitted peak counts per pixel above background (efficiency corrected).
        width: Fitted Gaussian sigma of the line (channels).
        band_rows: ``(first, last)`` ``index_1`` of each band (inclusive).
    """

    centre: np.ndarray
    error: np.ndarray
    amplitude: np.ndarray
    width: np.ndarray
    band_rows: List[Tuple[int, int]]

    @property
    def band_centres(self) -> np.ndarray:
        """Mean ``index_1`` of each band."""
        return np.array([(lo + hi) / 2 for lo, hi in self.band_rows])


@dataclass
class PositionResponse:
    """Physical position against channel, per band of rows.

    The position on the conversion plane is ``X(c) = sum_k a_k u**k`` with ``u = (c - reference) / half``,
    so the local pitch is ``dX/dc = (a_1 + 2 a_2 u + 3 a_3 u**2) / half``.

    Attributes:
        coefficients: ``a_k`` (m), shape ``(bands, degree + 1)``.
        covariance: Their covariance, shape ``(bands, degree + 1, degree + 1)``.
        chi2: Chi-square of the fit per band.
        dof: Degrees of freedom per band.
        residual: Position residual (m) of every slit position, shape ``(positions, bands)``.
        reference: Reference channel ``u = 0``.
        half: Channels from the reference to ``u = 1``.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    chi2: np.ndarray
    dof: np.ndarray
    residual: np.ndarray
    reference: float
    half: float

    @property
    def pitch(self) -> np.ndarray:
        """Local pitch at the reference channel (m per channel), per band."""
        return self.coefficients[:, 1] / self.half

    @property
    def pitch_error(self) -> np.ndarray:
        return np.sqrt(self.covariance[:, 1, 1]) / self.half

    @property
    def gradient(self) -> np.ndarray:
        """Relative change of the local pitch from the reference to ``u = 1`` (``2 a_2 / a_1``), per band."""
        if self.coefficients.shape[1] < 3:
            return np.zeros(len(self.coefficients))
        return 2 * self.coefficients[:, 2] / self.coefficients[:, 1]

    @property
    def gradient_error(self) -> np.ndarray:
        if self.coefficients.shape[1] < 3:
            return np.zeros(len(self.coefficients))
        a1, a2 = self.coefficients[:, 1], self.coefficients[:, 2]
        var = (
            4 * self.covariance[:, 2, 2] / a1**2
            + 4 * a2**2 * self.covariance[:, 1, 1] / a1**4
            - 8 * a2 * self.covariance[:, 1, 2] / a1**3
        )
        return np.sqrt(np.maximum(var, 0.0))

    def position(self, channel, band: int) -> np.ndarray:
        """Physical position (m) of a (fractional) channel in one band."""
        u = (np.asarray(channel, dtype=float) - self.reference) / self.half
        return np.polynomial.polynomial.polyval(u, self.coefficients[band])

    def local_pitch(self, channel, band: int) -> np.ndarray:
        """Local pitch (m per channel) at a (fractional) channel in one band."""
        u = (np.asarray(channel, dtype=float) - self.reference) / self.half
        derivative = np.polynomial.polynomial.polyder(self.coefficients[band])
        return np.polynomial.polynomial.polyval(u, derivative) / self.half


def _gaussian(x: np.ndarray, height: float, centre: float, sigma: float, background: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + background


def _line_model(channels: np.ndarray, exposure: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Expected summed counts per channel: exposure times (Gaussian line + flat background)."""
    return exposure * _gaussian(channels, *p)


def _fit_line(channels: np.ndarray, counts: np.ndarray, exposure: np.ndarray, guess_width: float) -> Optional[tuple]:
    """Poisson maximum-likelihood fit of a Gaussian line on a flat background to one band profile.

    ``counts`` are the raw counts summed over the band's usable pixels per channel and ``exposure`` the
    summed relative efficiency of those pixels, so the model is valid down to a few counts per channel
    (a chi-square with a variance floor is not). Returns None if no line is significant at 5 sigma.
    """
    signal = counts / exposure
    peak = int(np.argmax(signal))
    background = max(float(np.median(signal)), 1e-6)
    height = float(signal[peak]) - background
    if not height > 0:
        return None
    x0 = np.array([height, channels[peak], guess_width, background])

    def deviance(p):
        mu = np.maximum(_line_model(channels, exposure, p), 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(counts > 0, counts * np.log(counts / mu), 0.0)
        return np.sign(counts - mu) * np.sqrt(np.maximum(2 * (mu - counts + term), 0.0))

    lower = [0.0, channels[0], 0.2, 0.0]
    upper = [np.inf, channels[-1], float(len(channels)), np.inf]
    out = least_squares(
        deviance,
        np.clip(x0, lower, upper),
        bounds=(lower, upper),
        x_scale=[max(height, 1e-9), 1.0, 1.0, max(height, 1e-9)],
    )
    if not out.success:
        return None
    # Fisher information of the Poisson likelihood, from numerical derivatives of the expected counts
    mu = np.maximum(_line_model(channels, exposure, out.x), 1e-12)
    jac = np.empty((channels.size, 4))
    for k, step in enumerate((max(abs(out.x[0]), 1e-9) * 1e-4, 1e-4, 1e-4, max(abs(out.x[3]), 1e-9) * 1e-4)):
        shifted = out.x.copy()
        shifted[k] += step
        jac[:, k] = (_line_model(channels, exposure, shifted) - mu) / step
    covariance = np.linalg.pinv(jac.T @ (jac / mu[:, None]))
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not errors[0] > 0 or out.x[0] / errors[0] < 5 or not channels[0] < out.x[1] < channels[-1]:
        return None
    return out.x[1], float(errors[1]), out.x[0], out.x[2]


def find_line_centres(
    images: np.ndarray,
    rows_per_band: int = 32,
    keep: Optional[np.ndarray] = None,
    efficiency: Optional[np.ndarray] = None,
    half_window: int = 15,
    guess_width: float = 2.0,
) -> LineCentres:
    """Locate the bright slit line in every band of rows of every image.

    Each band's rows are summed into one profile along ``index_0`` (raw counts, with the summed pixel
    efficiency as the exposure), and a Gaussian on a flat background is fitted within ``half_window``
    channels of its maximum by Poisson maximum likelihood. The Gaussian centre of a symmetric line is its
    centre; the exponential conversion-depth tail shifts it slightly, in proportion to the obliquity,
    which mostly acts as a small change of magnification.

    Args:
        images: Counts, shape ``(positions, n_index_0, n_index_1)``.
        rows_per_band: Rows (``index_1``) summed per band.
        keep: 1 for pixels to use, 0 for masked ones, shape ``(n_index_0, n_index_1)``; None keeps all.
        efficiency: Relative pixel efficiency (e.g. a normalised vanadium flood), same shape; None for 1.
        half_window: Channels either side of the profile maximum used in the fit.
        guess_width: Starting Gaussian sigma (channels).

    Returns:
        The fitted line centres.

    Raises:
        ValueError: On mismatched shapes or non-positive band and window sizes.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim != 3:
        raise ValueError(f"images must have shape (positions, n_index_0, n_index_1), got {images.shape}")
    _, n_channels, n_rows = images.shape
    if rows_per_band < 1 or half_window < 3:
        raise ValueError("rows_per_band must be >= 1 and half_window >= 3")
    keep = np.ones((n_channels, n_rows)) if keep is None else np.asarray(keep, dtype=float)
    efficiency = np.ones((n_channels, n_rows)) if efficiency is None else np.asarray(efficiency, dtype=float)
    if keep.shape != (n_channels, n_rows) or efficiency.shape != (n_channels, n_rows):
        raise ValueError("keep and efficiency must have the shape of one image")
    use = (keep > 0) & (efficiency > 0)

    edges = list(range(0, n_rows, rows_per_band))
    band_rows = [(lo, min(lo + rows_per_band, n_rows) - 1) for lo in edges]
    shape = (images.shape[0], len(band_rows))
    centre, error, amplitude, width = (np.full(shape, np.nan) for _ in range(4))
    channels_all = np.arange(n_channels, dtype=float)
    for b, (lo, hi) in enumerate(band_rows):
        band_use = use[:, lo : hi + 1]
        exposure = np.where(band_use, efficiency[:, lo : hi + 1], 0.0).sum(axis=1)
        valid = exposure > 0
        for i, image in enumerate(images):
            counts = np.where(band_use, image[:, lo : hi + 1], 0.0).sum(axis=1)
            signal = np.where(valid, counts / np.where(valid, exposure, 1.0), -np.inf)
            peak = int(np.argmax(signal))
            window = slice(max(peak - half_window, 0), min(peak + half_window + 1, n_channels))
            sel = valid[window]
            if sel.sum() < 6:
                continue
            result = _fit_line(channels_all[window][sel], counts[window][sel], exposure[window][sel], guess_width)
            if result is not None:
                centre[i, b], error[i, b], amplitude[i, b], width[i, b] = result
    return LineCentres(centre, error, amplitude, width, band_rows)


def fit_position_response(
    centres: LineCentres,
    stage_positions: Union[Sequence[float], np.ndarray],
    geometry: SlitScanGeometry,
    degree: int = 3,
    reference: float = 511.5,
    half: float = 512.0,
) -> PositionResponse:
    """Fit physical position against line centre (channel) in every band.

    Physical position on the conversion plane is ``M * stage_position`` (the constant term absorbs
    the stage zero, the source position and any slant of the slit). Uncertainties of the centres are
    propagated through the local slope, in one reweighting pass.

    Args:
        centres: Output of :func:`find_line_centres`.
        stage_positions: Slit stage position of each image (m).
        geometry: Projection of the slit onto the conversion plane.
        degree: Polynomial degree (1: uniform pitch; 2: linear pitch gradient; 3: plus curvature).
        reference: Channel of ``u = 0`` (default: the detector centre).
        half: Channels from the reference to ``u = 1``.

    Returns:
        The position response per band (NaN for bands with too few lines).

    Raises:
        ValueError: If the number of stage positions does not match the images, or degree < 1.
    """
    stage = np.asarray(stage_positions, dtype=float)
    if stage.shape != (centres.centre.shape[0],):
        raise ValueError(f"need one stage position per image ({centres.centre.shape[0]}), got {stage.shape}")
    if degree < 1:
        raise ValueError(f"degree must be at least 1, got {degree}")
    position = geometry.magnification * stage
    n_bands = centres.centre.shape[1]
    coefficients = np.full((n_bands, degree + 1), np.nan)
    covariance = np.full((n_bands, degree + 1, degree + 1), np.nan)
    chi2 = np.full(n_bands, np.nan)
    dof = np.zeros(n_bands, dtype=int)
    residual = np.full(centres.centre.shape, np.nan)
    for b in range(n_bands):
        ok = np.isfinite(centres.centre[:, b]) & np.isfinite(centres.error[:, b])
        if ok.sum() < degree + 2:
            continue
        u = (centres.centre[ok, b] - reference) / half
        design = np.vander(u, degree + 1, increasing=True)
        slope = np.full(
            u.size, np.median(np.diff(position[ok]) / np.diff(centres.centre[ok, b])) if ok.sum() > 1 else 1.0
        )
        for _ in range(2):
            sigma = np.abs(slope) * centres.error[ok, b]
            weights = 1 / np.maximum(sigma, 1e-15)
            a, *_ = np.linalg.lstsq(design * weights[:, None], position[ok] * weights, rcond=None)
            slope = np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(a)) / half
        weighted = design * weights[:, None]
        cov = np.linalg.pinv(weighted.T @ weighted)
        res = position[ok] - design @ a
        coefficients[b], covariance[b] = a, cov
        chi2[b] = float(np.sum((res * weights) ** 2))
        dof[b] = int(ok.sum() - degree - 1)
        residual[ok, b] = res
    return PositionResponse(coefficients, covariance, chi2, dof, residual, reference, half)


def active_edges(flood: np.ndarray, rows: Optional[slice] = None, threshold: float = 0.5) -> Tuple[float, float]:
    """Channels (``index_0``) where the summed flood profile crosses ``threshold`` x its central level.

    Args:
        flood: Flood (or summed scan) counts, shape ``(n_index_0, n_index_1)``.
        rows: Rows (``index_1``) to sum; None for the middle half.
        threshold: Fraction of the median level of the central half of the profile.

    Returns:
        ``(low, high)`` crossing channels, linearly interpolated.

    Raises:
        ValueError: If the profile never reaches the threshold.
    """
    flood = np.asarray(flood, dtype=float)
    n_channels, n_rows = flood.shape
    rows = slice(n_rows // 4, 3 * n_rows // 4) if rows is None else rows
    profile = flood[:, rows].sum(axis=1)
    level = threshold * np.median(profile[n_channels // 4 : 3 * n_channels // 4])
    above = np.flatnonzero(profile > level)
    if above.size == 0:
        raise ValueError("the flood profile never reaches the threshold")
    i, j = above[0], above[-1]
    low = i - 1 + (level - profile[i - 1]) / (profile[i] - profile[i - 1]) if i > 0 else 0.0
    high = j + (profile[j] - level) / (profile[j] - profile[j + 1]) if j < n_channels - 1 else float(n_channels - 1)
    return float(low), float(high)


def source_distance_from_two_gaps(
    apparent_pitch_1: float, apparent_pitch_2: float, gap_1: float, gap_2: float
) -> float:
    """Source-to-window distance from two scans of the same detector at different slit gaps.

    With the magnification left out (``M = 1`` in :func:`fit_position_response`), each scan measures
    an apparent pitch ``p / M_i`` with ``M_i = (L + z) / (L - s_i)``, so their ratio is
    ``(L - s_1) / (L - s_2)`` whatever the pitch and conversion depth.

    Args:
        apparent_pitch_1: Stage metres per channel from the scan at ``gap_1``.
        apparent_pitch_2: Same at ``gap_2``.
        gap_1: Slit gap of the first scan (m).
        gap_2: Slit gap of the second scan (m); must differ from ``gap_1``.

    Returns:
        ``L`` (m).

    Raises:
        ValueError: If the gaps are equal or the pitches equal (no magnification difference).
    """
    if gap_1 == gap_2:
        raise ValueError("the two scans need different slit gaps")
    ratio = apparent_pitch_1 / apparent_pitch_2  # = M_2 / M_1 = (L - s_1) / (L - s_2)
    if ratio == 1:
        raise ValueError("equal apparent pitches: the gaps are too close to resolve the magnification")
    return (ratio * gap_2 - gap_1) / (ratio - 1)
