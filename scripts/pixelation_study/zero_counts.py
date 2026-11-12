"""
Zero-count pixel statistics for the HB2B delay-line detector.

Helpers for the pixelation-uncertainty study (docs/planning/PixelationUncertainty_Plan.md).
They test whether the zero-count pixels in a measurement follow Poisson
statistics once the detector's dead area (pixels with no vanadium counts) is
excluded, and how the zero fraction varies along the detector's horizontal (x)
and vertical (y) axes.

Typical workflow:

1. Reshape per-pixel counts with to_detector_image(); index ``[x, y]``.
2. Mark dead pixels with dead_pixel_mask(vanadium), the same rule reduction uses.
3. Predict each pixel's mean count with expected_counts(), using vanadium as the
   per-pixel efficiency within 2θ × η bins from angle_bin_index().
4. Compare observed and Poisson-expected zero fractions per detector line
   with zero_fraction_profile(), and look for periodic structure with
   dominant_period().

Pixel ordering (see docs/ground_truths.md): PyRS pixel ID = x_index * 1024 + y_index,
so ``counts.reshape(1024, 1024)[x, y]``.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NUM_PIXEL_1D = 1024
# Reduction drops pixels whose vanadium count is below this (reduce_hb2b_pyrs.histogram_by_numpy).
VANADIUM_DEAD_THRESHOLD = 0.9

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def to_detector_image(counts: npt.ArrayLike, num_pixel_1d: int = NUM_PIXEL_1D) -> FloatArray:
    """
    Reshape a 1D per-pixel array (pixel-ID order) into a detector image.

    Args:
        counts: Values in PyRS pixel-ID order, length ``num_pixel_1d**2``.
        num_pixel_1d: Pixels along each detector axis.

    Returns:
        Array of shape ``(num_pixel_1d, num_pixel_1d)`` indexed ``[x, y]``.

    Raises:
        ValueError: If ``counts`` does not have ``num_pixel_1d**2`` elements.
    """
    flat = np.asarray(counts, dtype=np.float64).ravel()
    if flat.size != num_pixel_1d**2:
        raise ValueError(f"Expected {num_pixel_1d**2} pixels, got {flat.size}")
    return flat.reshape(num_pixel_1d, num_pixel_1d)


def dead_pixel_mask(vanadium: npt.ArrayLike, threshold: float = VANADIUM_DEAD_THRESHOLD) -> BoolArray:
    """
    Flag pixels that recorded (almost) nothing in a vanadium flood.

    Args:
        vanadium: Vanadium counts per pixel.
        threshold: Pixels with counts below this are dead (default matches reduction).

    Returns:
        Boolean array, True where the pixel is dead.

    Raises:
        ValueError: If any vanadium count is negative or not finite.
    """
    van = np.asarray(vanadium, dtype=np.float64)
    if not np.all(np.isfinite(van)) or np.any(van < 0):
        raise ValueError("Vanadium counts must be finite and non-negative")
    return van < threshold


def smooth_along_axis(values: npt.ArrayLike, live: npt.ArrayLike, width: int, axis: int) -> FloatArray:
    """
    Box-average an image along one axis, using live pixels only.

    Vanadium counts carry their own Poisson noise (~1/√N per pixel). Used raw as an
    efficiency, that noise inflates the predicted zero count by about
    1 + λ²·var/2 because exp(-λ) is convex. Averaging along the horizontal axis,
    where the efficiency varies slowly, removes most of it while keeping the
    sharp row-to-row structure along the vertical axis.

    Args:
        values: 2D image.
        live: True for pixels to average; dead pixels get NaN.
        width: Window length in pixels (>= 1; 1 returns the input unchanged).
        axis: Axis to average along.

    Returns:
        Smoothed image, NaN at dead pixels.

    Raises:
        ValueError: If ``width`` < 1 or shapes differ.
    """
    from scipy.ndimage import uniform_filter1d

    v = np.asarray(values, dtype=np.float64)
    ok = np.asarray(live, dtype=bool)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if v.shape != ok.shape:
        raise ValueError("values and live must share one shape")
    weights = ok.astype(np.float64)
    num = uniform_filter1d(np.where(ok, v, 0.0), width, axis=axis, mode="constant")
    den = uniform_filter1d(weights, width, axis=axis, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


def angle_bin_index(two_theta: npt.ArrayLike, eta: npt.ArrayLike, two_theta_step: float, eta_step: float) -> IntArray:
    """
    Assign every pixel to a 2θ × η bin.

    Args:
        two_theta: Per-pixel 2θ (degrees).
        eta: Per-pixel η (degrees), same shape as ``two_theta``.
        two_theta_step: 2θ bin width (degrees), > 0.
        eta_step: η bin width (degrees), > 0.

    Returns:
        Non-negative integer bin label per pixel, same shape as the input.

    Raises:
        ValueError: If shapes differ or a step is not positive.
    """
    tth = np.asarray(two_theta, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    if tth.shape != eta_arr.shape:
        raise ValueError(f"2θ shape {tth.shape} != η shape {eta_arr.shape}")
    if two_theta_step <= 0 or eta_step <= 0:
        raise ValueError(f"Bin steps must be positive, got {two_theta_step}, {eta_step}")
    i_tth = np.floor((tth - tth.min()) / two_theta_step).astype(np.int64)
    i_eta = np.floor((eta_arr - eta_arr.min()) / eta_step).astype(np.int64)
    return i_tth * (int(i_eta.max()) + 1) + i_eta


def expected_counts(
    counts: npt.ArrayLike, efficiency: npt.ArrayLike, bin_index: npt.ArrayLike, valid: npt.ArrayLike
) -> FloatArray:
    """
    Predict each pixel's mean count, assuming intensity is uniform within a bin.

    Within bin *b* the sample intensity per unit efficiency is
    ``S_b = Σ N_i / Σ v_i`` over valid pixels, and pixel *i* expects
    ``λ_i = S_b · v_i``.

    Args:
        counts: Measured counts per pixel.
        efficiency: Relative efficiency per pixel (vanadium counts).
        bin_index: Integer bin label per pixel (angle_bin_index()).
        valid: True for pixels to use; invalid pixels get NaN.

    Returns:
        Expected counts per pixel, NaN where invalid or the bin has no efficiency.

    Raises:
        ValueError: If inputs do not share one shape.
    """
    n = np.asarray(counts, dtype=np.float64)
    v = np.asarray(efficiency, dtype=np.float64)
    b = np.asarray(bin_index, dtype=np.int64)
    ok = np.asarray(valid, dtype=bool)
    if not (n.shape == v.shape == b.shape == ok.shape):
        raise ValueError("counts, efficiency, bin_index and valid must share one shape")

    nbins = int(b.max()) + 1
    sum_n = np.bincount(b[ok], weights=n[ok], minlength=nbins)
    sum_v = np.bincount(b[ok], weights=v[ok], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(sum_v > 0, sum_n / sum_v, np.nan)
    lam = scale[b] * v
    lam[~ok] = np.nan
    return lam


@dataclass(frozen=True)
class ZeroProfile:
    """
    Observed vs Poisson-expected zero-count fraction along one detector axis.

    Attributes:
        n_valid: Valid pixels in each detector line.
        observed: Observed fraction of valid pixels with zero counts.
        expected: Poisson-expected fraction, mean of exp(-λ_i).
        z_score: (observed − expected zeros) / sqrt(Σ p_i(1−p_i)); NaN for empty lines.
        count_ratio: Σ N_i / Σ λ_i per line; 1 when the vanadium efficiency fits the line.
        expected_total: Σ λ_i per line; 1/√expected_total is the Poisson scatter of count_ratio.
    """

    n_valid: IntArray
    observed: FloatArray
    expected: FloatArray
    z_score: FloatArray
    count_ratio: FloatArray
    expected_total: FloatArray


def zero_fraction_profile(
    counts: npt.ArrayLike, expected: npt.ArrayLike, valid: npt.ArrayLike, axis: int
) -> ZeroProfile:
    """
    Compare observed and expected zero counts for every detector line.

    The zero count in a line is a sum of independent Bernoulli trials with
    p_i = exp(-λ_i), so its variance is Σ p_i(1 − p_i).

    Args:
        counts: Detector image ``[x, y]``.
        expected: Expected counts λ, same shape.
        valid: True for pixels to include.
        axis: 0 to sum over x (one value per y line), 1 to sum over y (one per x line).

    Returns:
        ZeroProfile with one entry per line along the kept axis.

    Raises:
        ValueError: If shapes differ or ``axis`` is not 0 or 1.
    """
    n = np.asarray(counts, dtype=np.float64)
    lam = np.asarray(expected, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(lam)
    if not (n.shape == lam.shape == ok.shape) or n.ndim != 2:
        raise ValueError("counts, expected and valid must be 2D images of one shape")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    p0 = np.where(ok, np.exp(-np.where(ok, lam, 0.0)), 0.0)
    zero = (n == 0) & ok
    n_valid = ok.sum(axis=axis)
    obs_zeros = zero.sum(axis=axis).astype(np.float64)
    exp_zeros = p0.sum(axis=axis)
    var = (p0 * (1.0 - p0)).sum(axis=axis)
    sum_n = np.where(ok, n, 0.0).sum(axis=axis)
    sum_lam = np.where(ok, lam, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.where(n_valid > 0, obs_zeros / n_valid, np.nan)
        expected_frac = np.where(n_valid > 0, exp_zeros / n_valid, np.nan)
        z = np.where(var > 0, (obs_zeros - exp_zeros) / np.sqrt(var), np.nan)
        ratio = np.where(sum_lam > 0, sum_n / sum_lam, np.nan)
    return ZeroProfile(n_valid.astype(np.int64), observed, expected_frac, z, ratio, sum_lam)


def dominant_period(signal: npt.ArrayLike, min_period: float = 2.0) -> tuple[float, float]:
    """
    Find the strongest periodic component of a 1D signal.

    Args:
        signal: Evenly sampled values (NaNs are replaced by the mean).
        min_period: Ignore periods shorter than this (samples).

    Returns:
        (period in samples, share of total non-DC power in that component).

    Raises:
        ValueError: If fewer than 8 finite samples are given.
    """
    x = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        raise ValueError("Need at least 8 finite samples")
    x = np.where(finite, x, np.nanmean(x)) - np.nanmean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(x.size)
    keep = (freq > 0) & (freq <= 1.0 / min_period)
    if not np.any(keep) or power[keep].sum() == 0:
        return float("nan"), 0.0
    k = np.argmax(np.where(keep, power, -1.0))
    return float(1.0 / freq[k]), float(power[k] / power[freq > 0].sum())
