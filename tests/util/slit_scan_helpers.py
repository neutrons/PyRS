"""Simulated slit scans for testing pyrs.calibration.slit_scan.

The detector response is a physical position ``X(c)`` along each row with local pitch
``p(c) = pitch * (1 + gradient * u + curvature * u**2)``, ``u = (c - 511.5) / 512``; pixel ``c`` covers
``[X(c - 0.5), X(c + 0.5)]``. Only ``|X| <= active_half_width`` detects neutrons.

A slit of width ``w`` at stage position ``x`` (plus ``tan(slant) * y`` along the slit) is projected from a
point source a distance ``L`` in front of the window onto conversion depth ``z`` behind it:
``X_line = x_source + (x - x_source) * (L + z) / (L - s)``. Depths are exponential with mean
``mean_depth`` (truncated at ``gas_depth``), sampled at quantiles. The line profile is the projected box
convolved with a Gaussian detector resolution, integrated over each pixel.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.special import ndtr

SQRT_2PI = np.sqrt(2 * np.pi)


@dataclass
class SlitScanTruth:
    pitch: float
    gradient: float
    curvature: float
    magnification_mean_depth: float  # (L + mean of the simulated depths) / (L - s)
    mean_depth: float  # mean of the simulated (truncated) depths
    active_half_width: float


def channel_position(channel, pitch: float, gradient: float = 0.0, curvature: float = 0.0, half: float = 512.0):
    """Physical position (m) of a fractional channel: the integral of the local pitch from 511.5."""
    u = (np.asarray(channel, dtype=float) - 511.5) / half
    return pitch * half * (u + gradient * u**2 / 2 + curvature * u**3 / 3)


def _box_gauss_cdf(x: np.ndarray, width: float, sigma: float) -> np.ndarray:
    """CDF of a unit-area box of ``width`` centred at 0 convolved with a Gaussian of ``sigma``."""

    def g(t):  # integral of the Gaussian CDF
        return t * ndtr(t / sigma) + sigma * np.exp(-0.5 * (t / sigma) ** 2) / SQRT_2PI

    return (g(x + width / 2) - g(x - width / 2)) / width


def simulate_slit_scan(
    stage_positions: Union[Sequence[float], np.ndarray],
    pitch: float = 0.314e-3,
    gradient: float = 0.0,
    curvature: float = 0.0,
    n_channels: int = 1024,
    n_rows: int = 64,
    row_pitch: float = 0.314e-3,
    source_to_window: float = 0.985,
    slit_gap: float = 0.005,
    slit_width: float = 0.75e-3,
    mean_depth: float = 0.005,
    gas_depth: float = 0.02,
    resolution_sigma: float = 0.8e-3,
    line_counts_per_row: float = 200.0,
    background: float = 0.02,
    efficiency_spread: float = 0.0,
    slant_deg: float = 0.0,
    source_offset: float = 0.0,
    active_half_width: float = 0.150,
    depth_samples: int = 24,
    seed: Optional[int] = 0,
):
    """Images of a slit scan, shape ``(positions, n_channels, n_rows)``, plus the efficiency map and truth.

    Args:
        stage_positions: Slit positions (m) relative to the detector centre line.
        line_counts_per_row: Expected counts in the line per row (before efficiency), for a slit fully
            inside the active area.
        background: Expected background counts per pixel.
        efficiency_spread: Relative r.m.s. of the per-pixel efficiency (0: uniform).
        seed: Random seed; None for no Poisson noise (expected counts returned).

    Returns:
        ``(images, efficiency, truth)``.
    """
    rng = np.random.default_rng(seed)
    efficiency = np.ones((n_channels, n_rows))
    if efficiency_spread > 0:
        efficiency = np.clip(1 + efficiency_spread * rng.standard_normal((n_channels, n_rows)), 0.1, None)
    edges = channel_position(np.arange(n_channels + 1) - 0.5, pitch, gradient, curvature)  # pixel boundaries
    active = np.abs(0.5 * (edges[:-1] + edges[1:])) <= active_half_width
    y = (np.arange(n_rows) - (n_rows - 1) / 2) * row_pitch
    # exponential conversion depths at quantiles, truncated at the gas depth
    q = (np.arange(depth_samples) + 0.5) / depth_samples
    cut = 1 - np.exp(-gas_depth / mean_depth)
    depths = -mean_depth * np.log(1 - q * cut)
    images = np.empty((len(stage_positions), n_channels, n_rows))
    for i, x in enumerate(stage_positions):
        expected = np.zeros((n_channels, n_rows))
        slit = x + np.tan(np.radians(slant_deg)) * y  # per row
        for z in depths:
            magnification = (source_to_window + z) / (source_to_window - slit_gap)
            centre = source_offset + (slit - source_offset) * magnification  # (n_rows,)
            cdf = _box_gauss_cdf(edges[:, None] - centre[None, :], slit_width * magnification, resolution_sigma)
            expected += np.diff(cdf, axis=0) / depth_samples
        expected = line_counts_per_row * expected * active[:, None] + background
        expected *= efficiency
        images[i] = expected if seed is None else rng.poisson(expected)
    mean = float(depths.mean())
    magnification = (source_to_window + mean) / (source_to_window - slit_gap)
    return images, efficiency, SlitScanTruth(pitch, gradient, curvature, magnification, mean, active_half_width)
