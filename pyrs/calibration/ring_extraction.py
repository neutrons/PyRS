"""
Ring extraction: find where each powder ring crosses the HB2B detector, in fractional pixels.

:func:`extract_ring_points` cuts each sub-run's detector image into strips, locates every reflection
of the sub-run's powder standard with one fit of all strips, fits it again in each strip (Mantid
``FitPeaks``), converts each fitted 2theta back to a column with the same geometry, and rejects
failed, imprecise and outlying fits. :class:`StripFit` records each strip fit for diagnostics. Part
of :mod:`pyrs.calibration.pixel_pos_calibration`.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mantid.simpleapi import DeleteWorkspace, mtd
from scipy.optimize import least_squares

from pyrs.core.detector_model import DetectorModel
from pyrs.calibration.powder_standards import HB2B_STANDARDS, PowderStandard, standard_for_sy
from pyrs.calibration.ring_points import _RING_FIELDS, RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks import FitEngineFactory  # type: ignore

__all__ = ["StripFit", "extract_ring_points"]


@dataclass
class StripFit:
    """Diagnostics of one ring fitted in one detector strip (see :func:`extract_ring_points`).

    Attributes:
        sub_run: Sub-run.
        strip: Strip number, counted along ``index_1``.
        row: Strip-centre ``index_1``, where the ring point is placed.
        powder: Powder standard name.
        hkl_squared: ``h^2 + k^2 + l^2`` of the reflection.
        bragg_two_theta: Expected 2theta (deg) at the starting wavelength.
        start_center: Ring 2theta (deg) found by the first, all-bands fit, on which the band's window
            was centred; NaN if that fit found no ring and the window was placed on the expectation.
        two_theta: Bin centres (deg) of the strip histogram.
        intensity: Strip histogram, counts per pixel (or per vanadium count).
        center: Fitted ring 2theta (deg); NaN if the fit failed.
        center_error: Its uncertainty (deg); NaN if unknown.
        index_0: Fractional column of the ring point; NaN if rejected.
        accepted: Whether the point is in the returned RingPoints.
        reason: Why it was rejected (empty if accepted).
    """

    sub_run: int
    strip: int
    row: float
    powder: str
    hkl_squared: int
    bragg_two_theta: float
    start_center: float
    two_theta: np.ndarray
    intensity: np.ndarray
    center: float
    center_error: float
    index_0: float = np.nan
    accepted: bool = False
    reason: str = ""


_SECANT_STEP_PX = 1e-3


def _column_at_two_theta(
    model: DetectorModel,
    target: float,
    row: float,
    arm_two_theta: float,
    l2: float,
    calibration: Optional[DENEXDetectorShift],
) -> float:
    """Fractional ``index_0`` on row ``row`` whose 2theta is ``target``; NaN if it is off the detector."""
    num_rows, num_columns = model.instrument_setup.detector_size
    columns = np.arange(num_columns, dtype=float)
    profile = model.two_theta(columns, row, arm_two_theta, l2, calibration)
    order = np.argsort(profile)
    if not profile[order[0]] <= target <= profile[order[-1]]:
        return np.nan
    column = float(np.interp(target, profile[order], columns[order]))
    # two secant refinements make the linear interpolation exact to well below 1e-6 pixel
    for _ in range(2):
        here, step = model.two_theta(column, row, arm_two_theta, l2, calibration), _SECANT_STEP_PX
        slope = (model.two_theta(column + step, row, arm_two_theta, l2, calibration) - here) / step
        column -= float((here - target) / slope)
    return column


_MIN_POINTS_FOR_OUTLIERS = 4  # a quadratic through 3 points has no residual to judge them by
_OUTLIER_F_SCALE_PX = 0.5  # soft-L1 transition: residuals beyond ~0.5 px count linearly, not squared
_MAD_TO_SIGMA = 1.4826  # median absolute deviation -> Gaussian sigma
_MIN_OUTLIER_SCALE_PX = 0.1


def _reject_outliers(points: List["_Candidate"], threshold: float, smooth: bool) -> None:
    """Flag points far from a quadratic ``index_0(index_1)`` through each ring; optionally smooth."""
    rings: Dict[Tuple[int, int], List["_Candidate"]] = {}
    for point in points:
        rings.setdefault((point.sub_run, point.hkl_squared), []).append(point)
    for ring in rings.values():
        if len(ring) < _MIN_POINTS_FOR_OUTLIERS:
            continue
        rows = np.array([p.index_1 for p in ring])
        columns = np.array([p.index_0 for p in ring])
        # robust (soft L1) quadratic: a plain least-squares fit is dragged towards the outlier it
        # should expose, shrinking that point's residual and inflating everyone else's
        scaled = (rows - rows.mean()) / max(np.ptp(rows), 1.0)
        start = np.polyfit(scaled, columns, 2)
        fit = least_squares(
            lambda c: np.polyval(c, scaled) - columns, start, loss="soft_l1", f_scale=_OUTLIER_F_SCALE_PX
        )
        smooth_columns = np.polyval(fit.x, scaled)
        residual = columns - smooth_columns
        # floor so a near-perfect ring (simulated, noise-free) does not reject on rounding scatter
        scale = max(_MAD_TO_SIGMA * np.median(np.abs(residual - np.median(residual))), _MIN_OUTLIER_SCALE_PX)
        for point, res, fitted in zip(ring, residual, smooth_columns):
            if abs(res) > threshold * scale:
                point.reject = f"outlier ({res:+.2f} px from the ring)"
            elif smooth:
                point.index_0 = float(fitted)


def _fit_band_histograms(
    name: str,
    pixel_sets: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    edges: np.ndarray,
    peak_function: str,
    background_function: str,
    wavelength: float,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Histogram each pixel set on ``edges`` (per-pixel normalised) and fit one peak in all of them.

    Each set is ``(2theta, weighted counts, weighted efficiency)`` of the pixels of one band (or of
    several bands combined). All sets share the bin grid, so a single Mantid ``FitPeaks`` call fits
    them all, one spectrum per set, over the whole grid.

    Returns:
        Bin centres, the histograms, and per set: fitted centre, centre error, significance
        (intensity / its error) and the fit-exclusion flag.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    fit_ws = HidraWorkspace(name)
    fit_ws.set_sub_runs(list(range(1, len(pixel_sets) + 1)))
    histograms = []
    for spectrum, (tth, weighted_counts, weighted_efficiency) in enumerate(pixel_sets, start=1):
        summed = np.histogram(tth, edges, weights=weighted_counts)[0]
        norm = np.histogram(tth, edges, weights=weighted_efficiency)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = np.where(norm > 0, summed / norm, 0.0)
            sigma = np.where(norm > 0, np.sqrt(np.maximum(summed, 1.0)) / norm, 0.0)
        histograms.append(intensity)
        # despite its name, the "variances" slot is handed to Mantid as DataE, i.e. sigma; reduction
        # also stores sigma there (reduce_hb2b_pyrs.py), see docs/ground_truths.md
        fit_ws.set_reduced_diffraction_data(spectrum, None, centres, intensity, sigma)

    # PyRS's standard fit (FitPeaks HighBackground=True): it ends in the same joint peak + background
    # fit as False, from better starting values, and is never worse here (docs/ground_truths.md)
    engine = FitEngineFactory.getInstance(
        fit_ws, peak_function, background_function, wavelength=wavelength, out_of_plane_angle=None
    )
    result = engine.fit_multiple_peaks([name], [edges[0]], [edges[-1]])
    values, errors = result.peakcollections[0].get_effective_params()
    excluded = np.asarray(result.peakcollections[0].exclude, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        significance = np.asarray(values["Intensity"], dtype=float) / np.asarray(errors["Intensity"], dtype=float)

    # FitPeaks leaves its workspaces in Mantid's data service
    for workspace_name in [engine._mtd_wksp.name()] + [str(ws) for ws in (result.fitted, result.difference)]:
        if workspace_name in mtd:
            DeleteWorkspace(workspace_name)

    return (
        centres,
        histograms,
        np.asarray(values["Center"], dtype=float),
        np.asarray(errors["Center"], dtype=float),
        significance,
        excluded,
    )


@dataclass
class _Candidate:
    """One reflection in one band while it is being extracted: a RingPoints entry and a StripFit."""

    sub_run: int
    strip: int
    index_1: float
    det_2theta: float
    l2: float
    d_spacing: float
    hkl_squared: int
    powder: str
    bragg_two_theta: float
    start_center: float
    sigma: float = np.nan
    index_0: float = np.nan
    center: float = np.nan
    two_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity: np.ndarray = field(default_factory=lambda: np.array([]))
    reject: str = ""

    def strip_fit(self) -> StripFit:
        return StripFit(
            sub_run=self.sub_run,
            strip=self.strip,
            row=self.index_1,
            powder=self.powder,
            hkl_squared=self.hkl_squared,
            bragg_two_theta=self.bragg_two_theta,
            start_center=self.start_center,
            two_theta=self.two_theta,
            intensity=self.intensity,
            center=self.center,
            center_error=self.sigma,
            index_0=np.nan if self.reject else self.index_0,
            accepted=not self.reject,
            reason=self.reject,
        )


@dataclass(frozen=True)
class _ExtractionSettings:
    """The tuning arguments of :func:`extract_ring_points`, validated once."""

    strips: int
    edge: int
    side_margin: int
    bin_width: float
    fit_half_width: float
    min_window_side: float
    peak_function: str
    background_function: str
    max_center_error: float
    min_significance: float
    outlier_threshold: float
    smooth: bool

    def __post_init__(self) -> None:
        if (
            self.strips < 1
            or self.edge < 0
            or self.side_margin < 0
            or not self.bin_width > 0
            or not self.fit_half_width > 0
        ):
            raise ValueError(
                "strips must be >= 1, edge and side_margin >= 0, and bin_width and fit_half_width positive"
            )
        if not self.min_significance >= 0:
            raise ValueError(f"min_significance must be non-negative, got {self.min_significance}")
        if not 0 < self.min_window_side <= self.fit_half_width:
            raise ValueError(f"min_window_side must be in (0, fit_half_width], got {self.min_window_side}")

    def window(self, center: float, tth: np.ndarray) -> Optional[Tuple[int, int]]:
        """Fit window around ``center`` clipped to the pixels' coverage, in whole bins; None if either
        side keeps less than ``min_window_side`` of data."""
        low = int(np.ceil(max(center - self.fit_half_width, tth.min()) / self.bin_width))
        high = int(np.floor(min(center + self.fit_half_width, tth.max()) / self.bin_width))
        if (
            center - low * self.bin_width < self.min_window_side
            or high * self.bin_width - center < self.min_window_side
        ):
            return None
        return low, high

    def edges(self, window: Tuple[int, int]) -> np.ndarray:
        """Bin edges (deg) of a window from :meth:`window`."""
        return np.arange(window[0], window[1] + 1) * self.bin_width


PixelSet = Tuple[np.ndarray, np.ndarray, np.ndarray]  # 2theta, weighted counts, weighted efficiency


def _band_pixels(
    two_theta: np.ndarray, counts: np.ndarray, weights: np.ndarray, efficiency: np.ndarray, bounds: np.ndarray
) -> List[PixelSet]:
    """Per band (rows ``bounds[k]`` to ``bounds[k + 1]``): 2theta, counts and efficiency of its unmasked pixels."""
    pixel_sets = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        band = slice(lo, hi)
        keep = weights[:, band].ravel() > 0
        pixel_sets.append(
            (
                two_theta[:, band].ravel()[keep],
                weights[:, band].ravel()[keep] * counts[:, band].ravel()[keep],
                weights[:, band].ravel()[keep] * efficiency[:, band].ravel()[keep],
            )
        )
    return pixel_sets


def _locate_ring(
    name: str, combined: PixelSet, bragg: float, settings: _ExtractionSettings, wavelength: float
) -> Tuple[float, bool]:
    """Pass 1: the ring's 2theta from one fit of all bands together, or ``bragg`` if none is found.

    The signal is strongest here, so the band windows (and the fits' starting centre, the window
    middle) sit on the ring even when the starting wavelength or geometry puts the prediction a peak
    width or more off. Returns the 2theta and whether the fit found the ring.
    """
    locate_window = settings.window(bragg, combined[0]) if combined[0].size else None
    if locate_window is None:
        return bragg, False
    edges = settings.edges(locate_window)
    _, _, center, _, significance, excluded = _fit_band_histograms(
        name, [combined], edges, settings.peak_function, settings.background_function, wavelength
    )
    if not excluded[0] and significance[0] >= settings.min_significance and edges[0] < center[0] < edges[-1]:
        return float(center[0]), True
    return bragg, False


def _judge_fit(
    candidate: _Candidate, excluded: bool, significance: float, edges: np.ndarray, settings: _ExtractionSettings
) -> str:
    """Why a band fit cannot be used ("" if it can); the checks run in this order."""
    center, error = candidate.center, candidate.sigma
    failed = excluded or not np.isfinite(center) or not np.isfinite(error) or error <= 0
    if failed or not np.isfinite(significance):  # no valid intensity uncertainty
        return "fit failed"
    if significance < settings.min_significance:
        return f"peak not significant ({significance:.1f} < {settings.min_significance} sigma)"
    if error > settings.max_center_error:
        return f"centre error {error:.3f} > {settings.max_center_error} deg"
    if not edges[0] < center < edges[-1]:
        return "centre outside the fit window"
    return ""


def _fit_band_group(
    model: DetectorModel,
    members: List[_Candidate],
    pixel_sets: List[PixelSet],
    band_window: Tuple[int, int],
    arm: float,
    calibration: Optional[DENEXDetectorShift],
    settings: _ExtractionSettings,
    wavelength: float,
) -> None:
    """Fit bands sharing one window (one FitPeaks call, bands as spectra) and place their points."""
    edges = settings.edges(band_window)
    first = members[0]
    centers, histograms, center, center_error, significance, excluded = _fit_band_histograms(
        f"ring_{first.sub_run}_{first.hkl_squared}_{band_window[0]}",
        pixel_sets,
        edges,
        settings.peak_function,
        settings.background_function,
        wavelength,
    )
    num_columns = model.instrument_setup.detector_size[1]
    for i, candidate in enumerate(members):
        candidate.center, candidate.sigma = float(center[i]), float(center_error[i])
        candidate.two_theta, candidate.intensity = centers, histograms[i]
        candidate.reject = _judge_fit(candidate, excluded[i], significance[i], edges, settings)
        if candidate.reject:
            continue
        column = _column_at_two_theta(model, candidate.center, candidate.index_1, arm, candidate.l2, calibration)
        candidate.index_0 = column
        if not np.isfinite(column):
            candidate.reject = "ring off the detector on this row"
        elif not settings.side_margin <= column <= num_columns - 1 - settings.side_margin:
            # the window was placed from pass 1 or the prediction; check the fitted point too
            candidate.reject = f"fitted ring within {settings.side_margin} px of the side edge"


def _extract_reflection(
    model: DetectorModel,
    sub_run: int,
    standard: PowderStandard,
    hkl2: int,
    bragg: float,
    band_pixels: List[PixelSet],
    rows_center: np.ndarray,
    det_2theta: float,
    l2: float,
    calibration: Optional[DENEXDetectorShift],
    settings: _ExtractionSettings,
    wavelength: float,
) -> List[_Candidate]:
    """All band candidates of one reflection in one sub-run, fitted and judged, in strip order."""
    arm = -det_2theta
    combined = tuple(np.concatenate([pixels[j] for pixels in band_pixels]) for j in range(3))
    target, located = _locate_ring(f"ring_{sub_run}_{hkl2}_locate", combined, bragg, settings, wavelength)

    # pass 2: choose bands for this reflection: a curved ring can be near the side edge in some only
    num_columns = model.instrument_setup.detector_size[1]
    candidates: Dict[int, _Candidate] = {}
    groups: Dict[Tuple[int, int], List[int]] = {}
    for k, row in enumerate(rows_center):
        predicted = _column_at_two_theta(model, target, row, arm, l2, calibration)
        if not np.isfinite(predicted):
            continue  # the ring does not cross this band
        candidate = _Candidate(
            sub_run=sub_run,
            strip=k,
            index_1=row,
            det_2theta=det_2theta,
            l2=l2,
            d_spacing=standard.lattice_a / np.sqrt(hkl2),
            hkl_squared=int(hkl2),
            powder=standard.name,
            bragg_two_theta=bragg,
            start_center=target if located else np.nan,
        )
        candidates[k] = candidate
        tth = band_pixels[k][0]
        if tth.size == 0:
            candidate.reject = "band has no unmasked pixels"
        elif not settings.side_margin <= predicted <= num_columns - 1 - settings.side_margin:
            candidate.reject = f"ring predicted within {settings.side_margin} px of the side edge"
        else:
            band_window = settings.window(target, tth)
            if band_window is None:
                candidate.reject = "fit window clipped by the detector edge"
            else:
                groups.setdefault(band_window, []).append(k)

    for band_window, members in groups.items():
        _fit_band_group(
            model,
            [candidates[k] for k in members],
            [band_pixels[k] for k in members],
            band_window,
            arm,
            calibration,
            settings,
            wavelength,
        )
    return [candidates[k] for k in sorted(candidates)]


def extract_ring_points(
    workspace,
    wavelength: float,
    calibration: Optional[DENEXDetectorShift] = None,
    standards: Optional[Sequence] = None,
    sub_runs: Optional[Sequence[int]] = None,
    strips: int = 15,
    edge: int = 40,
    side_margin: int = 40,
    bin_width: float = 0.03,
    fit_half_width: float = 2.0,
    min_window_side: float = 0.75,
    peak_function: str = "PseudoVoigt",
    background_function: str = "Linear",
    max_center_error: float = 0.05,
    min_significance: float = 5.0,
    outlier_threshold: float = 5.0,
    smooth: bool = False,
    mask: Optional[np.ndarray] = None,
    apply_default_mask: bool = True,
    vanadium: Optional[np.ndarray] = None,
    diagnostics: Optional[List[StripFit]] = None,
) -> RingPoints:
    """Find where each powder ring crosses the detector, in fractional pixel coordinates.

    For each sub-run the detector is cut into ``strips`` bands along ``index_1`` (rows ``edge`` to
    ``n_rows - edge``). Each band is histogrammed in 2theta using ``calibration`` and normalised by
    the number of pixels (or the vanadium counts) per bin, so the varying pixel density across a
    curved ring does not skew the peak. Each reflection of the sub-run's powder standard is first
    located by one fit on all bands together (high statistics), which puts the band windows on the
    observed ring even when the starting wavelength or geometry is off; it is then used in every
    band where that ring crosses at least ``side_margin`` pixels from the left and right edges; its
    fit window is clipped to the band's coverage, and needs ``min_window_side`` degrees of data on
    each side of the peak. Bands sharing a window are fitted together (one Mantid ``FitPeaks`` call
    through PyRS's peak-fitting engine), and each fitted 2theta is converted back to a
    column on the band's centre row *with the same geometry*, which makes the ring point nearly
    independent of how good ``calibration`` is. The fitted point must also lie ``side_margin``
    pixels inside the edges. Failed or imprecise fits are dropped, and one pass of outlier rejection per
    ring removes points far from a smooth ring.

    Args:
        workspace: HidraWorkspace with raw counts and the ``2theta``, ``sy`` (and optionally
            ``L2``) sample logs, e.g. from ``NeXusConvertingApp``.
        wavelength: Wavelength (A) used to place the expected reflections.
        calibration: Current geometry estimate (None: nominal).
        standards: Powder standards (default: ``HB2B_STANDARDS``); sub-runs whose ``sy`` matches
            none are skipped with a warning.
        sub_runs: Sub-runs to use (default: all).
        strips: Number of bands per sub-run.
        edge: Pixels excluded at the detector's top and bottom edges.
        side_margin: Minimum distance (pixels) of a ring point from the left/right edges, both as
            predicted and as fitted.
        bin_width: Histogram bin width (deg).
        fit_half_width: Half-width (deg) of the fit window around each expected reflection.
        min_window_side: Minimum data (deg) on each side of the expected peak in a clipped window,
            so the background is still constrained next to the detector edge.
        peak_function: Peak shape for the fit.
        background_function: Background for the fit.
        max_center_error: Reject fits whose centre uncertainty exceeds this (deg).
        min_significance: Reject fits whose peak intensity is below this many of its own standard
            errors. Needed because a fit to pure noise can report a small centre error by chance.
        outlier_threshold: Reject points further than this many robust sigmas (min 0.1 px) from a
            quadratic through their ring.
        smooth: Replace each accepted point's column by that quadratic (same rows; no resampling).
        mask: Detector mask in pixel-ID order, 1 = use, 0 = ignore. Takes precedence over the
            workspace's default mask.
        apply_default_mask: Without ``mask``, use the workspace's default mask, if it has one. To
            apply the instrument mask, give its XML file to ``NeXusConvertingApp`` when converting
            the run, which stores it as that default mask. False uses every pixel.
        vanadium: Vanadium counts in pixel-ID order for efficiency normalisation.
        diagnostics: If a list is given, one :class:`StripFit` per band and reflection is appended.

    Returns:
        RingPoints, with ``sigma`` set to each fit's centre uncertainty (deg).

    Raises:
        ValueError: On invalid arguments, or if no ring point is found.
    """
    if not wavelength > 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    settings = _ExtractionSettings(
        strips=strips,
        edge=edge,
        side_margin=side_margin,
        bin_width=bin_width,
        fit_half_width=fit_half_width,
        min_window_side=min_window_side,
        peak_function=peak_function,
        background_function=background_function,
        max_center_error=max_center_error,
        min_significance=min_significance,
        outlier_threshold=outlier_threshold,
        smooth=smooth,
    )
    standards = HB2B_STANDARDS if standards is None else standards

    setup = workspace.get_instrument_setup()
    model = DetectorModel(setup)
    num_rows, num_columns = setup.detector_size
    if 2 * edge + strips > num_rows:
        raise ValueError(f"{strips} strips do not fit between edges of {edge} pixels")
    if mask is None and apply_default_mask:
        mask = workspace.get_detector_mask(True)
    weights = np.ones(num_columns * num_rows) if mask is None else np.asarray(mask, dtype=float).copy()
    efficiency = np.ones_like(weights) if vanadium is None else np.asarray(vanadium, dtype=float)
    if weights.size != num_columns * num_rows or efficiency.size != weights.size:
        raise ValueError("mask and vanadium must have one value per detector pixel")
    weights[efficiency <= 0] = 0.0
    weights = weights.reshape(num_columns, num_rows)
    efficiency = efficiency.reshape(num_columns, num_rows)

    grid = np.meshgrid(np.arange(num_columns), np.arange(num_rows), indexing="ij")
    bounds = np.linspace(edge, num_rows - edge, strips + 1).round().astype(int)
    rows_center = (bounds[:-1] + bounds[1:] - 1) / 2.0

    candidates: List[_Candidate] = []
    for sub_run in workspace.get_sub_runs() if sub_runs is None else sub_runs:
        sub_run = int(sub_run)
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), standards)
        if standard is None:
            warnings.warn(f"sub-run {sub_run}: sy matches no powder standard; skipped", stacklevel=2)
            continue
        det_2theta = float(workspace.get_detector_2theta(sub_run))
        l2 = workspace.get_l2(sub_run)
        l2 = setup.arm_length if l2 is None else float(l2)
        counts = workspace.get_detector_counts(sub_run).astype(float).reshape(num_columns, num_rows)
        two_theta = model.two_theta(grid[0], grid[1], -det_2theta, l2, calibration)
        band_pixels = _band_pixels(two_theta, counts, weights, efficiency, bounds)

        for hkl2, bragg in zip(standard.hkl_squared, standard.two_theta(wavelength)):
            candidates.extend(
                _extract_reflection(
                    model,
                    sub_run,
                    standard,
                    hkl2,
                    bragg,
                    band_pixels,
                    rows_center,
                    det_2theta,
                    l2,
                    calibration,
                    settings,
                    wavelength,
                )
            )

    _reject_outliers([c for c in candidates if not c.reject], settings.outlier_threshold, settings.smooth)

    if diagnostics is not None:
        diagnostics.extend(candidate.strip_fit() for candidate in candidates)

    accepted = [c for c in candidates if not c.reject]
    if not accepted:
        raise ValueError("no ring points found: check the wavelength, sy logs and fit settings")
    return RingPoints(**{name: np.asarray([getattr(c, name) for c in accepted]) for name in _RING_FIELDS})
