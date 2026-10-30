"""
tests/util/ring_calibration_helpers.py

Synthetic ring points and synthetic detector images for testing the ring-based detector
calibration (:mod:`pyrs.calibration.pixel_pos_calibration`) without real data.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from pyrs.calibration.pixel_pos_calibration import DetectorModel, RingPoints
from pyrs.calibration.powder_standards import HB2B_STANDARDS, PowderStandard
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.peak_profile_utility import pseudo_voigt
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import HidraConstants

NI, FE, _MO = HB2B_STANDARDS

# Layout of calibration run HB2B_4971, read from the 2theta and sy sub-run logs of
# Calib_Testing/HB2B_4971.h5 (sy 62 = Ni, sy 12 = Fe). Each ring is walked across the detector:
# Ni 220 at 76/80, Fe 211 at 78-86, Ni 311 at 89-97, Ni 222 at 93/97, Fe 220 at 99/103.
RUN_4971_ARMS = (76.0, 80.0, 78.0, 80.0, 82.0, 86.0, 89.0, 93.0, 97.0, 99.0, 103.0)
RUN_4971_POWDERS = (NI, NI, FE, FE, FE, FE, NI, NI, NI, FE, FE)

# Alternative, for comparing scan designs: evenly spaced arms, alternating powders
UNIFORM_ARMS = tuple(np.linspace(76.0, 103.0, 11))
UNIFORM_POWDERS = tuple(NI if i % 2 == 0 else FE for i in range(11))


def synthetic_ring_points(
    instrument_setup: DENEXDetectorGeometry,
    true_params: Sequence[float],
    det_2theta: Sequence[float] = RUN_4971_ARMS,
    powders: Sequence[PowderStandard] = RUN_4971_POWDERS,
    strips: int = 10,
    edge: int = 40,
    l2: float = 0.985,
    noise_deg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> RingPoints:
    """Ring points that a perfect ring extraction would find for a known geometry.

    For each sub-run, reflection and strip row, the fractional column ``index_0`` where the pixel
    2theta equals the Bragg angle is solved for. Optional Gaussian noise (in 2theta, converted to
    columns) mimics peak-fit scatter.

    Args:
        instrument_setup: Detector geometry.
        true_params: Calibration vector ``[sx, sy, sz, rx, ry, rz, tth0, wavelength]``.
        det_2theta: Arm angle (DAS convention) of each sub-run.
        powders: Powder standard of each sub-run.
        strips: Number of strip rows per ring.
        edge: Pixels excluded at each detector edge.
        l2: Sample-detector distance (m) of every sub-run.
        noise_deg: Standard deviation of the ring-position noise, in 2theta degrees.
        rng: Random generator for the noise.

    Returns:
        RingPoints with ``sigma = noise_deg``.
    """
    model = DetectorModel(instrument_setup)
    calibration = DENEXDetectorShift(*true_params[:7])
    wavelength = true_params[7]
    num_rows, num_columns = instrument_setup.detector_size
    rows = edge + (np.arange(strips) + 0.5) * (num_rows - 2 * edge) / strips
    lo, hi = float(edge), float(num_columns - 1 - edge)
    rng = rng if rng is not None else np.random.default_rng(0)

    records = []
    for sub_run, (arm, powder) in enumerate(zip(det_2theta, powders), start=1):
        for d, hkl2, bragg in zip(powder.d_spacings(), powder.hkl_squared, powder.two_theta(wavelength)):
            for row in rows:

                def mismatch(column: float) -> float:
                    return float(model.two_theta(column, row, -arm, l2, calibration)) - bragg

                if mismatch(lo) * mismatch(hi) > 0:
                    continue  # this ring misses the detector at this row
                column = brentq(mismatch, lo, hi, xtol=1e-10)
                if noise_deg > 0:
                    step = mismatch(column + 0.5) - mismatch(column - 0.5)  # 2theta per column
                    column += rng.normal(0.0, noise_deg) / step
                records.append((sub_run, column, row, arm, l2, d, hkl2, powder.name))

    if not records:
        raise ValueError("no ring crosses the detector for these arm angles and powders")
    columns = list(zip(*records))
    return RingPoints(
        sub_run=np.asarray(columns[0]),
        index_0=np.asarray(columns[1]),
        index_1=np.asarray(columns[2]),
        det_2theta=np.asarray(columns[3]),
        l2=np.asarray(columns[4]),
        d_spacing=np.asarray(columns[5]),
        hkl_squared=np.asarray(columns[6]),
        powder=np.asarray(columns[7]),
        sigma=np.full(len(records), noise_deg),
    )


def concatenate_ring_points(*point_sets: RingPoints) -> RingPoints:
    """Join ring points of several runs, renumbering sub-runs so they stay distinct across runs.

    ``PixelCalibration`` groups points by their own arm angle and L2, so repeated sub-run numbers are
    safe; renumbering keeps the runs apart in diagnostics and plots (e.g. one scan at two
    sample-detector distances).
    """
    if not point_sets:
        raise ValueError("need at least one set of ring points")
    fields: Dict[str, List[np.ndarray]] = {name: [] for name in RingPoints.__dataclass_fields__}
    offset = 0
    for points in point_sets:
        for name in fields:
            values = getattr(points, name)
            fields[name].append(values + offset if name == "sub_run" else values)
        offset += int(points.sub_run.max())
    return RingPoints(**{name: np.concatenate(values) for name, values in fields.items()})


def synthetic_detector_image(
    instrument_setup: DENEXDetectorGeometry,
    true_params: Sequence[float],
    det_2theta: float,
    powder: PowderStandard,
    l2: float = 0.985,
    fwhm: float = 0.3,
    peak_height: float = 40.0,
    mixing: float = 0.5,
    background: float = 5.0,
    background_slope: float = 0.2,
    poisson: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Counts of one sub-run's powder exposure, rendered from a known detector geometry.

    Each pixel's 2theta comes from :class:`DetectorModel` with ``true_params``; every reflection
    of ``powder`` near the detector adds a pseudo-Voigt ring of the given height (counts per
    pixel) and FWHM (deg) at its Bragg angle, on a background sloping by ``background_slope``
    counts per degree across the detector.

    Args:
        instrument_setup: Detector geometry.
        true_params: Calibration vector ``[sx, sy, sz, rx, ry, rz, tth0, wavelength]``.
        det_2theta: Arm angle (DAS convention).
        powder: Powder standard in the beam.
        l2: Sample-detector distance (m).
        fwhm: Peak FWHM in 2theta (deg).
        peak_height: Peak height above background, in counts per pixel.
        mixing: Pseudo-Voigt Lorentzian fraction.
        background: Background at the arm angle, counts per pixel.
        background_slope: Background change per degree of 2theta.
        poisson: Draw Poisson counts; otherwise return the expected counts.
        rng: Random generator for the Poisson draw.

    Returns:
        1D counts in pixel-ID order (``index_0 * n_rows + index_1``).
    """
    model = DetectorModel(instrument_setup)
    num_rows, num_columns = instrument_setup.detector_size
    grid = np.meshgrid(np.arange(num_columns), np.arange(num_rows), indexing="ij")
    two_theta = model.two_theta(grid[0], grid[1], -det_2theta, l2, DENEXDetectorShift(*true_params[:7]))

    # pseudo_voigt takes the integrated intensity; convert from the requested peak height
    height_per_intensity = mixing * 2 / (np.pi * fwhm) + (1 - mixing) * np.sqrt(4 * np.log(2) / np.pi) / fwhm
    expected = background + background_slope * (two_theta - det_2theta)
    for bragg in powder.two_theta(true_params[7]):
        if two_theta.min() - 3 * fwhm < bragg < two_theta.max() + 3 * fwhm:
            expected = expected + pseudo_voigt(two_theta, peak_height / height_per_intensity, fwhm, mixing, bragg)
    expected = np.clip(expected, 0.0, None)

    if poisson:
        rng = rng if rng is not None else np.random.default_rng(0)
        expected = rng.poisson(expected).astype(float)
    return expected.reshape(-1)


def synthetic_calibration_workspace(
    instrument_setup: DENEXDetectorGeometry,
    true_params: Sequence[float],
    det_2theta: Sequence[float] = RUN_4971_ARMS,
    powders: Sequence[PowderStandard] = RUN_4971_POWDERS,
    l2: float = 0.985,
    **image_kwargs,
) -> HidraWorkspace:
    """A HidraWorkspace like a converted calibration run: one synthetic image per sub-run.

    Sub-runs are numbered from 1 and carry the ``2theta``, ``sy`` (the powder's position) and
    ``L2`` sample logs; ``image_kwargs`` go to :func:`synthetic_detector_image`.
    """
    workspace = HidraWorkspace("synthetic_calibration")
    sub_runs = np.arange(1, len(det_2theta) + 1)
    workspace.set_instrument_geometry(instrument_setup)
    workspace.set_sub_runs(sub_runs)
    workspace.set_sample_log(HidraConstants.TWO_THETA, sub_runs, np.asarray(det_2theta, dtype=float))
    workspace.set_sample_log("sy", sub_runs, np.array([powder.sy for powder in powders]))
    workspace.set_sample_log(HidraConstants.L2, sub_runs, np.full(sub_runs.size, l2))
    rng = image_kwargs.pop("rng", None) or np.random.default_rng(0)
    for sub_run, arm, powder in zip(sub_runs, det_2theta, powders):
        counts = synthetic_detector_image(instrument_setup, true_params, arm, powder, l2=l2, rng=rng, **image_kwargs)
        workspace.set_raw_counts(int(sub_run), counts)
    return workspace
