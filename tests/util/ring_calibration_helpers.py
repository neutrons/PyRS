"""
tests/util/ring_calibration_helpers.py

Synthetic ring points for testing the ring-based detector calibration
(:mod:`pyrs.calibration.pixel_pos_calibration`) without real data or Mantid.
"""

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from pyrs.calibration.pixel_pos_calibration import DetectorModel, RingPoints
from pyrs.calibration.powder_standards import HB2B_STANDARDS, PowderStandard
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

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
        sub_run=columns[0],
        index_0=columns[1],
        index_1=columns[2],
        det_2theta=columns[3],
        l2=columns[4],
        d_spacing=columns[5],
        hkl_squared=columns[6],
        powder=columns[7],
        sigma=np.full(len(records), noise_deg),
    )
