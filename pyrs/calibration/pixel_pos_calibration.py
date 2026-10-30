"""
Ring-based ("pixel position") detector calibration for HB2B.

Instead of re-reducing the data and re-fitting peaks for every trial geometry (as
:class:`pyrs.calibration.mantid_peakfit_calibration.FitCalibration` does), this approach fits each
Debye ring once, in strips of the raw detector image, and records where on the detector the ring
lies (fractional pixel coordinates). The global geometry (shift x/y/z, rotation x/y/z, 2theta_0,
wavelength) is then refined by computing 2theta at those pixels for each trial geometry.

The detector geometry here deliberately reproduces
:meth:`pyrs.core.reduce_hb2b_pyrs.ResidualStressInstrument.build_instrument` rather than defining its
own convention, so a calibration found here means the same thing when reduction applies it. That
equivalence is pinned by a unit test; see docs/ground_truths.md.

The implementation is split by step, and everything public is re-exported here:

- :mod:`pyrs.calibration.detector_model`: :class:`DetectorModel`, the forward model;
- :mod:`pyrs.calibration.ring_points`: :class:`RingPoints`;
- :mod:`pyrs.calibration.ring_extraction`: :func:`extract_ring_points`, :class:`StripFit`;
- :mod:`pyrs.calibration.pixel_refinement`: :class:`PixelCalibration`, :class:`StageResult` and the
  parameter names, units, error codes and recipe defaults.

Typical workflow:

1. Extract ring points (fractional pixel positions of fitted rings) with :func:`extract_ring_points`,
   which fits every ring in strips of each sub-run's raw detector image.
2. Refine the geometry with :meth:`PixelCalibration.iterate` (repeated ``CalibrateShift`` ->
   ``CalibrateRotation`` -> ``calibrate_wave_shift`` passes, then a joint polish), or with
   individual stage methods.
3. Optionally fix the 2theta_0 / detector-pose split with :meth:`PixelCalibration.set_two_theta_0`:
   at one sample-detector distance the ring fit cannot determine it (2theta_0 is a virtual swing
   of the detector, see docs/ground_truths.md).
4. Read the result from ``calibration_array`` / ``calibration_error_array``.

Example::

    ring_points = extract_ring_points(hidra_ws, wavelength=1.54)   # e.g. from NeXusConvertingApp
    calibration = PixelCalibration(ring_points, hidra_ws.get_instrument_setup(), wavelength=1.54)
    calibration.iterate(passes=3)            # Shift -> Rotation -> lambda+2theta_0, x3, then polish
    calibration.set_two_theta_0(0.0)         # optional: same detector, described with 2theta_0 = 0
    shift_x, shift_y, shift_z, rot_x, rot_y, rot_z, two_theta_0, wavelength = calibration.calibration_array
"""

from pyrs.calibration.detector_model import DetectorModel
from pyrs.calibration.pixel_refinement import (
    DEFAULT_RECIPE,
    DEFAULT_TOLERANCE,
    NOT_REFINED,
    PARAMETER_NAMES,
    PARAMETER_UNITS,
    UNDETERMINED,
    PixelCalibration,
    StageResult,
)
from pyrs.calibration.ring_extraction import StripFit, extract_ring_points
from pyrs.calibration.ring_points import RingPoints

__all__ = [
    "DetectorModel",
    "RingPoints",
    "PixelCalibration",
    "StageResult",
    "PARAMETER_NAMES",
    "PARAMETER_UNITS",
    "NOT_REFINED",
    "UNDETERMINED",
    "DEFAULT_RECIPE",
    "DEFAULT_TOLERANCE",
    "StripFit",
    "extract_ring_points",
]
