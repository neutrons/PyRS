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

Typical use::

    model = DetectorModel(hidra_ws.get_instrument_setup())
    two_theta = model.two_theta(index_0, index_1, -hidra_ws.get_detector_2theta(sub_run),
                                hidra_ws.get_l2(sub_run), DENEXDetectorShift(*params[:7]))
"""

from typing import Optional

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.utilities.convertdatatypes import to_float

__all__ = ["DetectorModel"]


class DetectorModel:
    """Lab-frame position, 2theta and eta of arbitrary (fractional) detector pixels.

    Pixel indices follow PyRS's 2D detector arrays, i.e.
    ``ResidualStressInstrument.get_pixels_2theta(1).reshape(n_columns, n_rows)`` and pixel ID
    ``index_0 * n_rows + index_1``: ``index_0`` runs across columns (X decreases with it) and
    ``index_1`` along rows (Y increases with it).

    The order of operations, and the rotation-matrix helpers, are those of
    ``ResidualStressInstrument.build_instrument``:

    1. shift the detector in X and Y;
    2. rotate about the detector centre by ``Rx @ Ry @ Rz``;
    3. push out along +Z by ``l2 + shift_z``;
    4. rotate about the sample (Y axis) by ``two_theta + two_theta_0``.

    Attributes:
        instrument_setup: Detector pixel counts, pixel sizes and default arm length.

    Example:
        >>> model = DetectorModel(DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False))
        >>> round(float(model.two_theta(511.5, 511.5, -90.0, None, None)), 6)  # detector centre
        90.0
    """

    def __init__(self, instrument_setup: DENEXDetectorGeometry) -> None:
        if not isinstance(instrument_setup, DENEXDetectorGeometry):
            raise TypeError(f"instrument_setup must be a DENEXDetectorGeometry, got {type(instrument_setup).__name__}")
        self.instrument_setup = instrument_setup
        # only used for its rotation helpers, so the arithmetic matches reduction exactly
        self._instrument = ResidualStressInstrument(instrument_setup)

    def local_positions(self, index_0, index_1) -> np.ndarray:
        """Uncalibrated pixel positions in the detector plane (Z = 0), in metres.

        Args:
            index_0: Column index or array of them (may be fractional).
            index_1: Row index or array of them, broadcastable against ``index_0``.

        Returns:
            Array of shape ``broadcast(index_0, index_1).shape + (3,)``.
        """
        num_rows, num_columns = self.instrument_setup.detector_size
        pixel_size_x, pixel_size_y = self.instrument_setup.pixel_dimension

        index_0, index_1 = np.broadcast_arrays(np.asarray(index_0, dtype=float), np.asarray(index_1, dtype=float))
        positions = np.zeros(index_0.shape + (3,), dtype=float)
        positions[..., 0] = (num_columns * 0.5 - 0.5) * pixel_size_x - index_0 * pixel_size_x
        positions[..., 1] = -(num_rows * 0.5 - 0.5) * pixel_size_y + index_1 * pixel_size_y

        return positions

    def lab_positions(
        self,
        index_0,
        index_1,
        two_theta: float,
        l2: Optional[float],
        calibration: Optional[DENEXDetectorShift],
    ) -> np.ndarray:
        """Lab-frame pixel positions for one detector arm position.

        Args:
            index_0: Column index or array of them (may be fractional).
            index_1: Row index or array of them.
            two_theta: Arm angle in degrees in the Mantid/PyRS convention, i.e. ``-get_detector_2theta()``.
            l2: Sample-to-detector distance in metres, or None for the instrument's arm length.
            calibration: Detector shift/rotation, or None for the nominal geometry.

        Returns:
            Array of shape ``broadcast(index_0, index_1).shape + (3,)``.
        """
        two_theta = to_float("2theta", two_theta)
        l2 = self.instrument_setup.arm_length if l2 is None else to_float("L2", l2, 1e-2)

        positions = self.local_positions(index_0, index_1)
        shape = positions.shape
        # ResidualStressInstrument._rotate_detector expects (N, M, 3)
        positions = positions.reshape(1, -1, 3)

        arm_l2 = l2
        if calibration is not None:
            if not isinstance(calibration, DENEXDetectorShift):
                raise TypeError(f"calibration must be a DENEXDetectorShift, got {type(calibration).__name__}")
            positions[:, :, 0] += calibration.center_shift_x
            positions[:, :, 1] += calibration.center_shift_y

            calib_matrix = self._instrument.generate_rotation_matrix(
                calibration.rotation_x * np.pi / 180.0,
                calibration.rotation_y * np.pi / 180.0,
                calibration.rotation_z * np.pi / 180.0,
            )
            positions = self._instrument._rotate_detector(positions, calib_matrix)

            two_theta += calibration.two_theta_0
            arm_l2 += calibration.center_shift_z

        positions[:, :, 2] += arm_l2

        if abs(two_theta) > 1.0e-7:
            two_theta_matrix = self._instrument._cal_rotation_matrix_y(np.deg2rad(two_theta))
            positions = self._instrument._rotate_detector(positions, two_theta_matrix)

        return positions.reshape(shape)

    def two_theta(self, index_0, index_1, two_theta: float, l2: Optional[float], calibration) -> np.ndarray:
        """Scattering angle 2theta (degrees) of the given pixels; arguments as :meth:`lab_positions`."""
        positions = self.lab_positions(index_0, index_1, two_theta, l2, calibration)
        norm = np.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2 + positions[..., 2] ** 2)
        return np.arccos(positions[..., 2] / norm) * 180 / np.pi

    def eta(self, index_0, index_1, two_theta: float, l2: Optional[float], calibration) -> np.ndarray:
        """Azimuthal angle eta (degrees, in (-180, 180]); arguments as :meth:`lab_positions`."""
        positions = self.lab_positions(index_0, index_1, two_theta, l2, calibration)
        eta = 180.0 - np.arctan2(positions[..., 1], positions[..., 0]) * 180 / np.pi
        return np.where(eta > 180.0, eta - 360, eta)
