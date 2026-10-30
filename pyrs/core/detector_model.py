"""
Forward model of the HB2B detector for the ring-based calibration.

:class:`DetectorModel` gives the lab position, 2theta and eta of arbitrary (fractional) detector
pixels for any trial calibration. It reproduces
:meth:`pyrs.core.reduce_hb2b_pyrs.ResidualStressInstrument.build_instrument` exactly (a unit test
pins bit-identity), so a calibration found with it means the same thing when reduction applies it;
see docs/ground_truths.md. Used by the ring-based calibration
(:mod:`pyrs.calibration.pixel_pos_calibration`); reduction itself does not use it (#1011).
"""

from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.utilities.convertdatatypes import to_float

__all__ = ["DetectorModel", "to_shift", "from_shift"]


def to_shift(values: Union[Sequence[float], np.ndarray]) -> DENEXDetectorShift:
    """The detector part (first 7 entries) of a calibration vector."""
    return DENEXDetectorShift(*values[:7])


def from_shift(shift: DENEXDetectorShift) -> np.ndarray:
    """The first 7 entries of a calibration vector, from a detector calibration."""
    return np.array(
        [
            shift.center_shift_x,
            shift.center_shift_y,
            shift.center_shift_z,
            shift.rotation_x,
            shift.rotation_y,
            shift.rotation_z,
            shift.two_theta_0,
        ]
    )


_GIMBAL_LIMIT_DEG = 89.0  # with_two_theta_0 refuses a Rot_y closer than this to +/-90 deg
_SHIFT_STEP_M = 1e-3


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

    # The only uses of ResidualStressInstrument's private helpers: reusing them keeps the arithmetic
    # bit-identical to reduction, and if they are renamed this is the one place to change (#1011/#1020).
    def _rotate(self, positions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return self._instrument._rotate_detector(positions, matrix)

    def _rotation_y(self, angle_rad: float) -> np.ndarray:
        return self._instrument._cal_rotation_matrix_y(angle_rad)

    def local_positions(self, index_0: ArrayLike, index_1: ArrayLike) -> np.ndarray:
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
        index_0: ArrayLike,
        index_1: ArrayLike,
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
            positions = self._rotate(positions, calib_matrix)

            two_theta += calibration.two_theta_0
            arm_l2 += calibration.center_shift_z

        positions[:, :, 2] += arm_l2

        if abs(two_theta) > 1.0e-7:  # the same threshold as build_instrument, to stay bit-identical
            two_theta_matrix = self._rotation_y(np.deg2rad(two_theta))
            positions = self._rotate(positions, two_theta_matrix)

        return positions.reshape(shape)

    def two_theta(
        self,
        index_0: ArrayLike,
        index_1: ArrayLike,
        two_theta: float,
        l2: Optional[float],
        calibration: Optional[DENEXDetectorShift],
    ) -> np.ndarray:
        """Scattering angle 2theta (degrees) of the given pixels; arguments as :meth:`lab_positions`."""
        positions = self.lab_positions(index_0, index_1, two_theta, l2, calibration)
        norm = np.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2 + positions[..., 2] ** 2)
        return np.arccos(positions[..., 2] / norm) * 180 / np.pi

    def eta(
        self,
        index_0: ArrayLike,
        index_1: ArrayLike,
        two_theta: float,
        l2: Optional[float],
        calibration: Optional[DENEXDetectorShift],
    ) -> np.ndarray:
        """Azimuthal angle eta (degrees, in (-180, 180]); arguments as :meth:`lab_positions`."""
        positions = self.lab_positions(index_0, index_1, two_theta, l2, calibration)
        eta = 180.0 - np.arctan2(positions[..., 1], positions[..., 0]) * 180 / np.pi
        return np.where(eta > 180.0, eta - 360, eta)

    def centre_position(self, calibration: Optional[DENEXDetectorShift], l2: float) -> np.ndarray:
        """Lab position (m) of the detector centre at arm angle 0.

        Arm angle 0, because the arm rotation is the same for every equivalent 2theta_0 description of
        the detector, so it drops out when two descriptions are compared.
        """
        num_rows, num_columns = self.instrument_setup.detector_size
        return self.lab_positions((num_columns - 1) / 2, (num_rows - 1) / 2, 0.0, l2, calibration)

    def with_two_theta_0(self, calibration: DENEXDetectorShift, two_theta_0: float, l2: float) -> DENEXDetectorShift:
        """The same detector described with a different 2theta_0.

        The arm is fixed; 2theta_0 (beam misalignment plus encoder error) turns the incident beam
        relative to the detector, which looks like a *virtual* swing of the detector about the
        sample: a uniform ring offset, a sideways move of the beam centre, a small distance change
        and a slight rotation. At one sample-detector distance that swing is exactly a change of
        the detector's rotations and shifts, so 2theta_0 can be traded for them without changing
        any pixel's 2theta or eta. This hands the virtual swing of the 2theta_0 change exactly to
        the rotations (``R -> Ry(-delta) @ R``, re-decomposed as ``Rx @ Ry @ Rz``) and re-solves
        the shifts so the detector centre stays where it was. See docs/ground_truths.md.

        Args:
            calibration: Current detector calibration.
            two_theta_0: New 2theta_0 (degrees).
            l2: Sample-detector distance (m) at which the equivalence is exact.

        Returns:
            Calibration with ``two_theta_0`` and compensated shifts and rotations.

        Raises:
            TypeError: If ``calibration`` is not a DENEXDetectorShift.
            ValueError: If the new rotation is within 1 deg of the Rx/Ry/Rz gimbal singularity
                (|Rot_y| = 90 deg), where the Rot_x/Rot_z split is ill-defined.
        """
        if not isinstance(calibration, DENEXDetectorShift):
            raise TypeError(f"calibration must be a DENEXDetectorShift, got {type(calibration).__name__}")
        two_theta_0 = to_float("2theta_0", two_theta_0)
        old = from_shift(calibration)
        delta = np.deg2rad(two_theta_0 - old[6])
        rotation = self._rotation_y(-delta) @ self._instrument.generate_rotation_matrix(*np.deg2rad(old[3:6]))
        if abs(rotation[0, 2]) > np.sin(np.deg2rad(_GIMBAL_LIMIT_DEG)):
            raise ValueError("compensated rotation is at the Rot_y = +/-90 deg singularity")

        new = old.copy()
        new[3] = np.rad2deg(np.arctan2(-rotation[1, 2], rotation[2, 2]))
        new[4] = np.rad2deg(np.arcsin(rotation[0, 2]))
        new[5] = np.rad2deg(np.arctan2(-rotation[0, 1], rotation[0, 0]))
        new[6] = two_theta_0

        # with the angles fixed the centre is affine in the three shifts, so one linear solve is exact
        target = self.centre_position(calibration, l2)
        base = self.centre_position(to_shift(new), l2)
        columns = []
        for k in range(3):
            moved = new.copy()
            moved[k] += _SHIFT_STEP_M
            columns.append((self.centre_position(to_shift(moved), l2) - base) / _SHIFT_STEP_M)
        new[:3] += np.linalg.solve(np.array(columns).T, target - base)

        return to_shift(new)
