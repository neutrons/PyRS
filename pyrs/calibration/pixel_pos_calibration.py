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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.utilities.convertdatatypes import to_float

__all__ = [
    "DetectorModel",
    "RingPoints",
    "PixelCalibration",
    "StageResult",
    "PARAMETER_NAMES",
    "DEFAULT_RECIPE",
    "DEFAULT_TOLERANCE",
    "StripFit",
    "extract_ring_points",
]

# order of the calibration vector, matching FitCalibration and the calibration JSON keys
PARAMETER_NAMES = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")

# PixelCalibration.iterate(): the production-style staged workflow and its convergence test
DEFAULT_RECIPE = ("CalibrateShift", "CalibrateRotation", "calibrate_wave_shift")
DEFAULT_TOLERANCE = 1e-3  # per pass, in mm (shifts), degrees (angles) and mA (wavelength)
_REPORT_UNITS = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1.0, 1e3])
_RECIPE_STAGES = (
    "calibrate_wave_length",
    "calibrate_wave_shift",
    "calibrate_shiftx",
    "calibrate_shifty",
    "calibrate_distance",
    "calibrate_tth0",
    "CalibrateShift",
    "CalibrateRotation",
    "CalibrateGeometry",
    "FullCalibration",
)


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

    def centre_position(self, calibration: Optional[DENEXDetectorShift], l2: float) -> np.ndarray:
        """Lab position (m) of the detector centre at arm angle 0 (the arm rotation is common to every split)."""
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
            ValueError: If the new rotation is at the Rx/Ry/Rz gimbal singularity (|Rot_y| = 90 deg).
        """
        if not isinstance(calibration, DENEXDetectorShift):
            raise TypeError(f"calibration must be a DENEXDetectorShift, got {type(calibration).__name__}")
        two_theta_0 = to_float("2theta_0", two_theta_0)
        old = np.array(
            [
                calibration.center_shift_x,
                calibration.center_shift_y,
                calibration.center_shift_z,
                calibration.rotation_x,
                calibration.rotation_y,
                calibration.rotation_z,
                calibration.two_theta_0,
            ]
        )
        delta = np.deg2rad(two_theta_0 - old[6])
        rotation = self._instrument._cal_rotation_matrix_y(-delta) @ self._instrument.generate_rotation_matrix(
            *np.deg2rad(old[3:6])
        )
        if abs(rotation[0, 2]) > np.sin(np.deg2rad(89.0)):
            raise ValueError("compensated rotation is at the Rot_y = +/-90 deg singularity")

        new = old.copy()
        new[3] = np.rad2deg(np.arctan2(-rotation[1, 2], rotation[2, 2]))
        new[4] = np.rad2deg(np.arcsin(rotation[0, 2]))
        new[5] = np.rad2deg(np.arctan2(-rotation[0, 1], rotation[0, 0]))
        new[6] = two_theta_0

        # with the angles fixed the centre is affine in the three shifts, so one linear solve is exact
        target = self.centre_position(calibration, l2)
        base = self.centre_position(DENEXDetectorShift(*new), l2)
        columns = []
        for k in range(3):
            moved = new.copy()
            moved[k] += 1e-3
            columns.append((self.centre_position(DENEXDetectorShift(*moved), l2) - base) / 1e-3)
        new[:3] += np.linalg.solve(np.array(columns).T, target - base)

        return DENEXDetectorShift(*new)


_RING_FIELDS = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")


@dataclass
class RingPoints:
    """Points on fitted Debye rings: where on the detector each ring was found.

    All attributes are 1D arrays of equal length, one entry per point.

    Attributes:
        sub_run: Sub-run the point was measured in.
        index_0: Fractional pixel index across columns (see :class:`DetectorModel`).
        index_1: Fractional pixel index along rows.
        det_2theta: Detector arm 2theta (degrees) **as logged by DAS**, i.e. ``get_detector_2theta``.
        l2: Sample-to-detector distance (m) of the sub-run.
        d_spacing: True d-spacing (Angstrom) of the reflection, from the powder standard.
        hkl_squared: ``h^2 + k^2 + l^2`` of the reflection.
        powder: Powder standard name.
        sigma: Uncertainty of the fitted ring position in 2theta (degrees); NaN if unknown.

    Example:
        >>> points = RingPoints(sub_run=[1], index_0=[500.2], index_1=[512.0], det_2theta=[85.0],
        ...                     l2=[0.985], d_spacing=[1.2467], hkl_squared=[8], powder=["Ni"])
        >>> len(points)
        1
    """

    sub_run: np.ndarray
    index_0: np.ndarray
    index_1: np.ndarray
    det_2theta: np.ndarray
    l2: np.ndarray
    d_spacing: np.ndarray
    hkl_squared: np.ndarray
    powder: np.ndarray
    sigma: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.sub_run = np.asarray(self.sub_run, dtype=int)
        for name in ("index_0", "index_1", "det_2theta", "l2", "d_spacing"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.hkl_squared = np.asarray(self.hkl_squared, dtype=int)
        self.powder = np.asarray(self.powder, dtype=str)
        if self.sigma is None:
            self.sigma = np.full(self.sub_run.shape, np.nan)
        self.sigma = np.asarray(self.sigma, dtype=float)

        lengths = {name: getattr(self, name).shape for name in _RING_FIELDS}
        if any(len(shape) != 1 for shape in lengths.values()) or len(set(lengths.values())) != 1:
            raise ValueError(f"RingPoints fields must be 1D arrays of equal length, got shapes {lengths}")
        if np.any(self.d_spacing <= 0):
            raise ValueError("RingPoints d_spacing must be positive")

    def __len__(self) -> int:
        return int(self.sub_run.size)

    def select(self, mask: np.ndarray) -> "RingPoints":
        """Subset of the points, by boolean mask or index array."""
        return RingPoints(**{name: getattr(self, name)[mask] for name in _RING_FIELDS})

    def save(self, path: Union[str, Path]) -> None:
        """Write the points to a ``.npz`` file, so refinement can be re-run without re-fitting rings."""
        np.savez(path, **{name: getattr(self, name) for name in _RING_FIELDS})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RingPoints":
        """Read points written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            missing = set(_RING_FIELDS) - set(data.files)
            if missing:
                raise ValueError(f"{path} is not a RingPoints file: missing {sorted(missing)}")
            return cls(**{name: data[name] for name in _RING_FIELDS})


@dataclass(frozen=True)
class StageResult:
    """Outcome of one refinement stage.

    Attributes:
        name: Stage name, e.g. ``"CalibrateShift"``.
        free_parameters: Names of the refined parameters.
        values: Refined values of those parameters.
        errors: One-sigma uncertainties, or -2 where the fit could not determine them.
        status: ``scipy.optimize.least_squares`` status (> 0 converged, 0 hit ``max_nfev``).
        rmse: Root-mean-square 2theta residual (degrees) after the stage.
        nfev: Number of residual evaluations.
    """

    name: str
    free_parameters: Tuple[str, ...]
    values: np.ndarray
    errors: np.ndarray
    status: int
    rmse: float
    nfev: int


class PixelCalibration:
    """Refine the HB2B detector geometry from ring points.

    For a trial calibration vector ``[shift_x, shift_y, shift_z, rot_x, rot_y, rot_z, two_theta_0,
    wavelength]`` the residual of each ring point is the 2theta of its pixel (from
    :class:`DetectorModel`) minus the Bragg angle of its reflection, ``2 asin(wavelength / 2d)``
    with the *true* d-spacing. Using the true d (not one derived from the nominal wavelength) is
    what makes the wavelength determinable.

    Stage methods mirror :class:`~pyrs.calibration.mantid_peakfit_calibration.FitCalibration`
    (names and default bounds), so the two can be driven by the same recipes.

    Attributes:
        ring_points: The observations.
        stages: One :class:`StageResult` per completed stage, in order.

    Example:
        >>> calibration = PixelCalibration(ring_points, instrument_setup, wavelength=1.54)  # doctest: +SKIP
        >>> calibration.CalibrateShift()  # doctest: +SKIP
        >>> calibration.calibration_array  # doctest: +SKIP
    """

    def __init__(
        self,
        ring_points: RingPoints,
        instrument_setup: DENEXDetectorGeometry,
        wavelength: float,
        max_nfev: Optional[int] = None,
        method: str = "trf",
        diff_step: Optional[float] = None,
    ) -> None:
        """
        Args:
            ring_points: Fitted ring positions; must not be empty.
            instrument_setup: Detector geometry used to build the :class:`DetectorModel`.
            wavelength: Starting wavelength (Angstrom), usually the nominal mono setting.
            max_nfev: Maximum residual evaluations per stage (None: scipy's default).
            method: ``least_squares`` method: ``"trf"``, ``"dogbox"`` or ``"lm"`` (``"lm"`` ignores bounds).
            diff_step: Relative finite-difference step for the Jacobian (absolute step
                ``diff_step * max(1, |x|)``); None uses scipy's default. The default is accurate
                because the rotation matrices are float64 (see docs/ground_truths.md).

        Raises:
            ValueError: If there are no ring points, the wavelength is not positive or the method is unknown.
        """
        if not isinstance(ring_points, RingPoints):
            raise TypeError(f"ring_points must be RingPoints, got {type(ring_points).__name__}")
        if len(ring_points) == 0:
            raise ValueError("PixelCalibration needs at least one ring point")
        if not wavelength > 0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        if method not in ("trf", "dogbox", "lm"):
            raise ValueError(f"method must be 'trf', 'dogbox' or 'lm', got {method!r}")
        if diff_step is not None and not diff_step > 0:
            raise ValueError(f"diff_step must be positive, got {diff_step}")

        self.ring_points = ring_points
        self._model = DetectorModel(instrument_setup)
        self._max_nfev = max_nfev
        self._method = method
        self._diff_step = diff_step

        self._calib = np.zeros(8)
        self._calib[7] = wavelength
        self._caliberr = np.full(8, -1.0)
        self._status = -1
        self.stages: List[StageResult] = []

        # one detector position per sub-run, so 2theta is evaluated sub-run by sub-run
        self._groups = []
        for sub_run in np.unique(ring_points.sub_run):
            index = np.flatnonzero(ring_points.sub_run == sub_run)
            self._groups.append((index, -float(ring_points.det_2theta[index[0]]), float(ring_points.l2[index[0]])))

    @property
    def calibration_array(self) -> np.ndarray:
        """Current calibration vector (copy), in :data:`PARAMETER_NAMES` order."""
        return self._calib.copy()

    @property
    def calibration_error_array(self) -> np.ndarray:
        """One-sigma errors (copy); -1 for parameters never refined, -2 if undetermined."""
        return self._caliberr.copy()

    @property
    def calibration_status(self) -> int:
        """``least_squares`` status of the last stage; -1 if nothing was refined."""
        return self._status

    @property
    def residual_rmse(self) -> List[float]:
        """2theta residual RMS (degrees) after each stage."""
        return [stage.rmse for stage in self.stages]

    @property
    def residual_sum(self) -> List[float]:
        """Sum of squared 2theta residuals (degrees^2) after each stage."""
        return [stage.rmse**2 * len(self.ring_points) for stage in self.stages]

    def set_calibration_array(self, params: Sequence[float]) -> None:
        """Set the starting calibration vector (8 values, :data:`PARAMETER_NAMES` order)."""
        params = np.asarray(params, dtype=float)
        if params.shape != (8,):
            raise ValueError(f"calibration array must have 8 values, got shape {params.shape}")
        if not params[7] > 0:
            raise ValueError(f"wavelength must be positive, got {params[7]}")
        self._calib = params.copy()

    def set_two_theta_0(self, two_theta_0: float, l2: Optional[float] = None) -> StageResult:
        """Re-express the current calibration with a chosen 2theta_0, keeping the same detector.

        The ring fit cannot tell 2theta_0 apart from the shifts and rotations it implies (see
        :meth:`DetectorModel.with_two_theta_0`), so its value has to come from outside the fit,
        e.g. an independent beam-alignment measurement, or a convention such as 0. The wavelength
        and every ring point's 2theta are unchanged. Errors of the shifts, rotations and 2theta_0
        are reset to -1 because they referred to the old split; re-run a stage with TTH_0 held
        (e.g. ``refine("errors", [0, 1, 2, 3, 4, 5, 7], ...)``) to get uncertainties for this one.

        Args:
            two_theta_0: The 2theta_0 (degrees) to describe the detector with.
            l2: Distance (m) at which the swap is exact; default: the ring points' L2. Sub-runs at
                other distances see a slightly different detector (that is what breaks the
                degeneracy in a two-distance scan).

        Returns:
            A :class:`StageResult` recorded in :attr:`stages` (no refinement, ``nfev = 0``).
        """
        if l2 is None:
            l2 = float(np.median(self.ring_points.l2))
        calibration = self._model.with_two_theta_0(DENEXDetectorShift(*self._calib[:7]), two_theta_0, l2)
        self._calib[:7] = [
            calibration.center_shift_x,
            calibration.center_shift_y,
            calibration.center_shift_z,
            calibration.rotation_x,
            calibration.rotation_y,
            calibration.rotation_z,
            calibration.two_theta_0,
        ]
        self._caliberr[:7] = -1.0
        stage = StageResult(
            name="set_two_theta_0",
            free_parameters=(),
            values=np.array([]),
            errors=np.array([]),
            status=self._status,
            rmse=float(np.sqrt(np.mean(self.residual(self._calib) ** 2))),
            nfev=0,
        )
        self.stages.append(stage)
        return stage

    def model_two_theta(self, params: Sequence[float]) -> np.ndarray:
        """2theta (degrees) of every ring point's pixel for calibration vector ``params``."""
        calibration = DENEXDetectorShift(*params[:7])
        two_theta = np.empty(len(self.ring_points))
        points = self.ring_points
        for index, arm_two_theta, l2 in self._groups:
            two_theta[index] = self._model.two_theta(
                points.index_0[index], points.index_1[index], arm_two_theta, l2, calibration
            )
        return two_theta

    def residual(self, params: Sequence[float]) -> np.ndarray:
        """Per-point 2theta residual (degrees): pixel 2theta minus Bragg angle.

        Fixed length (one entry per ring point) and free of side effects, as ``least_squares``
        requires.
        """
        params = np.asarray(params, dtype=float)
        # clipped so an extreme trial wavelength gives a large residual rather than NaN
        sin_theta = np.clip(params[7] / (2 * self.ring_points.d_spacing), -1.0, 1.0)
        bragg_two_theta = 2 * np.rad2deg(np.arcsin(sin_theta))
        return self.model_two_theta(params) - bragg_two_theta

    def refine(
        self,
        name: str,
        free_parameters: Sequence[Union[int, str]],
        lower: Sequence[float],
        upper: Sequence[float],
        initial_guess: Optional[Sequence[float]] = None,
    ) -> StageResult:
        """Refine any subset of the calibration vector, holding the others at their current values.

        The fixed stage methods (``CalibrateShift`` etc.) call this; use it directly to explore
        other combinations, e.g. ``refine("no rot_y", ["Shift_x", "TTH_0", "Lambda"], lo, hi)``.

        Note:
            Shift_x, Rot_y and TTH_0 are degenerate as a set: turning the arm about the sample,
            turning the detector about its own vertical axis and shifting it sideways can describe
            the same detector position, so refining all three together does not determine them
            individually (the 2theta map still is). See docs/ground_truths.md.

        Args:
            name: Label recorded in :attr:`stages`.
            free_parameters: Indices or :data:`PARAMETER_NAMES` of the parameters to refine.
            lower: Lower bound for each free parameter.
            upper: Upper bound for each free parameter.
            initial_guess: Starting values (default: the current values).

        Returns:
            The stage's :class:`StageResult`; the calibration vector and errors are updated.

        Raises:
            ValueError: On unknown parameter names, mismatched lengths, a starting value outside
                the bounds, or no more ring points than free parameters.
        """
        try:
            free = np.array([PARAMETER_NAMES.index(p) if isinstance(p, str) else int(p) for p in free_parameters])
        except ValueError:
            raise ValueError(f"{name}: free parameters must be indices or names from {PARAMETER_NAMES}")
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        x0 = self._calib[free] if initial_guess is None else np.asarray(initial_guess, dtype=float)
        if not (x0.shape == lower.shape == upper.shape == free.shape):
            raise ValueError(f"{name}: initial guess and bounds must have {free.size} values")
        if len(self.ring_points) <= free.size:
            raise ValueError(f"{name}: needs more ring points ({len(self.ring_points)}) than free parameters")
        if self._method != "lm":
            outside = [PARAMETER_NAMES[i] for i, x, lo, up in zip(free, x0, lower, upper) if not lo <= x <= up]
            if outside:
                raise ValueError(f"{name}: starting value outside bounds for {outside}")

        fixed = self._calib.copy()

        def stage_residual(x: np.ndarray) -> np.ndarray:
            params = fixed.copy()
            params[free] = x
            return self.residual(params)

        if self._method == "lm":
            out = least_squares(
                stage_residual, x0, method="lm", jac="2-point", diff_step=self._diff_step, max_nfev=self._max_nfev
            )
        else:
            out = least_squares(
                stage_residual,
                x0,
                bounds=(lower, upper),
                method=self._method,
                jac="3-point",
                diff_step=self._diff_step,
                max_nfev=self._max_nfev,
            )

        dof = out.fun.size - free.size
        try:
            variances = np.diag(np.linalg.inv(out.jac.T @ out.jac)) * np.sum(out.fun**2) / dof
        except np.linalg.LinAlgError:
            variances = np.full(free.size, np.nan)
        # a (near-)degenerate set, e.g. Shift_x + Rot_y + TTH_0, can give negative or NaN variances
        determined = np.isfinite(variances) & (variances >= 0)
        errors = np.where(determined, np.sqrt(np.where(determined, variances, 0.0)), -2.0)

        self._calib[free] = out.x
        self._caliberr[free] = errors
        self._status = int(out.status)
        stage = StageResult(
            name=name,
            free_parameters=tuple(PARAMETER_NAMES[i] for i in free),
            values=out.x.copy(),
            errors=errors,
            status=int(out.status),
            rmse=float(np.sqrt(np.mean(out.fun**2))),
            nfev=int(out.nfev),
        )
        self.stages.append(stage)
        return stage

    # Stage methods: names, free parameters and default bounds follow FitCalibration.

    def calibrate_wave_length(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the wavelength within +/-0.025 A of its current value."""
        wl = self._calib[7]
        return self.refine("calibrate_wave_length", [7], [wl - 0.025], [wl + 0.025], initial_guess)

    def calibrate_wave_shift(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine 2theta_0 (+/-5 deg) and the wavelength (+/-0.025 A)."""
        wl = self._calib[7]
        return self.refine("calibrate_wave_shift", [6, 7], [-5.0, wl - 0.025], [5.0, wl + 0.025], initial_guess)

    def calibrate_shiftx(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the X shift (+/-0.05 m)."""
        return self.refine("calibrate_shiftx", [0], [-0.05], [0.05], initial_guess)

    def calibrate_shifty(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the Y shift (+/-0.05 m)."""
        return self.refine("calibrate_shifty", [1], [-0.05], [0.05], initial_guess)

    def calibrate_distance(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the Z shift, i.e. the sample-detector distance correction (+/-0.1 m)."""
        return self.refine("calibrate_distance", [2], [-0.1], [0.1], initial_guess)

    def calibrate_tth0(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the 2theta offset (+/-5 deg)."""
        return self.refine("calibrate_tth0", [6], [-5.0], [5.0], initial_guess)

    def CalibrateShift(
        self, initalGuess: Optional[Sequence[float]] = None, bounds: Optional[Sequence[Sequence[float]]] = None
    ) -> StageResult:
        """Refine the X, Y, Z shifts (default bounds +/-0.05, +/-0.05, +/-0.15 m)."""
        if bounds is None:
            bounds = [[-0.05, -0.05, -0.15], [0.05, 0.05, 0.15]]
        return self.refine("CalibrateShift", [0, 1, 2], bounds[0], bounds[1], initalGuess)

    def CalibrateRotation(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the X, Y, Z rotations (+/-5 deg)."""
        return self.refine("CalibrateRotation", [3, 4, 5], [-5.0] * 3, [5.0] * 3, initalGuess)

    def CalibrateGeometry(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine shifts (+/-0.05 m), rotations and 2theta_0 (+/-5 deg); the wavelength is held."""
        return self.refine(
            "CalibrateGeometry",
            range(7),
            [-0.05, -0.05, -0.05, -5.0, -5.0, -5.0, -5.0],
            [0.05, 0.05, 0.05, 5.0, 5.0, 5.0, 5.0],
            initalGuess,
        )

    def FullCalibration(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine all 8 parameters (shifts +/-0.05/0.05/0.15 m, angles +/-5 deg, wavelength +/-0.05 A)."""
        wl = self._calib[7]
        return self.refine(
            "FullCalibration",
            range(8),
            [-0.05, -0.05, -0.15, -5.0, -5.0, -5.0, -5.0, wl - 0.05],
            [0.05, 0.05, 0.15, 5.0, 5.0, 5.0, 5.0, wl + 0.05],
            initalGuess,
        )

    def estimate_errors(self, hold: Sequence[Union[int, str]] = ("TTH_0",)) -> StageResult:
        """Uncertainties of the calibration at its current values, with ``hold`` fixed.

        Shift_x, Rot_y and TTH_0 are exactly degenerate at one detector distance (see :meth:`refine`),
        so a refinement of all eight parameters cannot give their errors (they are flagged -2). This
        refines every parameter except ``hold``, starting from the current values (normally the
        optimum, so the values barely move), to get their one-sigma errors. The held parameters'
        errors are set to -1 ("not refined"): their values are a choice along the degenerate set, not
        a measurement (docs/ground_truths.md).

        Args:
            hold: Parameters (indices or :data:`PARAMETER_NAMES`) kept fixed; default 2theta_0.

        Returns:
            The :class:`StageResult` of the error refinement, recorded in :attr:`stages`.
        """
        held = [PARAMETER_NAMES.index(p) if isinstance(p, str) else int(p) for p in hold]
        free = [i for i in range(8) if i not in held]
        # bounds around the current values: the start is always inside, and wide enough not to bind
        span = np.array([0.05, 0.05, 0.15, 5.0, 5.0, 5.0, 5.0, 0.05])[free]
        current = self._calib[free]
        stage = self.refine("estimate_errors", free, current - span, current + span)
        self._caliberr[held] = -1.0
        return stage

    def write_calibration(self, file_name: Union[str, Path]) -> None:
        """Write the calibration in the standard calibration JSON format (``write_calibration_to_json``).

        Values and errors are :attr:`calibration_array` and :attr:`calibration_error_array`; ``Status``
        is the scipy status of the last stage.

        Raises:
            RuntimeError: If nothing has been refined yet (reduction rejects ``Status`` < 0).
        """
        from pyrs.utilities.calibration_file_io import write_calibration_to_json

        if self._status < 0:
            raise RuntimeError("nothing has been refined: refusing to write a calibration with Status -1")
        values, errors = self._calib, self._caliberr
        write_calibration_to_json(
            DENEXDetectorShift(*values[:7]),
            DENEXDetectorShift(*errors[:7]),
            float(values[7]),
            float(errors[7]),
            int(self._status),
            str(file_name),
        )

    def iterate(
        self,
        recipe: Sequence[str] = DEFAULT_RECIPE,
        passes: Optional[int] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        max_passes: int = 50,
        polish: bool = True,
    ) -> int:
        """Run a staged recipe repeatedly, then optionally one joint refinement of everything.

        This is the iterative calibration workflow: e.g. shifts, then rotations, then wavelength and
        2theta_0, repeated. Each stage holds the others' parameters, so later passes re-identify the
        shifts and rotations after 2theta_0 and the wavelength have moved. The final ``polish``
        (:meth:`FullCalibration`) settles directions the stages only creep along.

        Args:
            recipe: Stage method names run in order in each pass.
            passes: Run exactly this many passes; None repeats until no parameter changes by more
                than ``tolerance`` between passes.
            tolerance: Convergence threshold in mm (shifts), degrees (angles) and mA (wavelength).
            max_passes: Upper limit when ``passes`` is None.
            polish: Finish with :meth:`FullCalibration`.

        Returns:
            Number of passes run (excluding the polish).

        Raises:
            ValueError: On an unknown stage name or a non-positive pass count or tolerance.
        """
        unknown = [name for name in recipe if name not in _RECIPE_STAGES]
        if unknown or not recipe:
            raise ValueError(f"recipe stages must be from {sorted(_RECIPE_STAGES)}, got {list(recipe)}")
        if passes is not None and passes < 1:
            raise ValueError(f"passes must be at least 1, got {passes}")
        if not tolerance > 0 or max_passes < 1:
            raise ValueError("tolerance must be positive and max_passes at least 1")

        n_pass = 0
        for n_pass in range(1, (passes or max_passes) + 1):
            before = self._calib.copy()
            for name in recipe:
                getattr(self, name)()
            if passes is None and np.max(np.abs(self._calib - before) * _REPORT_UNITS) < tolerance:
                break
        else:
            if passes is None:
                warnings.warn(f"iterate: not converged to {tolerance} after {max_passes} passes", stacklevel=2)

        if polish:
            self.FullCalibration()
        return n_pass


# ---------------------------------------------------------------- ring extraction


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


def _column_at_two_theta(
    model: DetectorModel, target: float, row: float, arm_two_theta: float, l2: float, calibration
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
        here, step = model.two_theta(column, row, arm_two_theta, l2, calibration), 1e-3
        slope = (model.two_theta(column + step, row, arm_two_theta, l2, calibration) - here) / step
        column -= float((here - target) / slope)
    return column


def _reject_outliers(points: List[dict], threshold: float, smooth: bool) -> None:
    """Flag points far from a quadratic ``index_0(index_1)`` through each ring; optionally smooth."""
    rings = {}
    for point in points:
        rings.setdefault((point["sub_run"], point["hkl_squared"]), []).append(point)
    for ring in rings.values():
        if len(ring) < 4:
            continue
        rows = np.array([p["index_1"] for p in ring])
        columns = np.array([p["index_0"] for p in ring])
        # robust (soft L1) quadratic: a plain least-squares fit is dragged towards the outlier it
        # should expose, shrinking that point's residual and inflating everyone else's
        scaled = (rows - rows.mean()) / max(np.ptp(rows), 1.0)
        start = np.polyfit(scaled, columns, 2)
        fit = least_squares(lambda c: np.polyval(c, scaled) - columns, start, loss="soft_l1", f_scale=0.5)
        smooth_columns = np.polyval(fit.x, scaled)
        residual = columns - smooth_columns
        # floor so a near-perfect ring (simulated, noise-free) does not reject on rounding scatter
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 0.1)
        for point, res, fitted in zip(ring, residual, smooth_columns):
            if abs(res) > threshold * scale:
                point["reject"] = f"outlier ({res:+.2f} px from the ring)"
            elif smooth:
                point["index_0"] = float(fitted)


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
    from pyrs.core.workspaces import HidraWorkspace
    from pyrs.peaks import FitEngineFactory  # type: ignore
    from mantid.simpleapi import DeleteWorkspace, mtd

    centres = 0.5 * (edges[:-1] + edges[1:])
    fit_ws = HidraWorkspace(name)
    fit_ws.set_sub_runs(list(range(1, len(pixel_sets) + 1)))
    histograms = []
    for spectrum, (tth, weighted_counts, weighted_efficiency) in enumerate(pixel_sets, start=1):
        summed = np.histogram(tth, edges, weights=weighted_counts)[0]
        norm = np.histogram(tth, edges, weights=weighted_efficiency)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = np.where(norm > 0, summed / norm, 0.0)
            variance = np.where(norm > 0, np.maximum(summed, 1.0) / norm**2, 0.0)
        histograms.append(intensity)
        fit_ws.set_reduced_diffraction_data(spectrum, None, centres, intensity, variance)

    engine = FitEngineFactory.getInstance(
        fit_ws,
        peak_function,
        background_function,
        wavelength=wavelength,
        out_of_plane_angle=None,
        high_background=False,
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


def extract_ring_points(
    workspace,
    wavelength: float,
    calibration: Optional[DENEXDetectorShift] = None,
    standards: Sequence = None,
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
    band where that ring crosses at least ``side_margin`` pixels from the left and right edges; its fit window is clipped to the band's coverage, and needs ``min_window_side`` degrees
    of data on each side of the peak. Bands sharing a window are fitted together (Mantid
    ``FitPeaks`` with ``HighBackground=False``), and each fitted 2theta is converted back to a
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
    from pyrs.calibration.powder_standards import HB2B_STANDARDS, standard_for_sy

    if not wavelength > 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if strips < 1 or edge < 0 or side_margin < 0 or not bin_width > 0 or not fit_half_width > 0:
        raise ValueError("strips must be >= 1, edge and side_margin >= 0, and bin_width and fit_half_width positive")
    if not min_significance >= 0:
        raise ValueError(f"min_significance must be non-negative, got {min_significance}")
    if not 0 < min_window_side <= fit_half_width:
        raise ValueError(f"min_window_side must be in (0, fit_half_width], got {min_window_side}")
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
    rows_centre = (bounds[:-1] + bounds[1:] - 1) / 2.0

    points: List[dict] = []
    for sub_run in workspace.get_sub_runs() if sub_runs is None else sub_runs:
        sub_run = int(sub_run)
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), standards)
        if standard is None:
            warnings.warn(f"sub-run {sub_run}: sy matches no powder standard; skipped", stacklevel=2)
            continue
        det_2theta = float(workspace.get_detector_2theta(sub_run))
        l2 = workspace.get_l2(sub_run)
        l2 = setup.arm_length if l2 is None else float(l2)
        arm = -det_2theta
        counts = workspace.get_detector_counts(sub_run).astype(float).reshape(num_columns, num_rows)
        two_theta = model.two_theta(*grid, arm, l2, calibration)

        bands = [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
        band_pixels = []  # per band: 2theta, counts and efficiency of its unmasked pixels
        for band in bands:
            keep = weights[:, band].ravel() > 0
            band_pixels.append(
                (
                    two_theta[:, band].ravel()[keep],
                    weights[:, band].ravel()[keep] * counts[:, band].ravel()[keep],
                    weights[:, band].ravel()[keep] * efficiency[:, band].ravel()[keep],
                )
            )

        combined = tuple(np.concatenate([pixels[j] for pixels in band_pixels]) for j in range(3))

        def window(centre: float, tth: np.ndarray) -> Optional[Tuple[int, int]]:
            """Fit window around ``centre`` clipped to the pixels' coverage, on whole bins; None if
            either side keeps less than ``min_window_side`` of data."""
            low = int(np.ceil(max(centre - fit_half_width, tth.min()) / bin_width))
            high = int(np.floor(min(centre + fit_half_width, tth.max()) / bin_width))
            if centre - low * bin_width < min_window_side or high * bin_width - centre < min_window_side:
                return None
            return low, high

        for hkl2, bragg in zip(standard.hkl_squared, standard.two_theta(wavelength)):
            # pass 1: locate the ring on all unmasked bands together, where the signal is strongest,
            # so the strip windows (and the fits' starting centre: the window middle) sit on it even
            # when the starting wavelength or geometry puts the prediction a peak width or more off
            target, located = bragg, False
            locate_window = window(bragg, combined[0]) if combined[0].size else None
            if locate_window is not None:
                edges = np.arange(locate_window[0], locate_window[1] + 1) * bin_width
                _, _, centre, _, significance, excluded = _fit_band_histograms(
                    f"ring_{sub_run}_{hkl2}_locate", [combined], edges, peak_function, background_function, wavelength
                )
                if not excluded[0] and significance[0] >= min_significance and edges[0] < centre[0] < edges[-1]:
                    target, located = float(centre[0]), True

            # pass 2: choose bands for this reflection: a curved ring can be near the side edge in some only
            records, groups = {}, {}
            for k in range(strips):
                predicted = _column_at_two_theta(model, target, rows_centre[k], arm, l2, calibration)
                if not np.isfinite(predicted):
                    continue  # the ring does not cross this band
                records[k] = {
                    "sub_run": sub_run,
                    "index_1": rows_centre[k],
                    "det_2theta": det_2theta,
                    "l2": l2,
                    "d_spacing": standard.lattice_a / np.sqrt(hkl2),
                    "hkl_squared": int(hkl2),
                    "powder": standard.name,
                    "sigma": np.nan,
                    "index_0": np.nan,
                    "reject": "",
                    "fit": (k, bragg, target if located else np.nan, np.array([]), np.array([]), np.nan),
                }
                tth = band_pixels[k][0]
                if tth.size == 0:
                    records[k]["reject"] = "band has no unmasked pixels"
                    continue
                if not side_margin <= predicted <= num_columns - 1 - side_margin:
                    records[k]["reject"] = f"ring predicted within {side_margin} px of the side edge"
                    continue
                band_window = window(target, tth)
                if band_window is None:
                    records[k]["reject"] = "fit window clipped by the detector edge"
                    continue
                groups.setdefault(band_window, []).append(k)

            # bands with the same window are fitted together (one FitPeaks call, bands as spectra)
            for (low, high), members in groups.items():
                edges = np.arange(low, high + 1) * bin_width
                centres, histograms, centre, centre_error, significance, excluded = _fit_band_histograms(
                    f"ring_{sub_run}_{hkl2}_{low}",
                    [band_pixels[k] for k in members],
                    edges,
                    peak_function,
                    background_function,
                    wavelength,
                )
                for i, k in enumerate(members):
                    point = records[k]
                    center, error = float(centre[i]), float(centre_error[i])
                    point["sigma"] = error
                    point["fit"] = (k, bragg, target if located else np.nan, centres, histograms[i], center)
                    failed = excluded[i] or not np.isfinite(center) or not np.isfinite(error) or error <= 0
                    if failed or not np.isfinite(significance[i]):  # no valid intensity uncertainty
                        point["reject"] = "fit failed"
                    elif significance[i] < min_significance:
                        point["reject"] = f"peak not significant ({significance[i]:.1f} < {min_significance} sigma)"
                    elif error > max_center_error:
                        point["reject"] = f"centre error {error:.3f} > {max_center_error} deg"
                    elif not edges[0] < center < edges[-1]:
                        point["reject"] = "centre outside the fit window"
                    else:
                        column = _column_at_two_theta(model, center, rows_centre[k], arm, l2, calibration)
                        point["index_0"] = column
                        if not np.isfinite(column):
                            point["reject"] = "ring off the detector on this row"
                        elif not side_margin <= column <= num_columns - 1 - side_margin:
                            # the window was placed from pass 1 or the prediction; check the fitted point too
                            point["reject"] = f"fitted ring within {side_margin} px of the side edge"

            points.extend(records[k] for k in sorted(records))

    _reject_outliers([p for p in points if not p["reject"]], outlier_threshold, smooth)

    if diagnostics is not None:
        for p in points:
            k, bragg, start_center, centres, intensity, center = p["fit"]
            diagnostics.append(
                StripFit(
                    sub_run=p["sub_run"],
                    strip=k,
                    row=p["index_1"],
                    powder=p["powder"],
                    hkl_squared=p["hkl_squared"],
                    bragg_two_theta=bragg,
                    start_center=start_center,
                    two_theta=centres,
                    intensity=intensity,
                    center=center,
                    center_error=p["sigma"],
                    index_0=np.nan if p["reject"] else p["index_0"],
                    accepted=not p["reject"],
                    reason=p["reject"],
                )
            )

    accepted = [p for p in points if not p["reject"]]
    if not accepted:
        raise ValueError("no ring points found: check the wavelength, sy logs and fit settings")
    return RingPoints(**{name: [p[name] for p in accepted] for name in _RING_FIELDS})
