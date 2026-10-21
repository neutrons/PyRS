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

1. Extract ring points (fractional pixel positions of fitted rings) into a :class:`RingPoints`.
2. Refine the geometry with :class:`PixelCalibration` stages, e.g. ``CalibrateShift()`` then
   ``CalibrateRotation()`` then ``calibrate_wave_shift()``.
3. Read the result from ``calibration_array`` / ``calibration_error_array``.

Example::

    calibration = PixelCalibration(ring_points, hidra_ws.get_instrument_setup(), wavelength=1.54)
    calibration.CalibrateShift()
    calibration.calibrate_wave_shift()
    shift_x, shift_y, shift_z, rot_x, rot_y, rot_z, two_theta_0, wavelength = calibration.calibration_array
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.utilities.convertdatatypes import to_float

__all__ = ["DetectorModel", "RingPoints", "PixelCalibration", "StageResult", "PARAMETER_NAMES"]

# order of the calibration vector, matching FitCalibration and the calibration JSON keys
PARAMETER_NAMES = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")


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
