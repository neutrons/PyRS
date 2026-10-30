"""
Refinement of the HB2B detector geometry and wavelength from ring points.

:class:`PixelCalibration` refines the 8-parameter calibration vector (:data:`PARAMETER_NAMES`) by
least squares on the 2theta residual of every ring point against its Bragg angle, in stages that
mirror :class:`pyrs.calibration.mantid_peakfit_calibration.FitCalibration`. Error-array codes are
:data:`NOT_REFINED` and :data:`UNDETERMINED`. Part of :mod:`pyrs.calibration.pixel_pos_calibration`.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative  # the finite-difference Jacobian least_squares uses

from pyrs.core.detector_model import DetectorModel, from_shift, to_shift
from pyrs.calibration.ring_points import RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

__all__ = [
    "PixelCalibration",
    "StageResult",
    "PARAMETER_NAMES",
    "PARAMETER_UNITS",
    "NOT_REFINED",
    "UNDETERMINED",
    "DEFAULT_RECIPE",
    "DEFAULT_TOLERANCE",
]

# order of the calibration vector, matching FitCalibration and the calibration JSON keys
PARAMETER_NAMES = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")
# display unit of each parameter and the factor from its stored unit (m, deg, A)
PARAMETER_UNITS = (("mm", 1e3),) * 3 + (("deg", 1.0),) * 4 + (("A", 1.0),)

# values of the error array that are codes, not uncertainties
NOT_REFINED = -1.0  # never refined, or held (e.g. 2theta_0 in estimate_errors)
UNDETERMINED = -2.0  # refined, but the fit cannot determine it (degenerate, or stopped at a bound)

# PixelCalibration.iterate(): the production-style staged workflow and its convergence test
DEFAULT_RECIPE = ("CalibrateShift", "CalibrateRotation", "calibrate_wave_shift")
DEFAULT_TOLERANCE = 1e-3  # per pass, in mm (shifts), degrees (angles) and mÅ (wavelength)
_TOLERANCE_UNITS = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1.0, 1e3])

# Stage methods of PixelCalibration: free parameters and bound half-widths, following
# FitCalibration. Bounds are +/- the half-width about zero for the shifts (m) and angles (deg), and
# about the current value for the wavelength (A). CalibrateGeometry's +/-0.05 m on Shift_z (the other
# stages allow 0.15 m) is FitCalibration's.
_STAGES: Dict[str, Tuple[Tuple[int, ...], Tuple[float, ...]]] = {
    "calibrate_wave_length": ((7,), (0.025,)),
    "calibrate_wave_shift": ((6, 7), (5.0, 0.025)),
    "calibrate_shiftx": ((0,), (0.05,)),
    "calibrate_shifty": ((1,), (0.05,)),
    "calibrate_distance": ((2,), (0.1,)),
    "calibrate_tth0": ((6,), (5.0,)),
    "CalibrateShift": ((0, 1, 2), (0.05, 0.05, 0.15)),
    "CalibrateRotation": ((3, 4, 5), (5.0, 5.0, 5.0)),
    "CalibrateGeometry": ((0, 1, 2, 3, 4, 5, 6), (0.05, 0.05, 0.05, 5.0, 5.0, 5.0, 5.0)),
    "FullCalibration": (tuple(range(8)), (0.05, 0.05, 0.15, 5.0, 5.0, 5.0, 5.0, 0.05)),
}


_METHODS = ("trf", "dogbox", "lm")  # scipy.optimize.least_squares methods


def _check_method(method: str) -> None:
    """Raise ValueError unless ``method`` is one of :data:`_METHODS`."""
    if method not in _METHODS:
        raise ValueError(f"method must be 'trf', 'dogbox' or 'lm', got {method!r}")


def _rmse(residual: np.ndarray) -> float:
    """Root-mean-square of a residual vector."""
    return float(np.sqrt(np.mean(residual**2)))


def _parameter_indices(parameters: Sequence[Union[int, str]], context: str) -> np.ndarray:
    """Indices into the calibration vector of ``parameters`` (indices or :data:`PARAMETER_NAMES`)."""
    indices = []
    for parameter in parameters:
        if isinstance(parameter, str):
            if parameter not in PARAMETER_NAMES:
                raise ValueError(f"{context}: unknown parameter {parameter!r}; use one of {PARAMETER_NAMES}")
            indices.append(PARAMETER_NAMES.index(parameter))
        elif isinstance(parameter, (int, np.integer)) and 0 <= parameter < len(PARAMETER_NAMES):
            indices.append(int(parameter))
        else:
            raise ValueError(
                f"{context}: parameters must be names from {PARAMETER_NAMES} or indices 0-7, got {parameter!r}"
            )
    return np.array(indices, dtype=int)


_RANK_RTOL = 1e-6  # Shift_x/Rot_y/TTH_0 at one L2: ~4e-10 relative; the next-weakest direction ~5e-2


def _parameter_errors(jacobian: np.ndarray, residual: np.ndarray, dof: int) -> np.ndarray:
    """One-sigma errors ``sqrt(diag(inv(JᵀJ)) · Σr² / dof)``, with -2 for undetermined parameters.

    Columns are scaled to unit norm first, so parameters in m, degrees and Å are compared fairly. If
    the scaled Jacobian is rank deficient (an exact degeneracy such as Shift_x + Rot_y + TTH_0 at one
    detector distance), every parameter whose removal restores the rank is undetermined and flagged
    -2, and the others' errors come from the pseudo-inverse, so the degenerate direction cannot
    inflate them. At full rank this is exactly the ``inv(JᵀJ)`` formula.
    """
    scale = np.linalg.norm(jacobian, axis=0)
    if dof <= 0 or not np.all(np.isfinite(jacobian)) or np.any(scale == 0):
        return np.full(jacobian.shape[1], UNDETERMINED)
    scaled = jacobian / scale

    def rank(matrix: np.ndarray) -> int:
        singular = np.linalg.svd(matrix, compute_uv=False)
        return int(np.sum(singular > _RANK_RTOL * singular[0]))

    full_rank = rank(scaled)
    undetermined = np.zeros(scaled.shape[1], dtype=bool)
    if full_rank < scaled.shape[1]:
        for column in range(scaled.shape[1]):
            if rank(np.delete(scaled, column, axis=1)) == full_rank:
                undetermined[column] = True
    covariance = np.linalg.pinv(scaled.T @ scaled, rcond=_RANK_RTOL**2, hermitian=True)
    variances = np.diag(covariance) / scale**2 * np.sum(residual**2) / dof
    return np.where(undetermined | ~(variances >= 0), UNDETERMINED, np.sqrt(np.abs(variances)))


@dataclass(frozen=True)
class StageResult:
    """Outcome of one refinement stage.

    Attributes:
        name: Stage name, e.g. ``"CalibrateShift"``.
        free_parameters: Names of the refined parameters.
        values: Refined values of those parameters.
        errors: One-sigma uncertainties, or :data:`UNDETERMINED` where the fit could not determine them.
        status: ``scipy.optimize.least_squares`` status (> 0 converged, 0 hit ``max_nfev``); stages
            that refine nothing (``set_two_theta_0``, ``errors_at_current``) carry the previous one,
            -1 if nothing has been refined.
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
            TypeError: If ``ring_points`` is not a :class:`RingPoints`.
            ValueError: If there are no ring points, the wavelength is not positive, the method is
                unknown or ``diff_step`` is not positive.
        """
        if not isinstance(ring_points, RingPoints):
            raise TypeError(f"ring_points must be RingPoints, got {type(ring_points).__name__}")
        if len(ring_points) == 0:
            raise ValueError("PixelCalibration needs at least one ring point")
        if not wavelength > 0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        _check_method(method)
        if diff_step is not None and not diff_step > 0:
            raise ValueError(f"diff_step must be positive, got {diff_step}")

        self.ring_points = ring_points
        self._model = DetectorModel(instrument_setup)
        self._max_nfev = max_nfev
        self._method = method
        self._diff_step = diff_step

        self._calib = np.zeros(8)
        self._calib[7] = wavelength
        self._caliberr = np.full(8, NOT_REFINED)
        self._status = -1
        self.stages: List[StageResult] = []
        # every warning a stage issued (stopped at a bound, not converged), also sent to ``warnings``;
        # callers read it instead of capturing warnings, which is not thread-safe
        self.warning_log: List[str] = []

        # 2theta is evaluated once per detector position (arm angle, L2), taken from the points
        # themselves rather than their sub-run number, which can repeat when runs are combined
        self._groups = []
        positions = np.column_stack([ring_points.det_2theta, ring_points.l2]).astype(float)
        for arm_two_theta, l2 in np.unique(positions, axis=0):
            index = np.flatnonzero((positions[:, 0] == arm_two_theta) & (positions[:, 1] == l2))
            self._groups.append((index, -float(arm_two_theta), float(l2)))

    @property
    def calibration_array(self) -> np.ndarray:
        """Current calibration vector (copy), in :data:`PARAMETER_NAMES` order."""
        return self._calib.copy()

    @property
    def calibration_error_array(self) -> np.ndarray:
        """One-sigma errors (copy); :data:`NOT_REFINED` (-1) or :data:`UNDETERMINED` (-2) where there is none."""
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

    def set_calibration_array(self, params: Union[Sequence[float], np.ndarray]) -> None:
        """Set the starting calibration vector (8 values, :data:`PARAMETER_NAMES` order).

        The errors and the status are reset (they described the previous values), so a calibration
        is not written with errors of values it no longer has.
        """
        values = np.asarray(params, dtype=float)
        if values.shape != (8,):
            raise ValueError(f"calibration array must have 8 values, got shape {values.shape}")
        if not values[7] > 0:
            raise ValueError(f"wavelength must be positive, got {values[7]}")
        self._calib = values.copy()
        self._caliberr = np.full(8, NOT_REFINED)
        self._status = -1

    def set_refinement(self, method: str = "trf", max_nfev: Optional[int] = None) -> None:
        """Change the ``least_squares`` method and evaluation limit used by later stages.

        Raises:
            ValueError: If ``method`` is not ``"trf"``, ``"dogbox"`` or ``"lm"``, or ``max_nfev`` < 1.
        """
        _check_method(method)
        if max_nfev is not None and int(max_nfev) < 1:
            raise ValueError(f"max_nfev must be at least 1, got {max_nfev}")
        self._method = method
        self._max_nfev = None if max_nfev is None else int(max_nfev)

    def checkpoint(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """The current values, errors, status and number of stages, for :meth:`restore`."""
        return self._calib.copy(), self._caliberr.copy(), self._status, len(self.stages)

    def restore(self, checkpoint: Tuple[np.ndarray, np.ndarray, int, int]) -> None:
        """Return to a :meth:`checkpoint`: its values, errors and status, dropping later stages."""
        values, errors, status, n_stages = checkpoint
        self._calib, self._caliberr, self._status = values.copy(), errors.copy(), status
        del self.stages[n_stages:]

    def _warn(self, message: str) -> None:
        self.warning_log.append(message)
        warnings.warn(message, stacklevel=3)

    def set_two_theta_0(self, two_theta_0: float, l2: Optional[float] = None) -> StageResult:
        """Re-express the current calibration with a chosen 2theta_0, keeping the same detector.

        The ring fit cannot tell 2theta_0 apart from the shifts and rotations it implies (see
        :meth:`DetectorModel.with_two_theta_0`), so its value has to come from outside the fit,
        e.g. an independent beam-alignment measurement, or a convention such as 0. The wavelength
        and every ring point's 2theta are unchanged. Errors of the shifts, rotations and 2theta_0
        are reset to -1 because they referred to the old split; :meth:`estimate_errors` (which holds
        TTH_0) gives uncertainties for this one.

        Args:
            two_theta_0: The 2theta_0 (degrees) to describe the detector with.
            l2: Distance (m) at which the swap is exact; default: the median L2 of the ring points.
                Sub-runs at other distances see a slightly different detector (that is what breaks
                the degeneracy in a two-distance scan).

        Returns:
            A :class:`StageResult` recorded in :attr:`stages` (no refinement, ``nfev = 0``).

        Raises:
            ValueError: If the new rotation is within 1 degree of Rot_y = +/-90 degrees.
        """
        if l2 is None:
            l2 = float(np.median(self.ring_points.l2))
        calibration = self._model.with_two_theta_0(to_shift(self._calib), two_theta_0, l2)
        self._calib[:7] = from_shift(calibration)
        self._caliberr[:7] = NOT_REFINED
        stage = StageResult(
            name="set_two_theta_0",
            free_parameters=(),
            values=np.array([]),
            errors=np.array([]),
            status=self._status,
            rmse=_rmse(self.residual(self._calib)),
            nfev=0,
        )
        self.stages.append(stage)
        return stage

    def model_two_theta(self, params: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """2theta (degrees) of every ring point's pixel for calibration vector ``params``."""
        calibration = to_shift(params)
        two_theta = np.empty(len(self.ring_points))
        points = self.ring_points
        for index, arm_two_theta, l2 in self._groups:
            two_theta[index] = self._model.two_theta(
                points.index_0[index], points.index_1[index], arm_two_theta, l2, calibration
            )
        return two_theta

    def residual(self, params: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """Per-point 2theta residual (degrees): pixel 2theta minus Bragg angle.

        Fixed length (one entry per ring point) and free of side effects, as ``least_squares``
        requires.
        """
        values = np.asarray(params, dtype=float)
        # clipped so an extreme trial wavelength gives a large residual rather than NaN
        sin_theta = np.clip(values[7] / (2 * self.ring_points.d_spacing), -1.0, 1.0)
        bragg_two_theta = 2 * np.rad2deg(np.arcsin(sin_theta))
        return self.model_two_theta(values) - bragg_two_theta

    def refine(
        self,
        name: str,
        free_parameters: Sequence[Union[int, str]],
        lower: Union[Sequence[float], np.ndarray],
        upper: Union[Sequence[float], np.ndarray],
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
            ValueError: On unknown parameters, mismatched lengths, a lower bound not below its upper
                bound, a starting value outside the bounds, or no more ring points than free parameters.
        """
        free = _parameter_indices(free_parameters, name)
        lower_bounds = np.asarray(lower, dtype=float)
        upper_bounds = np.asarray(upper, dtype=float)
        x0 = self._calib[free] if initial_guess is None else np.asarray(initial_guess, dtype=float)
        if not (x0.shape == lower_bounds.shape == upper_bounds.shape == free.shape):
            raise ValueError(f"{name}: initial guess and bounds must have {free.size} values")
        if not np.all(lower_bounds < upper_bounds):
            raise ValueError(f"{name}: every lower bound must be below its upper bound")
        if len(self.ring_points) <= free.size:
            raise ValueError(f"{name}: needs more ring points ({len(self.ring_points)}) than free parameters")
        if self._method != "lm":
            outside = [
                PARAMETER_NAMES[i] for i, x, lo, up in zip(free, x0, lower_bounds, upper_bounds) if not lo <= x <= up
            ]
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
                bounds=(lower_bounds, upper_bounds),
                method=self._method,
                jac="3-point",
                diff_step=self._diff_step,
                max_nfev=self._max_nfev,
            )

        errors = _parameter_errors(out.jac, out.fun, out.fun.size - free.size)
        # a parameter stopped by its bound is not at a minimum, so its covariance error means nothing
        at_bound = np.asarray(out.active_mask) != 0
        if np.any(at_bound):
            errors = np.where(at_bound, UNDETERMINED, errors)
            pinned = [PARAMETER_NAMES[i] for i in free[at_bound]]
            self._warn(f"{name}: {pinned} stopped at a bound; their errors are flagged -2")
        if out.status == 0:
            self._warn(f"{name}: stopped at max_nfev ({self._max_nfev}) before converging (status 0)")

        self._calib[free] = out.x
        self._caliberr[free] = errors
        self._status = int(out.status)
        stage = StageResult(
            name=name,
            free_parameters=tuple(PARAMETER_NAMES[i] for i in free),
            values=out.x.copy(),
            errors=errors,
            status=int(out.status),
            rmse=_rmse(out.fun),
            nfev=int(out.nfev),
        )
        self.stages.append(stage)
        return stage

    # Stage methods: names, free parameters and default bounds follow FitCalibration (see _STAGES).
    # ``initalGuess`` (sic) is FitCalibration's keyword, kept so the two are drop-in comparable.

    def _stage_bounds(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Default bounds of stage ``name``: about 0 for shifts and angles, about the current wavelength."""
        free, half_width = _STAGES[name]
        centre = np.where(np.asarray(free) == 7, self._calib[7], 0.0)
        return centre - np.asarray(half_width), centre + np.asarray(half_width)

    def _run_stage(self, name: str, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        lower, upper = self._stage_bounds(name)
        return self.refine(name, _STAGES[name][0], lower, upper, initial_guess)

    def calibrate_wave_length(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the wavelength within +/-0.025 A of its current value."""
        return self._run_stage("calibrate_wave_length", initial_guess)

    def calibrate_wave_shift(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine 2theta_0 (+/-5 deg) and the wavelength (+/-0.025 A)."""
        return self._run_stage("calibrate_wave_shift", initial_guess)

    def calibrate_shiftx(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the X shift (+/-0.05 m)."""
        return self._run_stage("calibrate_shiftx", initial_guess)

    def calibrate_shifty(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the Y shift (+/-0.05 m)."""
        return self._run_stage("calibrate_shifty", initial_guess)

    def calibrate_distance(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the Z shift, i.e. the sample-detector distance correction (+/-0.1 m)."""
        return self._run_stage("calibrate_distance", initial_guess)

    def calibrate_tth0(self, initial_guess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the 2theta offset (+/-5 deg)."""
        return self._run_stage("calibrate_tth0", initial_guess)

    def CalibrateShift(
        self, initalGuess: Optional[Sequence[float]] = None, bounds: Optional[Sequence[Sequence[float]]] = None
    ) -> StageResult:
        """Refine the X, Y, Z shifts (default bounds +/-0.05, +/-0.05, +/-0.15 m)."""
        if bounds is None:
            return self._run_stage("CalibrateShift", initalGuess)
        return self.refine("CalibrateShift", _STAGES["CalibrateShift"][0], bounds[0], bounds[1], initalGuess)

    def CalibrateRotation(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine the X, Y, Z rotations (+/-5 deg)."""
        return self._run_stage("CalibrateRotation", initalGuess)

    def CalibrateGeometry(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine shifts (+/-0.05 m), rotations and 2theta_0 (+/-5 deg); the wavelength is held."""
        return self._run_stage("CalibrateGeometry", initalGuess)

    def FullCalibration(self, initalGuess: Optional[Sequence[float]] = None) -> StageResult:
        """Refine all 8 parameters (shifts +/-0.05/0.05/0.15 m, angles +/-5 deg, wavelength +/-0.05 A)."""
        return self._run_stage("FullCalibration", initalGuess)

    def estimate_errors(self, hold: Sequence[Union[int, str]] = ("TTH_0",)) -> StageResult:
        """Uncertainties of the calibration at its current values, with ``hold`` fixed.

        Shift_x, Rot_y and TTH_0 are exactly degenerate at one detector distance (see :meth:`refine`),
        so a refinement of all eight parameters cannot give their errors (they are flagged
        :data:`UNDETERMINED`). This refines every parameter except ``hold``, starting from the
        current values (normally the optimum, so the values barely move), to get their one-sigma
        errors. The held parameters' errors are set to :data:`NOT_REFINED`: their values are a
        choice along the degenerate set, not a measurement (docs/ground_truths.md).

        Args:
            hold: Parameters (indices or :data:`PARAMETER_NAMES`) kept fixed; default 2theta_0.

        Returns:
            The :class:`StageResult` of the error refinement, recorded in :attr:`stages`.

        Raises:
            ValueError: On an unknown parameter in ``hold``.
        """
        held = _parameter_indices(hold, "estimate_errors")
        free = [i for i in range(8) if i not in held]
        # FullCalibration's widths about the current values: the start is inside, and they do not bind
        span = np.asarray(_STAGES["FullCalibration"][1])[free]
        current = self._calib[free]
        stage = self.refine("estimate_errors", free, current - span, current + span)
        self._caliberr[held] = NOT_REFINED
        return stage

    def errors_at_current(self, hold: Sequence[Union[int, str]] = ("TTH_0",)) -> StageResult:
        """Uncertainties of every parameter except ``hold`` at the current values, without refining.

        Unlike :meth:`estimate_errors`, no parameter moves: the errors come from the residual's
        Jacobian at the current values (the same finite differences and formula as :meth:`refine`).
        Use this after a recipe that refined only some parameters; ``estimate_errors`` would move
        all the others to their optimum. The status of the last refinement is kept.

        Raises:
            ValueError: On an unknown parameter in ``hold``.
        """
        held = _parameter_indices(hold, "errors_at_current")
        free = np.array([i for i in range(8) if i not in held], dtype=int)
        values = self._calib.copy()

        def free_residual(x: np.ndarray) -> np.ndarray:
            params = values.copy()
            params[free] = x
            return self.residual(params)

        residual = self.residual(values)
        jacobian = approx_derivative(free_residual, values[free], method="3-point", rel_step=self._diff_step)
        errors = _parameter_errors(jacobian, residual, residual.size - free.size)
        self._caliberr[free] = errors
        self._caliberr[held] = NOT_REFINED
        stage = StageResult(
            name="errors_at_current",
            free_parameters=tuple(PARAMETER_NAMES[i] for i in free),
            values=values[free].copy(),
            errors=errors,
            status=self._status,
            rmse=_rmse(residual),
            nfev=0,
        )
        self.stages.append(stage)
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
            to_shift(values),
            to_shift(errors),
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
            tolerance: Convergence threshold in mm (shifts), degrees (angles) and mÅ (wavelength).
            max_passes: Upper limit when ``passes`` is None.
            polish: Finish with :meth:`FullCalibration`.

        Returns:
            Number of passes run (excluding the polish).

        Raises:
            ValueError: On an unknown stage name or a non-positive pass count or tolerance.
        """
        unknown = [name for name in recipe if name not in _STAGES]
        if unknown or not recipe:
            raise ValueError(f"recipe stages must be from {sorted(_STAGES)}, got {list(recipe)}")
        if passes is not None and passes < 1:
            raise ValueError(f"passes must be at least 1, got {passes}")
        if not tolerance > 0 or max_passes < 1:
            raise ValueError("tolerance must be positive and max_passes at least 1")

        n_pass = 0
        for n_pass in range(1, (passes or max_passes) + 1):
            before = self._calib.copy()
            for name in recipe:
                getattr(self, name)()
            if passes is None and np.max(np.abs(self._calib - before) * _TOLERANCE_UNITS) < tolerance:
                break
        else:
            if passes is None:
                self._warn(f"iterate: not converged to {tolerance} after {max_passes} passes")

        if polish:
            self.FullCalibration()
        return n_pass
