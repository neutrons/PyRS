"""
Repeated ring extraction for the ring-based HB2B calibration (#1038).

Refining a fixed set of ring points ends at the same geometry from any start, but where
:func:`~pyrs.calibration.ring_extraction.extract_ring_points` looks for the rings depends on the geometry it
starts from. From a poor start a single extraction can find only part of the rings and give a geometry
that is wrong by tenths of a degree with a *better* RMS (HB2B_3489, docs/ground_truths.md). Extracting the
rings again where the refined geometry predicts them removes this.

The workflow, one *round* at a time: extract the rings, refine them (:meth:`PixelCalibration.iterate`), and
extract again at the refined geometry, until a round's step is less than :data:`CONVERGENCE_FRACTION` of the
data's statistical precision (:func:`statistical_precision`), after at least :data:`MIN_ROUNDS` and at most
``max_rounds`` rounds. The step is the larger of two changes from the previous round (user decision): the
detector's pixel 2theta (:func:`pixel_two_theta_change`, the geometry over the whole detector) and the
predicted ring positions at the ring points (:func:`ring_position_change`, which includes the wavelength:
the next extraction looks for each ring at its Bragg angle). On sparse runs one or two ring points drop in
and out between rounds, so the geometry keeps stepping by about the precision instead of settling: a
tolerance relative to the precision, not a fixed one, says when the steps are noise (settling study,
docs/ground_truths.md).

Typical use:

1. ``calibration = calibrate_rings(workspace, wavelength)``: extract, refine and repeat from the nominal
   geometry (or ``start``).
2. Or, with ring points already extracted and refined once, ``calibration = continue_rounds(calibration,
   workspace)``, as the calibration GUI does.
3. ``calibration.extraction_rounds`` records each round; it is part of the calibration's provenance.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pyrs.calibration.pixel_refinement import PARAMETER_NAMES, PixelCalibration
from pyrs.calibration.ring_extraction import StripFit, extract_ring_points
from pyrs.core.detector_model import to_shift

__all__ = [
    "DEFAULT_MAX_ROUNDS",
    "MIN_ROUNDS",
    "CONVERGENCE_FRACTION",
    "calibrate_rings",
    "continue_rounds",
    "statistical_precision",
    "pixel_two_theta_change",
    "ring_position_change",
]

DEFAULT_MAX_ROUNDS = 10  # user decision (#1038); HB2B_3489 needed up to 9 in the settling study
MIN_ROUNDS = 2  # the second round is the one that recovers from a bad first extraction
CONVERGENCE_FRACTION = 0.5  # a round converges when its step is < 0.5 x the precision
_EDGE, _GRID_STEP = 40, 8  # pixels compared: every 8th inside the 40-pixel edge the extraction skips


def statistical_precision(calibration: PixelCalibration) -> float:
    """Statistical precision of a refined calibration's pixel 2theta: RMS * sqrt(p / N) (degrees).

    With ``p`` = 8 refined parameters and ``N`` ring points, this is the typical uncertainty of the fitted
    ring positions; the spread of calibrations from different starts matches it (settling study).

    Raises:
        ValueError: If nothing has been refined yet.
    """
    if not calibration.stages:
        raise ValueError("the calibration has not been refined")
    return float(calibration.stages[-1].rmse * np.sqrt(len(PARAMETER_NAMES) / len(calibration.ring_points)))


def pixel_two_theta_change(calibration: PixelCalibration, before: np.ndarray, after: np.ndarray) -> float:
    """Largest change of pixel 2theta (degrees) between two calibration vectors, over the detector at every
    detector position (arm angle and L2) of ``calibration``'s ring points.

    Calibrations are compared by where they put the pixels, not parameter by parameter: Rot_y, Shift_x and
    2theta_0 are degenerate (docs/ground_truths.md).
    """
    model = calibration.model
    rows, columns = model.instrument_setup.detector_size
    grid = np.meshgrid(
        np.arange(_EDGE, rows - _EDGE, _GRID_STEP), np.arange(_EDGE, columns - _EDGE, _GRID_STEP), indexing="ij"
    )
    points = calibration.ring_points
    positions = np.unique(np.column_stack([points.det_2theta, points.l2]).astype(float), axis=0)
    shift_before, shift_after = to_shift(before[:7]), to_shift(after[:7])
    change = 0.0
    for arm_two_theta, l2 in positions:
        two_theta_before = model.two_theta(grid[0], grid[1], -arm_two_theta, l2, shift_before)
        two_theta_after = model.two_theta(grid[0], grid[1], -arm_two_theta, l2, shift_after)
        change = max(change, float(np.max(np.abs(two_theta_after - two_theta_before))))
    return change


def ring_position_change(calibration: PixelCalibration, before: np.ndarray, after: np.ndarray) -> float:
    """Largest change (degrees) of the predicted ring positions relative to ``calibration``'s ring points, i.e.
    of their 2theta residuals, between two calibration vectors: geometry and wavelength together.

    A round that mainly moves the wavelength moves where the next extraction looks for every ring by
    about tan(theta) * d(lambda) / lambda, which :func:`pixel_two_theta_change` does not see.
    """
    return float(np.max(np.abs(calibration.residual(after) - calibration.residual(before))))


def _check_max_rounds(max_rounds: int) -> None:
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")


def _round_record(
    number: int, calibration: PixelCalibration, pixel_step: Optional[float] = None, ring_step: Optional[float] = None
) -> Dict[str, Any]:
    return {
        "round": number,
        "ring_points": len(calibration.ring_points),
        "rmse_deg": float(calibration.stages[-1].rmse),
        "precision_deg": statistical_precision(calibration),
        # change from the previous round (None for the first): the larger of the pixel 2theta and the ring positions
        "step_deg": None if pixel_step is None or ring_step is None else max(pixel_step, ring_step),
        "pixel_step_deg": pixel_step,
        "ring_step_deg": ring_step,
        "values": [float(value) for value in calibration.calibration_array],
    }


def continue_rounds(
    calibration: PixelCalibration,
    workspace,
    strips: int = 5,
    sub_runs: Optional[Sequence[int]] = None,
    passes: int = 3,
    polish: bool = True,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    diagnostics: Optional[List[StripFit]] = None,
) -> PixelCalibration:
    """Refine ``calibration`` (round 1, its ring points as extracted), then extract the rings again at the
    refined geometry and refine, until converged (see the module docstring).

    Args:
        calibration: Ring points extracted from ``workspace``, with their starting calibration set.
        workspace: The calibration run the points came from (``HidraWorkspace``).
        strips: Strips per sub-run for the extraction.
        sub_runs: Sub-runs to extract (default: all).
        passes: Passes of :meth:`PixelCalibration.iterate` in each round.
        polish: Finish each round with the joint refinement.
        max_rounds: Most rounds, including the first; reaching it without converging warns.
        diagnostics: If given, replaced by the last round's strip fits.

    Returns:
        The calibration of the last round, a new object if there was more than one round. Its
        ``extraction_rounds`` lists every round, after those of an earlier call on ``calibration`` (the first
        record of each call carries the ``start`` its refinement began from); ``extraction_converged`` says
        whether they converged (None for one round); its ``warning_log`` keeps every round's warnings.

    Raises:
        ValueError: If ``max_rounds`` is less than 1, or a round extracts no ring point.
    """
    _check_max_rounds(max_rounds)
    earlier = list(calibration.extraction_rounds or [])  # e.g. a GUI recipe with "iterate" twice
    start = calibration.start  # later rounds start where the previous one ended: keep where this began
    calibration.iterate(passes=passes, polish=polish)
    rounds = [dict(_round_record(len(earlier) + 1, calibration), start=start)]
    converged: Optional[bool] = None
    for number in range(2, max_rounds + 1):
        values = calibration.calibration_array.copy()
        round_diagnostics: List[StripFit] = []
        points = extract_ring_points(
            workspace,
            float(values[7]),
            calibration=to_shift(values[:7]),
            strips=strips,
            sub_runs=sub_runs,
            diagnostics=round_diagnostics,
        )
        refined = PixelCalibration(
            points,
            workspace.get_instrument_setup(),
            wavelength=float(values[7]),
            max_nfev=calibration.max_nfev,
            method=calibration.method,
            diff_step=calibration.diff_step,
        )
        refined.set_calibration_array(values, source=f"round {number - 1} of the ring extraction")
        refined.warning_log.extend(calibration.warning_log)  # a round's warnings stay with the result
        refined.iterate(passes=passes, polish=polish)
        pixel_step = pixel_two_theta_change(refined, values, refined.calibration_array)
        ring_step = ring_position_change(refined, values, refined.calibration_array)
        step = max(pixel_step, ring_step)
        rounds.append(_round_record(len(earlier) + number, refined, pixel_step, ring_step))
        calibration = refined
        if diagnostics is not None:
            diagnostics[:] = round_diagnostics
        if number >= MIN_ROUNDS and step < CONVERGENCE_FRACTION * rounds[-1]["precision_deg"]:
            converged = True
            break
    else:
        if max_rounds >= MIN_ROUNDS:
            converged = False
            message = (
                f"ring extraction not converged after {max_rounds} rounds: the last round's step "
                f"{rounds[-1]['step_deg'] * 1e3:.1f} mdeg (pixel 2theta {rounds[-1]['pixel_step_deg'] * 1e3:.1f}, "
                f"ring positions {rounds[-1]['ring_step_deg'] * 1e3:.1f}) is more than {CONVERGENCE_FRACTION:g} x the "
                f"statistical precision {rounds[-1]['precision_deg'] * 1e3:.1f} mdeg"
            )
            calibration.warning_log.append(message)
            warnings.warn(message, stacklevel=2)
    calibration.extraction_rounds = earlier + rounds
    calibration.extraction_converged = converged
    return calibration


def calibrate_rings(
    workspace,
    wavelength: float,
    start: Optional[Sequence[float]] = None,
    strips: int = 5,
    sub_runs: Optional[Sequence[int]] = None,
    passes: int = 3,
    polish: bool = True,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    diagnostics: Optional[List[StripFit]] = None,
) -> PixelCalibration:
    """Ring-based calibration of one run with repeated extraction: what ``pyrs-pixel-calibration`` runs.

    Args:
        workspace: The calibration run (``HidraWorkspace``), its default mask applied.
        wavelength: Starting wavelength (Angstrom), usually the monochromator setting.
        start: Starting calibration vector (8 values; its wavelength replaces ``wavelength``); default:
            the nominal geometry.
        strips, sub_runs, passes, polish, max_rounds, diagnostics: As :func:`continue_rounds`.

    Returns:
        The refined calibration of the last round, with ``extraction_rounds``.

    Raises:
        ValueError: If no ring point is found, ``start`` is not 8 values or ``max_rounds`` < 1.
    """
    _check_max_rounds(max_rounds)  # before the extraction, which takes most of the time
    start_values = None if start is None else np.asarray(start, dtype=float)
    if start_values is not None:
        if start_values.shape != (8,):
            raise ValueError(f"start must have 8 values, got shape {start_values.shape}")
        wavelength = float(start_values[7])
    first_diagnostics: List[StripFit] = []
    points = extract_ring_points(
        workspace,
        wavelength,
        calibration=None if start_values is None else to_shift(start_values[:7]),
        strips=strips,
        sub_runs=sub_runs,
        diagnostics=first_diagnostics,
    )
    if diagnostics is not None:
        diagnostics[:] = first_diagnostics
    calibration = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=wavelength)
    if start_values is not None:
        calibration.set_calibration_array(start_values, source="given starting calibration")
    return continue_rounds(calibration, workspace, strips, sub_runs, passes, polish, max_rounds, diagnostics)
