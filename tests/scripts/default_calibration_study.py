"""
By-hand study (not collected by pytest) for #1038: should the ring-based calibration replace FitCalibration
as the default production calibration? Analysis only; nothing here changes either method.

Subcommands, each printing one ``RESULT {json}`` line per result for collation:

* ``ring RUN OUT.json``: the ring calibration as ``pyrs-pixel-calibration`` runs it (5 strips, 3 passes, joint
  polish, errors with 2theta_0 held), optionally from a perturbed start (C3: ``--perturb`` shifts in mm,
  rotations in deg, wavelength as a fraction) used for both the ring extraction and the refinement, or with
  ``--use-points`` for the refinement only, or ``--rounds N`` to extract the rings again at the refined geometry.
* ``fitcal RUN OUT.json``: a FitCalibration recipe as ``pyrs-calibration`` runs it (eta slices 3 deg, 512 bins),
  optionally from a starting calibration, with another ``least_squares`` method or ``max_nfev`` (C2 checks 1, 2).
  The final residual is re-scored once at the written values.
* ``profile RUN START.json``: either method's cost with Rot_y held at each given value and the other seven
  parameters re-refined (C2 check 3), to compare the valley's depth with FitCalibration's noise floor.
* ``line RUN A.json B.json``: both costs along the straight line between two calibrations (C2: replaces the
  Rot_y profile for FitCalibration, which is flat because Rot_y, Shift_x and 2theta_0 are degenerate).
* ``score RUN POINTS.npz``: the C1 checks for several calibrations of one run: ring-point 2theta RMS and strip
  drift (the ring method's target), FitCalibration's eta-wedge residual (FitCalibration's target), and, neutral
  to both, each reflection's offset from Bragg and FWHM in the whole-detector 1D patterns reduction produces.
* ``holdout RUN OUT_DIR``: #1074, judge on sub-runs not fitted. The run's sub-runs are split by arm angle
  (``--split alternate``: every other arm, interpolation; ``--split halves``: low against high arms,
  extrapolation; ``--split one-out``: all but the ``--fit-group``-th arm, leave-one-out) into group A
  (``--fit-group``) and B. Half a run leaves the geometry poorly determined (the half-run calibrations differ from
  the all-sub-run one by 8-44 x its precision), so leave-one-out, which keeps nearly every arm, is the main test. On A: the ring calibration (repeated extraction,
  ``calibrate_rings``) and that followed by a FitCalibration clean-up (``FullCalibration`` from it). Both, and
  the ring calibration of all sub-runs for reference, are scored on A and on B with both costs: the ring-point
  2theta RMS (points extracted once, at the all-sub-run ring geometry, so every candidate is judged on the
  same points) and FitCalibration's eta-wedge residual (its sub-runs restricted to the group).
* ``strain RUN``: C4, the strain a change of calibration implies, Delta d / d = Delta lambda / lambda
  - cot(theta) Delta theta, over the detector at each sub-run's arm angle: its mean (a uniform offset, which a
  d0 measured with the same calibration cancels) and its spread across the detector (which no d0 cancels).

Run from the repository root, e.g.:

    pixi run python tests/scripts/default_calibration_study.py ring RUN.nxs.h5 ring.json --points ring.npz --mask M.xml
    pixi run python tests/scripts/default_calibration_study.py fitcal RUN.nxs.h5 fit.json --recipe staged --mask M.xml
    pixi run python tests/scripts/default_calibration_study.py score RUN.nxs.h5 ring.npz --mask M.xml \
        --calibration ring=ring.json --calibration fitcal=fit.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit, least_squares

CAL_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")
ETA_SLICE, TTH_BINS = 3.0, 512  # pyrs-calibration's defaults
RECIPES = {
    "full": ("FullCalibration",),  # pyrs-calibration's default
    "staged": ("CalibrateShift", "CalibrateRotation", "calibrate_wave_shift", "FullCalibration"),  # #1038's
}
# FullCalibration's bounds: shifts (m), rotations and 2theta_0 (deg); the wavelength's is +/-0.05 A about the start
LOWER = np.array([-0.05, -0.05, -0.15, -5.0, -5.0, -5.0, -5.0])
UPPER = np.array([0.05, 0.05, 0.15, 5.0, 5.0, 5.0, 5.0])
ROT_Y = 4
WINDOW = 1.0  # deg either side of Bragg for a 1D peak fit


def result(kind: str, **values: Any) -> None:
    """One machine-readable line per result."""
    print("RESULT " + json.dumps({"kind": kind, **values}, default=float), flush=True)


def read_vector(path: str) -> np.ndarray:
    raw = json.loads(Path(path).read_text())
    return np.array([float(raw[key]) for key in CAL_KEYS])


def convert(run: str, mask: Optional[str]):
    from pyrs.core.nexus_conversion import NeXusConvertingApp

    return NeXusConvertingApp(run, mask).convert()


def mono_wavelength(workspace) -> float:
    return float(workspace.get_wavelength(False, False))


def bounded(start: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.r_[LOWER, start[7] - 0.05], np.r_[UPPER, start[7] + 0.05]


# --- ring ---------------------------------------------------------------------------------------------------


def cmd_ring(args: argparse.Namespace) -> None:
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints, extract_ring_points
    from pyrs.core.detector_model import to_shift

    workspace = convert(args.run, args.mask)
    mono = mono_wavelength(workspace)
    perturb = np.array(args.perturb, dtype=float)
    start = np.r_[perturb[:3] * 1e-3, perturb[3:6], 0.0, mono * (1.0 + perturb[6])]
    perturbed = bool(np.any(perturb))

    began = time.time()
    if args.use_points:  # refinement only: separates the refinement's start dependence from the extraction's
        points = RingPoints.load(args.use_points)
    else:
        points = extract_ring_points(
            workspace, start[7], calibration=to_shift(start[:7]) if perturbed else None, strips=args.strips
        )
    calibration = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=start[7])
    if perturbed:
        calibration.set_calibration_array(start, source=f"perturbed start {list(perturb)} (#1038 C3)")
    calibration.iterate(passes=3, polish=True)
    rounds = [
        {"points": len(points), "rmse": calibration.stages[-1].rmse, "values": list(calibration.calibration_array)}
    ]
    # further rounds (C3): extract the rings again where the refined geometry predicts them, refine from there
    for _ in range(args.rounds - 1):
        values = calibration.calibration_array.copy()
        points = extract_ring_points(workspace, values[7], calibration=to_shift(values[:7]), strips=args.strips)
        calibration = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=values[7])
        calibration.set_calibration_array(values, source="previous round (#1038 C3)")
        calibration.iterate(passes=3, polish=True)
        rounds.append(
            {"points": len(points), "rmse": calibration.stages[-1].rmse, "values": list(calibration.calibration_array)}
        )
    calibration.estimate_errors()
    elapsed = time.time() - began

    calibration.write_calibration(args.output, inputs={"driver": "tests/scripts/default_calibration_study.py ring"})
    if args.points:
        points.save(args.points)
    result(
        "ring",
        run=Path(args.run).name,
        label=args.label,
        mask=args.mask,
        use_points=args.use_points,
        perturb=list(perturb),
        rounds=rounds,
        seconds=elapsed,
        points=len(points),
        rings=points.num_rings,
        rmse=calibration.stages[-1].rmse,
        status=calibration.calibration_status,
        values=list(calibration.calibration_array),
        errors=list(calibration.calibration_error_array),
    )


# --- FitCalibration ----------------------------------------------------------------------------------------


def fit_calibration(run: str, mask: Optional[str]):
    from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

    return FitCalibration(nexus_file=run, mask_file=mask, eta_slice=ETA_SLICE, bins=TTH_BINS)


def cmd_fitcal(args: argparse.Namespace) -> None:
    fit = fit_calibration(args.run, args.mask)
    fit.set_refinement_method(args.method)
    fit.set_max_nfev(args.max_nfev)
    if args.start:
        fit.set_calibration_array(read_vector(args.start))

    began = time.time()
    for stage in RECIPES[args.recipe]:
        getattr(fit, stage)()
    elapsed = time.time() - began
    fit.write_calibration(args.output)

    rescored = np.asarray(fit.get_alignment_residual(fit.get_calib()))
    result(
        "fitcal",
        run=Path(args.run).name,
        label=args.label,
        recipe=args.recipe,
        start=args.start,
        method=args.method,
        max_nfev=args.max_nfev,
        seconds=elapsed,
        nfev=[step.get("nfev") for step in fit.steps],
        stage_status=[step.get("status") for step in fit.steps],
        rmse=float(np.sqrt(np.mean(rescored**2))),
        peaks=int(rescored.size),
        values=list(fit.get_calib()),
        errors=list(fit.calibration_error_array),
    )


# --- profile along Rot_y -------------------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> None:
    start = read_vector(args.start)
    free = [i for i in range(8) if i != ROT_Y]
    lower, upper = bounded(start)
    if args.method == "ring":
        from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints

        workspace = convert(args.run, args.mask)
        ring = PixelCalibration(RingPoints.load(args.points), workspace.get_instrument_setup(), wavelength=start[7])
        cost: Callable[[np.ndarray], np.ndarray] = ring.residual
        diff_step = None
    else:
        from pyrs.calibration.mantid_peakfit_calibration import DIFF_STEP

        fit = fit_calibration(args.run, args.mask)
        cost = lambda params: np.asarray(fit.get_alignment_residual(np.asarray(params, dtype=float)))  # noqa: E731
        diff_step = DIFF_STEP[free]

    for rot_y in args.rot_y:
        began = time.time()
        params = start.copy()
        params[ROT_Y] = rot_y

        def held(x: np.ndarray) -> np.ndarray:
            params[free] = x
            return cost(params)

        out = least_squares(
            held,
            np.clip(start[free], lower[free] + 1e-9, upper[free] - 1e-9),
            bounds=(lower[free], upper[free]),
            jac="3-point",
            diff_step=diff_step,
            max_nfev=args.max_nfev,
        )
        params[free] = out.x
        residual = cost(params)  # re-scored at the solution: FitCalibration's cost is noisy
        result(
            "profile",
            run=Path(args.run).name,
            method=args.method,
            rot_y=rot_y,
            rmse=float(np.sqrt(np.mean(residual**2))),
            nfev=int(out.nfev),
            status=int(out.status),
            seconds=time.time() - began,
            values=list(params),
        )


# --- line between two calibrations -------------------------------------------------------------------------


def cmd_line(args: argparse.Namespace) -> None:
    """Both costs along the straight line from calibration A (t = 0) to B (t = 1) in parameter space: whether
    FitCalibration's cost is lower towards the other geometry, and how rough it is on the way (C2)."""
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints

    start, end = read_vector(args.a), read_vector(args.b)
    workspace = convert(args.run, args.mask)
    ring = PixelCalibration(RingPoints.load(args.points), workspace.get_instrument_setup(), wavelength=start[7])
    fit = fit_calibration(args.run, args.mask)
    for t in np.linspace(args.t_min, args.t_max, args.steps):
        params = start + t * (end - start)
        wedge = np.asarray(fit.get_alignment_residual(params.copy()))
        rings = ring.residual(params)
        result(
            "line",
            run=Path(args.run).name,
            a=args.a,
            b=args.b,
            t=float(t),
            fitcal_rmse=float(np.sqrt(np.mean(wedge**2))),
            peaks=int(wedge.size),
            ring_rmse=float(np.sqrt(np.mean(rings**2))),
        )


# --- score (C1) ----------------------------------------------------------------------------------------------


def gaussian(x: np.ndarray, height: float, centre: float, sigma: float, a: float, b: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + a + b * (x - centre)


def fit_reflection(x: np.ndarray, y: np.ndarray, bragg: float) -> Optional[Tuple[float, float]]:
    """Centre and FWHM of the reflection near ``bragg`` (deg), or None if it cannot be fitted."""
    window = (x > bragg - WINDOW) & (x < bragg + WINDOW) & np.isfinite(y)
    if window.sum() < 10:
        return None
    xw, yw = x[window], y[window]
    background = float(np.median(np.r_[yw[:3], yw[-3:]]))
    guess = [float(yw.max()) - background, float(xw[np.argmax(yw)]), 0.2, background, 0.0]
    try:
        values, _ = curve_fit(gaussian, xw, yw, p0=guess, maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    if not (values[0] > 0 and abs(values[1] - bragg) < WINDOW and 0.02 < abs(values[2]) < 1.0):
        return None
    return float(values[1]), float(2.3548 * abs(values[2]))


def patterns_1d(workspace, calibration_file: Optional[str]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Whole-detector 1D pattern of every sub-run, as reduction produces it with this calibration."""
    from pyrs.core.powder_pattern import ReductionApp

    reducer = ReductionApp()
    reducer.load_hidra_workspace(workspace)
    sub_runs = list(workspace.get_sub_runs())
    reducer.reduce_data(
        sub_runs=sub_runs, instrument_file=None, calibration_file=calibration_file, mask=None, num_bins=1000
    )
    patterns = {}
    for sub_run in sub_runs:
        x, y, _ = reducer.get_diffraction_data(sub_run)
        patterns[int(sub_run)] = (np.asarray(x), np.asarray(y))
    return patterns


def peak_offsets(workspace, calibration_file: Optional[str], wavelength: float) -> List[Dict[str, Any]]:
    from pyrs.calibration.powder_standards import standard_for_sy

    peaks = []
    for sub_run, (x, y) in patterns_1d(workspace, calibration_file).items():
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)))
        if standard is None:
            continue
        for hkl2, bragg in zip(standard.hkl_squared, standard.two_theta(wavelength)):
            if not (np.isfinite(bragg) and x.min() + WINDOW < bragg < x.max() - WINDOW):
                continue
            fitted = fit_reflection(x, y, float(bragg))
            if fitted is not None:
                peaks.append(
                    {
                        "sub_run": sub_run,
                        "powder": standard.name,
                        "hkl2": int(hkl2),
                        "offset": fitted[0] - bragg,
                        "fwhm": fitted[1],
                    }
                )
    return peaks


def strip_drift(points, residual: np.ndarray) -> np.ndarray:
    """Per ring and sub-run: change of the residual from the bottom to the top strip (deg), from a line fit."""
    drifts = []
    for key in sorted(set(zip(points.sub_run.tolist(), points.hkl_squared.tolist()))):
        select = (points.sub_run == key[0]) & (points.hkl_squared == key[1])
        if select.sum() < 3:
            continue
        rows = points.index_1[select]
        drifts.append(np.polyfit(rows, residual[select], 1)[0] * (rows.max() - rows.min()))
    return np.array(drifts)


def labelled(items: Sequence[str]) -> Dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def cmd_score(args: argparse.Namespace) -> None:
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints

    files = labelled(args.calibration)
    workspace = convert(args.run, args.mask)
    vectors = {label: read_vector(path) for label, path in files.items()}
    points = RingPoints.load(args.points)
    ring = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=mono_wavelength(workspace))
    fit = None if args.no_fitcalibration else fit_calibration(args.run, args.mask)
    for label, vector in vectors.items():
        residual = ring.residual(vector)
        drift = strip_drift(points, residual)
        peaks = peak_offsets(workspace, files[label], vector[7])
        offsets = np.array([peak["offset"] for peak in peaks])
        scores = {
            "ring_rmse": float(np.sqrt(np.mean(residual**2))),
            "strip_drift_mean_abs": float(np.mean(np.abs(drift))),
            "strip_drift_max_abs": float(np.max(np.abs(drift))),
            "peak_offset_rms": float(np.sqrt(np.mean(offsets**2))),
            "peak_offset_mean": float(np.mean(offsets)),
            "peak_offset_spread": float(np.std(offsets)),
            "fwhm_mean": float(np.mean([peak["fwhm"] for peak in peaks])),
            "peaks_1d": len(peaks),
        }
        if fit is not None:
            wedge = np.asarray(fit.get_alignment_residual(vector.copy()))
            scores["fitcal_rmse"] = float(np.sqrt(np.mean(wedge**2)))
        result("score", run=Path(args.run).name, label=label, **scores, peaks=peaks)


# --- hold-out (#1074) ---------------------------------------------------------------------------------------


def split_sub_runs(workspace, split: str, fit_group: int) -> Tuple[List[int], List[int]]:
    """Sub-runs ordered by arm angle, split into the fitted group and the held-out one."""
    sub_runs = sorted(workspace.get_sub_runs(), key=lambda s: float(workspace.get_detector_2theta(s)))
    if split == "one-out":
        if not 0 <= fit_group < len(sub_runs):
            raise ValueError(f"one-out index must be 0 to {len(sub_runs) - 1}, got {fit_group}")
        held = sub_runs[fit_group]
        return [int(s) for s in sub_runs if s != held], [int(held)]
    if split == "alternate":
        groups = (sub_runs[0::2], sub_runs[1::2])
    else:
        half = len(sub_runs) // 2
        groups = (sub_runs[:half], sub_runs[half:])
    return [int(s) for s in groups[fit_group]], [int(s) for s in groups[1 - fit_group]]


def restricted_fit(run: str, mask: Optional[str], sub_runs: Sequence[int]):
    """FitCalibration over only ``sub_runs`` (others are neither reduced nor fitted)."""
    fit = fit_calibration(run, mask)
    keep = np.isin(np.asarray(fit.sub_runs), list(sub_runs))
    fit.set_keep_subrun_list(np.logical_and(keep, fit._keep_subrun_list))
    return fit


def cmd_holdout(args: argparse.Namespace) -> None:
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration, calibrate_rings, extract_ring_points
    from pyrs.core.detector_model import to_shift

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    workspace = convert(args.run, args.mask)
    mono = mono_wavelength(workspace)
    group_a, group_b = split_sub_runs(workspace, args.split, args.fit_group)
    label = f"{Path(args.run).name.split('.')[0]}_{args.split}_fit{args.fit_group}"

    began = time.time()
    full = calibrate_rings(workspace, mono)  # reference, and the geometry the scoring points are extracted at
    ring_a = calibrate_rings(workspace, mono, sub_runs=group_a)
    ring_a.estimate_errors()  # do the fitted sub-runs alone determine every parameter?
    ring_a.write_calibration(out / f"{label}_ring.json")
    ring_seconds = time.time() - began

    began = time.time()
    clean = restricted_fit(args.run, args.mask, group_a)
    clean.set_calibration_array(ring_a.calibration_array.copy())
    clean.FullCalibration()
    clean.write_calibration(str(out / f"{label}_cleanup.json"))
    cleanup_seconds = time.time() - began

    candidates = {
        "ring_A": ring_a.calibration_array.copy(),
        "cleanup_A": np.asarray(clean.get_calib(), dtype=float),
        "ring_all": full.calibration_array.copy(),
    }
    values = full.calibration_array
    scores: Dict[str, Dict[str, float]] = {name: {} for name in candidates}
    for group_name, group in (("A", group_a), ("B", group_b)):
        points = extract_ring_points(
            workspace, float(values[7]), calibration=to_shift(values[:7]), strips=5, sub_runs=group
        )
        ring_cost = PixelCalibration(points, workspace.get_instrument_setup(), wavelength=float(values[7]))
        wedge_cost = restricted_fit(args.run, args.mask, group)
        for name, vector in candidates.items():
            residual = ring_cost.residual(vector)
            wedge = np.asarray(wedge_cost.get_alignment_residual(vector.copy()))
            scores[name][f"ring_rmse_{group_name}"] = float(np.sqrt(np.mean(residual**2)))
            scores[name][f"fitcal_rmse_{group_name}"] = float(np.sqrt(np.mean(wedge**2)))
        scores["ring_A"][f"ring_points_{group_name}"] = len(points)
    result(
        "holdout",
        run=Path(args.run).name,
        split=args.split,
        fit_group=args.fit_group,
        sub_runs_A=group_a,
        sub_runs_B=group_b,
        arms_A=[float(workspace.get_detector_2theta(s)) for s in group_a],
        arms_B=[float(workspace.get_detector_2theta(s)) for s in group_b],
        ring_A_errors=list(ring_a.calibration_error_array),
        ring_A_rounds=len(ring_a.extraction_rounds or []),
        ring_A_converged=ring_a.extraction_converged,
        cleanup_status=int(clean._calibstatus),
        ring_seconds=ring_seconds,
        cleanup_seconds=cleanup_seconds,
        candidates={name: list(vector) for name, vector in candidates.items()},
        scores=scores,
    )


# --- strain (C4) ---------------------------------------------------------------------------------------------


def cmd_strain(args: argparse.Namespace) -> None:
    from pyrs.core.detector_model import DetectorModel, to_shift

    files = labelled(args.calibration)
    reference = read_vector(files.pop(args.reference))
    workspace = convert(args.run, None)
    model = DetectorModel(workspace.get_instrument_setup())
    grid = np.meshgrid(np.arange(40, 984, 4), np.arange(40, 984, 4), indexing="ij")  # inside the masked edge
    # L2 None: the nominal arm length (no L2 log), as reduction
    positions = sorted(
        {(float(workspace.get_detector_2theta(s)), workspace.get_l2(s)) for s in workspace.get_sub_runs()},
        key=lambda position: position[0],
    )
    for label, path in files.items():
        other = read_vector(path)
        for arm, l2 in positions:
            tth_ref = np.deg2rad(model.two_theta(grid[0], grid[1], -arm, l2, to_shift(reference[:7])))
            tth = np.deg2rad(model.two_theta(grid[0], grid[1], -arm, l2, to_shift(other[:7])))
            strain = (other[7] - reference[7]) / reference[7] - (tth - tth_ref) / 2.0 / np.tan(tth_ref / 2.0)
            micro = strain * 1e6
            result(
                "strain",
                run=Path(args.run).name,
                label=label,
                reference=args.reference,
                arm=arm,
                mean_ue=float(np.mean(micro)),
                spread_ue=float(np.percentile(micro, 97.5) - np.percentile(micro, 2.5)),
                max_dev_ue=float(np.max(np.abs(micro - np.mean(micro)))),
                max_dtth_deg=float(np.rad2deg(np.max(np.abs(tth - tth_ref)))),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    ring = sub.add_parser("ring")
    ring.add_argument("run")
    ring.add_argument("output")
    ring.add_argument("--points")
    ring.add_argument("--mask")
    ring.add_argument("--label", default="ring")
    ring.add_argument("--strips", type=int, default=5)
    ring.add_argument("--use-points", help="refine these ring points (.npz) instead of extracting them")
    ring.add_argument("--rounds", type=int, default=1, help="extraction + refinement rounds (C3)")
    ring.add_argument(
        "--perturb",
        type=float,
        nargs=7,
        default=[0.0] * 7,
        metavar=("DX", "DY", "DZ", "RX", "RY", "RZ", "DLAM"),
        help="start offsets: shifts (mm), rotations (deg), wavelength (fraction)",
    )
    ring.set_defaults(handler=cmd_ring)

    fitcal = sub.add_parser("fitcal")
    fitcal.add_argument("run")
    fitcal.add_argument("output")
    fitcal.add_argument("--mask")
    fitcal.add_argument("--label", default="fitcal")
    fitcal.add_argument("--recipe", choices=sorted(RECIPES), default="full")
    fitcal.add_argument("--start", help="starting calibration JSON (default: nominal)")
    fitcal.add_argument("--method", choices=("trf", "dogbox", "lm"), default="trf")
    fitcal.add_argument("--max-nfev", type=int)
    fitcal.set_defaults(handler=cmd_fitcal)

    profile = sub.add_parser("profile")
    profile.add_argument("run")
    profile.add_argument("start")
    profile.add_argument("--method", choices=("ring", "fitcal"), required=True)
    profile.add_argument("--points", help="ring points (.npz), for --method ring")
    profile.add_argument("--mask")
    profile.add_argument("--rot-y", type=float, nargs="+", required=True)
    profile.add_argument("--max-nfev", type=int)
    profile.set_defaults(handler=cmd_profile)

    line = sub.add_parser("line")
    line.add_argument("run")
    line.add_argument("a", help="calibration JSON at t = 0")
    line.add_argument("b", help="calibration JSON at t = 1")
    line.add_argument("--points", required=True, help="ring points (.npz) for the ring cost")
    line.add_argument("--mask")
    line.add_argument("--t-min", type=float, default=-0.2)
    line.add_argument("--t-max", type=float, default=1.2)
    line.add_argument("--steps", type=int, default=29)
    line.set_defaults(handler=cmd_line)

    score = sub.add_parser("score")
    score.add_argument("run")
    score.add_argument("points")
    score.add_argument("--mask")
    score.add_argument("--calibration", action="append", required=True, help="LABEL=FILE.json (repeatable)")
    score.add_argument("--no-fitcalibration", action="store_true")
    score.set_defaults(handler=cmd_score)

    holdout = sub.add_parser("holdout")
    holdout.add_argument("run")
    holdout.add_argument("output_dir")
    holdout.add_argument("--mask")
    holdout.add_argument("--split", choices=("alternate", "halves", "one-out"), default="one-out")
    holdout.add_argument(
        "--fit-group", type=int, default=0, help="alternate/halves: 0 or 1; one-out: index of the held-out arm"
    )
    holdout.set_defaults(handler=cmd_holdout)

    strain = sub.add_parser("strain")
    strain.add_argument("run")
    strain.add_argument("--calibration", action="append", required=True, help="LABEL=FILE.json (repeatable)")
    strain.add_argument("--reference", required=True)
    strain.set_defaults(handler=cmd_strain)

    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    args.handler(args)


if __name__ == "__main__":
    main()
