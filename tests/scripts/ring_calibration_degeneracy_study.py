"""
Compare conventions for the Shift_x / Rot_y / TTH_0 degeneracy of the ring-based calibration.

Turning the arm about the sample, turning the detector about its own vertical axis and shifting it
sideways can describe the same detector, so refining all three does not pin them individually (see
docs/ground_truths.md). This script compares, on synthetic ring points with the HB2B_4971 layout:

    A  all 8 parameters free
    B  Rot_y fixed at 0
    C  TTH_0 fixed at 0
    D  Shift_x fixed at 0
    E  all 8 free, but half the sub-runs at a second detector distance (L2)
    F  Rot_y fixed at 0; each TTH_0 step also re-solves the shifts so the detector centre stays put
       in the lab (TTH_0 then only turns the detector face; iterative mode only)
    G  as F with Rot_y free

in two modes:

    single     one refinement of all free parameters at once
    iterative  the production-style recipe CalibrateShift -> CalibrateRotation -> calibrate_wave_shift,
               repeated either a fixed number of passes or until the parameters stop changing; each
               convention removes its fixed parameter from the stage that would refine it

Run by hand (never collected by pytest):

    pixi run python tests/scripts/ring_calibration_degeneracy_study.py [--mode both] [--trials 20] [--noise 0.01]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyrs.calibration.pixel_pos_calibration import PARAMETER_NAMES, DetectorModel, PixelCalibration, RingPoints  # noqa: E402
from pyrs.core.instrument_geometry import DENEXDetectorShift  # noqa: E402
from tests.util.instrument_helpers import default_hb2b_geometry  # noqa: E402
from tests.util.ring_calibration_helpers import (  # noqa: E402
    RUN_4971_ARMS,
    RUN_4971_POWDERS,
    synthetic_ring_points,
)

NOMINAL_WAVELENGTH = 1.540
LOWER = np.array([-0.05, -0.05, -0.15, -5.0, -5.0, -5.0, -5.0, NOMINAL_WAVELENGTH - 0.05])
UPPER = np.array([0.05, 0.05, 0.15, 5.0, 5.0, 5.0, 5.0, NOMINAL_WAVELENGTH + 0.05])
UNITS = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 1.0, 1e3])  # report in mm, deg, mA
UNIT_LABELS = ["mm", "mm", "mm", "deg", "deg", "deg", "deg", "mA"]

# name: (fixed parameter indices, scan at two L2 values, compensate TTH_0 steps)
CONVENTIONS = {
    "A all free": ([], False, False),
    "B Rot_y=0": ([4], False, False),
    "C TTH_0=0": ([6], False, False),
    "D Shift_x=0": ([0], False, False),
    "E all free, 2 L2": ([], True, False),
    "F Rot_y=0, comp.": ([4], False, True),
    "G all free, comp.": ([], False, True),
}
CENTRE = 511.5  # centre pixel of the 1024 x 1024 detector

TRUE_GEOMETRIES = {
    "GUI-like": np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308]),
    "small": np.array([0.001, 0.001, -0.005, 0.1, 0.1, -0.2, 0.05, 1.5385]),
    "large": np.array([-0.012, 0.006, 0.040, -1.0, 1.5, 2.0, -1.5, 1.5250]),
}

MAP_ARMS = (60.0, 90.0, 120.0)

# iterative recipe: (stage name, parameters it refines)
RECIPE = (("CalibrateShift", [0, 1, 2]), ("CalibrateRotation", [3, 4, 5]), ("calibrate_wave_shift", [6, 7]))
FIXED_PASSES = (1, 2, 3, 5, 10)
TOLERANCE = 1e-3  # stop when no parameter changes by more than this (mm, deg, mA) between passes
MAX_PASSES = 50
L2_NEAR, L2_FAR = 0.985, 0.935


def make_rings(setup, params, two_l2, noise, rng):
    """Ring points for the 4971 layout; with two_l2, odd sub-runs sit at L2_FAR."""
    if not two_l2:
        return synthetic_ring_points(setup, params, noise_deg=noise, rng=rng, l2=L2_NEAR)
    parts = []
    for i, (arm, powder) in enumerate(zip(RUN_4971_ARMS, RUN_4971_POWDERS)):
        part = synthetic_ring_points(
            setup, params, [arm], [powder], noise_deg=noise, rng=rng, l2=L2_FAR if i % 2 else L2_NEAR
        )
        part.sub_run[:] = i + 1
        parts.append(part)
    fields = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")
    return RingPoints(**{f: np.concatenate([getattr(p, f) for p in parts]) for f in fields})


def fit(setup, rings, fixed):
    free = [i for i in range(8) if i not in fixed]
    calibration = PixelCalibration(rings, setup, wavelength=NOMINAL_WAVELENGTH)
    start = time.perf_counter()
    stage = calibration.refine("study", free, LOWER[free], UPPER[free])
    return calibration, stage, time.perf_counter() - start, free


def centre_in_lab(model, params):
    """Lab position of the detector centre pixel (the arm angle is irrelevant: it rotates both sides)."""
    return model.lab_positions(CENTRE, CENTRE, 0.0, L2_NEAR, DENEXDetectorShift(*params[:7]))


def compensate_shifts(model, params, reference_centre):
    """Return params with Shift_x/y/z re-solved so the detector centre sits at reference_centre.

    With the angles fixed the centre is exactly affine in the three shifts, so one linear solve is exact.
    """
    params = params.copy()
    base = centre_in_lab(model, params)
    columns = []
    for k in (0, 1, 2):
        moved = params.copy()
        moved[k] += 1e-3
        columns.append((centre_in_lab(model, moved) - base) / 1e-3)
    params[[0, 1, 2]] += np.linalg.solve(np.array(columns).T, reference_centre - base)
    return params


def compensated_wave_stage(calibration, model, fixed):
    """lambda + TTH_0 stage in which a TTH_0 step only turns the detector face: the shifts follow so the
    detector centre stays where it was in the lab."""
    start = calibration.calibration_array
    free = [i for i in (6, 7) if i not in fixed]
    reference_centre = centre_in_lab(model, start)

    def params_for(x):
        params = start.copy()
        params[free] = x
        return compensate_shifts(model, params, reference_centre)

    out = least_squares(
        lambda x: calibration.residual(params_for(x)), start[free], bounds=(LOWER[free], UPPER[free]), jac="3-point"
    )
    calibration.set_calibration_array(params_for(out.x))
    return out.nfev


def final_rmse(calibration):
    return float(np.sqrt(np.mean(calibration.residual(calibration.calibration_array) ** 2)))


def fit_iterative(setup, rings, fixed, passes=None, polish=False, compensate=False, model=None):
    """Run RECIPE repeatedly; passes=None means until changes < TOLERANCE (at most MAX_PASSES).

    With polish, finish with one joint refinement of every non-fixed parameter from the result.

    Returns the calibration, the number of passes run, total nfev, seconds and the per-pass history.
    """
    calibration = PixelCalibration(rings, setup, wavelength=NOMINAL_WAVELENGTH)
    history = [calibration.calibration_array]
    nfev, start = 0, time.perf_counter()
    for n_pass in range(1, (passes or MAX_PASSES) + 1):
        for stage_name, stage_params in RECIPE:
            free = [i for i in stage_params if i not in fixed]
            if compensate and stage_name == "calibrate_wave_shift":
                nfev += compensated_wave_stage(calibration, model, fixed)
            elif free:
                nfev += calibration.refine(stage_name, free, LOWER[free], UPPER[free]).nfev
        history.append(calibration.calibration_array)
        change = np.abs(history[-1] - history[-2]) * UNITS
        if passes is None and change.max() < TOLERANCE:
            break
    if polish:
        free = [i for i in range(8) if i not in fixed]
        nfev += calibration.refine("joint polish", free, LOWER[free], UPPER[free]).nfev
    return calibration, n_pass, nfev, time.perf_counter() - start, np.array(history)


def run_iterative(setup, model, truth, true_maps, trials, noise):
    print(
        "\n  ITERATIVE: Shift -> Rotation -> lambda+TTH_0, repeated"
        f" (fixed passes {FIXED_PASSES}, or until change < {TOLERANCE} mm|deg|mA, max {MAX_PASSES})"
    )
    print(
        "  convention         stop rule           | noise-free: passes  map err   rmse/opt  | noisy: passes  map err"
        " med/95%      rmse/opt  nfev   s/fit"
    )
    for name, (fixed, two_l2, compensate) in CONVENTIONS.items():
        exact_rings = make_rings(setup, truth, two_l2, 0.0, None)
        noisy_rings = [
            make_rings(setup, truth, two_l2, noise, np.random.default_rng(12345 + k)) for k in range(trials)
        ]
        # global optimum on the same data: one-shot refinement of every non-fixed parameter
        optimum = [fit(setup, rings, fixed)[1].rmse for rings in noisy_rings]
        rules = [(n, False) for n in FIXED_PASSES] + [(None, False), (3, True), (None, True)]
        for passes, polish in rules:
            rule = (f"{passes:2d} passes" if passes else "tolerance") + (" +polish" if polish else "")
            calibration, used, _, _, _ = fit_iterative(setup, exact_rings, fixed, passes, polish, compensate, model)
            exact_map = np.abs(two_theta_maps(model, calibration.calibration_array) - true_maps).max()
            exact_rmse = final_rmse(calibration)
            used_n, map_err, ratio, nfevs, secs = [], [], [], [], []
            for rings, best in zip(noisy_rings, optimum):
                calibration, n, nfev, seconds, _ = fit_iterative(
                    setup, rings, fixed, passes, polish, compensate, model
                )
                used_n.append(n)
                map_err.append(np.abs(two_theta_maps(model, calibration.calibration_array) - true_maps).max())
                ratio.append(final_rmse(calibration) / best)
                nfevs.append(nfev)
                secs.append(seconds)
            print(
                f"  {name:18s} {rule:19s} | {used:5d}  {exact_map:9.1e}  {exact_rmse:9.1e}  | "
                f"{np.median(used_n):5.0f}  {np.median(map_err):.4f}/{np.percentile(map_err, 95):.4f}"
                f"  {np.median(ratio):8.3f}  {np.median(nfevs):5.0f}  {np.median(secs):5.2f}"
            )


def two_theta_maps(model, params, step=8):
    grid = np.meshgrid(np.arange(0, 1024, step), np.arange(0, 1024, step), indexing="ij")
    shift = DENEXDetectorShift(*params[:7])
    return np.stack([model.two_theta(*grid, -arm, L2_NEAR, shift) for arm in MAP_ARMS])


def jacobian_condition(calibration, params, free):
    scale = 1.0 / UNITS
    columns = []
    for i in free:
        e = np.zeros(8)
        e[i] = 1e-4 * scale[i]
        columns.append((calibration.residual(params + e) - calibration.residual(params - e)) / (2 * e[i]) * scale[i])
    return np.linalg.cond(np.array(columns).T)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20, help="noisy trials per geometry and convention")
    parser.add_argument("--noise", type=float, default=0.01, help="ring-position scatter in 2theta (deg)")
    parser.add_argument("--mode", choices=["single", "iterative", "both"], default="both")
    parser.add_argument("--only", nargs="+", metavar="LETTER", help="run only these conventions, e.g. --only A G")
    args = parser.parse_args()
    if args.only:
        wanted = {letter.upper() for letter in args.only}
        for name in list(CONVENTIONS):
            if name[0] not in wanted:
                del CONVENTIONS[name]

    setup = default_hb2b_geometry()
    model = DetectorModel(setup)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    for geometry_name, truth in TRUE_GEOMETRIES.items():
        true_maps = two_theta_maps(model, truth)
        print(
            f"\n{'=' * 100}\nTrue geometry '{geometry_name}':", dict(zip(PARAMETER_NAMES, np.round(truth * UNITS, 3)))
        )
        if args.mode in ("iterative", "both"):
            run_iterative(setup, model, truth, true_maps, args.trials, args.noise)
        if args.mode == "iterative":
            continue
        print("\n  SINGLE: one refinement of all free parameters")
        for name, (fixed, two_l2, compensate) in CONVENTIONS.items():
            if compensate:
                continue  # a one-shot fit explores the same parameter space as without compensation
            # equivalent truth for this convention: noise-free fit with the same free parameters
            exact, exact_stage, _, free = fit(setup, make_rings(setup, truth, two_l2, 0.0, None), fixed)
            equivalent = exact.calibration_array
            in_bounds = bool(np.all((equivalent[free] > LOWER[free]) & (equivalent[free] < UPPER[free])))
            cond = jacobian_condition(exact, equivalent, free)
            equiv_map_err = np.abs(two_theta_maps(model, equivalent) - true_maps).max()

            rng = np.random.default_rng(12345)
            fitted, sigmas, map_errors, dropped_errors, times, nfevs, statuses = [], [], [], [], [], [], []
            for _ in range(args.trials):
                calibration, stage, seconds, _ = fit(setup, make_rings(setup, truth, two_l2, args.noise, rng), fixed)
                params = calibration.calibration_array
                fitted.append(params)
                sigmas.append(calibration.calibration_error_array)
                map_errors.append(np.abs(two_theta_maps(model, params) - true_maps).max())
                dropped = params.copy()
                dropped[6] = 0.0  # what reduction applied before #1031, when the reader dropped TTH_0 (#1021)
                dropped_errors.append(np.abs(two_theta_maps(model, dropped) - true_maps).max())
                times.append(seconds)
                nfevs.append(stage.nfev)
                statuses.append(stage.status)
            fitted, sigmas = np.array(fitted), np.array(sigmas)
            scatter = fitted.std(axis=0, ddof=1)
            bias = fitted.mean(axis=0) - equivalent
            with np.errstate(invalid="ignore", divide="ignore"):
                pulls = np.where(sigmas > 0, (fitted - equivalent) / sigmas, np.nan)
            pull_std = np.nanstd(pulls, axis=0, ddof=1)

            print(
                f"\n  [{name}]  noise-free: rmse {exact_stage.rmse:.1e} deg, map err {equiv_map_err:.1e} deg, "
                f"in bounds: {in_bounds}, cond(J) {cond:.1e}"
            )
            print(
                f"    converged {sum(s > 0 for s in statuses)}/{args.trials}, "
                f"nfev {np.median(nfevs):.0f}, {np.median(times):.2f} s/fit"
            )
            print(
                f"    2theta-map error (max over detector, arms {MAP_ARMS}): median {np.median(map_errors):.4f} deg, "
                f"95% {np.percentile(map_errors, 95):.4f}; with TTH_0 dropped on read: median {np.median(dropped_errors):.3f}"
            )
            print("    param      equivalent    scatter   |bias|/scatter  reported-sigma/scatter  pull-std")
            for i in free:
                print(
                    f"    {PARAMETER_NAMES[i]:8s} {equivalent[i] * UNITS[i]:10.3f} {UNIT_LABELS[i]:3s} "
                    f"{scatter[i] * UNITS[i]:9.4f}   {abs(bias[i]) / scatter[i]:8.2f}"
                    f"   {np.median(sigmas[:, i]) / scatter[i] if np.all(sigmas[:, i] > 0) else float('nan'):12.2f}"
                    f"          {pull_std[i]:6.2f}"
                )


if __name__ == "__main__":
    main()
