"""
By-hand study (not collected by pytest): can 2theta_0 be refined from ring data?

1. Profile: for each run, hold 2theta_0 at a grid of values, refine the other seven parameters
   (staged passes, then all-but-2theta_0 jointly) and report the RMS. At one detector distance
   2theta_0 is exactly degenerate with Shift_x/Rot_y (docs/ground_truths.md), so a flat profile is
   expected; a minimum would mean the data constrain it.
2. Joint fit of two runs: one shared detector pose (6 parameters) with a 2theta_0 and a wavelength per
   run. The absolute 2theta_0 is still degenerate, so run A's is held and the difference
   2theta_0(B) - 2theta_0(A) is refined; it is measurable because the shared pose must serve both.

Run from the repository root:

    pixi run python tests/scripts/two_theta_0_profile.py RUN_A.nxs.h5 RUN_B.nxs.h5 [--mask MASK.xml]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
from pyrs.core.nexus_conversion import NeXusConvertingApp

GRID = (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
UNITS = np.array([1e3, 1e3, 1e3, 1, 1, 1, 1, 1])  # mm, deg, A
NOT_TTH0 = ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "Lambda"]


def load(nexus: str, mask: str):
    workspace = NeXusConvertingApp(nexus, mask).convert()
    wavelength = float(workspace.get_wavelength(False, False))
    points = extract_ring_points(workspace, wavelength, strips=5)
    return Path(nexus).name.split(".")[0], points, workspace.get_instrument_setup(), wavelength


def refine_with_tth0_held(points, setup, wavelength: float, tth0: float) -> PixelCalibration:
    calibration = PixelCalibration(points, setup, wavelength=wavelength)
    params = calibration.calibration_array.copy()
    params[6] = tth0
    calibration.set_calibration_array(params)
    for _ in range(3):
        calibration.CalibrateShift()
        calibration.CalibrateRotation()
        calibration.calibrate_wave_length()
    wl = calibration.calibration_array[7]
    calibration.refine(
        "all but TTH_0",
        NOT_TTH0,
        [-0.1, -0.05, -0.15, -10.0, -10.0, -5.0, wl - 0.05],
        [0.1, 0.05, 0.15, 10.0, 10.0, 5.0, wl + 0.05],
    )
    return calibration


def profile(name: str, points, setup, wavelength: float) -> None:
    print(f"\n{name}: 2theta_0 held, the other seven refined ({len(points)} ring points)")
    print(f"  {'2th0':>6} {'RMS deg':>9} {'Sx mm':>8} {'Sy mm':>7} {'Sz mm':>8} {'Rx':>7} {'Ry':>7} {'Rz':>7} {'lambda':>8}")
    for tth0 in GRID:
        c = refine_with_tth0_held(points, setup, wavelength, tth0)
        p = c.calibration_array * UNITS
        rms = float(np.sqrt(np.mean(c.residual(c.calibration_array) ** 2)))
        print(
            f"  {tth0:6.2f} {rms:9.5f} {p[0]:8.2f} {p[1]:7.2f} {p[2]:8.2f} {p[3]:7.3f} {p[4]:7.3f} {p[5]:7.3f} {p[7]:8.5f}"
        )


def joint(run_a, run_b) -> None:
    (name_a, points_a, setup, wl_a), (name_b, points_b, _, wl_b) = run_a, run_b
    start_a = refine_with_tth0_held(points_a, setup, wl_a, 0.0)
    cal_a = PixelCalibration(points_a, setup, wavelength=wl_a)
    cal_b = PixelCalibration(points_b, setup, wavelength=wl_b)
    pose0 = start_a.calibration_array[:6]

    def unpack(x):
        pa = np.concatenate([x[:6], [0.0, x[6]]])  # run A's 2theta_0 held at 0
        pb = np.concatenate([x[:6], [x[7], x[8]]])
        return pa, pb

    def residual(x):
        pa, pb = unpack(x)
        return np.concatenate([cal_a.residual(pa), cal_b.residual(pb)])

    x0 = np.concatenate([pose0, [start_a.calibration_array[7], 0.0, wl_b]])
    out = least_squares(residual, x0, x_scale=[1e-3] * 3 + [0.1] * 3 + [1e-3, 0.1, 1e-3], max_nfev=2000)
    pa, pb = unpack(out.x)
    r_a, r_b = cal_a.residual(pa), cal_b.residual(pb)
    dof = out.fun.size - out.x.size
    jtj = out.jac.T @ out.jac
    cov = np.linalg.pinv(jtj) * float(out.fun @ out.fun) / dof
    err = np.sqrt(np.diag(cov))
    print(f"\nJoint fit, shared pose, {name_a} 2theta_0 held at 0 (status {out.status}, {out.nfev} evaluations)")
    print(f"  RMS {name_a} {np.sqrt(np.mean(r_a**2)):.5f} deg, {name_b} {np.sqrt(np.mean(r_b**2)):.5f} deg")
    labels = ["Shift_x mm", "Shift_y mm", "Shift_z mm", "Rot_x deg", "Rot_y deg", "Rot_z deg"]
    scale = [1e3, 1e3, 1e3, 1, 1, 1]
    for i, (label, s) in enumerate(zip(labels, scale)):
        print(f"  {label:11s} {out.x[i] * s:9.3f} +/- {err[i] * s:.3f}")
    print(f"  lambda {name_a} {out.x[6]:.5f} +/- {err[6]:.5f} A")
    print(f"  lambda {name_b} {out.x[8]:.5f} +/- {err[8]:.5f} A")
    print(f"  2theta_0({name_b}) - 2theta_0({name_a}) = {out.x[7]:.4f} +/- {err[7]:.4f} deg")
    singular = np.linalg.svd(out.jac / np.sqrt(dof), compute_uv=False)
    print(f"  Jacobian condition number {singular[0] / singular[-1]:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    runs = [load(args.run_a, args.mask), load(args.run_b, args.mask)]
    for run in runs:
        profile(*run)
    joint(*runs)


if __name__ == "__main__":
    main()
