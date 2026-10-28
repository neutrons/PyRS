"""
By-hand study (not collected by pytest): one detector for two calibration runs, with the first run's
2theta_0 locked at 0 (which removes the 2theta_0 / Shift_x / Rot_y degeneracy), the second run's beam
direction relative to it refined, a wavelength per run, the vertical pitch locked and the horizontal pitch
fitted as a diagnostic (it is not a calibration parameter). Shift_z is held, not refined with the pitch
(they trade almost exactly; user decision 2026-10-27): at the surveyed face (0) and with the wires 10 mm
behind it. Several Rot_y starts check that the fit converges to one solution (docs/ground_truths.md,
"Detection-plane tilt and the horizontal/vertical pixel pitch").

Input: ring points written by ``pyfai_ring_points_export.py`` (the first run is the 2theta_0 reference).
Run from the repository root:

    pixi run python tests/scripts/joint_two_theta_0_locked.py POINTS.npz [--py-mm 0.29296875]
"""

import argparse
import warnings

import numpy as np
from scipy.optimize import least_squares

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

FIELDS = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")
NOMINAL = 0.3 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("points")
    parser.add_argument("--py-mm", type=float, default=NOMINAL * 1e3, help="vertical pitch locked (mm)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    data = np.load(args.points)
    names = [str(n) for n in data["runs"]][:2]
    points = [RingPoints(**{f: data[f"{n}/{f}"] for f in FIELDS}) for n in names]
    wl = [float(data[f"{n}/mono_wavelength"]) for n in names]
    py = args.py_mm * 1e-3

    def calibrations(scale: float):
        geometry = DENEXDetectorGeometry(1024, 1024, scale * NOMINAL, py, 0.985, False)
        return [PixelCalibration(p, geometry, wavelength=w) for p, w in zip(points, wl)]

    def run(shift_z: float, rot_y0: float):
        # x: Shift_x, Shift_y, Rot_x, Rot_y, Rot_z, lambda_A, lambda_B, 2theta_0(B), horizontal pitch / nominal
        def residual(x):
            ca, cb = calibrations(x[8])
            pose = np.array([x[0], x[1], shift_z, x[2], x[3], x[4]])
            return np.concatenate([ca.residual(np.r_[pose, 0.0, x[5]]), cb.residual(np.r_[pose, x[7], x[6]])])

        x0 = np.array([0.0, 0.0, 0.0, rot_y0, 0.0, wl[0], wl[1], 0.0, 1.08])
        lo = [-0.15, -0.05, -10, -10, -5, wl[0] - 0.05, wl[1] - 0.05, -5, 0.95]
        hi = [0.15, 0.05, 10, 15, 5, wl[0] + 0.05, wl[1] + 0.05, 5, 1.2]
        out = least_squares(
            residual, x0, bounds=(lo, hi), x_scale=[1e-3, 1e-3, 0.1, 0.1, 0.1, 1e-3, 1e-3, 0.1, 1e-3],
            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000,
        )
        dof = out.fun.size - out.x.size
        error = np.sqrt(np.diag(np.linalg.pinv(out.jac.T @ out.jac)) * float(out.fun @ out.fun) / dof)
        n = points[0].index_0.size
        return out, error, (np.sqrt(np.mean(out.fun[:n] ** 2)), np.sqrt(np.mean(out.fun[n:] ** 2)))

    print(f"{names[0]} 2theta_0 = 0 (reference), {names[1]} offset free; vertical pitch {args.py_mm:.5f} mm")
    for shift_z in (0.0, 0.010):
        print(f"Shift_z = {shift_z * 1e3:+.0f} mm (held):")
        for rot_y0 in (0.0, 3.0, 6.0, 9.0):
            out, e, rms = run(shift_z, rot_y0)
            x = out.x
            print(
                f"  start Rot_y {rot_y0:3.0f}: chi2 {float(out.fun @ out.fun):.6f} RMS {rms[0]:.4f} / {rms[1]:.4f} deg |"
                f" px {x[8] * NOMINAL * 1e3:.4f} +/- {e[8] * NOMINAL * 1e3:.4f} mm, Rot_y {x[3]:+.3f} +/- {e[3]:.3f},"
                f" Rot_x {x[2]:+.2f}, Rot_z {x[4]:+.3f} deg, Shift_x {x[0] * 1e3:+.2f}, Shift_y {x[1] * 1e3:+.2f} mm,"
                f" 2theta_0({names[1]}) {x[7]:+.4f} +/- {e[7]:.4f} deg, lambda {x[5]:.5f} / {x[6]:.5f} A"
            )


if __name__ == "__main__":
    main()
