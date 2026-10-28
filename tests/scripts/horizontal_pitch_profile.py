"""
By-hand study (not collected by pytest): the horizontal pixel pitch implied by the ring points with the
vertical pitch locked (default: nominal 0.29297 mm), under the constraints discussed in
docs/ground_truths.md ("Detection-plane tilt and the horizontal/vertical pixel pitch"):

* survey: the surveyed face pose, Shift_z = 0 and Rot = 0 (Shift_x, Shift_y, 2theta_0, lambda free);
* 2theta_0 = 0 (removes the 2theta_0 / Shift_x / Rot_y degeneracy), Shift_z free or held at 0 (rotations,
  Shift_x/y and lambda free), each run alone and both runs jointly (one detector, both 2theta_0 = 0).

The horizontal pitch is scanned and a parabola through chi2 near its minimum (in units of the best fit's
residual variance) gives the best value and its 1-sigma error. All fits are direct least squares.

Input: ring points written by ``pyfai_ring_points_export.py``. Run from the repository root:

    pixi run python tests/scripts/horizontal_pitch_profile.py POINTS.npz [--py-mm 0.29296875]
"""

import argparse
import warnings
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

FIELDS = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")


def survey_fit(points: RingPoints, wavelength: float, px: float, py: float):
    calibration = PixelCalibration(points, DENEXDetectorGeometry(1024, 1024, px, py, 0.985, False), wavelength=wavelength)
    for _ in range(2):
        calibration.refine(
            "survey pose",
            ["Shift_x", "Shift_y", "TTH_0", "Lambda"],
            [-0.15, -0.05, -8.0, wavelength - 0.05],
            [0.15, 0.05, 8.0, wavelength + 0.05],
        )
    residual = calibration.residual(calibration.calibration_array)
    return float(residual @ residual), residual.size, calibration.calibration_array


def zero_two_theta_0_fit(runs: Sequence, px: float, py: float, shift_z_free: bool):
    """One detector pose for all given runs, every run's 2theta_0 = 0, a wavelength per run."""
    geometry = DENEXDetectorGeometry(1024, 1024, px, py, 0.985, False)
    calibrations = [PixelCalibration(points, geometry, wavelength=wl) for points, wl in runs]
    pose_index = [0, 1, 2, 3, 4, 5] if shift_z_free else [0, 1, 3, 4, 5]
    k = len(pose_index)

    def params(x):
        pose = np.zeros(6)
        pose[pose_index] = x[:k]
        return [np.concatenate([pose, [0.0, x[k + i]]]) for i in range(len(runs))]

    def residual(x):
        return np.concatenate([c.residual(p) for c, p in zip(calibrations, params(x))])

    pose_lo = np.array([-0.15, -0.05, -0.15, -10.0, -10.0, -5.0])[pose_index]
    wavelengths = np.array([wl for _, wl in runs])
    out = least_squares(
        residual,
        np.concatenate([np.zeros(k), wavelengths]),
        bounds=(np.concatenate([pose_lo, wavelengths - 0.05]), np.concatenate([-pose_lo, wavelengths + 0.05])),
        x_scale=[1e-3 if i < 3 else 0.1 for i in pose_index] + [1e-3] * len(runs),
    )
    rms = [float(np.sqrt(np.mean(c.residual(p) ** 2))) for c, p in zip(calibrations, params(out.x))]
    return float(out.fun @ out.fun), out.fun.size, params(out.x)[0], rms


def best_of(pxs: np.ndarray, chi2: np.ndarray, n: int):
    k = int(np.argmin(chi2))
    window = slice(max(k - 2, 0), min(k + 3, len(pxs)))
    a, b, _ = np.polyfit(pxs[window], chi2[window] / (chi2[k] / (n - 9)), 2)
    return (-b / (2 * a), 1 / np.sqrt(a)) if a > 0 else (pxs[k], np.inf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("points")
    parser.add_argument("--py-mm", type=float, default=0.3 / 1.024, help="vertical pitch locked (mm)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    data = np.load(args.points)
    py = args.py_mm * 1e-3
    names = [str(n) for n in data["runs"]]
    runs = {
        name: (RingPoints(**{f: data[f"{name}/{f}"] for f in FIELDS}), float(data[f"{name}/mono_wavelength"]))
        for name in names
    }
    pxs = np.linspace(0.300e-3, 0.330e-3, 13)
    print(f"vertical pitch locked at {args.py_mm:.5f} mm")

    for name in names:
        results = [survey_fit(*runs[name], px, py) for px in pxs]
        chi2 = np.array([r[0] for r in results])
        best, error = best_of(pxs, chi2, results[0][1])
        p = results[int(np.argmin(chi2))][2]
        print(
            f"{name} survey pose (Shift_z = 0, Rot = 0): px {best * 1e3:.4f} +/- {error * 1e3:.4f} mm,"
            f" needs 2theta_0 {p[6]:+.3f} deg, Shift_x {p[0] * 1e3:+.1f} mm, RMS {np.sqrt(chi2.min() / results[0][1]):.4f}"
        )

    for selection in [[name] for name in names] + ([names] if len(names) > 1 else []):
        for shift_z_free in (True, False):
            results = [zero_two_theta_0_fit([runs[n] for n in selection], px, py, shift_z_free) for px in pxs]
            chi2 = np.array([r[0] for r in results])
            best, error = best_of(pxs, chi2, results[0][1])
            k = int(np.argmin(chi2))
            p, rms = results[k][2], results[k][3]
            label = "+".join(selection) + (" | Shift_z free" if shift_z_free else " | Shift_z = 0")
            print(
                f"{label:32s} 2theta_0 = 0: px {best * 1e3:.4f} +/- {error * 1e3:.4f} mm | Shift_z {p[2] * 1e3:6.1f} mm,"
                f" Shift_x {p[0] * 1e3:+.1f} mm, Rot x/y/z {p[3]:+.2f} / {p[4]:+.2f} / {p[5]:+.2f} deg | RMS "
                + " / ".join(f"{v:.4f}" for v in rms)
            )


if __name__ == "__main__":
    main()
