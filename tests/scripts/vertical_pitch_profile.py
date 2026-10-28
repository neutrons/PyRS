"""
By-hand study (not collected by pytest): how well do the ring points constrain the vertical pixel pitch?

The horizontal pitch is held (default 0.314 mm, the mask-implied value), the vertical pitch is scanned,
and at each value the detector pose is refined, either freely (2theta_0 held at 0; it is degenerate with
Shift_x/Rot_y) or at the surveyed face pose (Shift_z = 0, Rot = 0; Shift_x, Shift_y, 2theta_0, lambda free).
A parabola through chi2 (in units of the best fit's residual variance) gives the best vertical pitch and
its 1-sigma error (docs/ground_truths.md, "Detection-plane tilt and the horizontal/vertical pixel pitch").

Input: ring points written by ``pyfai_ring_points_export.py``. Run from the repository root:

    pixi run python tests/scripts/vertical_pitch_profile.py POINTS.npz [--px-mm 0.314]
"""

import argparse
import warnings

import numpy as np

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

FIELDS = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")
FREE_POSE = ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "Lambda"]


def load_points(data, name: str) -> RingPoints:
    return RingPoints(**{field: data[f"{name}/{field}"] for field in FIELDS})


def fit(points: RingPoints, wavelength: float, px: float, py: float, survey: bool):
    """Sum of squared residuals and number of points after refining the pose at this pitch."""
    calibration = PixelCalibration(points, DENEXDetectorGeometry(1024, 1024, px, py, 0.985, False), wavelength=wavelength)
    if survey:
        for _ in range(2):
            calibration.refine(
                "survey pose",
                ["Shift_x", "Shift_y", "TTH_0", "Lambda"],
                [-0.15, -0.05, -8.0, wavelength - 0.05],
                [0.15, 0.05, 8.0, wavelength + 0.05],
            )
    else:
        for _ in range(3):
            calibration.CalibrateShift()
            calibration.CalibrateRotation()
            calibration.calibrate_wave_length()
        wl = calibration.calibration_array[7]
        calibration.refine(
            "free pose", FREE_POSE, [-0.15, -0.05, -0.15, -10, -10, -5, wl - 0.05], [0.15, 0.05, 0.15, 10, 10, 5, wl + 0.05]
        )
    residual = calibration.residual(calibration.calibration_array)
    return float(residual @ residual), residual.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("points")
    parser.add_argument("--px-mm", type=float, default=0.314, help="horizontal pitch held (mm)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    data = np.load(args.points)
    px = args.px_mm * 1e-3
    pys = np.linspace(0.290e-3, 0.338e-3, 9)
    for name in [str(n) for n in data["runs"]]:
        points = load_points(data, name)
        wavelength = float(data[f"{name}/mono_wavelength"])
        for survey in (False, True):
            results = [fit(points, wavelength, px, py, survey) for py in pys]
            chi2 = np.array([r[0] for r in results])
            n = results[0][1]
            variance = chi2.min() / (n - 8)
            a, b, _ = np.polyfit(pys, chi2 / variance, 2)
            best, error = (-b / (2 * a), 1 / np.sqrt(a)) if a > 0 else (pys[int(np.argmin(chi2))], np.inf)
            label = "survey pose (Shift_z = 0, Rot = 0)" if survey else "free pose (2theta_0 = 0)"
            rms = " / ".join(f"{np.sqrt(c / n):.4f}" for c in (chi2[0], chi2[len(pys) // 2], chi2[-1]))
            print(
                f"{name} {label:35s} px {args.px_mm:.4f} mm: vertical pitch {best * 1e3:.4f} +/- {error * 1e3:.4f} mm;"
                f" RMS at py = {pys[0] * 1e3:.3f} / {pys[len(pys) // 2] * 1e3:.3f} / {pys[-1] * 1e3:.3f} mm: {rms} deg"
            )


if __name__ == "__main__":
    main()
