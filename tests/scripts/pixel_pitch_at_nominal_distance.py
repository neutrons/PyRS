"""
By-hand study (not collected by pytest): the detector pixel pitch implied by a calibration run if the
detector sits at its nominal distance (Shift_z = 0).

Ring angles depend only on directions from the sample, so a detector with pitch p at distance L and one
with pitch k*p at distance k*L (shifts scaled by k, same rotations) give identical pixel 2theta/eta: ring
data alone cannot separate distance from pitch (docs/ground_truths.md). This holds Shift_z = 0 and scans
a common pitch scale k (both directions), refining the other parameters at each k, so the RMS minimum is
the pitch the data imply at the nominal distance. It also refines with x and y scales separately free.
Diagnostic only: the pixel pitch is not a parameter of the calibration.

Run from the repository root:

    pixi run python tests/scripts/pixel_pitch_at_nominal_distance.py RUN.nxs.h5 [--mask MASK.xml]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.nexus_conversion import NeXusConvertingApp

FREE = ["Shift_x", "Shift_y", "Rot_x", "Rot_y", "Rot_z", "Lambda"]  # Shift_z = 0, TTH_0 = 0 held


def scaled_setup(setup: DENEXDetectorGeometry, kx: float, ky: float) -> DENEXDetectorGeometry:
    num_rows, num_columns = setup.detector_size
    px, py = setup.pixel_dimension
    return DENEXDetectorGeometry(num_rows, num_columns, px * kx, py * ky, setup.arm_length, False)


def fit(points, setup, wavelength: float, kx: float, ky: float) -> PixelCalibration:
    calibration = PixelCalibration(points, scaled_setup(setup, kx, ky), wavelength=wavelength)
    for _ in range(3):
        calibration.calibrate_shiftx()
        calibration.calibrate_shifty()
        calibration.CalibrateRotation()
        calibration.calibrate_wave_length()
    wl = calibration.calibration_array[7]
    calibration.refine(
        "pose and lambda", FREE, [-0.1, -0.05, -10.0, -10.0, -5.0, wl - 0.05], [0.1, 0.05, 10.0, 10.0, 5.0, wl + 0.05]
    )
    return calibration


def rms(c: PixelCalibration) -> float:
    return float(np.sqrt(np.mean(c.residual(c.calibration_array) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    workspace = NeXusConvertingApp(args.run, args.mask).convert()
    wavelength = float(workspace.get_wavelength(False, False))
    setup = workspace.get_instrument_setup()
    points = extract_ring_points(workspace, wavelength, strips=5)
    px, py = setup.pixel_dimension
    name = Path(args.run).name.split(".")[0]
    print(f"{name}: nominal pitch {px * 1e3:.5f} x {py * 1e3:.5f} mm, arm {setup.arm_length} m, {len(points)} points")

    print(f"  {'k':>6} {'pitch mm':>9} {'RMS deg':>9} {'Sx mm':>7} {'Sy mm':>7} {'Rx':>7} {'Ry':>7} {'Rz':>7} {'lambda':>8}")
    for k in (1.00, 1.04, 1.06, 1.07, 1.08, 1.09, 1.10, 1.12):
        c = fit(points, setup, wavelength, k, k)
        p = c.calibration_array
        print(
            f"  {k:6.3f} {px * k * 1e3:9.5f} {rms(c):9.5f} {p[0] * 1e3:7.2f} {p[1] * 1e3:7.2f}"
            f" {p[3]:7.3f} {p[4]:7.3f} {p[5]:7.3f} {p[7]:8.5f}"
        )

    best = minimize_scalar(lambda k: rms(fit(points, setup, wavelength, k, k)), bounds=(0.95, 1.2), method="bounded",
                           options={"xatol": 1e-4})
    k = best.x
    print(f"  best common scale k = {k:.4f}: pitch {px * k * 1e3:.4f} mm, active width {px * k * 1024 * 1e3:.1f} mm"
          f" (nominal {px * 1024 * 1e3:.1f} mm)")

    # separate x / y scales: rows (index_1, along the ring) and columns (index_0, across rings)
    def rms_xy(kxy):
        return rms(fit(points, setup, wavelength, *kxy))

    grid = [(kx, ky) for kx in (k - 0.03, k, k + 0.03) for ky in (k - 0.1, k, k + 0.1)]
    print("  separate scales (kx, ky) -> RMS:")
    for kx, ky in grid:
        print(f"    kx {kx:.3f} ky {ky:.3f}: RMS {rms_xy((kx, ky)):.5f}")


if __name__ == "__main__":
    main()
