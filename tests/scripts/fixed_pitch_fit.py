"""
By-hand study (not collected by pytest): ring calibration of a run with the pixel pitch fixed at a
chosen value, e.g. the ~0.314 mm implied by the mask edges (300 mm active area over ~955 channels;
docs/ground_truths.md), compared with the nominal 0.29297 mm (300 mm / 1024).

For each pitch, three fits:

1. free geometry (staged passes, then all but 2theta_0, which is degenerate with Shift_x/Rot_y and is
   held at 0), also re-expressed at Rot_y = 0 (same detector) to compare with the survey;
2. survey distance: Shift_z = 0, rotations free, 2theta_0 held at 0;
3. survey pose: Shift_z = 0 and Rot = 0 (laser scanner); Shift_x, Shift_y, 2theta_0 and lambda free.

Diagnostic only: the pitch is set in the instrument setup handed to the fit; it is not refined and is
not a calibration parameter.

Run from the repository root:

    pixi run python tests/scripts/fixed_pitch_fit.py RUN.nxs.h5 [RUN2.nxs.h5 ...] [--mask MASK.xml]
        [--pitch X_MM[,Y_MM] ...]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

from pyrs.calibration.pixel_pos_calibration import DetectorModel, PixelCalibration, extract_ring_points
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.nexus_conversion import NeXusConvertingApp

DEFAULT_PITCHES = ("0.29296875", "0.314", "0.3132,0.3148")
HEADER = (
    f"    {'fit':26s} {'RMS deg':>8} {'Sx mm':>8} {'Sy mm':>7} {'Sz mm':>8} {'Rx':>7} {'Ry':>7} {'Rz':>7}"
    f" {'2th0':>7} {'lambda':>8}   residual by column octant (mdeg)"
)


def setup_with_pitch(setup: DENEXDetectorGeometry, px_mm: float, py_mm: float) -> DENEXDetectorGeometry:
    num_rows, num_columns = setup.detector_size
    return DENEXDetectorGeometry(num_rows, num_columns, px_mm * 1e-3, py_mm * 1e-3, setup.arm_length, False)


def rms(c: PixelCalibration, params=None) -> float:
    params = c.calibration_array if params is None else params
    return float(np.sqrt(np.mean(c.residual(params) ** 2)))


def octants(c: PixelCalibration, params) -> str:
    r = c.residual(params)
    which = np.digitize(c.ring_points.index_0, np.linspace(0, 1024, 9)) - 1
    return " ".join(f"{r[which == b].mean() * 1e3:+6.1f}" if np.any(which == b) else "    - " for b in range(8))


def row(label: str, c: PixelCalibration, params) -> str:
    p = np.asarray(params)
    return (
        f"    {label:26s} {rms(c, p):8.5f} {p[0] * 1e3:8.2f} {p[1] * 1e3:7.2f} {p[2] * 1e3:8.2f} {p[3]:7.3f}"
        f" {p[4]:7.3f} {p[5]:7.3f} {p[6]:7.3f} {p[7]:8.5f}   {octants(c, p)}"
    )


def fresh(points, setup, wavelength) -> PixelCalibration:
    return PixelCalibration(points, setup, wavelength=wavelength)


def fit_free(points, setup, wavelength) -> PixelCalibration:
    c = fresh(points, setup, wavelength)
    for _ in range(3):
        c.CalibrateShift()
        c.CalibrateRotation()
        c.calibrate_wave_length()
    wl = c.calibration_array[7]
    names = ["Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "Lambda"]
    c.refine("free, 2th0 = 0", names, [-0.15, -0.05, -0.15, -10, -10, -5, wl - 0.05], [0.15, 0.05, 0.15, 10, 10, 5, wl + 0.05])
    return c


def at_zero_rot_y(c: PixelCalibration, setup) -> np.ndarray:
    """The same detector re-expressed with 2theta_0 chosen so that Rot_y = 0."""
    model = DetectorModel(setup)
    p = c.calibration_array.copy()
    cal = DENEXDetectorShift(*p[:7])
    for _ in range(5):
        cal = model.with_two_theta_0(cal, cal.two_theta_0 + cal.rotation_y, setup.arm_length)
    q = np.array(
        [cal.center_shift_x, cal.center_shift_y, cal.center_shift_z, cal.rotation_x, cal.rotation_y, cal.rotation_z,
         cal.two_theta_0, p[7]]
    )
    return q


def fit_survey_distance(points, setup, wavelength) -> PixelCalibration:
    c = fresh(points, setup, wavelength)
    for _ in range(3):
        c.calibrate_shiftx()
        c.calibrate_shifty()
        c.CalibrateRotation()
        c.calibrate_wave_length()
    wl = c.calibration_array[7]
    names = ["Shift_x", "Shift_y", "Rot_x", "Rot_y", "Rot_z", "Lambda"]
    c.refine("Shift_z = 0", names, [-0.15, -0.05, -10, -10, -5, wl - 0.05], [0.15, 0.05, 10, 10, 5, wl + 0.05])
    return c


def fit_survey_pose(points, setup, wavelength) -> PixelCalibration:
    c = fresh(points, setup, wavelength)
    names = ["Shift_x", "Shift_y", "TTH_0", "Lambda"]
    for _ in range(2):
        c.refine("Shift_z = 0, Rot = 0", names, [-0.15, -0.05, -8, wavelength - 0.05], [0.15, 0.05, 8, wavelength + 0.05])
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--pitch", action="append", help="X_MM or X_MM,Y_MM (repeatable)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    pitches = []
    for text in args.pitch or DEFAULT_PITCHES:
        values = [float(v) for v in text.split(",")]
        pitches.append((values[0], values[-1]))

    for run in args.runs:
        workspace = NeXusConvertingApp(run, args.mask).convert()
        wavelength = float(workspace.get_wavelength(False, False))
        base = workspace.get_instrument_setup()
        points = extract_ring_points(workspace, wavelength, strips=5)
        print(f"\n{Path(run).name.split('.')[0]}: {len(points)} ring points, mono {wavelength} A, arm {base.arm_length} m")
        for px, py in pitches:
            setup = setup_with_pitch(base, px, py)
            print(f"  pitch {px:.5f} x {py:.5f} mm (x across columns, y along rows); active {px * 1024:.1f} x {py * 1024:.1f} mm")
            print(HEADER)
            c = fit_free(points, setup, wavelength)
            print(row("1 free (2th0 = 0)", c, c.calibration_array))
            print(row("1 free, same at Rot_y = 0", c, at_zero_rot_y(c, setup)))
            c = fit_survey_distance(points, setup, wavelength)
            print(row("2 Shift_z = 0", c, c.calibration_array))
            c = fit_survey_pose(points, setup, wavelength)
            print(row("3 Shift_z = 0, Rot = 0", c, c.calibration_array))


if __name__ == "__main__":
    main()
