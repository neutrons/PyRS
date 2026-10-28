"""
By-hand diagnostic (not collected by pytest): score several calibrations of one run on the same footing,
for #1018 (ring calibration against FitCalibration and the GUI calibration).

Each calibration JSON is read directly (shifts, rotations, TTH_0, Lambda; pitch 0.29297 mm), plus "none":
the nominal pose at the monochromator wavelength. For each:

1. ring-point 2theta RMS on the run's ring points (``pyrs-pixel-calibration --ring-points``), 2020
   lattice constants, and the drift of each ring's residual from the bottom to the top strip (#1018
   success criterion: the Delta-vs-strip drift is flattened);
2. FitCalibration's own eta-wedge peak residual (``get_alignment_residual``) with the mask, for the 2023
   (production) and 2020 lattice constants;
3. the largest pixel 2theta difference to a reference calibration over the detector at every sub-run's
   arm angle, compared as geometry objects (DetectorModel, as reduction builds the detector), and each
   calibration re-expressed at 2theta_0 = 0 (the same detector) so Shift_x / Rot_y are comparable.

Run from the repository root:

    pixi run python tests/scripts/score_calibrations.py RUN.nxs.h5 RING_POINTS.npz --mask MASK.xml \
        --calibration GUI=gui.json --calibration ring=ring.json [--reference ring] [--no-fitcalibration]
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from pyrs.calibration.pixel_pos_calibration import DetectorModel, PixelCalibration, RingPoints
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

CAL_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")
LATTICES = {"2023": [3.526314, 2.865579, 3.147664], "2020": [3.523799438, 2.8663982, 3.14719963]}
SETUP = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)


def calibration_vector(path: str) -> np.ndarray:
    raw = json.loads(Path(path).read_text())
    raw.setdefault("TTH_0", raw.get("two_theta_0", 0.0))
    return np.array([float(raw[k]) for k in CAL_KEYS])


def strip_drift(points: RingPoints, residual: np.ndarray) -> np.ndarray:
    """Per ring: residual change from the bottom to the top strip (deg), from a straight-line fit."""
    drifts = []
    for key in sorted(set(zip(points.sub_run.tolist(), points.hkl_squared.tolist()))):
        sel = (points.sub_run == key[0]) & (points.hkl_squared == key[1])
        if sel.sum() < 3:
            continue
        rows = points.index_1[sel]
        slope = np.polyfit(rows, residual[sel], 1)[0]
        drifts.append(slope * (rows.max() - rows.min()))
    return np.array(drifts)


def at_zero_two_theta_0(model: DetectorModel, params: np.ndarray) -> np.ndarray:
    shift = model.with_two_theta_0(DENEXDetectorShift(*params[:7]), 0.0, SETUP.arm_length)
    return np.array(
        [shift.center_shift_x, shift.center_shift_y, shift.center_shift_z, shift.rotation_x, shift.rotation_y,
         shift.rotation_z, shift.two_theta_0, params[7]]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("ring_points")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--calibration", action="append", default=[], help="LABEL=FILE.json (repeatable)")
    parser.add_argument("--reference", default=None, help="label for the pixel-2theta comparison (default: last)")
    parser.add_argument("--no-fitcalibration", action="store_true", help="skip the FitCalibration residual")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    points = RingPoints.load(args.ring_points)
    model = DetectorModel(SETUP)
    from pyrs.core.nexus_conversion import NeXusConvertingApp

    workspace = NeXusConvertingApp(args.run, args.mask).convert()
    mono = float(workspace.get_wavelength(False, False))
    arms = sorted({float(workspace.get_detector_2theta(s)) for s in workspace.get_sub_runs()})
    calibrations = {"none": np.r_[np.zeros(7), mono]}
    for item in args.calibration:
        label, path = item.split("=", 1)
        calibrations[label] = calibration_vector(path)
    reference = args.reference or list(calibrations)[-1]

    print(f"{Path(args.run).name}: {len(points)} ring points, mono {mono} A, arms {arms[0]:g}..{arms[-1]:g} deg")
    print("\ncalibrations as written and at 2theta_0 = 0 (same detector; mm, deg, A):")
    for label, p in calibrations.items():
        z = at_zero_two_theta_0(model, p)
        print(
            f"  {label:16s} Shift {p[0]*1e3:7.2f} {p[1]*1e3:6.2f} {p[2]*1e3:7.2f}  Rot {p[3]:+.3f} {p[4]:+.3f} {p[5]:+.3f}"
            f"  2th0 {p[6]:+.4f}  lambda {p[7]:.5f} | at 2th0 = 0: Shift_x {z[0]*1e3:7.2f}, Shift_z {z[2]*1e3:7.2f}, Rot_y {z[4]:+.3f}"
        )

    print("\n1. ring points (2020 lattice): 2theta RMS and per-ring strip drift (bottom -> top strip)")
    calibration = PixelCalibration(points, SETUP, wavelength=mono)
    for label, p in calibrations.items():
        residual = calibration.residual(p)
        drift = strip_drift(points, residual)
        print(
            f"  {label:16s} RMS {np.sqrt(np.mean(residual**2)):.4f} deg, mean {np.mean(residual):+.4f};"
            f" strip drift mean |d| {np.mean(np.abs(drift)):.4f}, max |d| {np.max(np.abs(drift)):.4f} deg ({drift.size} rings)"
        )

    print(f"\n3. largest pixel 2theta difference to '{reference}' over the detector at the run's arms:")
    grid = np.meshgrid(np.arange(0, 1024, 8), np.arange(0, 1024, 8), indexing="ij")
    ref = calibrations[reference]
    for label, p in calibrations.items():
        worst = 0.0
        for arm in arms:
            a = model.two_theta(grid[0], grid[1], -arm, SETUP.arm_length, DENEXDetectorShift(*ref[:7]))
            b = model.two_theta(grid[0], grid[1], -arm, SETUP.arm_length, DENEXDetectorShift(*p[:7]))
            worst = max(worst, float(np.max(np.abs(b - a))))
        print(f"  {label:16s} max |d 2theta| {worst:.4f} deg (lambda {p[7] - ref[7]:+.5f} A)")

    if args.no_fitcalibration:
        return
    from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

    print("\n2. FitCalibration eta-wedge peak residual (mask applied):")
    for lattice, values in LATTICES.items():
        fit = FitCalibration(nexus_file=args.run, mask_file=args.mask)
        fit._ref_lattice = values
        fit.get_powder_lines()
        for label, p in calibrations.items():
            start = time.time()
            residual = np.asarray(fit.get_alignment_residual(np.array(p, dtype=float)))
            print(
                f"  {lattice} lattice  {label:16s} RMS {np.sqrt(np.mean(residual**2)):.4f} deg over {residual.size}"
                f" peaks ({time.time() - start:.0f} s)"
            )


if __name__ == "__main__":
    main()
