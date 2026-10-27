"""
By-hand study (not collected by pytest): joint ring calibration of two runs taken with the same detector
geometry but different monochromator settings, so one shared detector pose (shifts, rotations) and a
2theta_0 and wavelength per run.

Variant (pixel pitch fixed per call): the shared pose (Shift_x, Shift_y, Shift_z, Rot_x, Rot_y_0, Rot_z)
free and both runs' 2theta_0 free within +/-0.5 deg; each run sees Rot_y = 2theta_0 + Rot_y_0 through the
2theta_0 swing. The absolute level of 2theta_0 against Rot_y_0 is still degenerate (Rot_y_0 + d with both
2theta_0 - d fits identically), so only the bound limits it; the difference between the runs is measured.

|2theta_0| < 0.5 deg and the survey pose are measured constraints (docs/ground_truths.md). The pixel
pitch is set in the instrument setup; it is not a calibration parameter.

Run from the repository root:

    pixi run python tests/scripts/joint_calibration_fit.py RUN_A.nxs.h5 RUN_B.nxs.h5 [--mask MASK.xml]
        [--pitch MM ...] [--scan-pitch]
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
from pyrs.calibration.powder_standards import HB2B_STANDARDS, PowderStandard
from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.nexus_conversion import NeXusConvertingApp

# lattice constants (A): 2020 = HB2B_STANDARDS; 2023 = FitCalibration's production set (docs/ground_truths.md)
LATTICES = {"2020": None, "2023": {"Ni": 3.526314, "Fe": 2.865579, "Mo": 3.147664}}


def standards_for(lattice: str):
    """HB2B powder standards with the chosen lattice constants.

    ``lattice`` is a named set (``2020``, ``2023``) or explicit values such as ``Ni=3.52942,Fe=2.8674``;
    powders not named keep their 2020 value.
    """
    values: Optional[Dict[str, float]]
    if "=" in lattice:
        values = {name.strip(): float(a) for name, a in (item.split("=") for item in lattice.split(","))}
        unknown = set(values) - {s.name for s in HB2B_STANDARDS}
        if unknown:
            raise ValueError(f"unknown powders {sorted(unknown)}")
    else:
        values = LATTICES[lattice]
    if values is None:
        return HB2B_STANDARDS
    return tuple(PowderStandard(s.name, s.structure, values.get(s.name, s.lattice_a), s.sy) for s in HB2B_STANDARDS)


POSE = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z")
POSE_SCALE = np.array([1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1])
POSE_LO = np.array([-0.15, -0.05, -0.15, -10.0, -10.0, -5.0])
POSE_HI = -POSE_LO
TTH0_LIMIT = 0.5
# variant -> (free pose indices, run A 2theta_0 free?, 2theta_0 bound)
VARIANTS = {
    "shared pose free, both 2th0 in +/-0.5 deg": ([0, 1, 2, 3, 4, 5], True, TTH0_LIMIT),
}


def load(nexus: str, mask: str, standards=None):
    workspace = NeXusConvertingApp(nexus, mask).convert()
    wavelength = float(workspace.get_wavelength(False, False))
    points = extract_ring_points(workspace, wavelength, standards=standards, strips=5)
    return Path(nexus).name.split(".")[0], points, workspace.get_instrument_setup(), wavelength


def setup_with_pitch(setup: DENEXDetectorGeometry, pitch_mm: float) -> DENEXDetectorGeometry:
    num_rows, num_columns = setup.detector_size
    return DENEXDetectorGeometry(num_rows, num_columns, pitch_mm * 1e-3, pitch_mm * 1e-3, setup.arm_length, False)


def joint_fit(runs, pitch_mm: float, free_pose, fit_tth0_a: bool, tth0_bound: float):
    cals = [PixelCalibration(points, setup_with_pitch(setup, pitch_mm), wavelength=wl) for _, points, setup, wl in runs]
    wls = [run[3] for run in runs]
    n_pose = len(free_pose)

    def unpack(x):
        pose = np.zeros(6)
        pose[free_pose] = x[:n_pose]
        rest = list(x[n_pose:])
        tth0_a = rest.pop(0) if fit_tth0_a else 0.0
        wl_a, tth0_b, wl_b = rest
        return np.concatenate([pose, [tth0_a, wl_a]]), np.concatenate([pose, [tth0_b, wl_b]])

    def residual(x):
        pa, pb = unpack(x)
        return np.concatenate([cals[0].residual(pa), cals[1].residual(pb)])

    tail0 = ([0.0] if fit_tth0_a else []) + [wls[0], 0.0, wls[1]]
    tail_lo = ([-tth0_bound] if fit_tth0_a else []) + [wls[0] - 0.05, -tth0_bound, wls[1] - 0.05]
    tail_hi = ([tth0_bound] if fit_tth0_a else []) + [wls[0] + 0.05, tth0_bound, wls[1] + 0.05]
    tail_scale = ([0.1] if fit_tth0_a else []) + [1e-3, 0.1, 1e-3]
    x0 = np.concatenate([np.zeros(n_pose), tail0])
    out = least_squares(
        residual,
        x0,
        bounds=(np.concatenate([POSE_LO[free_pose], tail_lo]), np.concatenate([POSE_HI[free_pose], tail_hi])),
        x_scale=np.concatenate([POSE_SCALE[free_pose], tail_scale]),
        max_nfev=3000,
    )
    dof = out.fun.size - out.x.size
    err_x = np.sqrt(np.diag(np.linalg.pinv(out.jac.T @ out.jac)) * float(out.fun @ out.fun) / dof)
    pa, pb = unpack(out.x)
    ea, eb = unpack(err_x)
    if not fit_tth0_a:
        ea[6] = np.nan
    return out, cals, (pa, pb), (ea, eb)


def octants(cal: PixelCalibration, params) -> str:
    r = cal.residual(params)
    which = np.digitize(cal.ring_points.index_0, np.linspace(0, 1024, 9)) - 1
    return " ".join(f"{r[which == b].mean() * 1e3:+6.1f}" if np.any(which == b) else "    - " for b in range(8))


def report(label: str, runs, pitch_mm: float, result) -> None:
    out, cals, (pa, pb), (ea, eb) = result
    at_bound = np.any(out.active_mask != 0)
    print(f"\n  {label} | pitch {pitch_mm:.5f} mm | status {out.status}, {out.nfev} evaluations"
          + ("  [a parameter is at its bound]" if at_bound else ""))
    print(
        "    shared: "
        + ", ".join(
            f"{n} {v * s:.3f} +/- {e * s:.3f}"
            for n, v, e, s in zip(POSE, pa[:6], np.nan_to_num(ea[:6]), [1e3, 1e3, 1e3, 1, 1, 1])
        )
        + "  (mm, deg)"
    )
    for (name, *_), c, p, e in zip(runs, cals, (pa, pb), (ea, eb)):
        tth0 = f"{p[6]:+.4f} +/- {e[6]:.4f}" if np.isfinite(e[6]) else f"{p[6]:+.4f} (held)"
        rms = np.sqrt(np.mean(c.residual(p) ** 2))
        print(
            f"    {name}: 2theta_0 {tth0} deg, lambda {p[7]:.5f} +/- {e[7]:.5f} A, RMS {rms:.5f} deg"
            f" | residual by column octant (mdeg): {octants(c, p)}"
        )
    singular = np.linalg.svd(out.jac, compute_uv=False)
    print(f"    Jacobian singular values: largest {singular[0]:.3g}, smallest {singular[-1]:.3g}"
          f" (ratio {singular[0] / singular[-1]:.3g})")
    if not np.isfinite(ea[6]):
        print(f"    2theta_0({runs[1][0]}) - 2theta_0({runs[0][0]}) = {pb[6] - pa[6]:+.4f} +/- {eb[6]:.4f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--pitch", type=float, action="append", help="pixel pitch in mm (repeatable)")
    parser.add_argument("--lattice", default="2020", help="2020, 2023 or e.g. Ni=3.52942,Fe=2.8674")
    parser.add_argument("--scan-pitch", action="store_true", help="also optimise the common pitch per variant")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    standards = standards_for(args.lattice)
    runs = [load(args.run_a, args.mask, standards), load(args.run_b, args.mask, standards)]
    print(f"{args.lattice} lattice constants")
    print(f"Joint fit of {runs[0][0]} ({len(runs[0][1])} points) and {runs[1][0]} ({len(runs[1][1])} points)")
    for pitch in args.pitch or [0.29296875, 0.314]:
        for label, (free_pose, fit_a, bound) in VARIANTS.items():
            report(label, runs, pitch, joint_fit(runs, pitch, free_pose, fit_a, bound))
    if args.scan_pitch:
        for label, (free_pose, fit_a, bound) in VARIANTS.items():
            if 2 not in free_pose:  # pitch is degenerate with a free Shift_z
                best = minimize_scalar(
                    lambda p: float(np.sum(joint_fit(runs, p, free_pose, fit_a, bound)[0].fun ** 2)),
                    bounds=(0.29, 0.33),
                    method="bounded",
                    options={"xatol": 1e-5},
                )
                report(label + ", pitch optimised", runs, best.x, joint_fit(runs, best.x, free_pose, fit_a, bound))


if __name__ == "__main__":
    main()
