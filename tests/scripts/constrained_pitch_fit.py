"""
By-hand study (not collected by pytest): ring fit with the detector pose fixed by the laser-scanner
survey (nominal distance, Shift_z = 0; face square to the arm, Rot_x = Rot_y = Rot_z = 0), the pixel
pitch free.

With Rot_y fixed the Shift_x / Rot_y / 2theta_0 degeneracy is broken, so 2theta_0 is refined. Free per
run: Shift_x, Shift_y, 2theta_0, wavelength; the pitch scale k (common, then kx/ky separately) is
optimised in an outer loop. A joint fit of two runs shares the pitch and the shifts (same detector) with
a 2theta_0 and wavelength per run. The residual pattern across the detector columns shows what a
uniform pitch cannot describe. Diagnostic only: the pixel pitch is not a calibration parameter.

Run from the repository root:

    pixi run python tests/scripts/constrained_pitch_fit.py RUN_A.nxs.h5 [RUN_B.nxs.h5] [--mask MASK.xml]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.nexus_conversion import NeXusConvertingApp

FREE = ["Shift_x", "Shift_y", "TTH_0", "Lambda"]


def scaled_setup(setup: DENEXDetectorGeometry, kx: float, ky: float) -> DENEXDetectorGeometry:
    num_rows, num_columns = setup.detector_size
    px, py = setup.pixel_dimension
    return DENEXDetectorGeometry(num_rows, num_columns, px * kx, py * ky, setup.arm_length, False)


def fit(run, kx: float, ky: float) -> PixelCalibration:
    _, points, setup, wavelength = run
    c = PixelCalibration(points, scaled_setup(setup, kx, ky), wavelength=wavelength)
    c.refine("survey pose", FREE, [-0.1, -0.05, -6.0, wavelength - 0.05], [0.1, 0.05, 6.0, wavelength + 0.05])
    c.refine("survey pose", FREE, [-0.1, -0.05, -6.0, wavelength - 0.05], [0.1, 0.05, 6.0, wavelength + 0.05])
    return c


def rms(c: PixelCalibration) -> float:
    return float(np.sqrt(np.mean(c.residual(c.calibration_array) ** 2)))


def report(c: PixelCalibration, label: str) -> None:
    p = c.calibration_array
    e = c.calibration_error_array
    print(
        f"  {label}: RMS {rms(c):.5f} deg | Shift_x {p[0] * 1e3:7.2f} +/- {e[0] * 1e3:.2f} mm,"
        f" Shift_y {p[1] * 1e3:6.2f} +/- {e[1] * 1e3:.2f} mm, 2theta_0 {p[6]:7.4f} +/- {e[6]:.4f} deg,"
        f" lambda {p[7]:.5f} +/- {e[7]:.5f} A"
    )


def residual_pattern(c: PixelCalibration) -> None:
    points = c.ring_points
    r = c.residual(c.calibration_array)
    edges = np.linspace(0, 1024, 9)
    which = np.digitize(points.index_0, edges) - 1
    cells = []
    for b in range(8):
        sel = which == b
        cells.append(f"{r[sel].mean() * 1e3:+6.1f}({sel.sum()})" if sel.any() else "   -   ")
    print("    mean residual (mdeg, n) by column octant 0..1024: " + " ".join(cells))


def load(nexus: str, mask: str):
    workspace = NeXusConvertingApp(nexus, mask).convert()
    wavelength = float(workspace.get_wavelength(False, False))
    points = extract_ring_points(workspace, wavelength, strips=5)
    return Path(nexus).name.split(".")[0], points, workspace.get_instrument_setup(), wavelength


def single(run) -> None:
    name = run[0]
    print(f"\n{name} ({len(run[1])} ring points): Shift_z = 0, Rot = 0 held")
    c = fit(run, 1.0, 1.0)
    report(c, "nominal pitch      ")
    residual_pattern(c)
    k = minimize_scalar(lambda k: rms(fit(run, k, k)), bounds=(0.95, 1.2), method="bounded", options={"xatol": 1e-5}).x
    c = fit(run, k, k)
    report(c, f"common k = {k:.4f}   ")
    residual_pattern(c)
    res = minimize(lambda kk: rms(fit(run, *kk)), [k, k], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7})
    kx, ky = res.x
    c = fit(run, kx, ky)
    report(c, f"kx {kx:.4f}, ky {ky:.4f}")
    residual_pattern(c)
    px = run[2].pixel_dimension[0] * 1e3
    print(f"    pitch: common {px * k:.4f} mm; x {px * kx:.4f} mm, y {px * ky:.4f} mm (nominal {px:.5f})")


def joint(run_a, run_b) -> None:
    (name_a, pa_, setup, wl_a), (name_b, pb_, _, wl_b) = run_a, run_b

    def residual_at(k, x):
        ca = PixelCalibration(pa_, scaled_setup(setup, k, k), wavelength=wl_a)
        cb = PixelCalibration(pb_, scaled_setup(setup, k, k), wavelength=wl_b)
        pa = np.array([x[0], x[1], 0, 0, 0, 0, x[2], x[3]])
        pb = np.array([x[0], x[1], 0, 0, 0, 0, x[4], x[5]])
        return np.concatenate([ca.residual(pa), cb.residual(pb)])

    def fit_at(k):
        x0 = [0.0, 0.0, 0.0, wl_a, 0.0, wl_b]
        return least_squares(lambda x: residual_at(k, x), x0, x_scale=[1e-3, 1e-3, 0.1, 1e-3, 0.1, 1e-3])

    k = minimize_scalar(lambda k: float(np.sum(fit_at(k).fun ** 2)), bounds=(0.95, 1.2), method="bounded",
                        options={"xatol": 1e-5}).x
    out = fit_at(k)
    n_a = len(pa_)
    dof = out.fun.size - out.x.size - 1
    err = np.sqrt(np.diag(np.linalg.pinv(out.jac.T @ out.jac)) * float(out.fun @ out.fun) / dof)
    print(f"\nJoint {name_a} + {name_b}: Shift_z = 0, Rot = 0 held; one pitch and shifts, 2theta_0 and lambda per run")
    print(
        f"  k = {k:.4f} (pitch {setup.pixel_dimension[0] * k * 1e3:.4f} mm) | RMS {name_a}"
        f" {np.sqrt(np.mean(out.fun[:n_a] ** 2)):.5f}, {name_b} {np.sqrt(np.mean(out.fun[n_a:] ** 2)):.5f} deg"
    )
    print(f"  Shift_x {out.x[0] * 1e3:.2f} +/- {err[0] * 1e3:.2f} mm, Shift_y {out.x[1] * 1e3:.2f} +/- {err[1] * 1e3:.2f} mm")
    print(f"  {name_a}: 2theta_0 {out.x[2]:.4f} +/- {err[2]:.4f} deg, lambda {out.x[3]:.5f} +/- {err[3]:.5f} A")
    print(f"  {name_b}: 2theta_0 {out.x[4]:.4f} +/- {err[4]:.4f} deg, lambda {out.x[5]:.5f} +/- {err[5]:.5f} A")
    print("  (errors at fixed k)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    runs = [load(r, args.mask) for r in args.runs]
    for run in runs:
        single(run)
    if len(runs) == 2:
        joint(*runs)


if __name__ == "__main__":
    main()
