"""
By-hand study (not collected by pytest): FitCalibration on HB2B_3510 with scipy's default finite-difference
step vs per-parameter steps, injected at run time only (no code change).

FitCalibration's residual has a noise floor of ~1e-3 deg per residual (re-binning + peak refits at every
evaluation), so scipy's default step (~6e-6 relative) gives noise-dominated derivatives for 2theta_0,
the rotations and Shift_y (docs/ground_truths.md). This compares the default with steps of 0.5 mm
(shifts), 0.01 deg (angles) and ~1.5e-4 A (wavelength).

Run from the repository root:

    pixi run python tests/scripts/fitcalibration_stepsize_study.py {default|per-parameter} {2023|2020} {nominal|perturbed} [RUN.nxs.h5]

The perturbed start (lambda 1.531 A) is meant for HB2B_3510 (mono 1.54 A).
"""

import sys
import time
from typing import Dict, List, Optional

import numpy as np

import pyrs.calibration.mantid_peakfit_calibration as mpc

NEXUS = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
MASK = "tests/data/HB2B_Mask_12-18-19.xml"
LATTICES = {"2023": None, "2020": [3.523799438, 2.8663982, 3.14719963]}
STARTS: Dict[str, Optional[np.ndarray]] = {
    "nominal": None,
    "perturbed": np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.5, 1.531]),
}
# relative steps (scipy: h = diff_step * max(1, |x|)): 0.5 mm, 0.01 deg, ~1.5e-4 A
PER_PARAMETER = np.array([5e-4, 5e-4, 5e-4, 1e-2, 1e-2, 1e-2, 1e-2, 1e-4])
STAGES = (
    ("CalibrateShift", [0, 1, 2]),
    ("CalibrateRotation", [3, 4, 5]),
    ("calibrate_wave_shift", [6, 7]),
    ("FullCalibration", list(range(8))),
)
UNITS = np.array([1e3, 1e3, 1e3, 1, 1, 1, 1, 1])  # report mm, deg, A


def main(mode: str, lattice: str, start: str, nexus: str = NEXUS) -> None:
    current: Dict[str, List[int]] = {"free": []}
    counter = {"n": 0}
    original = mpc.least_squares

    def wrapped(fun, x0, *args, **kwargs):
        def counted(*a, **k):
            counter["n"] += 1
            return fun(*a, **k)

        if mode == "per-parameter":
            kwargs["diff_step"] = PER_PARAMETER[current["free"]]
        return original(counted, x0, *args, **kwargs)

    mpc.least_squares = wrapped
    tag = f"[{nexus.split('/')[-1][:9]} | {mode:13s} | {lattice} | {start:9s}]"
    fc = mpc.FitCalibration(nexus_file=nexus, mask_file=MASK, max_nfev=40)
    if LATTICES[lattice] is not None:
        fc._ref_lattice = LATTICES[lattice]
        fc.get_powder_lines()
    start_values = STARTS[start]
    if start_values is not None:
        fc._calib = start_values.copy()
    t0 = time.time()
    for name, free in STAGES:
        current["free"], counter["n"] = free, 0
        t = time.time()
        getattr(fc, name)()
        r = fc.get_alignment_residual(np.array(fc._calib))
        print(
            f"{tag} {name:20s} {time.time() - t:5.0f}s {counter['n']:3d} evals  rmse {np.sqrt(np.mean(r**2)):.4f}"
            f"  status {fc._calibstatus}  params {np.round(np.array(fc._calib) * UNITS, 4)}",
            flush=True,
        )
    print(
        f"{tag} FINAL errors {np.round(np.array(fc._caliberr) * UNITS, 4)}  total {time.time() - t0:.0f}s", flush=True
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])
