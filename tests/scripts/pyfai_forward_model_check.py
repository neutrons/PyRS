"""
By-hand study, part 2 of 2 (not collected by pytest): check the ring calibration's forward model
(DetectorModel) against pyFAI's detector geometry, using maps written by ``pyfai_forward_model_export.py``.

1. Straight-through nominal detector (arm 0): pyFAI's analytic geometry (dist = arm length, PONI at the
   detector centre, no rotations) must reproduce our 2theta map with no fitting. This tests the pixel
   size and the pixel-centre convention.
2. Every geometry and arm angle: fit pyFAI's six-parameter rigid flat detector (dist, poni1, poni2,
   rot1, rot2, rot3) to our 2theta map. A residual at rounding level means our model is a rigid flat
   detector in pyFAI's sense. The pixel-index mapping (which of our indices is pyFAI's dim1 / dim2, and
   whether one is reversed) is chosen by the fit.
3. Compare our eta with pyFAI's chi on the fitted geometry to find their relation (sign / offset).
   2theta alone cannot tell a detector from its mirror image (the 2theta map is symmetric about the
   scattering plane); eta can.
4. Across arm angles of one calibration, report how the fitted pyFAI pose changes, i.e. what a pyFAI
   goniometer transformation for the HB2B arm must look like.

pyFAI is not a PyRS dependency; run in a throwaway environment from the repository root:

    pixi exec --spec pyfai --spec numexpr --spec scipy python tests/scripts/pyfai_forward_model_check.py MAPS.npz
"""

import logging
import sys
from itertools import product
from typing import Optional, Tuple

import numpy as np
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry
from scipy.optimize import least_squares

STEP = 16  # pixel sub-sampling for the fits; the final comparison uses every pixel
MAPPINGS = {
    # our [index_0, index_1] map -> pyFAI [dim1, dim2] array
    "dim1=index_1, dim2=index_0": lambda a: a.T,
    "dim1=index_1, dim2=1023-index_0": lambda a: a.T[:, ::-1],
    "dim1=1023-index_1, dim2=index_0": lambda a: a.T[::-1, :],
    "dim1=1023-index_1, dim2=1023-index_0": lambda a: a.T[::-1, ::-1],
}
PARAMS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


FULL = np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij")  # pyFAI (d1, d2) pixel indices
SUB = (FULL[0][::STEP, ::STEP], FULL[1][::STEP, ::STEP])


def tth_map(detector: Detector, x, pixels=FULL) -> np.ndarray:
    """pyFAI 2theta (deg) at pixel centres; integer indices are pixel centres in Geometry.tth."""
    return np.degrees(Geometry(*x, detector=detector).tth(*pixels))


def chi_map(detector: Detector, x, pixels=FULL) -> np.ndarray:
    return np.degrees(Geometry(*x, detector=detector).chi(*pixels))


def fit(detector: Detector, target: np.ndarray, x0) -> tuple:
    sub = target[::STEP, ::STEP]

    def residual(x):
        return (tth_map(detector, x, SUB) - sub).ravel()

    out = least_squares(residual, x0, x_scale=[1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    full = tth_map(detector, out.x) - target
    return out.x, float(np.max(np.abs(full))), float(np.sqrt(np.mean(full**2)))


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def main(path: str) -> None:
    logging.disable(logging.WARNING)
    data = np.load(path)
    arms = data["detector_arms"]
    arm_length = float(data["arm_length"])
    names = sorted({key.split("/")[0] for key in data.files if "/" in key})
    print(f"pyFAI forward-model check of {path}: geometries {names}, arms {list(arms)}")

    # 1. analytic straight-through check
    pitch = float(data["nominal/pitch"])
    detector = Detector(pixel1=pitch, pixel2=pitch, max_shape=(1024, 1024))
    ours = data["nominal/0/tth"]
    analytic = tth_map(detector, [arm_length, 512 * pitch, 512 * pitch, 0.0, 0.0, 0.0])
    print("\n1. arm 0, nominal, analytic pyFAI geometry (dist = arm length, PONI at the centre, no rotation):")
    for label, mapping in MAPPINGS.items():
        diff = analytic - mapping(ours)
        print(f"   {label:38s} max |d2theta| {np.max(np.abs(diff)):.3e} deg")

    # 2-3. fit pyFAI's rigid detector to every map
    print("\n2./3. pyFAI rigid-detector fit to our 2theta map (all pixels compared after the fit):")
    results = {}
    for name in names:
        pitch = float(data[f"{name}/pitch"])
        detector = Detector(pixel1=pitch, pixel2=pitch, max_shape=(1024, 1024))
        for arm in arms:
            key = f"{name}/{arm:g}"
            best: Optional[Tuple[str, np.ndarray, float, float]] = None
            for label, mapping in MAPPINGS.items():
                target = mapping(data[f"{key}/tth"])
                a = np.radians(arm)
                for r1, r2 in product((a, -a, 0.0), (a, -a, 0.0)):
                    if arm != 0 and r1 != 0 and r2 != 0:
                        continue
                    x0 = [arm_length, 512 * pitch, 512 * pitch, r1, r2, 0.0]
                    x, worst, rms = fit(detector, target, x0)
                    if best is None or worst < best[2]:
                        best = (label, x, worst, rms)
            assert best is not None
            label, x, worst, rms = best
            eta = MAPPINGS[label](data[f"{key}/eta"])
            chi = chi_map(detector, x)
            relations = {
                "eta = chi": wrap(eta - chi),
                "eta = -chi": wrap(eta + chi),
                "eta = chi + 180": wrap(eta - chi - 180),
                "eta = 180 - chi": wrap(eta + chi - 180),
                "eta = chi + 90": wrap(eta - chi - 90),
                "eta = 90 - chi": wrap(eta + chi - 90),
                "eta = chi - 90": wrap(eta - chi + 90),
                "eta = -90 - chi": wrap(eta + chi + 90),
            }
            rel, rel_diff = min(relations.items(), key=lambda kv: np.max(np.abs(kv[1])))
            results[key] = (label, x)
            print(
                f"   {key:18s} {label:38s} max |d2theta| {worst:.2e} rms {rms:.2e} deg | {rel} to {np.max(np.abs(rel_diff)):.1e} deg"
                f" | dist {x[0]:.6f} m poni1 {x[1] * 1e3:8.3f} poni2 {x[2] * 1e3:8.3f} mm"
                f" rot1 {np.degrees(x[3]):9.4f} rot2 {np.degrees(x[4]):9.4f} rot3 {np.degrees(x[5]):9.4f} deg"
            )

    # 3b. joint 2theta + eta fit: 2theta alone does not fix the rotation about the beam, eta does.
    # Mapping and eta = chi are taken from the nominal fits above.
    print("\n3b. pyFAI fit to 2theta and eta together (dim1=index_1, dim2=index_0, eta = chi):")
    for name in names:
        pitch = float(data[f"{name}/pitch"])
        detector = Detector(pixel1=pitch, pixel2=pitch, max_shape=(1024, 1024))
        calibration = data[f"{name}/calibration"]
        for arm in arms:
            if arm == 0:
                continue
            key = f"{name}/{arm:g}"
            tth = data[f"{key}/tth"].T
            eta = data[f"{key}/eta"].T

            def residual(x, tth=tth, eta=eta, detector=detector):
                d_tth = tth_map(detector, x, SUB) - tth[::STEP, ::STEP]
                d_eta = wrap(chi_map(detector, x, SUB) - eta[::STEP, ::STEP])
                return np.concatenate([d_tth.ravel(), d_eta.ravel()])

            x0 = [arm_length, 512 * pitch, 512 * pitch, np.radians(arm + calibration[6]), 0.0, 0.0]
            out = least_squares(residual, x0, x_scale=[1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2], xtol=1e-15, ftol=1e-15)
            x = out.x
            d_tth = np.max(np.abs(tth_map(detector, x) - tth))
            d_eta = np.max(np.abs(wrap(chi_map(detector, x) - eta)))
            results[key] = ("dim1=index_1, dim2=index_0", x)
            print(
                f"   {key:18s} max |d2theta| {d_tth:.1e} |d eta| {d_eta:.1e} deg | dist {x[0]:.6f} m"
                f" poni1 {x[1] * 1e3:8.3f} poni2 {x[2] * 1e3:8.3f} mm rot1 {np.degrees(x[3]):9.4f}"
                f" rot2 {np.degrees(x[4]):8.4f} rot3 {np.degrees(x[5]):8.4f} deg | rot1 - arm {np.degrees(x[3]) - arm:+.4f}"
            )

    # 4. how the pose follows the arm
    print("\n4. fitted pyFAI pose against the arm angle (arm > 0 only):")
    for name in names:
        rows = [(arm, results[f"{name}/{arm:g}"][1]) for arm in arms if arm != 0]
        if len(rows) < 2:
            continue
        arm_values = np.array([a for a, _ in rows])
        poses = np.array([x for _, x in rows])
        line = []
        for j, param in enumerate(PARAMS):
            values = poses[:, j] if j < 3 else np.degrees(poses[:, j])
            slope = np.polyfit(arm_values, values, 1)[0]
            unit = "m" if j == 0 else ("m" if j < 3 else "deg")
            line.append(f"{param} {values.min():.6g}..{values.max():.6g} {unit} (slope/arm {slope:+.4f})")
        print(f"   {name}: " + "; ".join(line))


if __name__ == "__main__":
    main(sys.argv[1])
