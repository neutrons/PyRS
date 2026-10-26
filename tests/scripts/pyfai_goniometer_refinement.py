"""
By-hand study (not collected by pytest): refine the HB2B detector geometry from our ring points with
pyFAI's geometry as the forward model and pyFAI's own parameters, as an independent check of the ring
calibration (docs/ground_truths.md).

Model (pyFAI conventions, verified in ``pyfai_forward_model_check.py``): pixel (index_0, index_1) is
pyFAI (d2, d1); a lab position is ``R . (p1 - poni1, p2 - poni2, dist)`` with ``R = R3 R2 R1``. The
detector pose in the arm frame is constant (dist, poni1, poni2, rot1_0, rot2_0, rot3_0), and the HB2B arm
turns it about pyFAI axis 1 outside it: ``R = R1(arm + delta_run) . R0`` (``delta_run``: the change of
beam direction between runs; 0 for the first run). The arm is not simply added to rot1, which pyFAI
applies first. Residual per ring point: pyFAI 2theta minus 2 asin(lambda / 2d), as in our refinement.
A rigid pose has no separate 2theta_0: rot1_0 carries -(2theta_0 + Rot_y).

Fits: each run alone (pose + lambda) and both runs jointly (one pose, lambda per run, delta for run B), at
the nominal 0.29297 mm and the mask-implied 0.314 mm pitch. Our refinements are converted to the same
pose for comparison (from the pyFAI forward-model check at arm 89 deg).

Run in a throwaway environment from the repository root:

    pixi exec --spec pyfai --spec numexpr --spec scipy python tests/scripts/pyfai_goniometer_refinement.py POINTS.npz
"""

import logging
import sys

import numpy as np
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry
from scipy.optimize import least_squares

PITCHES = (0.3 / 1024, 0.314e-3)
# our refinements as pyFAI poses at arm 89 deg (calib_diagnostics/pyfai_forward_model_check.txt, 3b)
OURS = {
    "free fit, 0.293 mm (3510)": (0.3 / 1024, 89.0, [0.909069, 0.152233, 0.079405, 84.6246, -0.1808, -0.0041]),
    "joint fit, 0.314 mm": (0.314e-3, 89.0, [0.970083, 0.164653, 0.058396, 83.0420, -0.1244, -0.2514]),
}


def r1(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def r2(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def r3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def compose(rot1: float, rot2: float, rot3: float) -> np.ndarray:
    return r3(rot3) @ r2(rot2) @ r1(rot1)


def decompose(matrix: np.ndarray) -> tuple:
    """pyFAI (rot1, rot2, rot3) of ``R3 R2 R1``: row 3 is (s2, -s1 c2, c1 c2), column 1 is c2 (c3, s3, s2)."""
    rot2 = np.arcsin(np.clip(matrix[2, 0], -1.0, 1.0))
    rot1 = np.arctan2(-matrix[2, 1], matrix[2, 2])
    rot3 = np.arctan2(matrix[1, 0], matrix[0, 0])
    return rot1, rot2, rot3


def pose_at_arm(pose, arm_rad: float) -> list:
    dist, poni1, poni2, rot1_0, rot2_0, rot3_0 = pose
    return [dist, poni1, poni2, *decompose(r1(arm_rad) @ compose(rot1_0, rot2_0, rot3_0))]


def arm_frame_pose(pose_at, arm_deg: float) -> list:
    """Inverse of ``pose_at_arm``: a pose fitted at one arm angle, expressed in the arm frame (deg)."""
    dist, poni1, poni2, a, b, c = pose_at
    matrix = r1(-np.radians(arm_deg)) @ compose(np.radians(a), np.radians(b), np.radians(c))
    return [dist, poni1, poni2, *np.degrees(decompose(matrix))]


class Run:
    def __init__(self, data, name: str, pitch: float):
        self.name = name
        self.mono = float(data[f"{name}/mono_wavelength"])
        self.d = data[f"{name}/d_spacing"]
        self.index_0 = data[f"{name}/index_0"]
        self.index_1 = data[f"{name}/index_1"]
        self.arm = data[f"{name}/det_2theta"]
        self.detector = Detector(pixel1=pitch, pixel2=pitch, max_shape=tuple(int(v) for v in data[f"{name}/detector_size"]))
        self.groups = [(arm, np.flatnonzero(self.arm == arm)) for arm in np.unique(self.arm)]

    def residual(self, pose, wavelength: float, delta_deg: float = 0.0) -> np.ndarray:
        tth = np.empty(self.d.size)
        for arm, index in self.groups:
            geometry = Geometry(*pose_at_arm(pose, np.radians(arm + delta_deg)), detector=self.detector)
            tth[index] = np.degrees(geometry.tth(self.index_1[index], self.index_0[index]))
        bragg = 2 * np.degrees(np.arcsin(np.clip(wavelength / (2 * self.d), -1, 1)))
        return tth - bragg


def errors(out) -> np.ndarray:
    dof = out.fun.size - out.x.size
    return np.sqrt(np.diag(np.linalg.pinv(out.jac.T @ out.jac)) * float(out.fun @ out.fun) / dof)


SCALE = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
POSE_LABEL = "dist {:.5f} m, poni1 {:.3f} mm, poni2 {:.3f} mm, rot1_0 {:+.4f}, rot2_0 {:+.4f}, rot3_0 {:+.4f} deg"


def show_pose(pose_rad_m) -> str:
    dist, poni1, poni2, a, b, c = pose_rad_m
    return POSE_LABEL.format(dist, poni1 * 1e3, poni2 * 1e3, *np.degrees([a, b, c]))


def main(path: str) -> None:
    logging.disable(logging.WARNING)
    data = np.load(path)
    names = [str(n) for n in data["runs"]]
    print(f"pyFAI goniometer refinement of {path}: runs {names}")
    for pitch in PITCHES:
        runs = [Run(data, name, pitch) for name in names]
        start = [0.985, 512 * pitch, 512 * pitch, 0.0, 0.0, 0.0]
        print(f"\npitch {pitch * 1e3:.5f} mm (start: nominal pose, mono wavelength)")
        for run in runs:
            out = least_squares(
                lambda x, run=run: run.residual(x[:6], x[6]), start + [run.mono], x_scale=SCALE + [1e-4], xtol=1e-14
            )
            e = errors(out)
            rms = np.sqrt(np.mean(out.fun**2))
            print(f"  {run.name} alone: RMS {rms:.5f} deg, {out.nfev} evaluations, lambda {out.x[6]:.5f} +/- {e[6]:.5f} A")
            print(f"    {show_pose(out.x[:6])}")
            print(f"    errors: dist {e[0] * 1e3:.2f} mm, poni1 {e[1] * 1e3:.2f} mm, poni2 {e[2] * 1e3:.2f} mm, rot "
                  + ", ".join(f"{v:.3f}" for v in np.degrees(e[3:6])) + " deg")
        if len(runs) == 2:
            a, b = runs

            def joint(x):
                return np.concatenate([a.residual(x[:6], x[6]), b.residual(x[:6], x[7], x[8])])

            out = least_squares(joint, start + [a.mono, b.mono, 0.0], x_scale=SCALE + [1e-4, 1e-4, 1e-3], xtol=1e-14)
            e = errors(out)
            ra, rb = out.fun[: a.d.size], out.fun[a.d.size :]
            print(f"  joint {a.name} + {b.name}: RMS {np.sqrt(np.mean(ra**2)):.5f} / {np.sqrt(np.mean(rb**2)):.5f} deg,"
                  f" lambda {out.x[6]:.5f} / {out.x[7]:.5f} A, delta({b.name}) {out.x[8]:+.4f} +/- {e[8]:.4f} deg")
            print(f"    {show_pose(out.x[:6])}")

    print("\nour refinements as pyFAI arm-frame poses (deg):")
    for label, (pitch, arm, pose) in OURS.items():
        dist, poni1, poni2, a_, b_, c_ = arm_frame_pose(pose, arm)
        print(f"  {label}: " + POSE_LABEL.format(dist, poni1 * 1e3, poni2 * 1e3, a_, b_, c_))


if __name__ == "__main__":
    main(sys.argv[1])
