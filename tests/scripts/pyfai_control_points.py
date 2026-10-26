"""
By-hand study (not collected by pytest): let pyFAI pick its own control points on the HB2B detector
images (``SingleGeometry.extract_cp``) and refine them with the pyFAI goniometer model of
``pyfai_goniometer_refinement.py``, as an independent check of our ring extraction (docs/ground_truths.md).

Start as our extraction does: the nominal pose and the monochromator wavelength. pyFAI looks for each
reflection within +/- 1/4 of the spacing to its neighbours around its predicted 2theta and keeps local
maxima of a blurred image (``Massif``), about ``pts_per_deg`` points per degree of azimuth. Then:

1. counts of control points per sub-run and reflection, against our ring points;
2. their residual under our refined geometry (the step-2 pose of each run alone), i.e. whether pyFAI's
   points sit on our rings;
3. a refinement from pyFAI's points alone (each run, and jointly), against ours.

Images and ring points come from ``pyfai_ring_points_export.py``. Run in a throwaway environment from the
repository root:

    pixi exec --spec pyfai --spec numexpr --spec scipy python tests/scripts/pyfai_control_points.py POINTS.npz \
        [--pts-per-deg 1.0] [--blur SIGMA_PX]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector
from pyFAI.goniometer import SingleGeometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyfai_goniometer_refinement import SCALE, Run, errors, pose_at_arm, show_pose  # noqa: E402

PITCH = 0.3 / 1024
# allowed h^2+k^2+l^2, as pyrs.calibration.powder_standards.REFLECTIONS
REFLECTIONS = {"FCC": (3, 4, 8, 11, 12, 16, 19), "BCC": (2, 4, 6, 8, 10, 12)}


def extract(data, name: str, pts_per_deg: float, blur: float) -> dict:
    """pyFAI control points of every sub-run, as ring-point arrays."""
    mono = float(data[f"{name}/mono_wavelength"])
    keep = data[f"{name}/mask_keep"]
    detector = Detector(pixel1=PITCH, pixel2=PITCH, max_shape=keep.T.shape)
    detector.mask = (keep.T == 0).astype(np.int8)
    nominal = [0.985, 512 * PITCH, 512 * PITCH, 0.0, 0.0, 0.0]
    columns: dict = {k: [] for k in ("index_0", "index_1", "det_2theta", "d_spacing", "sub_run", "powder", "hkl_squared")}
    for sub_run, arm, image, powder, a, structure in zip(
        data[f"{name}/image_sub_runs"],
        data[f"{name}/image_det_2theta"],
        data[f"{name}/images"],
        data[f"{name}/image_powder"],
        data[f"{name}/image_lattice"],
        data[f"{name}/image_structure"],
    ):
        if not powder:
            continue
        hkl = np.array(REFLECTIONS[str(structure)])
        d_spacings = a / np.sqrt(hkl)
        geometry = AzimuthalIntegrator(*pose_at_arm(nominal, np.radians(arm)), detector=detector, wavelength=mono * 1e-10)
        calibrant = Calibrant(dSpacing=list(d_spacings), wavelength=mono * 1e-10)
        picture = image.T if blur <= 0 else gaussian_filter(image.T, blur)
        single = SingleGeometry(f"{name}-{sub_run}", image=picture, calibrant=calibrant, detector=detector,
                                geometry=geometry)
        np.random.seed(int(sub_run))
        single.extract_cp(pts_per_deg=pts_per_deg)
        points = single.geometry_refinement.data
        if points is None or len(points) == 0:
            continue
        for d1, d2, ring in points:
            columns["index_1"].append(d1)
            columns["index_0"].append(d2)
            columns["det_2theta"].append(arm)
            columns["d_spacing"].append(d_spacings[int(ring)])
            columns["hkl_squared"].append(hkl[int(ring)])
            columns["sub_run"].append(int(sub_run))
            columns["powder"].append(str(powder))
    out = {f"{name}/{k}": np.asarray(v) for k, v in columns.items()}
    out[f"{name}/mono_wavelength"] = np.array(mono)
    out[f"{name}/detector_size"] = data[f"{name}/detector_size"]
    return out


def counts(points: dict, name: str) -> dict:
    keys = zip(points[f"{name}/sub_run"].tolist(), points[f"{name}/hkl_squared"].tolist())
    table: dict = {}
    for key in keys:
        table[key] = table.get(key, 0) + 1
    return table


def fit_alone(run: Run, robust: float = 0.0):
    """Pose and wavelength from one run's points; ``robust`` > 0: soft-L1 loss with that scale (deg)."""
    start = [0.985, 512 * PITCH, 512 * PITCH, 0.0, 0.0, 0.0, run.mono]
    kwargs = {"loss": "soft_l1", "f_scale": robust} if robust > 0 else {}
    out = least_squares(lambda x: run.residual(x[:6], x[6]), start, x_scale=SCALE + [1e-4], xtol=1e-14, **kwargs)
    return out, errors(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("points")
    parser.add_argument("--pts-per-deg", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur (pixels) before extraction; 0: none")
    parser.add_argument("--robust", type=float, default=0.0, help="soft-L1 scale (deg) for the pyFAI-point fit; 0: none")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    data = np.load(args.points)
    names = [str(n) for n in data["runs"]]
    print(
        f"pyFAI control points on {args.points}: pts_per_deg {args.pts_per_deg}, blur {args.blur} px,"
        f" robust {args.robust} deg"
    )

    ours_runs, theirs_runs = [], []
    for name in names:
        picked = extract(data, name, args.pts_per_deg, args.blur)
        ours, theirs = Run(data, name, PITCH), Run(picked, name, PITCH)
        ours_runs.append(ours)
        theirs_runs.append(theirs)
        ours_counts, theirs_counts = counts(data, name), counts(picked, name)
        print(f"\n{name}: pyFAI {theirs.d.size} control points, ours {ours.d.size} ring points")
        for key in sorted(set(ours_counts) | set(theirs_counts)):
            print(f"   sub-run {key[0]:2d} hkl^2 {key[1]:2d}: pyFAI {theirs_counts.get(key, 0):3d}, ours {ours_counts.get(key, 0):2d}")

        # 2. pyFAI's points under our refined geometry (our points, step-2 pose)
        out_ours, _ = fit_alone(ours)
        residual = theirs.residual(out_ours.x[:6], out_ours.x[6])
        median = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - median))
        print(f"   pyFAI points under our refined geometry: median {median:+.4f}, robust sigma {mad:.4f},"
              f" rms {np.sqrt(np.mean(residual**2)):.4f} deg; |r| > 0.2 deg: {np.sum(np.abs(residual) > 0.2)}")
        for key in sorted(theirs_counts):
            sel = (theirs.sub_run == key[0]) & (theirs.hkl == key[1])
            print(f"      sub-run {key[0]:2d} hkl^2 {key[1]:2d}: median residual {np.median(residual[sel]):+.4f} deg")

        # 3. refinement from pyFAI's points alone
        out, e = fit_alone(theirs, args.robust)
        r = theirs.residual(out.x[:6], out.x[6])
        print(
            f"   refined from pyFAI points: RMS {np.sqrt(np.mean(r**2)):.4f} deg, robust sigma"
            f" {1.4826 * np.median(np.abs(r - np.median(r))):.4f}, lambda {out.x[6]:.5f} A"
        )
        print(f"      pyFAI points: {show_pose(out.x[:6])}")
        print(f"      our points:   {show_pose(out_ours.x[:6])} (RMS {np.sqrt(np.mean(out_ours.fun**2)):.4f})")


if __name__ == "__main__":
    main()
