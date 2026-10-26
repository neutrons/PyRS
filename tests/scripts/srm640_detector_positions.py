"""
By-hand planning aid (not collected by pytest): detector arm positions that capture several Si (NIST SRM
640) reflections in one frame, for each HB2B monochromator wavelength.

The usable 2theta window of a frame is taken from the forward model on the detector centre row, inside
the mask rectangle minus the side margin that ring extraction keeps (docs/ground_truths.md). Geometry:
the joint 2020-lattice refinement of HB2B_3510 + HB2B_3489 with the 0.314 mm pitch. For each wavelength
the arm is scanned in 0.25 deg steps; a reflection counts as captured when it falls inside the window.

Run from the repository root:

    pixi run python tests/scripts/srm640_detector_positions.py [--a 5.431020511] [--arm-min 30] [--arm-max 150]
"""

import argparse
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from pyrs.calibration.pixel_pos_calibration import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

# mono setting, wavelength (A), relative flux (user, 2026-10-23)
SETTINGS = (
    ("Si 333", 1.452, 1600),
    ("Si 511", 1.452, 700),
    ("Si 422", 1.540, 1600),
    ("Si 331", 1.731, 1800),
    ("Si 400", 1.886, 1200),
    ("Si 311", 2.275, 650),
    ("Si 220", 2.667, 900),
)
# diamond-structure reflections: h^2+k^2+l^2 -> label (relative |F|^2: 4 for h+k+l = 4n, 2 for all odd)
SI_REFLECTIONS = {
    3: "111", 8: "220", 11: "311", 16: "400", 19: "331", 24: "422", 27: "333/511", 32: "440", 35: "531",
    40: "620", 43: "533", 48: "444", 51: "551/711", 56: "642", 59: "553/731", 64: "800",
}
# joint fit, 2020 lattice, pitch 0.314 mm (calib_diagnostics/joint_0.314_2020.txt), 3510's 2theta_0
CALIBRATION = DENEXDetectorShift(-0.003534, 0.000435, -0.009885, -0.266, 5.817, 0.120, 0.141)
PITCH = 0.314e-3
MASK_COLUMNS = (34, 991)  # HB2B_Mask_12-18-19.xml keeps index_0 34..991
SIDE_MARGIN = 40  # extract_ring_points default


def multiplicity_weight(q: int) -> int:
    """|F|^2 class of a diamond reflection: h+k+l = 4n (all even) -> 2, all odd -> 1 (relative)."""
    return 1 if q % 2 else 2


def window(model: DetectorModel, arm: float) -> tuple:
    columns = np.arange(MASK_COLUMNS[0] + SIDE_MARGIN, MASK_COLUMNS[1] - SIDE_MARGIN + 1)
    tth = model.two_theta(columns, np.full_like(columns, 512), -arm, 0.985, CALIBRATION)
    return float(tth.min()), float(tth.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", type=float, default=5.431020511, help="Si lattice constant (A); use the SRM certificate")
    parser.add_argument("--arm-min", type=float, default=30.0)
    parser.add_argument("--arm-max", type=float, default=150.0)
    parser.add_argument("--min-peaks", type=int, default=2)
    args = parser.parse_args()

    model = DetectorModel(DENEXDetectorGeometry(1024, 1024, PITCH, PITCH, 0.985, False))
    arms = np.arange(args.arm_min, args.arm_max + 1e-9, 0.25)
    windows = np.array([window(model, arm) for arm in arms])
    widths = windows[:, 1] - windows[:, 0]
    print(f"usable window width: {widths.min():.2f}-{widths.max():.2f} deg over arm {args.arm_min}-{args.arm_max} deg"
          f" (centre row, mask minus {SIDE_MARGIN} px margin)")

    for name, wavelength, flux in SETTINGS:
        peaks = []
        for q, label in SI_REFLECTIONS.items():
            s = wavelength * np.sqrt(q) / (2 * args.a)
            if s < 1:
                peaks.append((label, 2 * np.degrees(np.arcsin(s)), multiplicity_weight(q)))
        print(f"\n{name}, lambda {wavelength} A (flux {flux}): Si peaks "
              + ", ".join(f"{label} {t:.2f}" for label, t, _ in peaks if args.arm_min - 10 < t < args.arm_max + 10))
        # every arm position, the set of peaks it captures
        captured = []
        for arm, (lo, hi) in zip(arms, windows):
            inside = tuple(i for i, (_, t, _) in enumerate(peaks) if lo < t < hi)
            captured.append(inside)
        # maximal sets with >= min_peaks: report the arm range for each distinct set
        groups: Dict[Tuple[int, ...], List[float]] = {}
        for arm, inside in zip(arms, captured):
            if len(inside) >= args.min_peaks:
                groups.setdefault(inside, []).append(arm)
        maximal = [s for s in groups if not any(set(s) < set(t) for t in groups)]
        for s in sorted(maximal, key=lambda s: peaks[s[0]][1]):
            arm_range = groups[s]
            centre = float(np.mean([peaks[i][1] for i in s]))
            best = min(arm_range, key=lambda a: abs(a - centre))
            lo, hi = window(model, best)
            print(
                f"  {len(s)} peaks {'+'.join(peaks[i][0] for i in s):24s} arm {arm_range[0]:6.2f}-{arm_range[-1]:6.2f}"
                f"  (centred: arm {best:6.2f}, window {lo:6.2f}-{hi:6.2f})"
            )
        if not maximal:
            spacing = np.diff([t for _, t, _ in peaks])
            print(f"  no arm position captures {args.min_peaks} peaks (spacings {np.round(spacing, 1)} deg)")
        # minimum number of arm positions capturing every peak in range at least once
        in_range = [i for i, (_, t, _) in enumerate(peaks) if args.arm_min < t < args.arm_max]
        for n in range(1, 6):
            best_cover = None
            candidates = sorted(set(captured), key=len, reverse=True)[:40]
            for combo in combinations(candidates, n):
                if set(in_range) <= set().union(*combo):
                    best_cover = combo
                    break
            if best_cover:
                arms_used = [min(groups.get(c, [a for a, cc in zip(arms, captured) if cc == c]),
                                 key=lambda a: abs(a - np.mean([peaks[i][1] for i in c]))) for c in best_cover]
                print(f"  all {len(in_range)} peaks in {n} frames: arms "
                      + ", ".join(f"{a:.2f}" for a in sorted(arms_used)))
                break


if __name__ == "__main__":
    main()
