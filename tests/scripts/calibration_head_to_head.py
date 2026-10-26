"""
By-hand diagnostic (not collected by pytest): compare the detector geometry refined from two calibration
runs on equal terms.

Shift_x, Rot_y and 2theta_0 are exactly degenerate at one detector distance (docs/ground_truths.md), so
two calibrations with different 2theta_0 cannot be compared parameter by parameter. Each is re-expressed
(``DetectorModel.with_two_theta_0``, identical pixel 2theta/eta) at common 2theta_0 values first. The
two-ring spacing check then measures distance / pixel pitch directly from the data: the pixel gap between
two rings of one sub-run against the gap the nominal geometry predicts for their Bragg angles.

Run from the repository root:

    pixi run python tests/scripts/calibration_head_to_head.py RUN_A.nxs.h5 CAL_A.json RUN_B.nxs.h5 CAL_B.json \
        [--mask MASK.xml]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pyrs.calibration.pixel_pos_calibration import DetectorModel, extract_ring_points
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.nexus_conversion import NeXusConvertingApp

CAL_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0")


def _row(label: str, c: DENEXDetectorShift) -> str:
    return (
        f"  {label:28s} {c.center_shift_x * 1e3:8.2f} {c.center_shift_y * 1e3:7.2f} {c.center_shift_z * 1e3:8.2f}"
        f" {c.rotation_x:7.3f} {c.rotation_y:7.3f} {c.rotation_z:7.3f} {c.two_theta_0:7.3f}"
    )


def ring_gaps(workspace, wavelength: float, mask_applied: bool) -> list:
    """Pixel gap between the rings of every multi-ring sub-run, measured and nominal."""
    setup = workspace.get_instrument_setup()
    model = DetectorModel(setup)
    points = extract_ring_points(workspace, wavelength, strips=5)
    num_columns = setup.detector_size[1]
    columns = np.arange(num_columns)
    gaps = []
    for sub_run in sorted(set(points.sub_run.tolist())):
        p = points.select(points.sub_run == sub_run)
        rings = sorted(set(p.hkl_squared.tolist()))
        if len(rings) < 2:
            continue
        arm = -float(workspace.get_detector_2theta(sub_run))
        tth_row = model.two_theta(columns, np.full_like(columns, 512), arm, setup.arm_length, None)
        order = np.argsort(tth_row)
        for a, b in zip(rings[:-1], rings[1:]):
            measured = [p.index_0[p.hkl_squared == h].mean() for h in (a, b)]
            bragg = [2 * np.degrees(np.arcsin(wavelength / (2 * p.d_spacing[p.hkl_squared == h][0]))) for h in (a, b)]
            nominal = [np.interp(t, tth_row[order], columns[order]) for t in bragg]
            gaps.append(
                dict(
                    sub_run=sub_run,
                    powder=str(p.powder[0]),
                    hkl=(a, b),
                    d2theta=abs(bragg[1] - bragg[0]),
                    measured=abs(measured[1] - measured[0]),
                    nominal=abs(nominal[1] - nominal[0]),
                )
            )
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a")
    parser.add_argument("cal_a")
    parser.add_argument("run_b")
    parser.add_argument("cal_b")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()

    runs = []
    for run, cal in ((args.run_a, args.cal_a), (args.run_b, args.cal_b)):
        raw = json.loads(Path(cal).read_text())
        workspace = NeXusConvertingApp(run, args.mask).convert()
        runs.append(
            dict(
                name=Path(run).name.split(".")[0],
                raw=raw,
                cal=DENEXDetectorShift(*[raw[k] for k in CAL_KEYS]),
                workspace=workspace,
                model=DetectorModel(workspace.get_instrument_setup()),
                mono=float(workspace.get_wavelength(False, False)),
            )
        )

    header = f"  {'':28s} {'Sx mm':>8} {'Sy mm':>7} {'Sz mm':>8} {'Rx':>7} {'Ry':>7} {'Rz':>7} {'2th0':>7}"
    print("Refined calibrations as written, and re-expressed at common 2theta_0 (same detector):")
    print(header)
    common = sorted({0.0, *(round(r["cal"].two_theta_0, 3) for r in runs)})
    for r in runs:
        print(_row(f"{r['name']} as refined", r["cal"]))
    for t in common:
        for r in runs:
            l2 = r["workspace"].get_instrument_setup().arm_length
            print(_row(f"{r['name']} at 2th0 = {t:.3f}", r["model"].with_two_theta_0(r["cal"], t, l2)))
    for r in runs:
        print(
            f"  {r['name']}: lambda {r['raw']['Lambda']:.5f} +/- {r['raw']['error_Lambda']:.5f} A"
            f" (mono {r['mono']:.3f}, ratio {r['raw']['Lambda'] / r['mono']:.4f})"
        )

    print("\nTwo-ring spacing (refined lambda; nominal geometry L2 0.985 m, pixel 0.29297 mm):")
    for r in runs:
        gaps = ring_gaps(r["workspace"], float(r["raw"]["Lambda"]), args.mask is not None)
        for g in gaps:
            print(
                f"  {r['name']} sub-run {g['sub_run']:2d} {g['powder']:2s} hkl^2 {g['hkl']}: d2theta {g['d2theta']:.3f} deg,"
                f" gap {g['measured']:6.1f} px vs nominal {g['nominal']:6.1f} px, ratio {g['measured'] / g['nominal']:.4f}"
            )
        if gaps:
            ratios = np.array([g["measured"] / g["nominal"] for g in gaps])
            print(f"  {r['name']}: mean ratio {ratios.mean():.4f}  ->  L2 x 0.29297 mm / pitch = {0.985 * ratios.mean():.4f} m")
        else:
            print(f"  {r['name']}: no sub-run shows two rings")


if __name__ == "__main__":
    main()
