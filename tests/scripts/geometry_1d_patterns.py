"""
By-hand diagnostic (not collected by pytest): 1D diffraction patterns of two calibration runs under
three detector geometries:

* unrefined: nominal pose (no shifts or rotations), 2theta_0 = 0, pitch 0.29297 mm, mono wavelength;
* refined, pitch 0.29297 mm: joint fit of both runs (tests/scripts/joint_calibration_fit.py: shared pose,
  2theta_0 and wavelength per run, |2theta_0| <= 0.5 deg);
* refined, pitch 0.314 mm: the same joint fit with the mask-implied pitch.

Each sub-run's masked pixels are histogrammed in 2theta (the forward model, identical to reduction's
pixel map) as counts per pixel. Bragg positions 2*asin(lambda / 2d) use each geometry's wavelength. Each
reflection is fitted with a Gaussian on a linear background to give its offset from Bragg and its FWHM.

Run from the repository root:

    pixi run python tests/scripts/geometry_1d_patterns.py RUN_A.nxs.h5 RUN_B.nxs.h5 OUTPUT_DIR [--mask MASK.xml]
        [--lattice 2020|2023|Ni=A,Fe=A]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from pyrs.calibration.pixel_pos_calibration import DetectorModel  # noqa: E402
from pyrs.calibration.powder_standards import standard_for_sy  # noqa: E402
from pyrs.core.instrument_geometry import DENEXDetectorShift  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_calibration_fit import TTH0_LIMIT, joint_fit, load, setup_with_pitch, standards_for  # noqa: E402

NOMINAL_PITCH, MASK_PITCH = 0.29296875, 0.314
BIN = 0.03
WINDOW = 1.0  # deg either side of Bragg for the peak close-ups and fits
STYLES = {  # label -> colour, line style (categorical slots 2, 1, 3)
    "unrefined (nominal, 0.293 mm)": ("#eb6834", "-"),
    "refined, 0.293 mm": ("#2a78d6", "-"),
    "refined, 0.314 mm": ("#1baf7a", "--"),
}


HKL = {2: "110", 3: "111", 4: "200", 6: "211", 8: "220", 10: "310", 11: "311", 12: "222", 16: "400", 19: "331"}
CLOSE_UP = 1.5  # deg either side of the refined prediction in the close-ups
# the two refined geometries share one wavelength, so one set of refined predictions is drawn
PREDICTED = ("unrefined (nominal, 0.293 mm)", "refined, 0.293 mm")


def mark_predictions(ax, curves, only=None, values=False) -> None:
    """Vertical line and a labelled marker at each predicted peak 2theta, per wavelength."""
    transform = ax.get_xaxis_transform()
    for level, label in zip((0.97, 0.84), PREDICTED):
        colour = STYLES[label][0]
        _, _, bragg, _, hkls = curves[label]
        for b, hkl in zip(bragg, hkls):
            if only is not None and hkl != only:
                continue
            ax.axvline(b, color=colour, ls=":", lw=1.0)
            ax.plot([b], [level], marker="v", ms=6, color=colour, transform=transform, clip_on=False)
            text = f"{hkl} {b:.2f}°" if values else hkl
            ax.annotate(text, (b, level), xytext=(5, -3), textcoords="offset points", xycoords=transform,
                        fontsize=6.5, color="#1a1a19", va="top")


def gaussian(x, height, centre, sigma, a, b):
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + a + b * (x - centre)


def histogram(tth: np.ndarray, counts: np.ndarray, keep: np.ndarray):
    t, c = tth[keep], counts[keep]
    edges = np.arange(np.floor(t.min() / BIN) * BIN, t.max() + BIN, BIN)
    summed, _ = np.histogram(t, edges, weights=c)
    pixels, _ = np.histogram(t, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (edges[:-1] + edges[1:]), np.where(pixels > 0, summed / pixels, np.nan)


def fit_peak(x, y, bragg):
    sel = (np.abs(x - bragg) < WINDOW) & np.isfinite(y)
    if sel.sum() < 10:
        return None
    xs, ys = x[sel], y[sel]
    p0 = [ys.max() - np.median(ys), xs[np.argmax(ys)], 0.15, np.median(ys), 0.0]
    try:
        p, _ = curve_fit(gaussian, xs, ys, p0=p0, maxfev=5000)
    except RuntimeError:
        return None
    if not (abs(p[1] - bragg) < WINDOW and 0.01 < abs(p[2]) < 1.0 and p[0] > 0):
        return None
    return p[1], 2.3548 * abs(p[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("output_dir")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--lattice", default="2020", help="2020, 2023 or e.g. Ni=3.52942,Fe=2.8674")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pyrs.core.nexus_conversion import NeXusConvertingApp

    standards = standards_for(args.lattice)
    print(f"{args.lattice} lattice constants: " + ", ".join(f"{s.name} {s.lattice_a}" for s in standards))
    runs = [load(args.run_a, args.mask, standards), load(args.run_b, args.mask, standards)]
    geometries = {"unrefined (nominal, 0.293 mm)": (NOMINAL_PITCH, [np.zeros(8), np.zeros(8)])}
    for pitch, label in ((NOMINAL_PITCH, "refined, 0.293 mm"), (MASK_PITCH, "refined, 0.314 mm")):
        _, _, params, _ = joint_fit(runs, pitch, [0, 1, 2, 3, 4, 5], True, TTH0_LIMIT)
        geometries[label] = (pitch, list(params))
    for i, run in enumerate(runs):
        geometries["unrefined (nominal, 0.293 mm)"][1][i][7] = run[3]

    print("Geometries (Shift mm, Rot deg, 2theta_0 deg, lambda A):")
    for label, (pitch, params) in geometries.items():
        for run, p in zip(runs, params):
            print(
                f"  {label:30s} {run[0]}: Shift {p[0] * 1e3:7.2f} {p[1] * 1e3:6.2f} {p[2] * 1e3:7.2f}"
                f"  Rot {p[3]:6.3f} {p[4]:6.3f} {p[5]:6.3f}  2th0 {p[6]:+.4f}  lambda {p[7]:.5f}"
            )

    print("\nPeak fits: centre - Bragg (deg) and FWHM (deg) per geometry")
    for i, (name, _, base_setup, _) in enumerate(runs):
        workspace = NeXusConvertingApp(args.run_a if i == 0 else args.run_b, args.mask).convert()
        num_rows, num_columns = base_setup.detector_size
        grid = np.meshgrid(np.arange(num_columns), np.arange(num_rows), indexing="ij")
        mask = workspace.get_detector_mask(True)
        keep = (np.ones(num_columns * num_rows) if mask is None else np.asarray(mask, float)) > 0
        models = {label: DetectorModel(setup_with_pitch(base_setup, pitch)) for label, (pitch, _) in geometries.items()}

        panels = []
        for sub_run in [int(s) for s in workspace.get_sub_runs()]:
            standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), standards)
            if standard is None:
                continue
            arm = -float(workspace.get_detector_2theta(sub_run))
            counts = workspace.get_detector_counts(sub_run).astype(float)
            curves = {}
            for label, (_, params) in geometries.items():
                p = params[i]
                cal = None if not np.any(p[:7]) else DENEXDetectorShift(*p[:7])
                tth = models[label].two_theta(grid[0], grid[1], arm, base_setup.arm_length, cal).ravel()
                x, y = histogram(tth, counts, keep)
                bragg = standard.two_theta(p[7])
                hkls = standard.hkl_squared[: bragg.size]
                inside = np.isfinite(bragg) & (bragg > x.min() + 0.3) & (bragg < x.max() - 0.3)
                bragg, hkl_labels = bragg[inside], [HKL[int(h)] for h in hkls[inside]]
                curves[label] = (x, y, bragg, [fit_peak(x, y, b) for b in bragg], hkl_labels)
            panels.append((sub_run, standard.name, -arm, curves))

        # full patterns
        ncols = 2
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows), constrained_layout=True, squeeze=False)
        for ax, (sub_run, powder, det, curves) in zip(axes.flat, panels):
            for label, (x, y, *_rest) in curves.items():
                colour, style = STYLES[label]
                ax.plot(x, y, color=colour, ls=style, lw=1.2, label=label)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
            mark_predictions(ax, curves)
            ax.set_title(f"{name} sub-run {sub_run}  {powder}  arm 2θ {det:.1f}°", fontsize=9)
            ax.set_xlabel("2θ (deg)", fontsize=8)
            ax.set_ylabel("counts / pixel", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(
            f"{name} ({args.lattice} lattice): 1D patterns; ▼ predicted peak 2θ = 2·asin(λ/2d) at each geometry's λ"
            " (orange: unrefined, mono λ; blue: refined λ, both pitches)",
            fontsize=10,
        )
        fig.savefig(out_dir / f"{name}_1d_patterns.png", dpi=120)
        plt.close(fig)

        # peak close-ups in absolute 2theta, centred on the refined prediction
        rings = [(s, pw, det, k) for s, pw, det, cv in panels for k in range(len(cv["refined, 0.293 mm"][2]))]
        ncols = 4
        nrows = int(np.ceil(len(rings) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.7 * nrows), constrained_layout=True, squeeze=False)
        for ax, (sub_run, powder, det, k) in zip(axes.flat, rings):
            curves = dict(next(cv for s, _, _, cv in panels if s == sub_run))
            centre = curves["refined, 0.293 mm"][2][k]
            hkl = curves["refined, 0.293 mm"][4][k]
            line = f"  {name} sub-run {sub_run:2d} {powder} {hkl}:"
            for label, (x, y, bragg, fits, hkl_labels) in curves.items():
                colour, style = STYLES[label]
                sel = np.abs(x - centre) < CLOSE_UP
                ax.plot(x[sel], y[sel], color=colour, ls=style, lw=1.2, label=label)
                if hkl in hkl_labels:
                    j = hkl_labels.index(hkl)
                    f = fits[j]
                    line += f"  {label.split(' (')[0]:18s} pred {bragg[j]:8.4f} " + (
                        f"peak-pred {f[0] - bragg[j]:+.4f} / FWHM {f[1]:.4f}" if f else "   fit failed                 "
                    )
            print(line)
            ax.set_xlim(centre - CLOSE_UP, centre + CLOSE_UP)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
            mark_predictions(ax, curves, only=hkl, values=True)
            ax.set_title(f"sub-run {sub_run} {powder} {hkl}", fontsize=8)
            ax.set_xlabel("2θ (deg)", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(rings):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(
            f"{name} ({args.lattice} lattice): each reflection; ▼ predicted 2θ at each geometry's λ (orange: unrefined, blue: refined)",
            fontsize=10,
        )
        fig.savefig(out_dir / f"{name}_1d_peaks.png", dpi=120)
        plt.close(fig)
    print(f"\nfigures written to {out_dir}")


if __name__ == "__main__":
    main()
