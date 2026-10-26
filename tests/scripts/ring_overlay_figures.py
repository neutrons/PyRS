"""
By-hand diagnostic (not collected by pytest): detector images of a calibration run with the predicted
Debye rings overlaid, for the nominal geometry and a refined calibration, plus the extracted ring points.

The predicted rings are the Bragg angles 2*asin(lambda / 2d) of each powder standard's reflections,
drawn as contours of the forward model's pixel 2theta map (DetectorModel, identical to reduction's).
A refined calibration that describes the detector puts its rings on the intensity ridges.

Run from the repository root:

    pixi run python tests/scripts/ring_overlay_figures.py RUN.nxs.h5 CALIBRATION.json OUTPUT_DIR [--mask MASK.xml]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from pyrs.calibration.pixel_pos_calibration import DetectorModel, extract_ring_points  # noqa: E402
from pyrs.calibration.powder_standards import HB2B_STANDARDS, standard_for_sy  # noqa: E402
from pyrs.core.instrument_geometry import DENEXDetectorShift  # noqa: E402
from pyrs.core.nexus_conversion import NeXusConvertingApp  # noqa: E402

NOMINAL, REFINED, POINTS = "#eb6834", "#2a78d6", "#1baf7a"
CAL_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nexus_file")
    parser.add_argument("calibration_json")
    parser.add_argument("output_dir")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads(Path(args.calibration_json).read_text())
    refined = DENEXDetectorShift(*[raw[k] for k in CAL_KEYS])
    refined_wavelength = float(raw["Lambda"])

    workspace = NeXusConvertingApp(args.nexus_file, args.mask).convert()
    nominal_wavelength = float(workspace.get_wavelength(False, False))
    setup = workspace.get_instrument_setup()
    model = DetectorModel(setup)
    num_rows, num_columns = setup.detector_size
    grid = np.meshgrid(np.arange(num_columns), np.arange(num_rows), indexing="ij")
    mask = workspace.get_detector_mask(True)
    mask = np.ones(num_columns * num_rows) if mask is None else np.asarray(mask, dtype=float)
    mask = mask.reshape(num_columns, num_rows)

    # the driver extracts with the nominal geometry and the mono wavelength; do the same
    points = extract_ring_points(workspace, nominal_wavelength, strips=5)

    sub_runs = [int(s) for s in workspace.get_sub_runs()]
    panels = []
    for sub_run in sub_runs:
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), HB2B_STANDARDS)
        if standard is None:
            continue
        arm = -float(workspace.get_detector_2theta(sub_run))
        l2 = workspace.get_l2(sub_run)
        l2 = setup.arm_length if l2 is None else float(l2)
        counts = workspace.get_detector_counts(sub_run).astype(float).reshape(num_columns, num_rows)
        counts = np.where(mask > 0, counts, np.nan)
        panels.append(
            dict(
                sub_run=sub_run,
                standard=standard,
                det_2theta=-arm,
                counts=counts,
                nominal=model.two_theta(grid[0], grid[1], arm, l2, None),
                refined=model.two_theta(grid[0], grid[1], arm, l2, refined),
                points=points.select(points.sub_run == sub_run),
            )
        )

    def draw(ax, panel, label_rings: bool) -> None:
        counts = panel["counts"]
        positive = counts[np.isfinite(counts) & (counts > 0)]
        vmin, vmax = (np.percentile(positive, [5, 99.9]) if positive.size else (1, 10))
        ax.imshow(
            counts.T,
            origin="lower",
            cmap="gray_r",
            norm=LogNorm(vmin=max(vmin, 0.5), vmax=max(vmax, vmin * 2 + 1)),
            interpolation="nearest",
        )
        for key, wavelength, colour, style in (
            ("nominal", nominal_wavelength, NOMINAL, "--"),
            ("refined", refined_wavelength, REFINED, "-"),
        ):
            tth = panel[key]
            bragg = panel["standard"].two_theta(wavelength)
            bragg = np.sort(bragg[np.isfinite(bragg) & (bragg > np.nanmin(tth)) & (bragg < np.nanmax(tth))])
            if bragg.size:
                ax.contour(tth.T, levels=bragg, colors=colour, linestyles=style, linewidths=1.2)
            if label_rings and key == "refined":
                for value in bragg:
                    rows = np.argmin(np.abs(tth - value), axis=0)
                    ax.annotate(
                        f"{value:.2f}°",
                        (rows[num_rows - 20], num_rows - 20),
                        color="#1a1a19",
                        fontsize=7,
                        ha="center",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                    )
        p = panel["points"]
        if len(p):
            ax.plot(p.index_0, p.index_1, "o", ms=5, mfc=POINTS, mec="white", mew=0.8)
        ax.set_title(
            f"sub-run {panel['sub_run']}  {panel['standard'].name}  arm 2θ {panel['det_2theta']:.1f}°", fontsize=9
        )
        ax.set_xlabel("pixel column (index_0)", fontsize=8)
        ax.set_ylabel("pixel row (index_1)", fontsize=8)
        ax.tick_params(labelsize=7)

    legend = [
        Line2D([], [], color=NOMINAL, ls="--", label=f"nominal geometry, λ {nominal_wavelength:.5f} Å"),
        Line2D([], [], color=REFINED, ls="-", label=f"refined geometry, λ {refined_wavelength:.5f} Å"),
        Line2D([], [], marker="o", ls="", mfc=POINTS, mec="white", label="extracted ring points"),
    ]
    suptitle = (
        f"{Path(args.nexus_file).name}: Shift x/y/z {raw['Shift_x'] * 1e3:.1f}/{raw['Shift_y'] * 1e3:.1f}/"
        f"{raw['Shift_z'] * 1e3:.1f} mm, Rot x/y/z {raw['Rot_x']:.2f}/{raw['Rot_y']:.2f}/{raw['Rot_z']:.2f}°, "
        f"2θ₀ {raw['TTH_0']:.3f}°"
    )

    for panel in panels:
        fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
        draw(ax, panel, label_rings=True)
        ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.9)
        fig.suptitle(suptitle, fontsize=8)
        fig.savefig(out / f"rings_sub_run_{panel['sub_run']:02d}.png", dpi=150)
        plt.close(fig)

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        draw(ax, panel, label_rings=False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(suptitle, fontsize=9)
    fig.savefig(out / "rings_all_sub_runs.png", dpi=120)
    plt.close(fig)
    print(f"{len(panels)} sub-runs written to {out}")


if __name__ == "__main__":
    main()
