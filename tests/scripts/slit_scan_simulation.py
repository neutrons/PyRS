"""
By-hand study (not collected by pytest): planning a bright-line slit scan of the HB2B detector with
simulated data (tests/util/slit_scan_helpers.py) analysed by pyrs.calibration.slit_scan.

1. Precision against counts: pitch and pitch-gradient precision for a full 1024-row detector, 21 slit
   positions over +/-140 mm, bands of 32 rows, as a function of the counts in the line per row. Counts
   are also given as the equivalent flood (counts per pixel per position without the slit sheet): the
   line takes ~ slit width / pitch of the flood.
2. Systematics: pitch and gradient bias from a wrong assumed conversion depth or slit gap, a slanted
   slit, a source off the detector normal, 5% pixel-efficiency spread (corrected with the flood),
   background.
3. Source distance from two slit gaps.
4. Figures, one per simulated case (one slit position, the recovered local pitch per band against the
   truth, the residuals), plus summaries of precision against counts and of the systematic biases.

A pitch gradient of ~1.5% from the centre to the edge mimics the ~6 deg rotation-like asymmetry of the
ring calibrations (a 6 deg face tilt changes the local distance by ~ +/-1.6% across +/-150 mm).

Run from the repository root:

    pixi run python tests/scripts/slit_scan_simulation.py OUTPUT_DIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyrs.calibration.slit_scan import (  # noqa: E402
    SlitScanGeometry,
    find_line_centres,
    fit_position_response,
    source_distance_from_two_gaps,
)
from tests.util.slit_scan_helpers import channel_position, simulate_slit_scan  # noqa: E402

STAGE = np.linspace(-0.14, 0.14, 21)
PITCH, GRADIENT = 0.314e-3, 0.015
ROWS = 1024
BAND = 32
SLIT_WIDTH = 0.75e-3


def combined(response):
    """Weighted mean pitch and gradient over bands, with their errors."""
    ok = np.isfinite(response.pitch) & np.isfinite(response.pitch_error) & (response.pitch_error > 0)
    if not ok.any():
        return np.nan, np.nan, np.nan, np.nan  # too few lines found in every band
    w = 1 / response.pitch_error[ok] ** 2
    pitch = np.sum(w * response.pitch[ok]) / np.sum(w)
    pitch_error = 1 / np.sqrt(np.sum(w))
    if response.coefficients.shape[1] < 3:
        return pitch, pitch_error, 0.0, 0.0
    wg = 1 / response.gradient_error[ok] ** 2
    gradient = np.sum(wg * response.gradient[ok]) / np.sum(wg)
    return pitch, pitch_error, gradient, 1 / np.sqrt(np.sum(wg))


def run(degree=2, depth_assumed=None, gap_assumed=0.005, stage=STAGE, **kwargs):
    images, efficiency, truth = simulate_slit_scan(stage, n_rows=ROWS, slit_width=SLIT_WIDTH, **kwargs)
    if depth_assumed is None:
        depth_assumed = truth.mean_depth
    geometry = SlitScanGeometry(0.985, gap_assumed, depth_assumed)
    centres = find_line_centres(images, rows_per_band=BAND, efficiency=efficiency)
    response = fit_position_response(centres, stage, geometry, degree=degree)
    return images, centres, response, truth


def plot_case(images, centres, response, true_pitch, true_gradient, title: str, path: Path) -> None:
    """One slit position, the recovered local pitch per band against the truth, and the residuals."""
    fig, (ax_image, ax_pitch, ax_res) = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    found = np.isfinite(centres.centre).sum(axis=1)
    k = int(np.argmax(found)) if found.any() else len(images) // 2
    k = min(range(len(images)), key=lambda i: abs(i - len(images) * 2 // 3)) if found[len(images) * 2 // 3] else k
    ax_image.imshow(images[k].T, origin="lower", cmap="gray_r", aspect="auto", interpolation="nearest")
    if np.isfinite(centres.centre[k]).any():
        middle = np.nanmean(centres.centre[k])
        ax_image.set_xlim(middle - 60, middle + 60)
    ax_image.set_title(f"slit position {k + 1} of {len(images)} (zoomed)", fontsize=9)
    ax_image.set_xlabel("index_0 (channel)")
    ax_image.set_ylabel("index_1 (row)")

    channels = np.arange(60, 964)
    true_local = true_pitch * (1 + true_gradient * (channels - 511.5) / 512)
    good = np.isfinite(response.coefficients).all(axis=1)
    for band in np.flatnonzero(good):
        ax_pitch.plot(channels, response.local_pitch(channels, band) * 1e3, color="#2a78d6", lw=0.6, alpha=0.35)
    ax_pitch.plot(channels, true_local * 1e3, color="#eb6834", lw=2, ls="--", label="truth")
    ax_pitch.plot([], [], color="#2a78d6", label=f"recovered, {good.sum()} bands of {BAND} rows")
    ax_pitch.set_xlabel("index_0 (channel)")
    ax_pitch.set_ylabel("local pitch (mm)")
    ax_pitch.set_title("local pitch", fontsize=9)
    ax_pitch.legend(fontsize=8)

    ok = np.isfinite(response.residual)
    ax_res.plot(centres.centre[ok], response.residual[ok] * 1e6, "o", ms=2.5, color="#2a78d6", alpha=0.5)
    ax_res.axhline(0, color="#888888", lw=0.8)
    ax_res.set_xlabel("line centre, index_0 (channel)")
    ax_res.set_ylabel("position residual (um)")
    chi2 = np.nansum(response.chi2) / max(int(np.sum(response.dof[good])), 1)
    ax_res.set_title(f"fit residuals, all bands (chi2/dof {chi2:.2f})", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_").replace("__", "_")


def main(output: str) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    flood_per_line = SLIT_WIDTH * 1.01 / PITCH  # flood pixels' worth of counts that pass the slit
    print(f"slit width {SLIT_WIDTH * 1e3} mm ~ {flood_per_line:.2f} pixels; {len(STAGE)} positions; bands of {BAND} rows")
    figures = []

    print("\n1. precision against counts (gradient 1.5% simulated, quadratic fit, background 0.02/pixel)")
    print(f"   {'line counts/row':>15} {'flood cts/px':>12} {'pitch (mm)':>22} {'gradient':>18} {'band pitch err':>15}")
    levels = (0.5, 1, 2, 5, 10, 30)
    precision = []
    for counts in levels:
        images, centres, response, _ = run(gradient=GRADIENT, line_counts_per_row=counts, seed=int(10 * counts))
        pitch, pitch_error, gradient, gradient_error = combined(response)
        precision.append((pitch_error, gradient_error))
        band_error = np.where(np.isfinite(response.pitch_error), response.pitch_error, np.inf)
        print(
            f"   {counts:15g} {counts / flood_per_line:12.2f} {pitch * 1e3:.6f} +/- {pitch_error * 1e3:.6f}"
            f"   {gradient:+.5f} +/- {gradient_error:.5f} {np.median(band_error) / PITCH * 100:12.4f} %"
        )
        path = out / f"slit_scan_counts_{str(counts).replace('.', 'p')}_per_row.png"
        plot_case(images, centres, response, PITCH, GRADIENT,
                  f"{counts:g} counts/row in the line ({counts / flood_per_line:.2f} flood counts/pixel); truth 0.314 mm,"
                  " 1.5% gradient", path)
        figures.append(path)

    print("\n2. systematics (5 counts/row; truth pitch 0.314 mm, gradient 0.015)")
    cases = {
        "nominal (depth, gap known)": dict(),
        "assumed depth 2 mm too deep": dict(depth_assumed=0.0067),
        "assumed depth 2 mm too shallow": dict(depth_assumed=0.0027),
        "assumed slit gap 1 mm too large": dict(gap_assumed=0.006),
        "slit slanted 0.5 deg": dict(slant_deg=0.5),
        "source 20 mm off the centre normal": dict(source_offset=0.02),
        "5% efficiency spread (flood-corrected)": dict(efficiency_spread=0.05),
        "background 0.2/pixel": dict(background=0.2),
        "uniform-pitch detector, quadratic fit": dict(gradient=0.0),
    }
    biases = []
    for label, extra in cases.items():
        kwargs = dict(gradient=GRADIENT, line_counts_per_row=5, seed=7)
        kwargs.update(extra)
        images, centres, response, _ = run(**kwargs)
        pitch, pitch_error, gradient, gradient_error = combined(response)
        true_gradient = kwargs["gradient"]
        biases.append(((pitch / PITCH - 1) * 100, pitch_error / PITCH * 100, gradient - true_gradient, gradient_error))
        print(
            f"   {label:40s} pitch {(pitch / PITCH - 1) * 100:+.3f} % (+/- {pitch_error / PITCH * 100:.3f})"
            f"   gradient {gradient - true_gradient:+.5f} (+/- {gradient_error:.5f}) vs truth"
        )
        path = out / f"slit_scan_systematic_{slug(label)}.png"
        plot_case(images, centres, response, PITCH, true_gradient, f"systematic: {label} (5 counts/row)", path)
        figures.append(path)

    print("\n3. source distance from two scans (slit 5 mm and 55 mm from the window, 5 counts/row)")
    apparent = []
    for gap, stage in ((0.005, STAGE), (0.055, STAGE * 0.9)):
        images, centres, response, truth = run(stage=stage, depth_assumed=0.0, gap_assumed=0.0, slit_gap=gap,
                                               gradient=GRADIENT, line_counts_per_row=5, seed=11)
        apparent.append(combined(response)[:2])
        path = out / f"slit_scan_two_gaps_{gap * 1e3:.0f}mm.png"
        plot_case(images, centres, response, PITCH / truth.magnification_mean_depth, GRADIENT,
                  f"slit {gap * 1e3:.0f} mm from the window, analysed with M = 1 (apparent pitch)", path)
        figures.append(path)
    (p1, e1), (p2, e2) = apparent
    distance = source_distance_from_two_gaps(p1, p2, 0.005, 0.055)
    r = p1 / p2  # error by propagation through the ratio
    d_distance = abs((0.055 - 0.005) / (r - 1) ** 2) * r * np.hypot(e1 / p1, e2 / p2)
    print(f"   L = {distance * 1e3:.1f} +/- {d_distance * 1e3:.1f} mm (truth 985.0)")

    # summaries
    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    count_levels = np.array(levels)
    pe = np.array([p for p, _ in precision]) / PITCH * 100
    ge = np.array([g for _, g in precision])
    ax_p.loglog(count_levels, pe, "o-", color="#2a78d6", lw=2, ms=8)
    ax_p.set_xlabel("counts per row in the line, per slit position")
    ax_p.set_ylabel("pitch uncertainty (%)")
    ax_p.set_title("mean pitch precision (the 0.293 vs 0.314 mm question is 7%)", fontsize=9)
    ax_g.loglog(count_levels, ge, "o-", color="#2a78d6", lw=2, ms=8)
    ax_g.axhline(GRADIENT, color="#eb6834", ls="--", lw=1.5)
    ax_g.annotate("1.5% gradient (~6 deg tilt equivalent)", (count_levels[0], GRADIENT), xytext=(0, 4),
                  textcoords="offset points", fontsize=8)
    ax_g.set_xlabel("counts per row in the line, per slit position")
    ax_g.set_ylabel("gradient uncertainty")
    ax_g.set_title("pitch-gradient precision", fontsize=9)
    path = out / "slit_scan_precision_vs_counts.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(path)

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    labels = list(cases)
    y = np.arange(len(labels))[::-1]
    b = np.array(biases)
    ax_p.errorbar(b[:, 0], y, xerr=b[:, 1], fmt="o", color="#2a78d6", ms=7, capsize=3)
    ax_p.axvline(0, color="#888888", lw=0.8)
    ax_p.set_yticks(y, labels, fontsize=8)
    ax_p.set_xlabel("pitch bias (%)")
    ax_p.set_title("pitch bias, 5 counts/row", fontsize=9)
    ax_g.errorbar(b[:, 2], y, xerr=b[:, 3], fmt="o", color="#2a78d6", ms=7, capsize=3)
    ax_g.axvline(0, color="#888888", lw=0.8)
    ax_g.set_yticks(y, [""] * len(labels))
    ax_g.set_xlabel("gradient bias")
    ax_g.set_title("pitch-gradient bias", fontsize=9)
    path = out / "slit_scan_systematics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(path)

    print(f"\n{len(figures)} figures in {out}:")
    for path in figures:
        print(f"   {path.name}")
    print(f"(edge check: 300 mm at 0.314 mm spans channels {511.5 - 0.150 / PITCH:.1f}..{511.5 + 0.150 / PITCH:.1f};"
          f" X(0) = {channel_position(0, PITCH, GRADIENT) * 1e3:.1f} mm)")


if __name__ == "__main__":
    main(sys.argv[1])
