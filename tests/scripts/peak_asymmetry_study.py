"""
By-hand study (not collected by pytest): why FitCalibration's eta-wedge residual and the ring calibration
disagree (#1018, docs/ground_truths.md). Hypothesis: the peaks are asymmetric (axial divergence: low-angle
tails below 2theta = 90 deg, high-angle tails above), so a symmetric profile fitted over a wide window
with a sloping background is pulled by an amount that depends on the window and the function, with a sign
change near 90 deg.

All under one geometry (a calibration JSON, pixel pitch 0.29297 mm):

1. whole-detector 1D pattern of each sub-run (masked pixels histogrammed in 2theta, counts per pixel), each
   reflection fitted four ways: Gaussian + constant over +/-1 deg; Gaussian + linear over +/-3 deg;
   pseudo-Voigt + linear over +/-3 deg (FitCalibration's choice); split pseudo-Voigt (separate left/right
   widths) + linear over +/-3 deg, whose width ratio measures the asymmetry;
2. FitCalibration's own eta-wedge patterns (its reduction, mask, 5 wedges of 3 deg), fitted with the narrow
   Gaussian and the wide pseudo-Voigt, beside the ring-point residuals of the strips (2020 lattice).

Run from the repository root:

    pixi run python tests/scripts/peak_asymmetry_study.py RUN.nxs.h5 CALIBRATION.json RING_POINTS.npz --mask MASK.xml
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pyrs.calibration.pixel_pos_calibration import DetectorModel, PixelCalibration, RingPoints
from pyrs.calibration.powder_standards import HB2B_STANDARDS, standard_for_sy
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

CAL_KEYS = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")
SETUP = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)
BIN = 0.03


def gaussian(x, h, c, s, a, b=0.0):
    return h * np.exp(-0.5 * ((x - c) / s) ** 2) + a + b * (x - c)


def pseudo_voigt(x, h, c, w, eta, a, b):
    """Area-free pseudo-Voigt with FWHM w and Lorentzian fraction eta, plus a linear background."""
    g = np.exp(-4 * np.log(2) * ((x - c) / w) ** 2)
    lorentz = 1 / (1 + 4 * ((x - c) / w) ** 2)
    return h * ((1 - eta) * g + eta * lorentz) + a + b * (x - c)


def split_pseudo_voigt(x, h, c, wl, wr, eta, a, b):
    w = np.where(x < c, wl, wr)
    g = np.exp(-4 * np.log(2) * ((x - c) / w) ** 2)
    lorentz = 1 / (1 + 4 * ((x - c) / w) ** 2)
    return h * ((1 - eta) * g + eta * lorentz) + a + b * (x - c)


def fits(x, y, centre):
    """Peak centre (and split-PV width ratio) from the four profiles; NaN where a fit fails."""
    out = {}
    ok = np.isfinite(y)
    for label, half, model, p0, bounds in (
        ("gauss_1", 1.0, lambda x, h, c, s, a: gaussian(x, h, c, s, a), lambda h, b: [h, centre, 0.2, b],
         ([0, centre - 1, 0.02, -np.inf], [np.inf, centre + 1, 1.0, np.inf])),
        ("gauss_3", 3.0, gaussian, lambda h, b: [h, centre, 0.2, b, 0.0],
         ([0, centre - 1, 0.02, -np.inf, -np.inf], [np.inf, centre + 1, 1.0, np.inf, np.inf])),
        ("pv_3", 3.0, pseudo_voigt, lambda h, b: [h, centre, 0.5, 0.5, b, 0.0],
         ([0, centre - 1, 0.05, 0, -np.inf, -np.inf], [np.inf, centre + 1, 2.0, 1, np.inf, np.inf])),
        ("spv_3", 3.0, split_pseudo_voigt, lambda h, b: [h, centre, 0.5, 0.5, 0.5, b, 0.0],
         ([0, centre - 1, 0.05, 0.05, 0, -np.inf, -np.inf], [np.inf, centre + 1, 2.0, 2.0, 1, np.inf, np.inf])),
    ):
        sel = ok & (np.abs(x - centre) < half)
        if sel.sum() < 12:
            out[label] = (np.nan, np.nan)
            continue
        xs, ys = x[sel], y[sel]
        background = float(np.median(ys))
        try:
            p, _ = curve_fit(model, xs, ys, p0=p0(ys.max() - background, background), bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            out[label] = (np.nan, np.nan)
            continue
        out[label] = (p[1], p[3] / p[2] if label == "spv_3" else np.nan)  # split PV: right / left width
    return out


def histogram(tth: np.ndarray, counts: np.ndarray):
    edges = np.arange(np.floor(tth.min() / BIN) * BIN, tth.max() + BIN, BIN)
    summed, _ = np.histogram(tth, edges, weights=counts)
    pixels, _ = np.histogram(tth, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (edges[:-1] + edges[1:]), np.where(pixels > 0, summed / pixels, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("calibration")
    parser.add_argument("ring_points")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    raw = json.loads(Path(args.calibration).read_text())
    raw.setdefault("TTH_0", raw.get("two_theta_0", 0.0))
    params = np.array([float(raw[k]) for k in CAL_KEYS])
    shift = DENEXDetectorShift(*params[:7])
    wavelength = params[7]
    model = DetectorModel(SETUP)

    from pyrs.core.nexus_conversion import NeXusConvertingApp

    workspace = NeXusConvertingApp(args.run, args.mask).convert()
    mask = workspace.get_detector_mask(True)
    keep = (np.ones(1024 * 1024) if mask is None else np.asarray(mask, float)) > 0
    grid = np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij")

    print(f"1. whole-detector 1D, geometry {Path(args.calibration).name} (lambda {wavelength:.5f} A):")
    print("   peak centre - Bragg (mdeg) per profile, and the split-PV right/left width ratio")
    print(f"   {'sub-run':>7} {'refl':>8} {'Bragg':>8} {'G +-1':>7} {'G+lin +-3':>9} {'PV +-3':>7} {'sPV +-3':>8} {'R/L':>5}")
    rows = []
    for sub_run in [int(s) for s in workspace.get_sub_runs()]:
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), HB2B_STANDARDS)
        if standard is None:
            continue
        arm = -float(workspace.get_detector_2theta(sub_run))
        tth = model.two_theta(grid[0], grid[1], arm, SETUP.arm_length, shift).ravel()
        counts = workspace.get_detector_counts(sub_run).astype(float)
        x, y = histogram(tth[keep], counts[keep])
        for hkl, bragg in zip(standard.hkl_squared, standard.two_theta(wavelength)):
            if not (x.min() + 1 < bragg < x.max() - 1):
                continue
            f = fits(x, y, bragg)
            rows.append((bragg, {k: v[0] - bragg for k, v in f.items()}, f["spv_3"][1]))
            print(
                f"   {sub_run:7d} {standard.name + ' ' + str(hkl):>8} {bragg:8.3f} "
                + " ".join(f"{(f[k][0] - bragg) * 1e3:+8.1f}" for k in ("gauss_1", "gauss_3", "pv_3", "spv_3"))
                + f" {f['spv_3'][1]:5.2f}"
            )
    tt = np.array([r[0] for r in rows])
    for k in ("gauss_3", "pv_3", "spv_3"):
        d = np.array([r[1][k] - r[1]["gauss_1"] for r in rows])
        good = np.isfinite(d)
        slope = np.polyfit(tt[good], d[good], 1)[0]
        print(f"   {k} - gauss_1: trend {slope * 1e3 * 10:+.1f} mdeg per 10 deg of 2theta, range {np.nanmin(d)*1e3:+.1f}..{np.nanmax(d)*1e3:+.1f} mdeg")
    ratio = np.array([r[2] for r in rows])
    print("   split-PV width ratio R/L by 2theta: " + ", ".join(f"{t:.0f}: {q:.2f}" for t, q in zip(tt, ratio)))

    # 2. FitCalibration's eta-wedge patterns under the same geometry
    from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

    fit = FitCalibration(nexus_file=args.run, mask_file=args.mask)  # its own powders: the 2020 set (#1022)
    fit.set_inst_shifts_wl(params)
    fit.reduce_data()
    masks = list(fit._hidra_ws.reduction_masks)
    order = np.argsort([float(m.split("_")[1]) for m in masks])
    print(f"\n2. FitCalibration eta wedges ({', '.join(masks[i] for i in order)}), first reflection of each sub-run,")
    print("   centre - Bragg (mdeg): narrow Gaussian +-1 | wide pseudo-Voigt +-3")
    gauss_by_wedge, pv_by_wedge = [], []
    for i_run, sub_run in enumerate(fit.sub_runs):
        bragg = fit.get_diff_peaks(sub_run - 1)
        g_row, p_row = [], []
        for i in order:
            x, y, _ = fit.reducer.get_diffraction_data(sub_run, mask_id=masks[i])
            peaks = bragg[(bragg > x.min() + 1) & (bragg < x.max() - 1)][:1]
            if peaks.size == 0:
                g_row.append(np.nan)
                p_row.append(np.nan)
                continue
            f = fits(np.asarray(x), np.asarray(y, float), peaks[0])
            g_row.append(f["gauss_1"][0] - peaks[0])
            p_row.append(f["pv_3"][0] - peaks[0])
        gauss_by_wedge.append(g_row)
        pv_by_wedge.append(p_row)
        print(f"   sub-run {sub_run:2d}: " + " ".join(f"{v*1e3:+6.1f}" for v in g_row) + "  |  " + " ".join(f"{v*1e3:+6.1f}" for v in p_row))
    g, p = np.array(gauss_by_wedge), np.array(pv_by_wedge)
    print("   mean by wedge, narrow Gaussian: " + " ".join(f"{v*1e3:+6.1f}" for v in np.nanmean(g, axis=0))
          + f"   (RMS {np.sqrt(np.nanmean(g**2))*1e3:.1f} mdeg)")
    print("   mean by wedge, wide PV:         " + " ".join(f"{v*1e3:+6.1f}" for v in np.nanmean(p, axis=0))
          + f"   (RMS {np.sqrt(np.nanmean(p**2))*1e3:.1f} mdeg)")

    points = RingPoints.load(args.ring_points)
    residual = PixelCalibration(points, SETUP, wavelength=wavelength).residual(params)
    strips = np.unique(np.round(points.index_1))
    print("   ring points by strip (row: mean residual mdeg): " + ", ".join(
        f"{s:.0f}: {np.mean(residual[np.round(points.index_1) == s])*1e3:+.1f}" for s in strips))


if __name__ == "__main__":
    main()
