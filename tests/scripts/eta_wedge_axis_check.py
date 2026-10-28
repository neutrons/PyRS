"""
By-hand check (not collected by pytest): do PyRS's eta-wedge (texture) reductions put each wedge's
intensities on the right 2theta axis? (docs/ground_truths.md, "eta-wedge reductions share one 2theta axis")

``ReductionManager.reduce_sub_run_texture`` bins every eta wedge over the 2theta range of that wedge's own
pixels, but ``HidraWorkspace`` keeps one 2theta axis per sub-run, which each wedge overwrites: all wedges
are returned with the axis of the last one written. This script, for one geometry (a calibration JSON):

1. reduces the run as FitCalibration does (mask, 512 bins, 3 deg wedges) and lists the 2theta axis each
   wedge comes back with;
2. histograms each wedge itself on its own axis (the forward model DetectorModel, identical to reduction's
   pixel map) and fits the first reflection of every sub-run (Gaussian, +/-1 deg) on both;
3. predicts the shift from the axis mix-up, x_last + (x - x_own) * (range_last / range_own), and compares
   it with the observed difference.

Run from the repository root:

    pixi run python tests/scripts/eta_wedge_axis_check.py RUN.nxs.h5 CALIBRATION.json --mask MASK.xml
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peak_asymmetry_study import CAL_KEYS, SETUP, fits  # noqa: E402

from pyrs.calibration.pixel_pos_calibration import DetectorModel  # noqa: E402
from pyrs.calibration.powder_standards import HB2B_STANDARDS, standard_for_sy  # noqa: E402
from pyrs.core.instrument_geometry import DENEXDetectorShift  # noqa: E402

ETA_STEP, NUM_BINS = 3.0, 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("calibration")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    raw = json.loads(Path(args.calibration).read_text())
    raw.setdefault("TTH_0", raw.get("two_theta_0", 0.0))
    params = np.array([float(raw[k]) for k in CAL_KEYS])
    shift, wavelength = DENEXDetectorShift(*params[:7]), params[7]

    from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

    fit = FitCalibration(nexus_file=args.run, mask_file=args.mask, eta_slice=ETA_STEP, bins=NUM_BINS)
    fit.set_inst_shifts_wl(params)
    fit.reduce_data()
    masks = list(fit._hidra_ws.reduction_masks)
    centres = [float(m.split("_")[-1]) for m in masks]
    last = masks[-1]  # written last: its axis is the one every wedge gets
    workspace = fit._hidra_ws
    keep_mask = workspace.get_detector_mask(True)
    keep = (np.ones(1024 * 1024) if keep_mask is None else np.asarray(keep_mask, float)) > 0
    model = DetectorModel(SETUP)
    grid = np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij")

    print(f"wedges in write order: {masks} (last written: {last})")
    observed, predicted = [], []
    for sub_run in [int(s) for s in workspace.get_sub_runs()]:
        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)), HB2B_STANDARDS)
        if standard is None:
            continue
        arm = -float(workspace.get_detector_2theta(sub_run))
        tth = model.two_theta(grid[0], grid[1], arm, SETUP.arm_length, shift).ravel()
        eta = model.eta(grid[0], grid[1], arm, SETUP.arm_length, shift).ravel()
        counts = workspace.get_detector_counts(sub_run).astype(float)
        axes, own_fit, reduced_fit, ranges = {}, {}, {}, {}
        for mask_id, c in zip(masks, centres):
            x_reduced, y_reduced, _ = fit.reducer.get_diffraction_data(sub_run, mask_id=mask_id)
            axes[mask_id] = (float(x_reduced[0]), float(x_reduced[-1]))
            select = keep & (eta <= c + ETA_STEP / 2) & (eta >= c - ETA_STEP / 2)
            t = tth[select]
            ranges[mask_id] = (t.min(), t.max())
            edges = np.linspace(t.min(), t.max(), NUM_BINS + 1)
            x_own = 0.5 * (edges[1:] + edges[:-1])
            y_own = np.histogram(t, edges, weights=counts[select])[0].astype(float)
            bragg = standard.two_theta(wavelength)
            peak = bragg[(bragg > x_own.min() + 1) & (bragg < x_own.max() - 1)][:1]
            if peak.size == 0:
                continue
            own_fit[mask_id] = fits(x_own, y_own, peak[0])["gauss_1"][0]
            reduced_fit[mask_id] = fits(np.asarray(x_reduced), np.asarray(y_reduced, float), peak[0])["gauss_1"][0]
        a_last, b_last = ranges[last]
        line_obs, line_pred = [], []
        for mask_id in masks:
            if mask_id not in own_fit:
                continue
            a, b = ranges[mask_id]
            x0 = own_fit[mask_id]
            line_obs.append(reduced_fit[mask_id] - x0)
            line_pred.append(a_last + (x0 - a) * (b_last - a_last) / (b - a) - x0)
        observed += line_obs
        predicted += line_pred
        same_axis = len({axes[m] for m in masks}) == 1
        print(
            f"sub-run {sub_run:2d}: all wedges returned with one axis: {same_axis}; own ranges "
            + ", ".join(f"{m} {ranges[m][0]:.2f}-{ranges[m][1]:.2f}" for m in masks)
        )
        print("   peak shift from the axis (mdeg) observed " + " ".join(f"{v*1e3:+6.1f}" for v in line_obs)
              + " | predicted " + " ".join(f"{v*1e3:+6.1f}" for v in line_pred))
    o, p = np.array(observed), np.array(predicted)
    print(f"\nobserved shift RMS {np.sqrt(np.mean(o**2))*1e3:.1f} mdeg, max {np.max(np.abs(o))*1e3:.1f} mdeg;"
          f" observed - predicted RMS {np.sqrt(np.mean((o - p)**2))*1e3:.2f} mdeg")


if __name__ == "__main__":
    main()
