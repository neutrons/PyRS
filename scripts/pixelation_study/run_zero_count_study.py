"""
Zero-count pixel study: vanadium dead-pixel mask and zero fraction along the detector wires.

For every sample run this script

1. sums the raw counts per pixel (HiDRA project file or raw ``.nxs.h5`` event file),
2. excludes dead pixels, defined as < 0.9 counts in the vanadium flood (the reduction rule),
3. predicts each pixel's mean count from its 2θ × η bin and its vanadium efficiency
   (vanadium box-averaged along x, ``--efficiency-smooth``, to suppress its own
   counting noise; 1 uses raw vanadium),
4. compares observed zero-count pixels with the Poisson expectation exp(-λ),
   overall and per detector column (x, horizontal position) and row (y, vertical
   position), and
5. writes a JSON summary and figures to ``--out``.

Example (from the repository root):

    pixi run python scripts/pixelation_study/run_zero_count_study.py \\
        --vanadium tests/data/HB2B_931.h5 \\
        --run tests/data/HB2B_938.h5 --run tests/data/HB2B_1118.nxs.h5 \\
        --mask tests/data/HB2B_Mask_12-18-19.xml --out pixelation_results

See docs/planning/PixelationUncertainty_Plan.md and docs/ground_truths.md.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pixelation_study.zero_counts import (  # noqa: E402
    NUM_PIXEL_1D,
    FloatArray,
    ZeroProfile,
    angle_bin_index,
    dead_pixel_mask,
    dominant_period,
    expected_counts,
    smooth_along_axis,
    to_detector_image,
    zero_fraction_profile,
)

PIXEL_SIZE_MM = 300.0 / NUM_PIXEL_1D
PROFILE_FIELDS = ("n_valid", "observed", "expected", "z_score", "count_ratio", "expected_total")


def load_run(path: Path) -> tuple[FloatArray, float, str]:
    """
    Sum a run's counts over all sub-runs.

    Args:
        path: HiDRA project file (``.h5``) or raw event NeXus file (``.nxs.h5``).

    Returns:
        (detector image ``[x, y]``, detector-arm 2θ in degrees, run title).

    Raises:
        ValueError: If the file holds no raw per-pixel counts.
    """
    with h5py.File(path, "r") as h5:
        if "entry" in h5:
            entry = h5["entry"]
            ids = entry["bank1_events/event_id"][()]
            counts = np.bincount(ids, minlength=NUM_PIXEL_1D**2)[: NUM_PIXEL_1D**2]
            two_theta = float(np.median(entry["DASlogs/2theta/value"][()]))
            title = entry["title"][0].decode()
        elif "raw data/sub-runs" in h5 and len(h5["raw data/sub-runs"]) > 0:
            sub_runs = h5["raw data/sub-runs"]
            counts = sum(sub_runs[k]["counts"][()] for k in sub_runs)
            two_theta = float(np.median(h5["raw data/logs/2theta"][()]))
            title = h5["raw data/logs/run_title"][0].decode() if "run_title" in h5["raw data/logs"] else ""
        else:
            raise ValueError(f"{path} holds no raw per-pixel counts")
    return to_detector_image(counts), two_theta, title


def pixel_angles(two_theta_arm: float) -> tuple[FloatArray, FloatArray]:
    """
    Compute uncalibrated per-pixel 2θ and η with PyRS's own reduction geometry.

    Args:
        two_theta_arm: Detector-arm 2θ (degrees).

    Returns:
        (2θ image, η image), both indexed ``[x, y]``, degrees.
    """
    from pyrs.core.instrument_geometry import DENEXDetectorGeometry
    from pyrs.core.nexus_conversion import ARM_LENGTH, PIXEL_SIZE
    from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument

    setup = DENEXDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D, PIXEL_SIZE, PIXEL_SIZE, ARM_LENGTH, False)
    instrument = ResidualStressInstrument(setup)
    instrument.build_instrument(two_theta_arm)
    return (
        to_detector_image(instrument.get_pixels_2theta(1)),
        to_detector_image(instrument.get_eta_values(1)),
    )


def load_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    """Return a ``[x, y]`` boolean image, True where the Mantid XML mask masks a pixel."""
    if path is None:
        return None
    from pyrs.core.mask_util import load_mantid_mask

    # is_mask=True: listed detector IDs become 0, all others 1.
    keep = load_mantid_mask(NUM_PIXEL_1D**2, str(path), is_mask=True)
    return to_detector_image(keep) == 0


def active_lines(dead: np.ndarray, max_dead_fraction: float) -> tuple[slice, slice]:
    """Return the x and y ranges whose lines are at most ``max_dead_fraction`` dead."""
    x_ok = np.nonzero(dead.mean(axis=1) <= max_dead_fraction)[0]
    y_ok = np.nonzero(dead.mean(axis=0) <= max_dead_fraction)[0]
    return slice(int(x_ok.min()), int(x_ok.max()) + 1), slice(int(y_ok.min()), int(y_ok.max()) + 1)


def profile_summary(profile: ZeroProfile, lines: slice) -> dict[str, Any]:
    """Summarise a per-line profile restricted to the active lines."""
    obs = profile.observed[lines]
    exp = profile.expected[lines]
    z = profile.z_score[lines]
    period, share = dominant_period(obs - exp)
    ratio = profile.count_ratio[lines]
    ratio_period, ratio_share = dominant_period(ratio)
    finite = np.isfinite(z)
    return {
        "observed_min": float(np.nanmin(obs)),
        "observed_max": float(np.nanmax(obs)),
        "residual_rms": float(np.sqrt(np.nanmean((obs - exp) ** 2))),
        "z_rms": float(np.sqrt(np.mean(z[finite] ** 2))),
        "lines_abs_z_gt_5": int(np.sum(np.abs(z[finite]) > 5)),
        "n_lines": int(finite.sum()),
        "residual_period_pixels": period,
        "residual_period_mm": period * PIXEL_SIZE_MM,
        "residual_period_power_share": share,
        "count_ratio_rel_std": float(np.nanstd(ratio) / np.nanmean(ratio)),
        "count_ratio_poisson_rel_std": float(np.sqrt(np.nanmean(1.0 / profile.expected_total[lines]))),
        "count_ratio_excess_rel_std": float(
            np.sqrt(
                max(np.nanvar(ratio) / np.nanmean(ratio) ** 2 - np.nanmean(1.0 / profile.expected_total[lines]), 0.0)
            )
        ),
        "count_ratio_period_pixels": ratio_period,
        "count_ratio_period_power_share": ratio_share,
    }


def plot_run(
    name: str,
    counts: FloatArray,
    dead: np.ndarray,
    by_x: ZeroProfile,
    by_y: ZeroProfile,
    out: Path,
) -> None:
    """Save a zero-map image and per-line observed/expected profiles."""
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    zero_map = np.where(dead, 0.5, (counts == 0).astype(float))
    axes[0].imshow(zero_map.T, origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"{name}: zero pixels (black), dead (grey)")
    axes[0].set_xlabel("x index (horizontal)")
    axes[0].set_ylabel("y index (vertical)")
    for ax, prof, label in (
        (axes[1], by_x, "x index (one value per column)"),
        (axes[2], by_y, "y index (one value per row)"),
    ):
        idx = np.arange(prof.observed.size)
        ax.plot(idx, prof.observed, lw=0.8, label="observed")
        ax.plot(idx, prof.expected, lw=0.8, label="Poisson expected")
        ax.set_xlabel(label)
        ax.set_ylabel("zero-count fraction (live pixels)")
        ax.legend()
    axes[3].plot(np.arange(by_y.count_ratio.size), by_y.count_ratio, lw=0.8)
    axes[3].axhline(1.0, color="k", lw=0.5)
    axes[3].set_xlabel("y index (one value per row)")
    axes[3].set_ylabel("measured / predicted counts per row")
    fig.tight_layout()
    fig.savefig(out / f"{name}_zero_profiles.png", dpi=120)
    plt.close(fig)


@click.command()
@click.option("--vanadium", type=click.Path(exists=True, path_type=Path), required=True, help="Vanadium flood run.")
@click.option("--run", "runs", type=click.Path(exists=True, path_type=Path), multiple=True, required=True)
@click.option("--mask", type=click.Path(exists=True, path_type=Path), default=None, help="Production XML mask.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("pixelation_results"), show_default=True)
@click.option("--tth-step", default=0.05, show_default=True, help="2θ bin width (deg) for λ estimates.")
@click.option("--eta-step", default=2.0, show_default=True, help="η bin width (deg) for λ estimates.")
@click.option("--efficiency-smooth", default=15, show_default=True, help="Vanadium box width along x (pixels).")
@click.option("--max-dead-fraction", default=0.05, show_default=True, help="Max dead share of an analysed line.")
def main(
    vanadium: Path,
    runs: tuple[Path, ...],
    mask: Optional[Path],
    out: Path,
    tth_step: float,
    eta_step: float,
    efficiency_smooth: int,
    max_dead_fraction: float,
) -> None:
    """Test the vanadium dead-pixel mask and profile zero counts along the detector wires."""
    out.mkdir(parents=True, exist_ok=True)
    van, _, van_title = load_run(vanadium)
    dead = dead_pixel_mask(van)
    x_lines, y_lines = active_lines(dead, max_dead_fraction)
    prod_mask = load_mask(mask)
    efficiency = np.nan_to_num(smooth_along_axis(van, ~dead, efficiency_smooth, axis=0))

    van_live = np.where(dead, np.nan, van)
    van_x = np.nanmean(van_live[x_lines, y_lines], axis=1)
    van_y = np.nanmean(van_live[x_lines, y_lines], axis=0)
    summary: dict[str, Any] = {
        "vanadium": {
            "file": str(vanadium),
            "title": van_title,
            "mean_counts_live": float(np.nanmean(van_live)),
            "dead_fraction": float(dead.mean()),
            "dead_inside_active_lines": int(dead[x_lines, y_lines].sum()),
            "active_x": [x_lines.start, x_lines.stop - 1],
            "active_y": [y_lines.start, y_lines.stop - 1],
            "efficiency_rel_std_by_x": float(np.std(van_x) / np.mean(van_x)),
            "efficiency_rel_std_by_y": float(np.std(van_y) / np.mean(van_y)),
            "efficiency_period_x_pixels": dominant_period(van_x)[0],
            "efficiency_period_y_pixels": dominant_period(van_y)[0],
        },
        "settings": {
            "tth_step": tth_step,
            "eta_step": eta_step,
            "efficiency_smooth": efficiency_smooth,
            "max_dead_fraction": max_dead_fraction,
        },
        "runs": {},
    }
    if prod_mask is not None:
        summary["vanadium"]["production_mask"] = {
            "masked_fraction": float(prod_mask.mean()),
            "dead_and_masked": int((dead & prod_mask).sum()),
            "dead_not_masked": int((dead & ~prod_mask).sum()),
            "masked_not_dead": int((prod_mask & ~dead).sum()),
        }

    for path in runs:
        counts, two_theta, title = load_run(path)
        name = path.name.split(".")[0]
        tth, eta = pixel_angles(two_theta)
        live = ~dead
        lam = expected_counts(counts, efficiency, angle_bin_index(tth, eta, tth_step, eta_step), live)
        analysed = np.zeros_like(live)
        analysed[x_lines, y_lines] = live[x_lines, y_lines]
        by_x = zero_fraction_profile(counts, lam, analysed, axis=1)
        by_y = zero_fraction_profile(counts, lam, analysed, axis=0)

        zeros = counts == 0
        p0 = np.exp(-lam[analysed])
        obs_total = float(zeros[analysed].sum())
        exp_total = float(p0.sum())
        summary["runs"][name] = {
            "file": str(path),
            "title": title,
            "two_theta_arm": two_theta,
            "mean_counts_live": float(counts[live].mean()),
            "zero_fraction_all_pixels": float(zeros.mean()),
            "zero_fraction_live_pixels": float(zeros[live].mean()),
            "share_of_zeros_that_are_dead": float((zeros & dead).sum() / max(zeros.sum(), 1)),
            "dead_pixels_with_counts": int((dead & ~zeros).sum()),
            "counts_in_dead_pixels": float(counts[dead].sum()),
            "analysed_observed_zeros": obs_total,
            "analysed_expected_zeros": exp_total,
            "analysed_z": (obs_total - exp_total) / float(np.sqrt((p0 * (1 - p0)).sum())),
            "by_x": profile_summary(by_x, x_lines),
            "by_y": profile_summary(by_y, y_lines),
        }
        plot_run(name, counts, dead, by_x, by_y, out)
        np.savez_compressed(
            out / f"{name}_profiles.npz",
            **{f"x_{k}": getattr(by_x, k) for k in PROFILE_FIELDS},
            **{f"y_{k}": getattr(by_y, k) for k in PROFILE_FIELDS},
        )
        click.echo(f"{name}: done")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(x_lines.start, x_lines.stop), van_x / van_x.mean(), lw=0.8)
    ax[0].set_xlabel("x index (horizontal)")
    ax[1].plot(np.arange(y_lines.start, y_lines.stop), van_y / van_y.mean(), lw=0.8)
    ax[1].set_xlabel("y index (vertical)")
    for a in ax:
        a.set_ylabel("relative vanadium counts (live pixels)")
    fig.tight_layout()
    fig.savefig(out / "vanadium_efficiency_profiles.png", dpi=120)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
