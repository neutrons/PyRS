"""
The reference script's per-run pole-figure CSV (``pole_figure_<run>_<hkl>.csv``), the input of an MTEX ODF fit.

The column names are the reference script's, including ``plate_volume_in_gauge_mm3``, so existing MTEX scripts read
the files as they are. chi is the logged chi (cradle chi minus ``chi_offset_deg``). Rates are counts/s, F and volumes
mm^3, paths mm.
Typical workflow:

1. Build the pole figures with ``core.polefig`` (one ``PolePeak`` per run and hkl).
2. Write them with ``write_all`` into the ODF work folder.
3. Fit the ODF from the CSVs with MTEX.

Example:
    >>> paths = write_all("work", pole_figures)
    >>> paths[0].name
    'pole_figure_1234_111.csv'
"""

import csv
from pathlib import Path

import numpy as np

from ..core.polefig import PolePeak

COLUMNS = ["run", "hkl", "chi", "phi", "pole_vx", "pole_vy", "pole_vz", "count_s", "raw_net_per_s", "raw_sigma_per_s",
           "F_mm3", "plate_volume_in_gauge_mm3", "mean_path_mm", "sampled_vy_offset_mm", "sampled_vy_rms_mm",
           "correction", "corrected_per_s", "corrected_sigma_per_s", "ignored"]  # fmt: skip


def csv_name(r: PolePeak) -> str:
    """
    File name ``pole_figure_<run>_<hkl>.csv`` of one pole figure.

    A combined pole figure is named after its first run (MTEX ODF scripts read the run and hkl from the name).

    Args:
        r: Pole figure of one run (or combined runs) and one hkl.

    Returns:
        The file name, without a folder.
    """
    run = r.source_runs[0] if r.source_runs else r.run
    return f"pole_figure_{run}_{r.hkl}.csv"


def write_pole_figure_csv(path: str | Path, r: PolePeak) -> None:
    """
    Write one pole figure as the reference script writes it: a header, then one row per sub-run.

    Correction and corrected columns are empty where the correction is not finite (the gauge misses the sample).

    Args:
        path: Output CSV file.
        r: Pole figure to write.
    """
    h, ign, g = r.poles(), r.ignored, r.gauge
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for i in range(r.n_points):
            c = r.correction[i]
            fin = bool(np.isfinite(c))
            w.writerow([r.run, r.hkl, r.chi[i] - r.chi_offset_deg, r.phi[i], *np.round(h[i], 5), round(r.count_s[i], 2),
                        round(r.rate[i], 4), round(r.sigma[i], 4), round(g.F[i], 4), round(g.volume_in_sample[i], 3),
                        round(g.mean_path[i], 3), round(g.vy_offset[i], 3), round(g.vy_rms[i], 3),
                        round(c, 4) if fin else "", round(r.rate[i] * c, 4) if fin else "",
                        round(r.sigma[i] * c, 4) if fin else "", int(ign[i])])  # fmt: skip


def write_all(folder: str | Path, results: list[PolePeak]) -> list[Path]:
    """
    Write every pole figure to ``folder`` (created if needed), named by ``csv_name``.

    Args:
        folder: Output folder.
        results: Pole figures to write.

    Returns:
        The paths written, in the order of ``results``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for r in results:
        p = folder / csv_name(r)
        write_pole_figure_csv(p, r)
        out.append(p)
    return out
