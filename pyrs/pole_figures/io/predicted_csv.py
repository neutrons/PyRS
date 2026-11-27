"""
The reference script's predicted_correction.csv: one row per plane and grid point of a prediction.

Example:
    >>> write_predicted_csv("predicted_correction.csv", pred, cutoff=3.0)
"""

import csv
from pathlib import Path

import numpy as np

from ..core.prediction import Prediction
from ..text import check_label
from .retry import write_atomic

#: the reference script's columns, with ``plate_volume_in_gauge_mm3`` named ``sample_volume_in_gauge_mm3``
COLUMNS = ["hkl", "two_theta", "omega", "chi", "phi", "pole_vx", "pole_vy", "pole_vz", "mean_path_mm",
           "sample_volume_in_gauge_mm3", "correction", "ignored"]  # fmt: skip


def write_predicted_csv(path: str | Path, pred: Prediction, *, cutoff: float) -> None:
    """
    Write the predicted corrections as a CSV: one row per plane and grid point.

    The columns follow the reference script's predicted_correction.csv, except that its
    ``plate_volume_in_gauge_mm3`` is named ``sample_volume_in_gauge_mm3`` here (the sample need not be a plate).
    chi is the logged chi (the cradle chi minus the mount's ``chi_offset_deg``). Paths are in mm, volumes in mm^3.
    ``ignored`` is 1 where the correction is above the cutoff. The file is written whole or not at all
    (``retry.write_atomic``).

    Args:
        path: Output CSV file.
        pred: Result of ``core.prediction.predict``.
        cutoff: Correction above which a point is marked ignored (``pred.cutoff`` is the one the prediction was
            made with).

    Raises:
        ValueError: If an hkl label holds a character other than letters, digits, space and + _ . -.
        OSError: If the file cannot be written (``retry.write_atomic``).
    """
    cmax = float(cutoff)
    for q in pred.planes:
        check_label(q.plane.hkl, what="hkl")
    rows: list[list[object]] = [list(COLUMNS)]
    for q in pred.planes:
        g, chi_offset_deg = q.gauge, q.mount.conv.chi_offset_deg
        for c, f, h, L, V, cc in zip(q.chi, q.phi, q.poles(), g.mean_path, g.volume_in_sample, q.correction,
                                     strict=True):  # fmt: skip
            rows.append([q.plane.hkl, round(q.plane.two_theta, 3), round(q.plane.omega, 3), c - chi_offset_deg, f,
                         *np.round(h, 5), round(L, 3) if np.isfinite(L) else "", round(V, 3),
                         round(cc, 4) if np.isfinite(cc) else "", int(cc > cmax)])  # fmt: skip

    def write(tmp: Path) -> None:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)

    write_atomic(path, write)
