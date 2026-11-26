"""
Pole-figure table scans from a prediction: keep only the useful chi, phi points and set each count time.

One table scan per plane (one run per reflection). Points whose predicted correction is
above the cutoff are left out, so no time is spent where the beam path is too long. By default the count time of each
kept point scales with its correction (time = base x correction, whole seconds), so every kept point gets about the
same corrected statistics. A fixed time per point and a cap per point are options.

Order: chi ring by ring, with phi going up on one ring and down on the next ("serpentine"), so the cradle never swings
phi back from 355 to 0 between rings. Left-out points are simply not visited.

Layout (the HB-2B table-scan file convention): row 1 holds the
column names, row 2 sets the run title, phase label, omega and 2theta, then one row per point with chi, phi and the
count time. A blank cell leaves that motor or PV unchanged. chi is the logged (goniometer) chi.

Typical workflow:

1. Predict the correction on the planned grid (``prediction.predict``).
2. Plan each plane with :func:`plan_scan`.
3. Name the run with :func:`run_title` and build the rows with :func:`table_rows`, then save them as a spreadsheet.

Example:
    >>> scan = plan_scan(prediction.planes[0], cutoff=3.0, base_s=30.0)  # doctest: +SKIP
    >>> rows = table_rows(scan, run_title("IN718 plate", scan), "IN718")  # doctest: +SKIP
    >>> format_duration(scan.total_s)  # doctest: +SKIP
    '5 h 12 min'
"""

import math
from dataclasses import dataclass

import numpy as np

from .prediction import Plane, PlanePrediction

TITLE, PHASE = "HB2B:SMS:RunInfo:RunTitle", "HB2B:CS:hklPhase"
COLUMNS: tuple[str, ...] = (TITLE, PHASE, "omega", "2theta", "chi", "phi", "Wait For", "Value")
Row = list[object]


@dataclass(frozen=True, kw_only=True)
class ScanPoint:
    """
    One measurement of a table scan.

    Attributes:
        chi: Logged (goniometer) chi (deg).
        phi: Phi (deg).
        correction: Predicted path-length correction F(0, 0) / F.
        count_s: Count time (s).
    """

    chi: float
    phi: float
    correction: float
    count_s: int


@dataclass(kw_only=True)
class PlaneScan:
    """
    The table scan for one plane.

    Attributes:
        plane: hkl, 2theta and omega.
        points: Kept points in measurement order (chi rings, phi alternating up and down when serpentine).
        n_grid: Number of grid points before the cutoff.
        base_s: Count time at correction 1 (s).
        cutoff: Points with a correction above this were left out.
        serpentine: phi alternates direction from one chi ring to the next.
        chi_offset_deg: Cradle chi = logged chi + offset (deg), the mount's offset used for the logged chi of the
            points (a record: the points already hold the logged chi).
        scaled: Count time = base x correction (True) or base for every point (False).
        max_s: Cap on the count time per point (s, inf: no cap).
    """

    plane: Plane
    points: list[ScanPoint]
    n_grid: int
    base_s: float
    cutoff: float
    serpentine: bool
    chi_offset_deg: float
    scaled: bool
    max_s: float

    @property
    def n_capped(self) -> int:
        """Points whose time was cut by the cap."""
        return sum(1 for p in self.points if self.scaled and self.base_s * p.correction > self.max_s + 0.5)

    @property
    def n_skipped(self) -> int:
        """Grid points left out by the cutoff."""
        return self.n_grid - len(self.points)

    @property
    def total_s(self) -> int:
        """Total count time (s), without motor moves."""
        return int(sum(p.count_s for p in self.points))


def count_time(correction: float, *, base_s: float, scaled: bool = True, max_s: float = math.inf) -> int:
    """
    Count time for one point: base x correction (or base when not scaled), capped, whole seconds, at least 1.

    Args:
        correction: Predicted correction of the point.
        base_s: Count time at correction 1 (s).
        scaled: Scale the time with the correction (default: True).
        max_s: Cap on the count time (s, default: inf, no cap).

    Returns:
        The count time (s), rounded to a whole second.

    Example:
        >>> count_time(2.4, base_s=30.0, max_s=60.0)
        60
    """
    t = base_s * correction if scaled else base_s
    t = min(t, max_s)
    return max(1, int(round(t)))


def measurement_order(chi: np.ndarray, phi: np.ndarray, *, serpentine: bool = True) -> np.ndarray:
    """
    Indices in measurement order: chi rings from low to high, phi up, then down on the next ring when serpentine.

    Only rings that have points count for the alternation, so left-out rings do not break it.

    Args:
        chi: Chi of each point (deg). Values that round to the same 1e-6 deg are one ring.
        phi: Phi of each point (deg).
        serpentine: Alternate the phi direction from ring to ring (default: True).

    Returns:
        Indices into chi and phi.

    Example:
        >>> measurement_order(np.array([0.0, 0.0, 10.0, 10.0]), np.array([0.0, 90.0, 0.0, 90.0]))
        array([0, 1, 3, 2])
    """
    rings = np.round(np.asarray(chi, float), 6)
    out: list[int] = []
    up = True
    for c in np.unique(rings):
        idx = np.flatnonzero(rings == c)
        idx = idx[np.argsort(np.asarray(phi, float)[idx], kind="stable")]
        out.extend(idx if (up or not serpentine) else idx[::-1])
        up = not up
    return np.asarray(out, int)


def plan_scan(q: PlanePrediction, *, cutoff: float, base_s: float, serpentine: bool = True, scaled: bool = True,
              max_s: float = math.inf) -> PlaneScan:  # fmt: skip
    """
    Keep the points with correction <= cutoff, give each a count time, and put them in measurement order.

    Args:
        q: Prediction for one plane on the planned chi/phi grid. Its mount's ``conv.chi_offset_deg`` turns the
            predicted cradle chi into the logged chi that the table holds.
        cutoff: Leave out points whose correction is above this (the hatched cells) or not finite.
        base_s: Count time at correction 1 (s).
        serpentine: Alternate the phi direction from ring to ring (default: True). False keeps phi increasing on
            every ring.
        scaled: Count time = base x correction (default: True). False gives every point the base time.
        max_s: Cap on the count time per point (s, default: inf, no cap).

    Returns:
        The table scan of the plane.

    Raises:
        ValueError: If base_s is not positive or max_s is below 1 s.
    """
    if base_s <= 0:
        raise ValueError(f"the count time at correction 1 must be positive, got {base_s!r} s")
    if not max_s >= 1:
        raise ValueError(f"the cap on the count time must be at least 1 s, got {max_s!r} s")
    keep = np.isfinite(q.correction) & (q.correction <= cutoff)
    idx = np.flatnonzero(keep)
    order = idx[measurement_order(q.chi[idx], q.phi[idx], serpentine=serpentine)]
    conv = q.mount.conv
    pts = [ScanPoint(chi=float(conv.logged_chi(q.chi[i])) + 0.0, phi=float(q.phi[i]),
                     correction=float(q.correction[i]),
                     count_s=count_time(float(q.correction[i]), base_s=base_s, scaled=scaled, max_s=max_s))
           for i in order]  # fmt: skip
    return PlaneScan(plane=q.plane, points=pts, n_grid=len(q.chi), base_s=float(base_s), cutoff=float(cutoff),
                     serpentine=serpentine, chi_offset_deg=float(conv.chi_offset_deg), scaled=scaled,
                     max_s=float(max_s))  # fmt: skip


def phi_travel(scan: PlaneScan) -> float:
    """
    Total phi motion between consecutive points (a rough measure of move time).

    Args:
        scan: The table scan.

    Returns:
        Sum of the phi steps (deg), 0 for fewer than two points.
    """
    phi = np.array([p.phi for p in scan.points])
    return float(np.abs(np.diff(phi)).sum()) if len(phi) > 1 else 0.0


def safe_text(value: str) -> str:
    """
    Text that a spreadsheet will not read as a formula.

    Args:
        value: Cell text.

    Returns:
        The text with a leading apostrophe when it starts with =, +, -, @, a tab or a carriage return.
    """
    return f"'{value}" if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def run_title(prefix: str, scan: PlaneScan) -> str:
    """
    Run title for the table scan.

    Args:
        prefix: The user's text in front (sample name). Empty gives no prefix.
        scan: The table scan.

    Returns:
        E.g. 'IN718 plate | {222} pole figure | 2theta 112 omega 56 | 600 of 649 points (correction <= 3)'.
    """
    head = f"{prefix.strip()} | " if prefix.strip() else ""
    pl = scan.plane
    return (f"{head}{{{pl.hkl}}} pole figure | 2theta {pl.two_theta:g} omega {pl.omega:g} | "
            f"{len(scan.points)} of {scan.n_grid} points (correction <= {scan.cutoff:g})")  # fmt: skip


def table_rows(scan: PlaneScan, title: str, phase_label: str) -> list[Row]:
    """
    All rows of the table scan (see the layout in the module docstring).

    Args:
        scan: The table scan.
        title: Run title (row 2).
        phase_label: Phase label (row 2).

    Returns:
        The column names, the run row (title, phase, omega, 2theta), then one row per point (chi, phi, "seconds",
        count time). Empty cells are None. Angles are rounded to 4 decimals.
    """
    pl = scan.plane
    blank: Row = [None] * len(COLUMNS)
    head = list(blank)
    head[:4] = [safe_text(title), safe_text(phase_label), round(pl.omega, 4), round(pl.two_theta, 4)]
    rows: list[Row] = [list(COLUMNS), head]
    for p in scan.points:
        r = list(blank)
        r[4:] = [round(p.chi, 4), round(p.phi, 4), "seconds", p.count_s]
        rows.append(r)
    return rows


def format_duration(seconds: float) -> str:
    """
    A duration as short text.

    Args:
        seconds: The duration (s).

    Returns:
        Seconds below one minute, else whole minutes and hours, e.g. '45 s', '12 min' or '1 h 05 min'.

    Example:
        >>> format_duration(3900)
        '1 h 05 min'
    """
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"
