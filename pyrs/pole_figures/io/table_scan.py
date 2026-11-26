"""
Write HB-2B table-scan workbooks (.xlsx) from ``core.table_scan`` plans, and the text helpers for them.

Layout (the HB-2B table-scan file convention): row 1 holds the column names, row 2 sets the run title, phase label,
omega and 2theta, then one row per point with chi, phi and the count time. A blank cell leaves that motor or PV
unchanged. chi is the logged (goniometer) chi.

Typical workflow:

1. Plan the scan of each plane with ``core.table_scan`` (one ``PlaneScan`` per plane).
2. Write one workbook per plane with ``write_table_scans``.

Example:
    >>> paths = write_table_scans("scans", scans, prefix="IN718 plate", phase_name="IN718")
    >>> paths[0].name
    'IN718_plate_111.xlsx'
"""

import re
import time
from pathlib import Path
from typing import Literal

import openpyxl

from ..core.hb2b import PHASE_PV, RUN_TITLE_PV
from ..core.table_scan import PlaneScan

#: column whose value says what each point waits for, and the "wait for" condition a count time is given in
WAIT_FOR_COLUMN, VALUE_COLUMN = "Wait For", "Value"
WaitFor = Literal["seconds"]
WAIT_SECONDS: WaitFor = "seconds"
COLUMNS: tuple[str, ...] = (RUN_TITLE_PV, PHASE_PV, "omega", "2theta", "chi", "phi", WAIT_FOR_COLUMN, VALUE_COLUMN)
Row = list[object]


def write_xlsx(path: str | Path, rows: list[Row], *, retries: int = 20, wait_s: float = 1.5) -> None:
    """
    Write rows to the first sheet of a new workbook, retrying while a sync client or Excel briefly holds the file.

    Args:
        path: Output .xlsx file.
        rows: Table rows, header first (:func:`table_rows`).
        retries: Number of save attempts before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).

    Raises:
        OSError: If the file is still locked after the last attempt.
        RuntimeError: If the new workbook has no sheet.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is None:
        raise RuntimeError(f"openpyxl made a workbook without a sheet for {path}")
    for r in rows:
        ws.append(r)
    for k in range(retries):
        try:
            wb.save(path)
            return
        except OSError:
            if k == retries - 1:
                raise
            time.sleep(wait_s)


def scan_filename(prefix: str, hkl: str) -> str:
    """
    File name of the workbook of one plane.

    Runs of characters other than letters and digits in ``prefix`` become one underscore. An empty prefix gives
    "pole_figure".

    Args:
        prefix: Sample or run-title prefix.
        hkl: Plane, for example "222".

    Returns:
        The file name, without a folder.

    Example:
        >>> scan_filename("IN718 plate", "222")
        'IN718_plate_222.xlsx'
        >>> scan_filename("", "222")
        'pole_figure_222.xlsx'
    """
    stem = re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_") or "pole_figure"
    return f"{stem}_{hkl}.xlsx"


def write_table_scans(folder: str | Path, scans: list[PlaneScan], *, prefix: str, phase_name: str) -> list[Path]:
    """
    Write one workbook per plane to ``folder`` (created if needed). Planes with no kept points are skipped.

    Args:
        folder: Output folder.
        scans: Table scans, one per plane.
        prefix: Prefix of the run titles and the file names.
        phase_name: Phase name, written with the hkl in each workbook.

    Returns:
        The paths written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for sc in scans:
        if not sc.points:
            continue
        path = folder / scan_filename(prefix, sc.plane.hkl)
        label = f"{phase_name} {sc.plane.hkl}".strip()
        write_xlsx(path, table_rows(sc, title=run_title(prefix, sc), phase_label=label))
        out.append(path)
    return out


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


def table_rows(scan: PlaneScan, *, title: str, phase_label: str) -> list[Row]:
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
        r[4:] = [round(p.chi, 4), round(p.phi, 4), WAIT_SECONDS, p.count_s]
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
