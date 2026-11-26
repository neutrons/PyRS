"""
Write HB-2B table-scan workbooks (.xlsx) from ``core.table_scan`` plans.

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

from pathlib import Path
from typing import Literal

import openpyxl

from ..core.hb2b import PHASE_PV, RUN_TITLE_PV
from ..core.table_scan import PlaneScan
from ..text import file_part
from .retry import RETRIES, WAIT_S, write_atomic

#: column whose value says what each point waits for, and the "wait for" condition a count time is given in
WAIT_FOR_COLUMN, VALUE_COLUMN = "Wait For", "Value"
WaitFor = Literal["seconds"]
WAIT_SECONDS: WaitFor = "seconds"
COLUMNS: tuple[str, ...] = (RUN_TITLE_PV, PHASE_PV, "omega", "2theta", "chi", "phi", WAIT_FOR_COLUMN, VALUE_COLUMN)
Row = list[object]


def write_xlsx(path: str | Path, rows: list[Row], *, retries: int = RETRIES, wait_s: float = WAIT_S) -> None:
    """
    Write rows to the first sheet of a new workbook, retrying while a sync client or Excel briefly holds the file.

    Text is written unchanged. Text that starts with "=" (which openpyxl would store as a formula) is stored as text
    with Excel's quote prefix, so Excel shows it as typed and does not run it. The workbook is written to a temporary
    file and moved over ``path`` (``retry.write_atomic``), so a failed save leaves no partial workbook.

    Args:
        path: Output .xlsx file.
        rows: Table rows, header first (:func:`table_rows`).
        retries: Attempts to replace a locked file before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).

    Raises:
        PermissionError: If the file is still locked after the last attempt.
        OSError: At once for any other file error.
        RuntimeError: If the new workbook has no sheet.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is None:
        raise RuntimeError(f"openpyxl made a workbook without a sheet for {path}")
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and cell.value.startswith("="):
                cell.data_type = "s"
                cell.quotePrefix = True
    write_atomic(path, wb.save, retries=retries, wait_s=wait_s)


def scan_filename(prefix: str, hkl: str) -> str:
    """
    File name of the workbook of one plane.

    Runs of characters other than letters and digits in ``prefix`` and ``hkl`` become one underscore
    (``text.file_part``), so the name never holds a folder. An empty prefix gives "pole_figure".

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
    stem = file_part(prefix) or "pole_figure"
    return f"{stem}_{file_part(hkl)}.xlsx"


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
    head[:4] = [title, phase_label, round(pl.omega, 4), round(pl.two_theta, 4)]
    rows: list[Row] = [list(COLUMNS), head]
    for p in scan.points:
        r = list(blank)
        r[4:] = [round(p.chi, 4), round(p.phi, 4), WAIT_SECONDS, p.count_s]
        rows.append(r)
    return rows
