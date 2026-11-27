"""
File output: pole-figure CSVs, the predicted CSV and table-scan workbooks.

- ``polefig_csv``: the reference script's per-run pole-figure CSVs, the input of an MTEX ODF fit.
- ``predicted_csv``: the reference script's predicted_correction.csv.
- ``table_scan``: one .xlsx table-scan workbook per plane.
- ``retry``: every file is written to a temporary file and moved over its name, retried while the file is locked.

Example:
    >>> from pyrs.pole_figures.io.polefig_csv import write_all
    >>> paths = write_all("work", pole_figures)
"""
