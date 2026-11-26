"""
File output: pole-figure CSVs, mount matrices as text and table-scan workbooks.

- ``polefig_csv``: the reference script's per-run pole-figure CSVs, the input of an MTEX ODF fit.
- ``matrix_text``: mount matrices as text and back.
- ``table_scan``: one .xlsx table-scan workbook per plane.

Example:
    >>> from pyrs.pole_figures.io.polefig_csv import write_all
    >>> paths = write_all("work", pole_figures)
"""
