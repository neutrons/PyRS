"""Text helpers of the table-scan workbooks: file names, durations, and text cells kept as typed."""

from pathlib import Path

import openpyxl
import pytest

from pyrs.pole_figures.io.table_scan import format_duration, scan_filename, write_xlsx


@pytest.mark.parametrize(("seconds", "want"), [(45, "45 s"), (600, "10 min"), (3 * 3600 + 7 * 60, "3 h 07 min")])
def test_format_duration(seconds: float, want: str) -> None:
    # Arrange
    duration = seconds

    # Act
    text = format_duration(duration)

    # Assert
    assert text == want


@pytest.mark.parametrize("text", ["=1+1", "-10C anneal | {200} pole figure", "+x", "@y", "-2 0 0"])
def test_write_xlsx_leading_symbol_kept_as_text(text: str, tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "scan.xlsx"

    # Act
    write_xlsx(path, [["title", "phase"], [text, "IN718 200"]])

    # Assert
    ws = openpyxl.load_workbook(path).active
    assert ws is not None
    cell = ws["A2"]
    assert cell.value == text
    assert cell.data_type == "s"
    assert not str(cell.value).startswith("'")


def test_write_xlsx_formula_text_has_quote_prefix(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "scan.xlsx"

    # Act
    write_xlsx(path, [["=1+1", "-10C"]])

    # Assert
    ws = openpyxl.load_workbook(path).active
    assert ws is not None
    assert ws["A1"].quotePrefix
    assert not ws["B1"].quotePrefix


def test_write_xlsx_plain_title_unchanged(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "scan.xlsx"
    row = ["IN718 plate | {222} pole figure", "IN718 222", 56.0]

    # Act
    write_xlsx(path, [row])

    # Assert
    ws = openpyxl.load_workbook(path).active
    assert ws is not None
    assert [c.value for c in ws[1]] == row
    assert not ws["A1"].quotePrefix


def test_scan_filename_joins_prefix_words() -> None:
    # Arrange
    prefix = "IN718 plate / A"

    # Act
    name = scan_filename(prefix, "222")

    # Assert
    assert name == "IN718_plate_A_222.xlsx"


def test_scan_filename_empty_prefix() -> None:
    # Arrange
    prefix = ""

    # Act
    name = scan_filename(prefix, "311")

    # Assert
    assert name == "pole_figure_311.xlsx"
