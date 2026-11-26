"""Table-scan workbooks: file names, text cells kept as typed, and saves that leave no partial file."""

from pathlib import Path
from typing import Any

import openpyxl
import pytest

from pyrs.pole_figures.io.table_scan import scan_filename, write_xlsx


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


def test_scan_filename_path_in_hkl_stays_in_folder() -> None:
    # Arrange
    hkl = "../../../tmp/evil"

    # Act
    name = scan_filename("IN718", hkl)

    # Assert
    assert name == "IN718_tmp_evil.xlsx"
    assert "/" not in name and "\\" not in name and ".." not in name


def test_write_xlsx_failed_save_leaves_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: an older workbook, then a save that writes part of the file and fails
    path = tmp_path / "scan.xlsx"
    write_xlsx(path, [["old"]])

    def broken_save(self: Any, filename: Any) -> None:
        Path(filename).write_bytes(b"PK partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(openpyxl.Workbook, "save", broken_save)

    # Act / Assert
    with pytest.raises(OSError, match="No space"):
        write_xlsx(path, [["new"]])
    ws = openpyxl.load_workbook(path).active
    assert ws is not None and ws["A1"].value == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["scan.xlsx"]
