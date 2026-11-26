"""Text helpers of the table-scan workbooks: file names, durations and formula-safe cells."""

import pytest

from pyrs.pole_figures.io.table_scan import format_duration, safe_text, scan_filename


@pytest.mark.parametrize(("seconds", "want"), [(45, "45 s"), (600, "10 min"), (3 * 3600 + 7 * 60, "3 h 07 min")])
def test_format_duration(seconds: float, want: str) -> None:
    # Arrange
    duration = seconds

    # Act
    text = format_duration(duration)

    # Assert
    assert text == want


def test_safe_text_quotes_formula() -> None:
    # Arrange
    text = "=SUM(A1)"

    # Act
    safe = safe_text(text)

    # Assert
    assert safe == "'=SUM(A1)"


def test_safe_text_plain_unchanged() -> None:
    # Arrange
    text = "IN718 plate"

    # Act
    safe = safe_text(text)

    # Assert
    assert safe == text


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
