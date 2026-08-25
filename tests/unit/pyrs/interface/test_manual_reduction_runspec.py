"""Unit tests for the manual-reduction run-number specification parser.

These tests exercise `parse_run_numbers`/`is_run_specification` directly, with no
file I/O or HFIR archive access needed -- kept out of the HFIR-gated integration
suite for exactly that reason (originally split out of
tests/integration/test_batch_reduction.py; see plans/test-framework.md).
"""

import pytest

from pyrs.interface.manual_reduction.manual_reduction_model import is_run_specification, parse_run_numbers


def test_parse_run_numbers_single_run_returns_one_element_list() -> None:
    """Test that a single run number parses to a one-element list."""
    # Arrange
    text = "938"

    # Act
    result = parse_run_numbers(text)

    # Assert
    assert result == [938]


def test_parse_run_numbers_dash_range_returns_inclusive_list() -> None:
    """Test that a dash range expands to an inclusive list of run numbers."""
    # Arrange
    text = "938-940"

    # Act
    result = parse_run_numbers(text)

    # Assert
    assert result == [938, 939, 940]


def test_parse_run_numbers_comma_separated_returns_ordered_list() -> None:
    """Test that comma-separated runs parse in the order given."""
    # Arrange
    text = "938,945,950"

    # Act
    result = parse_run_numbers(text)

    # Assert
    assert result == [938, 945, 950]


def test_parse_run_numbers_mixed_range_and_list_with_spaces_returns_combined_list() -> None:
    """Test that ranges and individual runs can be combined, with spaces ignored."""
    # Arrange
    text = "938-940, 945"

    # Act
    result = parse_run_numbers(text)

    # Assert
    assert result == [938, 939, 940, 945]


def test_parse_run_numbers_stray_comma_returns_list_with_empty_tokens_skipped() -> None:
    """Test that empty tokens from stray commas are skipped."""
    # Arrange
    text = "938,,940"

    # Act
    result = parse_run_numbers(text)

    # Assert
    assert result == [938, 940]


def test_parse_run_numbers_non_integer_token_raises_value_error() -> None:
    """Test that a non-integer token raises `ValueError`."""
    # Arrange
    text = "938-abc"

    # Act / Assert
    with pytest.raises(ValueError):
        parse_run_numbers(text)


def test_parse_run_numbers_blank_or_whitespace_returns_empty_list() -> None:
    """Test that blank and whitespace-only input return an empty list, not an error."""
    # Arrange / Act / Assert
    assert parse_run_numbers("") == []
    assert parse_run_numbers("   ") == []


def test_is_run_specification_digit_dash_comma_strings_returns_true() -> None:
    """Test that digit/dash/comma strings (including spaced dash ranges) are recognized as run specs."""
    # Arrange / Act / Assert
    assert is_run_specification("938")
    assert is_run_specification("938-940,945")
    assert is_run_specification(" 938 - 940 ")


def test_is_run_specification_path_or_blank_returns_false() -> None:
    """Test that file paths, blank input, and whitespace-only input are not run specs."""
    # Arrange / Act / Assert
    assert not is_run_specification("tests/data/HB2B_938.nxs.h5")
    assert not is_run_specification("")
    assert not is_run_specification("   ")
