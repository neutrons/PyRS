"""Unit tests for the manual-reduction run-number specification parser."""

import pytest

from pyrs.interface.manual_reduction.manual_reduction_model import is_run_specification, parse_run_numbers


def test_parse_run_numbers_single():
    """A single run number parses to a one-element list."""
    assert parse_run_numbers("938") == [938]


def test_parse_run_numbers_dash_range_is_inclusive():
    """A dash range includes both endpoints."""
    assert parse_run_numbers("938-940") == [938, 939, 940]


def test_parse_run_numbers_comma_list():
    """Comma-separated runs parse in order."""
    assert parse_run_numbers("938,945,950") == [938, 945, 950]


def test_parse_run_numbers_mixed_range_and_list():
    """Ranges and individual runs can be combined and spaces are ignored."""
    assert parse_run_numbers("938-940, 945") == [938, 939, 940, 945]


def test_parse_run_numbers_trailing_comma_ignored():
    """Empty tokens from stray commas are skipped."""
    assert parse_run_numbers("938,,940") == [938, 940]


def test_parse_run_numbers_invalid_raises():
    """A non-integer token raises ValueError."""
    with pytest.raises(ValueError):
        parse_run_numbers("938-abc")


def test_is_run_specification_accepts_run_specs():
    """Digit/dash/comma strings are recognized as run specs."""
    assert is_run_specification("938")
    assert is_run_specification("938-940,945")
    assert is_run_specification(" 938 - 940 ")


def test_is_run_specification_rejects_paths_and_empty():
    """File paths and empty input are not run specs."""
    assert not is_run_specification("tests/data/HB2B_938.nxs.h5")
    assert not is_run_specification("")
    assert not is_run_specification("   ")
