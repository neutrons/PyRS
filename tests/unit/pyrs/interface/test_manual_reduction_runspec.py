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


# The following moved from tests/integration/test_batch_reduction.py — they test the
# same two functions and need no HFIR access, so they don't belong in an integration
# suite gated on archive availability. See plans/test-framework.md.


def test_parse_run_numbers_range():
    """A dash range is expanded to inclusive list."""
    assert parse_run_numbers("1017-1019") == [1017, 1018, 1019]


def test_parse_run_numbers_comma_and_range():
    """Mixed comma and range parses correctly."""
    assert parse_run_numbers("1017,1019-1021") == [1017, 1019, 1020, 1021]


def test_is_run_specification_run_numbers():
    assert is_run_specification("1017")
    assert is_run_specification("1017-1019")
    assert is_run_specification("1017, 1019")


def test_is_run_specification_rejects_path():
    assert not is_run_specification("/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5")
