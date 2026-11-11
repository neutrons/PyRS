"""Tests for pyrs/utilities/NXstress/peak_tag.py"""

import pytest

from pyrs.utilities.NXstress._peaks import _Peaks
from pyrs.utilities.NXstress.peak_tag import UNKNOWN_HKL, peak_tag_key


@pytest.mark.parametrize("tag", ["Fe 211", "Fe211", "Ni 111", "Al 002"])
def test_peak_tag_key_matches_writer(tag: str) -> None:
    """The key is the one the NXstress writer files the tag under."""
    assert peak_tag_key(tag) == _Peaks._parse_peak_tag(tag)


def test_peak_tag_key_spacing_does_not_matter() -> None:
    assert peak_tag_key("Fe 211") == peak_tag_key("Fe211") == ("Fe", (2, 1, 1))


@pytest.mark.parametrize("tag", ["peak0", "peak12", "Fe 21"])
def test_peak_tag_key_unparsable_keyed_by_tag(tag: str) -> None:
    """A tag the writer cannot parse is keyed by itself, so it collides only with an identical tag."""
    assert peak_tag_key(tag) == (tag, UNKNOWN_HKL)
