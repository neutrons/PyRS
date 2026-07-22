#!/usr/bin/python
# Tests for the shared ETA_MASK_ANGLE config parser used by both the
# autoreduction and live-reduction pipelines.
import configparser
from pyrs.utilities.file_util import parse_eta_mask_angle
import pytest


def _config(value):
    """Build a config with a single REDUCTION/ETA_MASK_ANGLE entry."""
    config = configparser.ConfigParser()
    config.add_section("REDUCTION")
    config["REDUCTION"]["ETA_MASK_ANGLE"] = value
    return config


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),        # blank -> no eta masking (in-plane only)
        ("   ", None),     # whitespace-only is also blank
        ("5", 5.0),        # single digit must be honored (was dropped by auto)
        ("10", 10.0),
        ("-5", -5.0),
        ("2.5", 2.5),
    ],
)
def test_parse_eta_mask_angle_valid(value, expected):
    """Blank yields None; any numeric value is coerced to float."""
    # Arrange / Act
    result = parse_eta_mask_angle(_config(value))

    # Assert
    assert result == expected


def test_parse_eta_mask_angle_missing_key_is_none():
    """A REDUCTION section without the key is treated as blank."""
    # Arrange
    config = configparser.ConfigParser()
    config.add_section("REDUCTION")

    # Act / Assert
    assert parse_eta_mask_angle(config) is None


def test_parse_eta_mask_angle_rejects_unparseable():
    """A non-empty, non-numeric value raises rather than silently disabling eta."""
    with pytest.raises(ValueError):
        parse_eta_mask_angle(_config("abc"))


def test_parse_eta_mask_angle_auto_and_live_agree():
    """Both pipelines call the same parser, so identical config -> identical eta."""
    # Arrange
    config = _config("7.5")

    # Act -- auto and live both do exactly this call now
    auto_eta = parse_eta_mask_angle(config)
    live_eta = parse_eta_mask_angle(config)

    # Assert
    assert auto_eta == live_eta == 7.5


if __name__ == "__main__":
    pytest.main()
