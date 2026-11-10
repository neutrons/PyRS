"""Tests for pyrs/interface/peak_fitting/peak_tags.py"""

import pytest

from pyrs.interface.peak_fitting.peak_tags import peak_tags_from_labels


@pytest.mark.parametrize(
    "labels, n_peaks, expected",
    [
        (["Peak0", "Peak1"], 2, ["peak0", "peak1"]),
        (["Fe 211", "Peak1"], 2, ["Fe 211", "peak1"]),
        (["  Fe 211 ", ""], 2, ["Fe 211", "peak1"]),
        ([None], 2, ["peak0", "peak1"]),
        ([], 0, []),
    ],
)
def test_peak_tags_from_labels(labels: list, n_peaks: int, expected: list) -> None:
    """Edited labels become tags; blank or auto-generated `Peak<N>` labels keep
    the historical `peak<N>` tag."""
    # Act
    tags = peak_tags_from_labels(labels, n_peaks)

    # Assert
    assert tags == expected


@pytest.mark.parametrize(
    "labels",
    [
        ["Fe 211", "Fe 211"],
        # A typed lower-case `peak1` collides with the default tag of the second range.
        ["peak1", "Peak1"],
    ],
)
def test_peak_tags_from_labels_duplicates_raises(labels: list) -> None:
    # Act / Assert
    with pytest.raises(ValueError, match="distinct label"):
        peak_tags_from_labels(labels, len(labels))


@pytest.mark.parametrize(
    "labels",
    [["Fe 211", "Fe211"], ["Fe 211", "Peak1", " Fe211"]],
    ids=["adjacent", "with-default-between"],
)
def test_peak_tags_from_labels_same_nxstress_peak_raises(labels: list) -> None:
    """Different labels that NXstress files under one peak are refused before fitting."""
    # Act / Assert
    with pytest.raises(ValueError, match=r"name the same peak: 'Fe 211' and 'Fe211' \(phase 'Fe', hkl 2 1 1\)"):
        peak_tags_from_labels(labels, len(labels))


def test_peak_tags_from_labels_different_nxstress_peaks() -> None:
    """Same phase, different planes; and legacy tags, which never collide with each other."""
    # Act
    tags = peak_tags_from_labels(["Fe 211", "Fe 110", "Peak2", "Peak3"], 4)

    # Assert
    assert tags == ["Fe 211", "Fe 110", "peak2", "peak3"]
