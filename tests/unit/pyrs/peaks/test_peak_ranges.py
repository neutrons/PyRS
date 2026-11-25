"""Unit tests for `pyrs.peaks.peak_ranges`."""

import json

import pytest

from pyrs.peaks.peak_ranges import DEFAULT_D0, PeakRange, PeakRangeFileError, read_peak_range_json


def _write(path, entries):
    path.write_text(json.dumps({str(i): entry for i, entry in enumerate(entries)}))
    return path


def test_read_empty_d0_list_uses_default(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [{"peak_range": [87.415, 92.486], "peak_label": "IN718_311", "d0": []}])

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert peaks == [PeakRange(label="IN718_311", x_min=87.415, x_max=92.486, d0=DEFAULT_D0)]
    assert DEFAULT_D0 == 1.0


@pytest.mark.parametrize(
    "d0, expected",
    [(1.17, [1.17, 1.17]), ([1.1, 1.2], [1.1, 1.2]), ([1.1], [DEFAULT_D0, DEFAULT_D0])],
    ids=["scalar", "one-per-peak", "wrong-length"],
)
def test_read_resolves_d0(tmp_path, d0, expected):
    # Arrange
    entries = [
        {"peak_range": [78.0, 85.0], "peak_label": "A", "d0": d0},
        {"peak_range": [90.0, 95.0], "peak_label": "B", "d0": d0},
    ]
    filename = _write(tmp_path / "p.json", entries)

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert [p.d0 for p in peaks] == expected
    assert all(type(p.d0) is float for p in peaks)


def test_read_keeps_order_and_windows(tmp_path):
    # Arrange
    entries = [
        {"peak_range": [90.0, 95.0], "peak_label": "B", "d0": 1.2},
        {"peak_range": [78.0, 85.0], "peak_label": "A", "d0": 1.1},
    ]
    filename = _write(tmp_path / "p.json", entries)

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert [(p.label, p.x_min, p.x_max) for p in peaks] == [("B", 90.0, 95.0), ("A", 78.0, 85.0)]


def test_read_missing_d0_raises(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [{"peak_range": [78.0, 85.0], "peak_label": "A"}])

    # Act / Assert
    with pytest.raises(ValueError, match="peak '0': missing d0"):
        read_peak_range_json(filename)


@pytest.mark.parametrize("d0", [None, "", "1.1", True, float("nan"), 0.0, -1.2, [None, 1.0]], ids=repr)
def test_read_bad_d0_raises(tmp_path, d0):
    # Arrange
    filename = _write(tmp_path / "p.json", [{"peak_range": [78.0, 85.0], "peak_label": "A", "d0": d0}] * 2)

    # Act / Assert
    with pytest.raises(ValueError, match=r"p\.json, peak '0': d0 must be"):
        read_peak_range_json(filename)


@pytest.mark.parametrize("window", [[85.0, 78.0], [78.0], [78.0, None], "78-85", [78.0, float("inf")]], ids=repr)
def test_read_bad_window_raises(tmp_path, window):
    # Arrange
    filename = _write(tmp_path / "p.json", [{"peak_range": window, "peak_label": "A", "d0": 1.0}])

    # Act / Assert
    with pytest.raises(ValueError, match="peak_range must"):
        read_peak_range_json(filename)


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "expected an object of peaks, got list"),
        ('{"0": 5}', "peak '0': expected an object, got int"),
        ("", "not valid JSON"),
    ],
    ids=["syntax", "top-level-list", "entry-not-object", "empty-file"],
)
def test_read_malformed_file_raises(tmp_path, content, message):
    # Arrange
    filename = tmp_path / "p.json"
    filename.write_text(content)

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match=message):
        read_peak_range_json(filename)


def test_read_missing_file_raises(tmp_path):
    # Arrange
    filename = tmp_path / "absent.json"

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="absent.json: cannot read"):
        read_peak_range_json(filename)
