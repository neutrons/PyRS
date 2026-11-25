"""Unit tests for `pyrs.peaks.peak_ranges`.

The contract under test: valid files read as before, an empty or wrong-length d0 list
defaults to `DEFAULT_D0`, and *every* other bad input -- file, path or content -- raises
`PeakRangeFileError` and nothing else, because the peak fitting window and autoreduction
catch that one type and carry on.
"""

import json
import os

import pytest

from pyrs.peaks.peak_ranges import (
    DEFAULT_D0,
    MAX_FILE_BYTES,
    PeakRange,
    PeakRangeFileError,
    read_peak_range_json,
)


def _entry(label="A", window=(78.0, 85.0), d0=1.0):
    return {"peak_range": list(window), "peak_label": label, "d0": d0}


def _write(path, entries):
    path.write_text(json.dumps({str(i): entry for i, entry in enumerate(entries)}))
    return path


def _write_raw(path, content):
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def test_default_d0_is_one():
    # Arrange / Act / Assert
    assert DEFAULT_D0 == 1.0


def test_read_empty_d0_list_uses_default(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry("IN718_311", (87.415, 92.486), [])])

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert peaks == [PeakRange(label="IN718_311", x_min=87.415, x_max=92.486, d0=DEFAULT_D0)]


@pytest.mark.parametrize(
    "d0, expected",
    [(1.17, [1.17, 1.17]), (2, [2.0, 2.0]), ([1.1, 1.2], [1.1, 1.2]), ([1.1], [DEFAULT_D0, DEFAULT_D0])],
    ids=["float", "int", "one-per-peak", "wrong-length"],
)
def test_read_resolves_d0(tmp_path, d0, expected):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry("A", d0=d0), _entry("B", (90.0, 95.0), d0=d0)])

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert [p.d0 for p in peaks] == expected
    assert all(type(p.d0) is float for p in peaks)


def test_read_keeps_order_and_windows(tmp_path):
    # Arrange
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"b": _entry("B", (90.0, 95.0), 1.2), "a": _entry("A", (78.0, 85.0), 1.1)}))

    # Act
    peaks = read_peak_range_json(path)

    # Assert
    assert [(p.label, p.x_min, p.x_max) for p in peaks] == [("B", 90.0, 95.0), ("A", 78.0, 85.0)]


def test_read_accepts_bom_and_non_ascii_label(tmp_path):
    # Arrange
    content = "﻿" + json.dumps({"0": _entry("γ' 111")}, ensure_ascii=False)
    filename = _write_raw(tmp_path / "p.json", content.encode("utf-8"))

    # Act
    peaks = read_peak_range_json(filename)

    # Assert
    assert peaks[0].label == "γ' 111"


def test_read_accepts_str_path(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry()])

    # Act
    peaks = read_peak_range_json(str(filename))

    # Assert
    assert len(peaks) == 1


def test_read_missing_field_names_the_peak(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry(), {"peak_range": [78.0, 85.0], "peak_label": "B"}])

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="peak '1': missing d0"):
        read_peak_range_json(filename)


@pytest.mark.parametrize(
    "d0",
    [None, "", "1.1", True, float("nan"), float("inf"), float("-inf"), 0.0, -1.2, [None, 1.0], [1.0, None], {}],
    ids=repr,
)
def test_read_bad_d0_raises(tmp_path, d0):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry("A", d0=d0), _entry("B", (90.0, 95.0), d0=d0)])
    position = "'1'" if d0 == [1.0, None] else "'0'"

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match=rf"peak {position}: d0 must be"):
        read_peak_range_json(filename)


def test_read_bad_later_peak_names_it(tmp_path):
    # Arrange
    filename = _write(tmp_path / "p.json", [_entry("A"), _entry("B", d0=None)])

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="peak '1': d0 must be"):
        read_peak_range_json(filename)


@pytest.mark.parametrize(
    "window",
    [[85.0, 78.0], [78.0], [78.0, None], "78-85", ["78", "85"], [True, 90.0], [78.0, float("inf")], [78.0, 10**400]],
    ids=repr,
)
def test_read_bad_window_raises(tmp_path, window):
    # Arrange
    filename = _write(tmp_path / "p.json", [{**_entry(), "peak_range": window}])

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="peak_range must"):
        read_peak_range_json(filename)


@pytest.mark.parametrize("label", [None, "", "   ", 5, ["x"], "a\nb", "\x1b[31m", "\ud800"], ids=repr)
def test_read_bad_label_raises(tmp_path, label):
    # Arrange
    filename = _write(tmp_path / "p.json", [{**_entry(), "peak_label": label}])

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="peak_label"):
        read_peak_range_json(filename)


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "not valid JSON"),
        ("", "not valid JSON"),
        ("[1, 2]", "expected an object of peaks, got list"),
        ("{}", "holds no peaks"),
        ('{"0": 5}', "peak '0': expected an object, got int"),
        ('{"a": {"peak_range": [1, 2], "peak_label": "x", "d0": 1}, "a": {}}', "key 'a' appears more than once"),
        (b"\xff\xfe\x00bad", "not valid UTF-8"),
        ("[" * 100_000 + "]" * 100_000, "cannot be parsed as JSON"),
        ('{"0": {"peak_range": [1, 2], "peak_label": "x", "d0": ' + "9" * 5000 + "}}", "cannot be parsed as JSON"),
        ('{"0": {"peak_range": [1, 2], "peak_label": "x", "d0": ' + "9" * 400 + "}}", "integer too large"),
    ],
    ids=[
        "syntax",
        "empty-file",
        "top-level-list",
        "no-peaks",
        "entry-not-object",
        "duplicate-key",
        "invalid-utf8",
        "deep-nesting",
        "5000-digit-int",
        "400-digit-int",
    ],
)
def test_read_malformed_content_raises(tmp_path, content, message):
    # Arrange
    filename = _write_raw(tmp_path / "p.json", content)

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match=message):
        read_peak_range_json(filename)


def test_read_missing_file_raises(tmp_path):
    # Arrange
    filename = tmp_path / "absent.json"

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="absent.json.*cannot read"):
        read_peak_range_json(filename)


def test_read_directory_raises(tmp_path):
    # Arrange / Act / Assert
    with pytest.raises(PeakRangeFileError, match="cannot read the file"):
        read_peak_range_json(tmp_path)


@pytest.mark.timeout(10)
def test_read_fifo_raises_instead_of_hanging(tmp_path):
    # Arrange
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="not a regular file"):
        read_peak_range_json(fifo)


def test_read_oversized_file_raises(tmp_path):
    # Arrange
    filename = _write_raw(tmp_path / "big.json", b" " * (MAX_FILE_BYTES + 1))

    # Act / Assert
    with pytest.raises(PeakRangeFileError, match="larger than"):
        read_peak_range_json(filename)


@pytest.mark.parametrize("path", [None, 5, True, b"p.json", 1.5, ["p.json"]], ids=repr)
def test_read_non_path_argument_raises(path):
    # Arrange / Act / Assert
    with pytest.raises(PeakRangeFileError, match="must be given as a path"):
        read_peak_range_json(path)


def test_read_path_with_nul_raises():
    # Arrange / Act / Assert
    with pytest.raises(PeakRangeFileError, match="cannot read"):
        read_peak_range_json("a\0b.json")


def test_read_error_message_has_no_control_characters(tmp_path):
    # Arrange
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"a\nFAKE: ok\x1b[2J": 5}))

    # Act
    with pytest.raises(PeakRangeFileError) as raised:
        read_peak_range_json(path)

    # Assert
    message = str(raised.value)
    assert "\n" not in message and "\x1b" not in message


def test_read_long_key_is_cut_in_message(tmp_path):
    # Arrange
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"k" * 100_000: 5}))

    # Act
    with pytest.raises(PeakRangeFileError) as raised:
        read_peak_range_json(path)

    # Assert
    assert len(str(raised.value)) < 400
