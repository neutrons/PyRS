"""Read peak-range JSON files, the fit windows and reference d-spacings of a peak list.

A peak-range file names each peak to fit, the 2-theta window to fit it in, and the
reference d-spacing (d0) its strain is measured against. It is written by the peak
fitting tool and read both by that tool and by autoreduction, so the rule for filling
in a missing d0 lives here, once.

The reader is written for callers that must not fail on a user's file, such as
autoreduction and the peak fitting window: whatever the file or the path holds, the
only exception that leaves `read_peak_range_json` is `PeakRangeFileError`.

Typical workflow::

    from pyrs.peaks.peak_ranges import read_peak_range_json

    for peak in read_peak_range_json("IN718_311.json"):
        result = engine.fit_peaks(peak_tag=peak.label, x_min=peak.x_min, x_max=peak.x_max)
        result.peakcollections[0].set_d_reference(values=peak.d0)

See [docs/ground_truths.md](../../docs/ground_truths.md), "Autoreduced, peak-fit files" (#1147).
"""

import json
import math
import os
import stat
from pathlib import Path
from typing import NamedTuple, Union

# The d0 the peak fitting tool gives a newly drawn peak range. It is a placeholder, not a
# measured reference: a strain computed against it is meaningless until a real d0 is entered.
DEFAULT_D0 = 1.0

# A real file is a few hundred bytes; anything near this is not a peak-range file.
MAX_FILE_BYTES = 1_000_000

_MAX_SHOWN = 80  # characters of a file's own text echoed in an error message


class PeakRangeFileError(ValueError):
    """A peak-range file cannot be used: unreadable, not JSON, or holding a malformed peak.

    The message names the file (and the peak). Callers that must keep going on a bad user
    file, such as autoreduction and the peak fitting window, catch this and nothing broader.
    """


class PeakRange(NamedTuple):
    """One peak of a peak-range file."""

    label: str
    x_min: float  # lower edge of the fit window, 2-theta degrees
    x_max: float  # upper edge of the fit window, 2-theta degrees
    d0: float  # reference d-spacing, Angstrom


def _shown(value: object) -> str:
    """`repr` of text taken from a file, cut short, so a message stays one safe line."""
    text = repr(value)
    return text if len(text) <= _MAX_SHOWN else text[: _MAX_SHOWN - 3] + "..."


def _number(value: object, *, what: str, where: str) -> float:
    """`value` as a finite float, or a `PeakRangeFileError` naming `what` and `where`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PeakRangeFileError(f"{where}: {what} must be a finite number, got {_shown(value)}")
    try:
        number = float(value)  # an int beyond the float range raises OverflowError
    except OverflowError:
        raise PeakRangeFileError(f"{where}: {what} must be a finite number, got an integer too large to use") from None
    if not math.isfinite(number):
        raise PeakRangeFileError(f"{where}: {what} must be a finite number, got {_shown(value)}")
    return number


def _read_json(filename: Union[str, Path]) -> object:
    """The parsed JSON of `filename`, or a `PeakRangeFileError`.

    The file is opened without blocking and must be a regular file under `MAX_FILE_BYTES`,
    so a FIFO, a device or a huge file cannot hang the caller or exhaust its memory.
    """
    if not isinstance(filename, (str, os.PathLike)):
        raise PeakRangeFileError(f"the peak-range file must be given as a path, got {_shown(filename)}")
    name = _shown(os.fspath(filename))

    try:
        descriptor = os.open(filename, os.O_RDONLY | os.O_NONBLOCK)
    except (OSError, ValueError) as err:  # ValueError: a NUL byte in the path
        raise PeakRangeFileError(f"{name}: cannot read the file ({err})") from err
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise PeakRangeFileError(f"{name}: not a regular file")
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as err:
        raise PeakRangeFileError(f"{name}: cannot read the file ({err})") from err
    if len(raw) > MAX_FILE_BYTES:
        raise PeakRangeFileError(f"{name}: larger than {MAX_FILE_BYTES} bytes, which no peak-range file is")

    try:
        text = raw.decode("utf-8-sig")  # a byte-order mark is common from Windows editors
    except UnicodeDecodeError as err:
        raise PeakRangeFileError(f"{name}: not valid UTF-8 ({err})") from err

    def no_duplicate_keys(pairs: list) -> dict:
        seen: set = set()
        for key, _ in pairs:
            if key in seen:
                raise PeakRangeFileError(f"{name}: key {_shown(key)} appears more than once")
            seen.add(key)
        return dict(pairs)

    try:
        return json.loads(text, object_pairs_hook=no_duplicate_keys)
    except PeakRangeFileError:
        raise
    except json.JSONDecodeError as err:
        raise PeakRangeFileError(f"{name}: not valid JSON ({err})") from err
    except (RecursionError, MemoryError, ValueError, TypeError) as err:
        # nesting too deep, or a number with thousands of digits
        raise PeakRangeFileError(f"{name}: cannot be parsed as JSON ({type(err).__name__})") from err


def read_peak_range_json(filename: Union[str, Path]) -> list[PeakRange]:
    """Read a peak-range JSON file.

    Files written before #925 stored the whole d0 list on every peak. A list holding
    one value per peak is resolved to that peak's own value; any other list (in practice
    the empty list) falls back to `DEFAULT_D0`, the d0 the UI assigns to a newly drawn
    peak range. Nothing else is defaulted: a d0 that is `null`, an empty string, NaN,
    not a number, or not positive is an error, because a guessed d0 gives a meaningless
    strain.

    Args:
        filename: Path of the JSON file.

    Returns:
        One `PeakRange` per peak, in file order.

    Raises:
        PeakRangeFileError: For any file or path that cannot be used: not a path, missing,
            unreadable, not a regular file, too large, not UTF-8 or JSON, a repeated key,
            no peaks, or a peak that lacks `peak_range`, `peak_label` or `d0` or holds a
            malformed one (window not two finite numbers with x_min < x_max, d0 not a
            finite positive number, label not a non-empty printable string). The message
            names the file and the peak. No other exception leaves this function.
    """
    data = _read_json(filename)
    name = _shown(os.fspath(filename))
    if not isinstance(data, dict):
        raise PeakRangeFileError(f"{name}: expected an object of peaks, got {type(data).__name__}")
    if not data:
        raise PeakRangeFileError(f"{name}: the file holds no peaks")

    peaks = []
    for position, key in enumerate(data.keys()):
        where = f"{name}, peak {_shown(key)}"
        entry = data[key]
        if not isinstance(entry, dict):
            raise PeakRangeFileError(f"{where}: expected an object, got {type(entry).__name__}")
        missing = [field for field in ("peak_range", "peak_label", "d0") if field not in entry]
        if missing:
            raise PeakRangeFileError(f"{where}: missing {', '.join(missing)}")

        label = entry["peak_label"]
        if not isinstance(label, str) or not label.strip() or not label.isprintable():
            raise PeakRangeFileError(f"{where}: peak_label must be a non-empty printable string, got {_shown(label)}")
        try:
            label.encode("utf-8")
        except UnicodeEncodeError:
            raise PeakRangeFileError(f"{where}: peak_label is not valid text, got {_shown(label)}") from None

        window = entry["peak_range"]
        if not isinstance(window, list) or len(window) != 2:
            raise PeakRangeFileError(f"{where}: peak_range must be [x_min, x_max], got {_shown(window)}")
        x_min, x_max = (_number(v, what="peak_range", where=where) for v in window)
        if not x_min < x_max:
            raise PeakRangeFileError(f"{where}: peak_range must have x_min < x_max, got {_shown(window)}")

        d0 = entry["d0"]
        if isinstance(d0, list):
            d0 = d0[position] if len(d0) == len(data) else DEFAULT_D0
        d0 = _number(d0, what="d0", where=where)
        if d0 <= 0.0:
            raise PeakRangeFileError(f"{where}: d0 must be positive, got {_shown(d0)}")

        peaks.append(PeakRange(label=label, x_min=x_min, x_max=x_max, d0=d0))
    return peaks
