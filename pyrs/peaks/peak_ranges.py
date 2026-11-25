"""Read peak-range JSON files, the fit windows and reference d-spacings of a peak list.

A peak-range file names each peak to fit, the 2-theta window to fit it in, and the
reference d-spacing (d0) its strain is measured against. It is written by the peak
fitting tool and read both by that tool and by autoreduction, so the rule for filling
in a missing d0 lives here, once.

Typical workflow::

    from pyrs.peaks.peak_ranges import read_peak_range_json

    for peak in read_peak_range_json("IN718_311.json"):
        result = engine.fit_peaks(peak_tag=peak.label, x_min=peak.x_min, x_max=peak.x_max)
        result.peakcollections[0].set_d_reference(values=peak.d0)

See [docs/ground_truths.md](../../docs/ground_truths.md), "Autoreduced, peak-fit files" (#1147).
"""

import json
import math
from pathlib import Path
from typing import NamedTuple, Union

# The d0 the peak fitting tool gives a newly drawn peak range. It is a placeholder, not a
# measured reference: a strain computed against it is meaningless until a real d0 is entered.
DEFAULT_D0 = 1.0


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


def _number(value: object, *, what: str, where: str) -> float:
    """`value` as a finite float, or a `ValueError` naming `what` and `where`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PeakRangeFileError(f"{where}: {what} must be a finite number, got {value!r}")
    return float(value)


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
        PeakRangeFileError: If the file cannot be read or is not JSON, is not an object of
            peaks, or a peak lacks `peak_range`, `peak_label` or `d0`, or any of them is
            malformed (window not two finite numbers with x_min < x_max, d0 not a finite
            positive number). The message names the file and the peak.
    """
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
    except (OSError, UnicodeDecodeError) as err:
        raise PeakRangeFileError(f"{filename}: cannot read the file ({err})") from err
    except json.JSONDecodeError as err:
        raise PeakRangeFileError(f"{filename}: not valid JSON ({err})") from err
    if not isinstance(data, dict):
        raise PeakRangeFileError(f"{filename}: expected an object of peaks, got {type(data).__name__}")

    peaks = []
    for position, key in enumerate(data.keys()):
        where = f"{filename}, peak '{key}'"
        entry = data[key]
        if not isinstance(entry, dict):
            raise PeakRangeFileError(f"{where}: expected an object, got {type(entry).__name__}")
        missing = [name for name in ("peak_range", "peak_label", "d0") if name not in entry]
        if missing:
            raise PeakRangeFileError(f"{where}: missing {', '.join(missing)}")

        window = entry["peak_range"]
        if not isinstance(window, list) or len(window) != 2:
            raise PeakRangeFileError(f"{where}: peak_range must be [x_min, x_max], got {window!r}")
        x_min, x_max = (_number(v, what="peak_range", where=where) for v in window)
        if not x_min < x_max:
            raise PeakRangeFileError(f"{where}: peak_range must have x_min < x_max, got {window!r}")

        d0 = entry["d0"]
        if isinstance(d0, list):
            d0 = d0[position] if len(d0) == len(data) else DEFAULT_D0
        d0 = _number(d0, what="d0", where=where)
        if d0 <= 0.0:
            raise PeakRangeFileError(f"{where}: d0 must be positive, got {d0!r}")

        peaks.append(PeakRange(label=str(entry["peak_label"]), x_min=x_min, x_max=x_max, d0=d0))
    return peaks
