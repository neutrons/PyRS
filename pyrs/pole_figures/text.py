"""
Short text shared by the figures, the files and the GUI: durations, 3 x 3 matrices, labels and file-name parts.

A leaf module (numpy and the standard library only), so ``plots``, ``io`` and the GUI all use it without depending
on each other.

Example:
    >>> format_duration(3900)
    '1 h 05 min'
    >>> format_matrix(np.eye(3))
    '1 0 0; 0 1 0; 0 0 1'
    >>> parse_matrix("0 0 1; 1 0 0; 0 1 0")
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    >>> file_part("../2 0 0")
    '2_0_0'
"""

import math
import re

import numpy as np
import numpy.typing as npt

#: a run or hkl label written to a CSV: letters, digits, space and + _ . - only (so no cell starts with = or @)
LABEL_PATTERN = re.compile(r"[A-Za-z0-9 +_.-]+")


def format_duration(seconds: float) -> str:
    """
    A duration as short text.

    Args:
        seconds: The duration (s).

    Returns:
        Seconds below one minute, else whole minutes and hours, e.g. '45 s', '12 min' or '1 h 05 min'.

    Example:
        >>> format_duration(3900)
        '1 h 05 min'
    """
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def parse_matrix(text: str) -> tuple[float, ...] | None:
    """
    Read a 3 x 3 matrix typed as text, rows split by ';' or new lines and numbers by spaces or commas.

    Args:
        text: The matrix, e.g. '0 0 1; 1 0 0; 0 1 0'.

    Returns:
        The 9 numbers row by row, or None for empty text (no matrix given: the mount uses up_axis and q_axis).

    Raises:
        ValueError: If the text is not 3 rows of 3 numbers, or a number is not finite (nan, inf).

    Example:
        >>> parse_matrix("0 0 1; 1 0 0; 0 1 0")
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    """
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    if not rows:
        return None
    vals = [[float(x) for x in r.replace(",", " ").split()] for r in rows]
    if len(vals) != 3 or any(len(r) != 3 for r in vals):
        raise ValueError(f"the sample-to-instrument matrix needs 3 rows of 3 numbers, e.g. 0 0 1; 1 0 0; 0 1 0, "
                         f"got {text!r}")  # fmt: skip
    if not all(math.isfinite(x) for r in vals for x in r):
        raise ValueError(f"the sample-to-instrument matrix must hold finite numbers, got {text!r}")
    return tuple(float(x) for r in vals for x in r)


def format_matrix(m: npt.ArrayLike) -> str:
    """
    Write a 3 x 3 matrix as text that :func:`parse_matrix` reads back.

    Args:
        m: The matrix.

    Returns:
        Rows split by '; ', 4 significant digits, values within 5e-5 of 0 shown as 0 (so no '-0').

    Example:
        >>> format_matrix(np.eye(3))
        '1 0 0; 0 1 0; 0 0 1'
    """
    m = np.asarray(m, float) + 0.0
    return "; ".join(" ".join(f"{v:.4g}" if abs(v) > 5e-5 else "0" for v in row) for row in m)


def check_label(value: str, *, what: str) -> str:
    """
    Check a run or hkl label before it is written to a CSV.

    Only letters, digits, space and ``+ _ . -`` are allowed, so a spreadsheet never reads a cell as a formula
    (no ``=``, ``@`` or ``|``) and the label is safe in a file name.

    Args:
        value: The label, for example "1234", "1234+1235" or "222".
        what: What the label is, for the message ("run", "hkl").

    Returns:
        The label, unchanged.

    Raises:
        ValueError: If the label is empty or has another character.

    Example:
        >>> check_label("1234+1235", what="run")
        '1234+1235'
    """
    if not LABEL_PATTERN.fullmatch(value):
        raise ValueError(f"{what} label {value!r} may only hold letters, digits, spaces and + _ . -")
    return value


def file_part(text: str) -> str:
    """
    Text made safe as part of a file name: each run of characters other than letters and digits becomes "_".

    Leading and trailing underscores are dropped, so no path separator or ".." survives.

    Args:
        text: The text, for example a sample name, run or hkl.

    Returns:
        The file-name part (empty when the text holds no letter or digit).

    Example:
        >>> file_part("IN718 plate / A")
        'IN718_plate_A'
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
