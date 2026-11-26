"""
A 3 x 3 matrix as text and back, for the sample-to-instrument (mount) matrix in plots and text fields.

Example:
    >>> format_matrix(np.eye(3))
    '1 0 0; 0 1 0; 0 0 1'
    >>> parse_matrix("0 0 1; 1 0 0; 0 1 0")
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
"""

import numpy as np
import numpy.typing as npt


def parse_matrix(text: str) -> tuple[float, ...] | None:
    """
    Read a 3 x 3 matrix typed as text, rows split by ';' or new lines and numbers by spaces or commas.

    Args:
        text: The matrix, e.g. '0 0 1; 1 0 0; 0 1 0'.

    Returns:
        The 9 numbers row by row, or None for empty text (no matrix given: the mount uses up_axis and q_axis).

    Raises:
        ValueError: If the text is not 3 rows of 3 numbers.

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
