"""
Array and progress-callback types shared by the core modules, and the no-op callbacks.

A long job reports progress through a plain callable, so the core stays free of Qt: ``Progress`` takes the fraction
done and a message, ``CountProgress`` takes the number of items done. The no-op callbacks are the defaults where no
progress is wanted.

Typical workflow:

1. A job takes ``progress: Progress = no_progress``.
2. Each of its parts counts items and reports them through :func:`part_progress`.

Example:
    >>> seen = []
    >>> tick = part_progress(lambda f, m: seen.append((f, m)), part=1, parts=2, n=20, label="{311}", unit="points")
    >>> tick(10)
    >>> seen
    [(0.75, '{311}: 10 of 20 points')]
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
#: ``progress(fraction, message)``: fraction of the whole job done (0 to 1) and what is being done
Progress = Callable[[float, str], None]
#: ``progress(n_done)``: number of items finished so far
CountProgress = Callable[[int], None]

PROGRESS_EVERY = 10  # report every this many items (and the last), so a fast loop does not flood the GUI


def no_progress(fraction: float, message: str) -> None:
    """
    Progress callback that ignores every report (the default where no progress is wanted).

    Args:
        fraction: Fraction of the job done, 0 to 1.
        message: What is being done.
    """


def no_count_progress(done: int) -> None:
    """
    Progress callback that ignores the count (the default where no progress is wanted).

    Args:
        done: Number of items finished so far.
    """


def part_progress(progress: Progress, *, part: int, parts: int, n: int, label: str, unit: str) -> CountProgress:
    """
    Count progress of one of ``parts`` equal parts of a job, passed on as the fraction of the whole job.

    Args:
        progress: ``progress(fraction, message)`` of the whole job.
        part: Index of this part, 0 to parts - 1.
        parts: Number of parts in the job.
        n: Number of items in this part.
        label: Start of the message, e.g. "{311}".
        unit: Name of the items, e.g. "points".

    Returns:
        ``tick(done)``: reports every ``PROGRESS_EVERY`` items and the last one as
        ``progress((part * n + done) / (parts * n), "<label>: <done> of <n> <unit>")``.
    """

    def tick(done: int) -> None:
        if done % PROGRESS_EVERY == 0 or done == n:
            progress((part * n + done) / (parts * n), f"{label}: {done} of {n} {unit}")

    return tick
