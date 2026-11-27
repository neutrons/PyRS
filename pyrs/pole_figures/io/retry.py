"""
Safe file writes: a temporary file in the same folder, moved over the final name, retried while the file is locked.

On Windows a viewer, Excel or a sync client can hold a file for a moment. Replacing it then fails with a sharing
violation (``PermissionError``), so the move is retried, as the reference script's ``save`` does. Any other
``OSError`` (missing folder, disk full, read-only file system) is raised at once. A write that fails part way leaves
no partial file under the final name, and an older file there stays as it was.

A leaf module (standard library only): the figure, GIF and table writers in ``plots`` and ``io`` all use it.

Example:
    >>> write_atomic("table.csv", lambda tmp: tmp.write_text("a,b", encoding="utf-8"))
"""

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

RETRIES = 20  # attempts before a locked file's error is raised
WAIT_S = 1.5  # wait between attempts (s)


def retry_on_locked(write: Callable[[], object], *, retries: int = RETRIES, wait_s: float = WAIT_S) -> None:
    """
    Call ``write()``, retrying while it fails with ``PermissionError`` (a file held by another program on Windows).

    Args:
        write: The write to try (its return value is not used).
        retries: Number of attempts before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).

    Raises:
        PermissionError: If the file is still locked after the last attempt.
        OSError: At once for any other file error.
        ValueError: If ``retries`` is below 1.
    """
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")
    for k in range(retries):
        try:
            write()
            return
        except PermissionError:
            if k == retries - 1:
                raise
            time.sleep(wait_s)


def temp_path(path: Path) -> Path:
    """
    A new temporary file name next to ``path``, with the same extension (writers pick the format from it).

    Args:
        path: The final file.

    Returns:
        A hidden name in the same folder, for example ``.figure.3f2a9c1b.tmp.png`` for ``figure.png``.
    """
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")


def write_atomic(
    path: str | Path,
    write: Callable[[Path], object],
    *,
    retries: int = RETRIES,
    wait_s: float = WAIT_S,
) -> None:
    """
    Write a file through a temporary file in the same folder, then move it over ``path``.

    The move (``os.replace``) is retried while the final file is locked (``retry_on_locked``). If ``write`` or the
    move fails, the temporary file is removed and the error is raised. ``path`` then holds its old content, or does
    not exist.

    Args:
        path: The final file.
        write: Writes the whole file to the path it is given (its return value is not used).
        retries: Attempts of the move before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).

    Raises:
        PermissionError: If the final file is still locked after the last attempt.
        OSError: At once for any other file error, for example a missing folder.
    """
    final = Path(path)
    tmp = temp_path(final)
    try:
        write(tmp)
        retry_on_locked(lambda: os.replace(tmp, final), retries=retries, wait_s=wait_s)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
