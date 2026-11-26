"""
Save a figure whole or not at all, retrying while a viewer or a sync client briefly holds the file.

As the reference script's ``save``, through ``io.retry.write_atomic``: the figure goes to a temporary file in the
same folder, which then replaces the final file.

Example:
    >>> from matplotlib.figure import Figure
    >>> from pyrs.pole_figures.plots.save import save_figure
    >>> save_figure(Figure(), "empty.png", dpi=100)
"""

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from ..io.retry import RETRIES, WAIT_S, write_atomic


def save_figure(fig: Figure, path: str | Path, *, retries: int = RETRIES, wait_s: float = WAIT_S, **kw: Any) -> None:
    """
    Call ``fig.savefig`` on a temporary file next to ``path``, then move it over ``path``.

    Args:
        fig: Figure to save.
        path: Output file. The format follows the extension (the temporary file keeps it).
        retries: Attempts to replace a locked file before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).
        **kw: Passed to ``Figure.savefig`` (for example ``dpi`` or ``bbox_inches``).

    Raises:
        PermissionError: If the file is still locked after the last attempt.
        OSError: At once for any other file error.
    """
    write_atomic(path, lambda tmp: fig.savefig(tmp, **kw), retries=retries, wait_s=wait_s)
