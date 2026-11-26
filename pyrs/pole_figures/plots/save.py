"""
Save a figure, retrying while a viewer or a sync client briefly holds the file (as the reference script's ``save``).

Example:
    >>> from matplotlib.figure import Figure
    >>> from pyrs.pole_figures.plots.save import save_figure
    >>> save_figure(Figure(), "empty.png", dpi=100)
"""

import time
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str | Path, *, retries: int = 20, wait_s: float = 1.5, **kw: Any) -> None:
    """
    Call ``fig.savefig(path, **kw)``, retrying on OSError.

    Args:
        fig: Figure to save.
        path: Output file. The format follows the extension.
        retries: Number of attempts before the error is raised (default: 20).
        wait_s: Wait between attempts, in seconds (default: 1.5).
        **kw: Passed to ``Figure.savefig`` (for example ``dpi`` or ``bbox_inches``).

    Raises:
        OSError: If the file is still locked after the last attempt.
    """
    for k in range(retries):
        try:
            fig.savefig(path, **kw)
            return
        except OSError:
            if k == retries - 1:
                raise
            time.sleep(wait_s)
