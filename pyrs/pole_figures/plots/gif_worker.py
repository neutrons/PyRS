"""
Worker-process side of the parallel beam-path GIF frames.

Workflow, driven by ``beam_gif`` through ``core.parallel.run_chunks``:

1. ``init`` runs once in each worker process and builds the GIF figure (a ``beam_frame.FrameRenderer``).
2. ``render`` draws each chunk of frames sent to that process and returns the images.

The start-up arguments arrive as bytes, so a new process unpickles them in ``init``, after it has started.

Example:
    >>> payload = pickle.dumps((mesh, renderer_kw, order))  # as beam_gif builds it
    >>> init(payload)
    >>> frames = render([(0, order[0]), (1, order[1])])
"""

import pickle
from typing import Any

from ..core.parallel import lower_priority
from .beam_frame import FrameRenderer

_STATE: dict[str, Any] = {}


def init(payload: bytes) -> None:
    """
    Build this process's figure once.

    Args:
        payload: Pickled tuple (mesh, ``FrameRenderer`` keyword arguments, frame order). The order lists the
            orientation index of every frame.
    """
    lower_priority()
    mesh, kw, order = pickle.loads(payload)
    _STATE["renderer"] = FrameRenderer(mesh, **kw)
    _STATE["order"] = order


def render(chunk: list[tuple[int, int]]) -> list[tuple[int, Any]]:
    """
    Draw a chunk of frames with the figure built by ``init``.

    Args:
        chunk: Pairs (frame number k, orientation index).

    Returns:
        Pairs (k, palette GIF image), one per frame of the chunk.
    """
    r, order = _STATE["renderer"], _STATE["order"]
    return [(k, r.frame(k, i, order)) for k, i in chunk]
