"""
Worker-process side of the parallel beam-path GIF frames.

Kept free of heavy imports at module level: on Windows a new process must unpickle its start-up arguments before the
parent starts the next one, so matplotlib is imported inside ``init`` (after start-up) and the arguments arrive as
bytes. The processes then start together instead of one after another. Workflow, driven by
``beam_gif.beam_path_gif`` through ``core.parallel.run_chunks``:

1. ``init`` runs once in each worker process and builds the GIF figure (a ``beam_gif.FrameRenderer``).
2. ``render`` draws each chunk of frames sent to that process and returns the images.

Example:
    >>> payload = pickle.dumps((mesh, renderer_kw, order))  # as beam_path_gif builds it
    >>> init(payload)
    >>> frames = render([(0, order[0]), (1, order[1])])
"""

import pickle
from typing import Any

_STATE: dict[str, Any] = {}


def init(payload: bytes) -> None:
    """
    Build this process's figure once.

    Args:
        payload: Pickled tuple (mesh, ``FrameRenderer`` keyword arguments, frame order). The order lists the
            orientation index of every frame.
    """
    # deferred: beam_gif imports this module (a real cycle), and both pull in matplotlib and numpy, an expensive
    # load that would make each spawned worker unpickle slowly and so start one after another (module docstring)
    from ..core.parallel import lower_priority
    from .beam_gif import FrameRenderer

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
