"""
Built figures kept for the plot panels, so going back to a plot does not rebuild it.

A presenter asks :meth:`FigureCache.get` for a key made of everything the figure depends on. The cache returns the
stored figure, or calls the builder and keeps the result. The least recently used figure is dropped past the size
limit.

Example:
    >>> from matplotlib.figure import Figure
    >>> cache = FigureCache(size=2)
    >>> fig = cache.get(("raw", "111"), Figure)
    >>> cache.get(("raw", "111"), Figure) is fig
    True
"""

from collections import OrderedDict
from collections.abc import Callable, Hashable

from matplotlib.figure import Figure

MAX_FIGURES = 12


class FigureCache:
    """
    Most recently used figures by key (plot type, plane, and every input the figure depends on).

    Args:
        size: Most figures kept (default: MAX_FIGURES, 12).

    Attributes:
        size: Most figures kept. The least recently used one is dropped first.
    """

    def __init__(self, size: int = MAX_FIGURES) -> None:
        self.size = size
        self._figs: OrderedDict[Hashable, Figure] = OrderedDict()

    def get(self, key: Hashable, build: Callable[[], Figure | None]) -> Figure | None:
        """
        The cached figure for key, or build, keep and return it (None is not kept).

        Args:
            key: Hashable description of the figure and every input it depends on.
            build: Called with no arguments when the key is not cached. It may return None (nothing to show).

        Returns:
            The cached or newly built figure, or None when ``build`` returned None.
        """
        fig = self.cached(key)
        if fig is not None:
            return fig
        fig = build()
        if fig is not None:
            self.put(key, fig)
        return fig

    def cached(self, key: Hashable) -> Figure | None:
        """
        The cached figure for key (marked as most recently used), or None when it is not cached.

        Args:
            key: Hashable description of the figure.

        Returns:
            The figure, or None.
        """
        if key not in self._figs:
            return None
        self._figs.move_to_end(key)
        return self._figs[key]

    def put(self, key: Hashable, fig: Figure) -> None:
        """
        Keep a figure built elsewhere (for example in a background job) under key.

        Args:
            key: Hashable description of the figure.
            fig: The figure.
        """
        self._figs[key] = fig
        self._figs.move_to_end(key)
        while len(self._figs) > self.size:
            self._figs.popitem(last=False)

    def clear(self) -> None:
        """Drop every figure kept."""
        self._figs.clear()

    def __len__(self) -> int:
        """Number of figures kept."""
        return len(self._figs)
