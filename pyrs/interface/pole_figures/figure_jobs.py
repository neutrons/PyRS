"""
Build plot-panel figures off the GUI thread.

A presenter asks :meth:`BackgroundFigures.show` for a figure by key. A cached figure is shown at once. Otherwise the
panel shows "Drawing ..." while a :class:`~.worker.Worker` builds the figure on a one-thread pool, so the window
stays responsive. The figure is kept in the :class:`~.figure_cache.FigureCache` and shown when it is still the newest
request. A newer request, or :meth:`BackgroundFigures.cancel`, supersedes an older one, so a stale figure never
replaces a newer choice.

The builder must make its figure with the object-oriented API (``matplotlib.figure.Figure``, never pyplot) and must
use only inputs captured on the GUI thread. Drawing to the Qt canvas stays on the GUI thread (``PlotPanel``).

Example:
    >>> figures = BackgroundFigures(panel, FigureCache())  # doctest: +SKIP
    >>> figures.show(("summary", 3.0), lambda: predicted_figure(pred, cutoff=3.0), name="the summary")  # doctest: +SKIP
"""

from collections.abc import Callable, Hashable

from matplotlib.figure import Figure
from qtpy.QtCore import QObject, QThreadPool, Signal  # type: ignore[attr-defined]

from .figure_cache import FigureCache
from .widgets.plot_panel import PlotPanel
from .worker import Worker


def _ignore() -> None:
    """Default for ``on_shown``: nothing to do."""


class BackgroundFigures(QObject):
    """
    Figures for one plot panel, built in the background, newest request wins.

    Args:
        panel: The plot panel that shows the figures.
        cache: Where built figures are kept.
        parent: Owner QObject, or None (default: None).

    Attributes:
        shown: Signal with the key of a figure just shown in the panel.
        panel: The plot panel.
        cache: The figure cache.
    """

    shown = Signal(object)

    def __init__(self, panel: PlotPanel, cache: FigureCache, *, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.panel, self.cache = panel, cache
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)  # one build at a time: matplotlib is not thread-safe across figures
        self._token = 0  # the newest request
        self._generation = 0  # bumped by clear(): results of older inputs are not cached
        self._running: dict[int, Worker] = {}

    def show(self, key: Hashable, build: Callable[[], Figure | None], *, name: str,
             on_shown: Callable[[], None] = _ignore) -> None:  # fmt: skip
        """
        Show the figure for key: at once when cached, else after a background build.

        Args:
            key: Hashable description of the figure and every input it depends on.
            build: Makes the figure (or None for nothing to show). Runs on a worker thread.
            name: What is drawn, for the "Drawing ..." note, for example "the pole figures".
            on_shown: Called on the GUI thread after the figure is shown (default: nothing).
        """
        self._token += 1
        fig = self.cache.cached(key)
        if fig is not None:
            self.panel.show_figure(fig)
            on_shown()
            self.shown.emit(key)
            return
        token, generation = self._token, self._generation
        self.panel.show_message(f"Drawing {name} ...")

        def job(progress: Callable[[float, str], None]) -> Figure | None:
            del progress  # a figure build reports no progress
            if token != self._token:
                return None  # superseded while it waited: skip the work
            return build()

        worker = Worker(job)
        worker.signals.finished.connect(
            lambda f: self._on_built(key, token=token, generation=generation, fig=f, on_shown=on_shown)
        )
        worker.signals.failed.connect(lambda tb: self._on_failed(token, tb))
        self._running[token] = worker
        self._pool.start(worker)

    def _on_built(self, key: Hashable, *, token: int, generation: int, fig: Figure | None,
                  on_shown: Callable[[], None]) -> None:  # fmt: skip
        self._running.pop(token, None)
        if fig is not None and generation == self._generation:
            self.cache.put(key, fig)
        if token != self._token:
            return  # a newer request owns the panel
        if fig is None:
            self.panel.show_message("Nothing to show.")
            return
        self.panel.show_figure(fig)
        on_shown()
        self.shown.emit(key)

    def _on_failed(self, token: int, tb: str) -> None:
        self._running.pop(token, None)
        if token == self._token:
            self.panel.show_message(f"Could not draw the plot: {tb.strip().splitlines()[-1]}")

    def cancel(self) -> None:
        """Supersede the pending request: its figure is cached but not shown (the panel shows something else)."""
        self._token += 1

    def clear(self) -> None:
        """Drop every cached figure. Figures still being built from the old inputs are not cached or shown."""
        self._generation += 1
        self._token += 1
        self.cache.clear()

    @property
    def busy(self) -> bool:
        """True while a figure is being built or waits to be built."""
        return bool(self._running)

    def wait(self, msecs: int = 30_000) -> bool:
        """
        Block until the pool is idle (for scripts and tests). The results arrive with the next event processing.

        Args:
            msecs: Most time to wait (ms, default: 30000).

        Returns:
            True when the pool finished in time.
        """
        return bool(self._pool.waitForDone(msecs))
