"""
Run long core jobs off the GUI thread and report progress.

Typical workflow:

1. Wrap a core function in a :class:`Worker`. It is called with an extra ``progress(fraction, message)`` keyword.
2. Connect ``worker.signals.progress``, ``finished`` and ``failed`` to the presenter.
3. Start it with ``QThreadPool.globalInstance().start(worker)``.

Example:
    >>> from qtpy.QtCore import QThreadPool
    >>> def job(n, *, progress):
    ...     progress(1.0, "done")
    ...     return n * 2
    >>> worker = Worker(job, 21)
    >>> worker.signals.finished.connect(print)  # doctest: +SKIP
    >>> QThreadPool.globalInstance().start(worker)  # doctest: +SKIP
"""

import traceback
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QObject, QRunnable, Signal, Slot  # type: ignore[attr-defined]


class WorkerSignals(QObject):
    """
    Signals emitted by :class:`Worker`.

    ``progress`` carries a fraction in [0, 1] and a message. ``finished`` carries the job result. ``failed`` carries
    the formatted traceback.

    Attributes:
        progress: Fraction done in [0, 1] and a status message.
        finished: The job's return value.
        failed: The formatted traceback of the exception the job raised.
    """

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """
    Run ``fn(*args, progress=callback, **kwargs)`` in a ``QThreadPool``.

    The job function receives a ``progress(fraction, message)`` callable, so core code stays free of Qt.

    Args:
        fn: The job. It must accept a ``progress`` keyword argument.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Attributes:
        signals: The :class:`WorkerSignals` to connect before the worker starts.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Call the job and emit ``finished`` with its result, or ``failed`` with the traceback."""
        try:
            result = self.fn(*self.args, progress=self.signals.progress.emit, **self.kwargs)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)
