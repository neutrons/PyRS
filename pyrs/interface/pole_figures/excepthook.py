"""
Show unexpected errors in a message box instead of letting PyQt end the application.

PyQt5 (5.5 and later) and PyQt6 call ``sys.excepthook`` for an exception raised in a slot. With Python's default
hook PyQt then aborts the application. :func:`install_excepthook` installs a hook that logs the error, shows it in a
QMessageBox, and calls the hook that was installed before (it never replaces it silently). The same is done for
``threading.excepthook`` (errors in Python threads). The box is shown on the GUI thread.

When the window runs inside another application (the PyRS main window), that application may own the hook already.
Then the hook only logs and passes the error on to the previous hook, which shows it as the host prefers.

Example:
    >>> install_excepthook(show_dialog=True)  # doctest: +SKIP
"""

import logging
import sys
import threading
import traceback
from collections.abc import Callable
from types import TracebackType
from typing import Any

from qtpy.QtCore import QObject, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtWidgets import QApplication, QMessageBox

LOG = logging.getLogger("pyrs.pole_figures")
TITLE = "Unexpected error"

ExceptHook = Callable[[type[BaseException], BaseException, TracebackType | None], Any]


class _Relay(QObject):
    """Carries an error text to the GUI thread, where the message box is shown."""

    error = Signal(str)

    def __init__(self, dialog: Callable[[str], None]) -> None:
        super().__init__()
        self.dialog = dialog
        self.error.connect(self._show)

    @Slot(str)
    def _show(self, text: str) -> None:
        self.dialog(text)


def show_error(text: str) -> None:
    """
    Show an error text in a message box (GUI thread only). Does nothing without a QApplication.

    Args:
        text: The formatted traceback.
    """
    if QApplication.instance() is None:
        return
    last = text.strip().splitlines()[-1] if text.strip() else "unknown error"
    box = QMessageBox(QMessageBox.Icon.Critical, TITLE, f"{last}\n\nThe window keeps running. The details are in "
                      "the log.", QMessageBox.StandardButton.Ok)  # fmt: skip
    box.setDetailedText(text)
    box.exec()


class ErrorHook:
    """
    Replacement for ``sys.excepthook`` and ``threading.excepthook`` that logs, can show a box, and chains.

    Args:
        show_dialog: Show each error in a message box.
        previous: The ``sys.excepthook`` to call after logging.
        previous_thread: The ``threading.excepthook`` to call after logging.

    Attributes:
        show_dialog: Whether errors are shown in a message box.
        dialog: Shows an error text on the GUI thread (default: ``show_error``).
        previous: The hook called after this one (sys).
        previous_thread: The hook called after this one (threading).
    """

    def __init__(self, *, show_dialog: bool, previous: ExceptHook,
                 previous_thread: Callable[[Any], Any]) -> None:  # fmt: skip
        self.show_dialog, self.previous, self.previous_thread = show_dialog, previous, previous_thread
        self.dialog: Callable[[str], None] = show_error
        self._relay: _Relay | None = None

    def _report(self, text: str) -> None:
        LOG.error("Unexpected error:\n%s", text)
        app = QApplication.instance()
        if not self.show_dialog or app is None:
            return
        if threading.current_thread() is threading.main_thread():  # the GUI thread
            self.dialog(text)
            return
        if self._relay is None:
            self._relay = _Relay(lambda t: self.dialog(t))
            self._relay.moveToThread(app.thread())
        self._relay.error.emit(text)  # queued to the GUI thread

    def __call__(self, exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        """Log and report an exception, then pass it to the previous ``sys.excepthook``."""
        if not issubclass(exc_type, KeyboardInterrupt):
            self._report("".join(traceback.format_exception(exc_type, exc, tb)))
        self.previous(exc_type, exc, tb)  # the host's hook, or Python's, which prints the traceback

    def thread_hook(self, args: Any) -> None:
        """``threading.excepthook``: log and report an exception of a Python thread, then pass it on."""
        if args.exc_type is SystemExit:
            return
        self._report("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        self.previous_thread(args)


def install_excepthook(*, show_dialog: bool) -> ErrorHook:
    """
    Install the error hook for ``sys.excepthook`` and ``threading.excepthook``, chained to the hooks already there.

    Installing again returns the hook already installed (nothing is chained twice).

    Args:
        show_dialog: Show errors in a message box. Use True when this package owns the QApplication, or when no
            other hook is installed. With False the hook only logs and passes errors on.

    Returns:
        The installed hook.
    """
    if isinstance(sys.excepthook, ErrorHook):
        return sys.excepthook
    hook = ErrorHook(show_dialog=show_dialog, previous=sys.excepthook, previous_thread=threading.excepthook)
    sys.excepthook = hook
    threading.excepthook = hook.thread_hook
    return hook


def uninstall_excepthook() -> None:
    """Put back the hooks that were there before :func:`install_excepthook` (for tests and embedding hosts)."""
    hook = sys.excepthook
    if isinstance(hook, ErrorHook):
        sys.excepthook = hook.previous
        if getattr(threading.excepthook, "__self__", None) is hook:
            threading.excepthook = hook.previous_thread
