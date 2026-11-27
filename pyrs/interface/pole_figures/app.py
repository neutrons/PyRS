"""
QApplication entry point.

Use :func:`main` to run the GUI as its own application, or :func:`open_window` to open the window inside a Qt
application that is already running (the PyRS main window). Both install an error hook (``excepthook``): an
exception in a slot is logged and shown in a message box instead of ending the application. Inside a host that has
its own ``sys.excepthook`` the hook only logs and passes the error on to the host's hook.

Example:
    >>> from pyrs.interface.pole_figures.app import open_window
    >>> window = open_window(parent=pyrs_main_window, guide="pyrs")  # doctest: +SKIP
"""

import sys
from collections.abc import Sequence

from qtpy.QtWidgets import QApplication, QWidget

from .excepthook import install_excepthook
from .help import GuideVariant, check_variant
from .main_window import MainWindow


def main() -> int:
    """
    Start the GUI with the command-line arguments (``sys.argv``) and return the Qt exit code.

    Returns:
        Exit code from the Qt event loop.
    """
    return run(sys.argv)


def run(argv: Sequence[str]) -> int:
    """
    Start the GUI and return the Qt exit code.

    Args:
        argv: Command-line arguments passed to Qt. They are used only when no QApplication exists yet.

    Returns:
        Exit code from the Qt event loop.
    """
    owns = QApplication.instance() is None
    app = QApplication.instance() or QApplication(list(argv))
    install_excepthook(show_dialog=owns or sys.excepthook is sys.__excepthook__)
    app.setApplicationName("HIDRA Pole Figures")
    window = MainWindow()
    window.show()
    return int(app.exec_())


def open_window(parent: QWidget | None = None, *, guide: GuideVariant = "pyrs") -> MainWindow:
    """
    Open the pole-figure window inside a running Qt application (e.g. from the PyRS main window).

    The caller keeps the returned window referenced; Qt closes it with its parent, if one is given.

    Args:
        parent: Owner window, or None for no owner (default: None). The pole-figure window stays a separate
            top-level window.
        guide: User guide the ? button opens: "pyrs" (the guide that starts from the PyRS main window) or
            "standalone" (default: "pyrs").

    Returns:
        The window, already shown.

    Raises:
        RuntimeError: If no QApplication exists yet.
        ValueError: If the guide variant is unknown.
    """
    if QApplication.instance() is None:
        raise RuntimeError("open_window needs a running QApplication; use main() to start the GUI on its own")
    install_excepthook(show_dialog=sys.excepthook is sys.__excepthook__)  # the host's own hook stays in charge
    window = MainWindow(parent, guide=check_variant(guide))
    window.show()
    window.raise_()
    return window
