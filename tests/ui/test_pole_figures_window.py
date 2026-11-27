"""
The main window and its shared parts.

Workflow tabs, opening from a host application, the user-guide button, the wheel guard, the background worker and
the plot panel's toolbar.
"""

import sys
import threading
from typing import Any

import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt, QThreadPool  # type: ignore[attr-defined]
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QMainWindow, QSpinBox

from pyrs.interface.pole_figures import main_window
from pyrs.interface.pole_figures.app import open_window
from pyrs.interface.pole_figures.help import GuideResult
from pyrs.interface.pole_figures.main_window import TAB_NAMES, MainWindow
from pyrs.interface.pole_figures.widgets.plot_panel import PlotPanel
from pyrs.interface.pole_figures.worker import Worker

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _job(x: int, progress: Any) -> int:
    progress(0.5, "half")
    return x * 2


def _bad_job(progress: Any) -> None:
    raise ValueError("boom")


def _wheel(widget: Any, steps: int = 1) -> bool:
    """Send a wheel turn (up, ``steps`` notches) to ``widget``; return whether it was accepted."""
    center = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        center,
        QPointF(widget.mapToGlobal(center.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)
    return bool(event.isAccepted())


def test_init_workflow_tabs_in_order(window: MainWindow) -> None:
    # Arrange
    tabs = window.tabs

    # Act
    names = [tabs.tabText(i) for i in range(tabs.count())]

    # Assert: the five workflow tabs, then the ODF tab when this copy has it
    assert names[: len(TAB_NAMES)] == list(TAB_NAMES)
    assert names[len(TAB_NAMES) :] == (["MTEX"] if main_window.odf_tab_installed() else [])
    assert "Figures" not in names and "Analysis" not in names


def test_worker_job_result_emitted(qtbot: QtBot) -> None:
    # Arrange
    worker = Worker(_job, 21)

    # Act
    with qtbot.waitSignal(worker.signals.finished, timeout=5000) as blocker:
        QThreadPool.globalInstance().start(worker)

    # Assert
    assert blocker.args == [42]


def test_worker_job_error_emits_traceback(qtbot: QtBot) -> None:
    # Arrange
    worker = Worker(_bad_job)

    # Act
    with qtbot.waitSignal(worker.signals.failed, timeout=5000) as blocker:
        QThreadPool.globalInstance().start(worker)

    # Assert
    assert "ValueError: boom" in blocker.args[0]


def test_show_figure_cached_again_no_stale_toolbar(qtbot: QtBot) -> None:
    """A cached figure shown again must not keep the deleted toolbar's mouse-move callback."""
    # Arrange
    panel = PlotPanel()
    qtbot.addWidget(panel)
    first, second = Figure(), Figure()
    first.add_subplot().plot([0, 1])
    second.add_subplot()
    errors: list[BaseException] = []

    # Act
    panel.show_figure(first)
    panel.show_figure(second)
    panel.show_figure(first)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)  # delete the old toolbars now
    assert panel.canvas is not None
    first._canvas_callbacks.exception_handler = errors.append  # type: ignore[attr-defined]
    MouseEvent("motion_notify_event", panel.canvas, 10, 10)._process()  # type: ignore[attr-defined]

    # Assert
    assert errors == []


def test_open_window_host_owns_window(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host app (PyRS) opens the window as a separate top-level window owned by its own."""
    # Arrange
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # open_window chains an error hook: put it back after
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    host = QMainWindow()
    qtbot.addWidget(host)

    # Act
    window = open_window(host)

    # Assert
    assert isinstance(window, MainWindow) and window.guide == "pyrs"  # opened from a host: the PyRS guide
    assert window.parent() is host
    assert window.isVisible() and window.isWindow()
    window.close()


def test_open_window_unknown_guide_raises(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    host = QMainWindow()
    qtbot.addWidget(host)

    # Act / Assert
    with pytest.raises(ValueError, match="'web'"):
        open_window(host, guide="web")  # type: ignore[arg-type]


def test_show_help_opens_current_tab_section(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    asked: list[str] = []

    def fake_open(section: str, *, variant: str) -> GuideResult:
        asked.append(f"{variant}:{section}")
        return GuideResult.OPENED

    monkeypatch.setattr(main_window, "open_guide", fake_open)

    # Act
    window.tabs.setCurrentWidget(window.data_tab)
    window.help_button.click()
    window.tabs.setCurrentWidget(window.sample_tab)
    window.show_help()
    window.show_guide("start")

    # Assert
    assert window.tabs.cornerWidget() is window.help_button and window.help_button.text() == "?"
    assert asked == ["standalone:data", "standalone:sample", "standalone:start"]


def test_build_menus_f1_opens_step_guide(window: MainWindow) -> None:
    # Arrange
    menus = [m.menu() for m in window.menuBar().actions() if m.menu()]

    # Act
    actions = [a for menu in menus for a in menu.actions()]

    # Assert
    assert [a.text() for a in actions if a.shortcut().toString() == "F1"] == ["User guide for this &step"]


def test_wheel_guard_unfocused_inputs_ignore_wheel(window: MainWindow, qtbot: QtBot) -> None:
    """Scrolling over an unfocused spin box or combo box leaves it alone (the event goes to the panel)."""
    # Arrange
    window.show()
    qtbot.waitExposed(window)
    spin = next(s for s in window.sample_tab.findChildren(QAbstractSpinBox) if s.isEnabled())
    combo = next(c for c in window.findChildren(QComboBox) if c.isEnabled() and c.count() > 1)
    window.setFocus()
    before, index = spin.value(), combo.currentIndex()

    # Act
    accepted = (_wheel(spin), _wheel(combo))

    # Assert
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert combo.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert accepted == (False, False)  # ignored, so Qt passes it on to the scroll area
    assert spin.value() == before and combo.currentIndex() == index

    # Act: after a click (focus) the wheel steps the value as before
    window.activateWindow()
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus, timeout=2000)
    _wheel(spin)

    # Assert
    assert spin.value() != before


def test_wheel_guard_host_widget_untouched(window: MainWindow, qtbot: QtBot) -> None:
    """A host application's own widget (for example PyRS) keeps the default wheel behavior."""
    # Arrange
    other = QSpinBox()
    qtbot.addWidget(other)
    other.show()

    # Act
    _wheel(other)

    # Assert
    assert other.value() == 1
