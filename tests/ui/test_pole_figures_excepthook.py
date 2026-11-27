"""The error hook: logs, shows a box when the app is ours, and always chains to the hook that was there before."""

import sys
import threading
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QMainWindow

from pyrs.interface.pole_figures.app import open_window
from pyrs.interface.pole_figures.excepthook import ErrorHook, install_excepthook, uninstall_excepthook

pytestmark = [pytest.mark.gui, pytest.mark.integration]


class Recorder:
    """A stand-in for the host's (PyRS's) sys.excepthook."""

    def __init__(self) -> None:
        self.calls: list[type[BaseException]] = []

    def __call__(self, exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        self.calls.append(exc_type)


@pytest.fixture
def host_hook(monkeypatch: pytest.MonkeyPatch) -> Iterator[Recorder]:
    """A host hook in sys.excepthook. Both hooks are put back after the test."""
    rec = Recorder()
    monkeypatch.setattr(sys, "excepthook", rec)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)  # the host's thread hook: quiet
    yield rec
    uninstall_excepthook()


def _raise() -> tuple[type[BaseException], BaseException, TracebackType | None]:
    try:
        raise ValueError("bad value in a slot")
    except ValueError:
        return sys.exc_info()  # type: ignore[return-value]


def test_call_owned_app_shows_dialog_and_chains(host_hook: Recorder, qapp: Any) -> None:
    # Arrange
    hook = install_excepthook(show_dialog=True)
    shown: list[str] = []
    hook.dialog = shown.append

    # Act
    sys.excepthook(*_raise())

    # Assert
    assert len(shown) == 1 and "ValueError: bad value in a slot" in shown[0]
    assert host_hook.calls == [ValueError] and hook.previous is host_hook


def test_call_embedded_only_logs_and_chains(host_hook: Recorder, caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    hook = install_excepthook(show_dialog=False)
    shown: list[str] = []
    hook.dialog = shown.append

    # Act
    sys.excepthook(*_raise())

    # Assert
    assert shown == [] and host_hook.calls == [ValueError]
    assert "bad value in a slot" in caplog.text


def test_install_excepthook_twice_keeps_one_hook(host_hook: Recorder) -> None:
    # Arrange
    first = install_excepthook(show_dialog=True)

    # Act
    second = install_excepthook(show_dialog=False)

    # Assert
    assert second is first and first.previous is host_hook


def test_thread_hook_error_shown_on_gui_thread(host_hook: Recorder, qtbot: QtBot) -> None:
    # Arrange
    hook = install_excepthook(show_dialog=True)
    seen: list[Any] = []
    hook.dialog = lambda text: seen.append((threading.current_thread(), text))

    def fail() -> None:
        raise RuntimeError("bad thread")

    # Act
    t = threading.Thread(target=fail)
    t.start()
    t.join()
    qtbot.waitUntil(lambda: bool(seen), timeout=5000)

    # Assert
    assert seen[0][0] is threading.main_thread() and "RuntimeError: bad thread" in seen[0][1]


def test_open_window_host_hook_stays_in_charge(host_hook: Recorder, qtbot: QtBot) -> None:
    # Arrange
    host = QMainWindow()
    qtbot.addWidget(host)

    # Act
    window = open_window(host)

    # Assert
    hook = sys.excepthook
    assert isinstance(hook, ErrorHook) and not hook.show_dialog and hook.previous is host_hook
    window.close()
