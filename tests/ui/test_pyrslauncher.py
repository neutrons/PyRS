from pyrs.interface import pyrs_main
from pyrs.interface.pole_figures.main_window import MainWindow as PoleFigureWindow
from pyrs.interface.pyrs_main import PyRSLauncher
from qtpy import QtCore
from qtpy.QtWidgets import QMessageBox  # type: ignore
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 100


@pytest.fixture
def main_window(qtbot):
    """A fresh PyRS launcher window for each test, closed afterwards (see `tests/ui/conftest.py`)."""
    window = PyRSLauncher()
    qtbot.addWidget(window)
    return window, qtbot


def test_launcher(main_window):
    main_window, qtbot = main_window
    main_window.show()
    qtbot.wait(wait)

    assert main_window.isVisible()
    assert main_window.manual_reduction_window is None
    assert main_window.peak_fit_window is None

    # click the manual reduction button and check that the UI has opened
    qtbot.mouseClick(main_window.pushButton_manualReduction, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    assert main_window.manual_reduction_window is not None
    assert main_window.manual_reduction_window.isVisible()

    # click the peak fitting button and check that the UI has opened
    qtbot.mouseClick(main_window.pushButton_fitPeaks, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    assert main_window.peak_fit_window is not None
    assert main_window.peak_fit_window.isVisible()
    main_window.peak_fit_window.close()
    main_window.manual_reduction_window.close()
    main_window.close()


def test_pole_figure_button_opens_window_once(main_window):
    """The button opens the pole-figure window with the PyRS guide, and a second click raises the same window."""
    # Arrange
    window, qtbot = main_window
    window.show()

    # Act
    qtbot.mouseClick(window.pushButton_poleFigures, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    first = window.pole_figure_window
    qtbot.mouseClick(window.pushButton_poleFigures, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Assert
    assert isinstance(first, PoleFigureWindow) and first.isVisible() and first.isWindow()
    assert first.parent() is window and first.guide == "pyrs"
    assert window.pole_figure_window is first
    assert window.pushButton_poleFigures.text() == "Path-Length-Corrected Pole Figures"
    first.close()


def test_pole_figure_open_error_shows_message(main_window, monkeypatch):
    """An error while opening the window is shown in a message box, and PyRS keeps running."""
    # Arrange
    window, qtbot = main_window
    window.show()
    shown = []

    def broken_open(parent, *, guide):
        raise RuntimeError(f"cannot build the window for {guide}")

    monkeypatch.setattr(pyrs_main, "open_pole_figure_window", broken_open)
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: shown.append((title, text)))

    # Act
    qtbot.mouseClick(window.pushButton_poleFigures, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Assert
    assert window.pole_figure_window is None and window.isVisible()
    assert shown == [("Pole figures", "Could not open the pole-figure window:\n\ncannot build the window for pyrs")]
