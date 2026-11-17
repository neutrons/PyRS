from pyrs.interface.pyrs_main import PyRSLauncher
from qtpy import QtCore
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


def test_pole_figure_button_disabled_without_package(qtbot, monkeypatch):
    """Without the optional package the button is grayed out and says why."""
    # Arrange
    import pyrs.interface.pyrs_main as pyrs_main

    monkeypatch.setattr(pyrs_main, "pole_figures_available", lambda: False)

    # Act
    window = PyRSLauncher()
    qtbot.addWidget(window)

    # Assert
    assert not window.pushButton_poleFigures.isEnabled()
    assert "hidra-pathlength-polefigs" in window.pushButton_poleFigures.toolTip()


def test_pole_figure_button_opens_window(main_window):
    """The button opens the pole-figure window once and raises it on a second click."""
    pytest.importorskip("hidra_pathlength_polefigs")
    # Arrange
    main_window, qtbot = main_window
    main_window.show()
    assert main_window.pole_figure_window is None

    # Act
    qtbot.mouseClick(main_window.pushButton_poleFigures, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    first = main_window.pole_figure_window
    qtbot.mouseClick(main_window.pushButton_poleFigures, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Assert
    assert first is not None and first.isVisible()
    assert main_window.pole_figure_window is first
    first.close()
    main_window.close()
