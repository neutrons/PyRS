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
