from pyrs.interface.pyrs_main import PyRSLauncher
from qtpy import QtCore

wait = 100


def test_launcher(qtbot):
    main_window = PyRSLauncher()
    qtbot.addWidget(main_window)
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
