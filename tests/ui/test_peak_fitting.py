from pyrs.interface.peak_fitting import fitpeakswindow
from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import pytest
import os
from tests.conftest import ON_GITHUB_ACTIONS  # set to True when running on build servers

wait = 300


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason="UI tests segfault on GitHub Actions")
def test_peak_fitting(qtbot, tmpdir):
    fit_peak_core = pyrscore.PyRsCore()
    window = fitpeakswindow.FitPeaksWindow(None, fit_peak_core=fit_peak_core)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # This is the handle modal dialogs
    def handle_dialog(filename):
        # get a reference to the dialog and handle it here
        dialog = window.findChild(QtWidgets.QFileDialog)
        # get a File Name field
        lineEdit = dialog.findChild(QtWidgets.QLineEdit)
        # Type in file to load and press enter
        qtbot.keyClicks(lineEdit, filename)
        qtbot.wait(wait)
        qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)

    # Browser Exp. Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "tests/data/HB2B_1423.h5"))
    qtbot.mouseClick(window.ui.pushButton_browseHDF, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # First get canvas
    canvas = window.ui.frame_PeakView.children()[1]._myCanvas
    # The get start and end mouse points to drag select
    start_x, start_y = canvas.figure.axes[0].transData.transform((78, 500))
    end_x, end_y = canvas.figure.axes[0].transData.transform((85, 500))

    # Drag select with mouse control
    qtbot.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(start_x), int(start_y)))
    qtbot.wait(wait)
    qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(end_x), int(end_y)))
    qtbot.wait(wait)

    # Peak Ranges Load ...
    peak_range_filename = "tests/data/peak_range.json"
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, peak_range_filename))
    qtbot.mouseClick(window.ui.pushButton_load_peak_range, QtCore.Qt.LeftButton)
    # qtbot.mouseClick(window.ui.pushButton_save_peak_range, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Fit Peak(s)
    qtbot.mouseClick(window.ui.pushButton_FitPeaks, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Select peak range, drag select
    qtbot.mouseClick(window.ui.radioButton_listSubRuns, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, str("0"))
    qtbot.wait(wait)
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, QtCore.Qt.Key_Enter)

    # Export CSV ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, ""))
    qtbot.mouseClick(window.ui.pushButton_exportCSV, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # check output CSV
    assert os.path.isfile("HB2B_1423.csv")
    file_contents = open("HB2B_1423.csv").readlines()
    # check number of lines
    assert len(file_contents) == 127
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=np.float64, sep=',')) == 33

    # look at 1D results plot
    line = window.ui.graphicsView_fitResult.canvas().get_axis(0, 0, True).lines[0]
    assert line.get_xdata().min() == 1
    assert line.get_xdata().max() == 87
    assert line.get_ydata().min() == pytest.approx(-42.000572)
    assert line.get_ydata().max() == pytest.approx(37.999313)

    # change plot to sx vs d-spacing and check data
    # Scroll ComboBox down to d-spacing
    for _ in range(15):
        qtbot.keyClick(window.ui.comboBox_yaxisNames, QtCore.Qt.Key_Down)
        qtbot.wait(wait)
    # Scroll to sx
    qtbot.keyClick(window.ui.comboBox_xaxisNames, QtCore.Qt.Key_Down)
    qtbot.wait(wait)

    line = window.ui.graphicsView_fitResult.canvas().get_axis(0, 0, True).lines[0]
    assert line.get_xdata().min() == pytest.approx(-42.000572)
    assert line.get_xdata().max() == pytest.approx(37.999313)
    assert line.get_ydata().min() == pytest.approx(1.169515, rel=1e-5)
    assert line.get_ydata().max() == pytest.approx(1.170074, rel=1e-5)

    # Change to Peak Type to Gaussian
    qtbot.keyClick(window.ui.comboBox_peakType, QtCore.Qt.Key_Down)
    qtbot.wait(wait)
    # Fit Peak(s)
    qtbot.mouseClick(window.ui.pushButton_FitPeaks, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Export CSV ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, ""))
    qtbot.mouseClick(window.ui.pushButton_exportCSV, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # check output CSV
    assert os.path.isfile("HB2B_1423.csv")
    file_contents = open("HB2B_1423.csv").readlines()
    # check number of lines
    assert len(file_contents) == 127
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=np.float64, sep=',')) == 33

    # look at 1D results plot
    line = window.ui.graphicsView_fitResult.canvas().get_axis(0, 0, True).lines[0]
    assert line.get_xdata().min() == 1
    assert line.get_xdata().max() == 87
    assert line.get_ydata().min() == pytest.approx(-42.000572)
    assert line.get_ydata().max() == pytest.approx(37.999313)

    # change plot to sx vs d-spacing and check data
    # Scroll ComboBox down to d-spacing
    for _ in range(15):
        qtbot.keyClick(window.ui.comboBox_yaxisNames, QtCore.Qt.Key_Down)
        qtbot.wait(wait)
    # Scroll to sx
    qtbot.keyClick(window.ui.comboBox_xaxisNames, QtCore.Qt.Key_Down)
    qtbot.wait(wait)

    line = window.ui.graphicsView_fitResult.canvas().get_axis(0, 0, True).lines[0]
    assert line.get_xdata().min() == pytest.approx(-42.000572)
    assert line.get_xdata().max() == pytest.approx(37.999313)
    assert line.get_ydata().min() == pytest.approx(1.169389, rel=2e-2)
    assert line.get_ydata().max() == pytest.approx(1.170393, rel=2e-2)
    os.remove("HB2B_1423.csv")

    for _ in range(15):
        qtbot.keyClick(window.ui.comboBox_zaxisNames_2dplot, QtCore.Qt.Key_Down)
        qtbot.wait(wait)

    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, str("5"))
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, str(":"))
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, str("5"))
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, str("0"))
    qtbot.wait(wait)
    qtbot.keyClick(window.ui.lineEdit_subruns_2dplot, QtCore.Qt.Key_Enter)

    qtbot.mouseClick(window.ui.radioButton_contour, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.mouseClick(window.ui.radioButton_3dline, QtCore.Qt.LeftButton)
    qtbot.wait(wait)


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason="UI tests segfault on GitHub Actions")
def test_peak_selection(qtbot, tmpdir):
    fit_peak_core = pyrscore.PyRsCore()
    window = fitpeakswindow.FitPeaksWindow(None, fit_peak_core=fit_peak_core)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # This is the handle modal dialogs
    def handle_dialog(filename):
        # get a reference to the dialog and handle it here
        dialog = window.findChild(QtWidgets.QFileDialog)
        # get a File Name field
        lineEdit = dialog.findChild(QtWidgets.QLineEdit)
        # Type in file to load and press enter
        qtbot.keyClicks(lineEdit, filename)
        qtbot.wait(wait)
        qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)

    # Browser Exp. Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "tests/data/HB2B_1423.h5"))
    qtbot.mouseClick(window.ui.pushButton_browseHDF, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # First get canvas
    canvas = window.ui.frame_PeakView.children()[1]._myCanvas
    # The get start and end mouse points to drag select
    start_x1, start_y1 = canvas.figure.axes[0].transData.transform((78, 500))
    end_x1, end_y1 = canvas.figure.axes[0].transData.transform((81.5, 500))
    start_x2, start_y2 = canvas.figure.axes[0].transData.transform((82, 500))
    end_x2, end_y2 = canvas.figure.axes[0].transData.transform((85, 500))

    # Drag select with mouse control
    qtbot.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(start_x1), int(start_y1)))
    qtbot.wait(wait)
    qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(end_x1), int(end_y1)))
    qtbot.wait(wait)

    # Drag select with mouse control
    qtbot.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(start_x2), int(start_y2)))
    qtbot.wait(wait)
    qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(end_x2), int(end_y2)))
    qtbot.wait(wait)
