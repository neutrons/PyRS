from pyrs.interface.peak_fitting import fitpeakswindow
from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import json
import pytest
import os
from tests.conftest import ON_GITHUB_ACTIONS  # set to True when running on build servers

wait = 100


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason='Test hangs on github CI')
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

    # Select peak range, drag select

    # First get canvas
    canvas = window.ui.frame_PeakView.children()[1]._myCanvas
    # The get start and end mouse points to drag select
    start_x, start_y = canvas.figure.axes[0].transData.transform((78, 500))
    end_x, end_y = canvas.figure.axes[0].transData.transform((85, 500))
    # Drag select with mouse control
    qtbot.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(start_x, start_y))
    qtbot.wait(wait)
    qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(end_x, end_y))
    qtbot.wait(wait)

    # Peak Ranges Save ...
    peak_range_filename = tmpdir.join("peak_range.json")
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, str(peak_range_filename)))
    qtbot.mouseClick(window.ui.pushButton_save_peak_range, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # check peak range data in json file
    with open(peak_range_filename, 'r') as json_file:
        data = json.load(json_file)
    assert len(data) == 1
    assert '0' in data
    p0 = data['0']
    assert p0['peak_label'] == 'Peak0'
    assert len(p0['d0']) == 0
    start, end = p0['peak_range']
    assert start == pytest.approx(78, abs=0.1)
    assert end == pytest.approx(85, abs=0.1)

    # Fit Peak(s)
    qtbot.mouseClick(window.ui.pushButton_FitPeaks, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Export CSV ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, str(tmpdir)))
    qtbot.mouseClick(window.ui.pushButton_exportCSV, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # check output CSV
    assert os.path.isfile(tmpdir.join("HB2B_1423.csv"))
    file_contents = open(tmpdir.join("HB2B_1423.csv")).readlines()
    # check number of lines
    assert len(file_contents) == 127
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=float, sep=',')) == 33

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
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, str(tmpdir)))
    qtbot.mouseClick(window.ui.pushButton_exportCSV, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # check output CSV
    assert os.path.isfile(tmpdir.join("HB2B_1423.csv"))
    file_contents = open(tmpdir.join("HB2B_1423.csv")).readlines()
    # check number of lines
    assert len(file_contents) == 127
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=float, sep=',')) == 33

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
    assert line.get_ydata().min() == pytest.approx(1.169389, rel=1e-5)
    assert line.get_ydata().max() == pytest.approx(1.170393, rel=1e-5)
