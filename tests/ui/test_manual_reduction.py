from qtpy import QtCore
import os
import pytest
import matplotlib
matplotlib.use("Agg")
from pyrs.interface.manual_reduction import manualreductionwindow  # noqa E402


wait = 100

@pytest.fixture(scope="session")
def manual_reduction_window(my_qtbot):
    r"""
    Fixture for the detector calibration window. Creating the window with a session scope and reusing it for all tests.
    This is done to avoid the segmentation fault error that occurs when the window is created with a function scope.
    """
    window = manualreductionwindow.ManualReductionWindow(None)
    return window, my_qtbot
    

def test_manual_reduction(tmpdir, manual_reduction_window):
    window, qtbot = manual_reduction_window
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # set data
    qtbot.keyClicks(window.ui.lineEdit_runNumber, "tests/data/HB2B_938.nxs.h5")
    qtbot.wait(wait)

    # set mask
    qtbot.mouseClick(window.ui.checkBox_defaultMaskFile, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    for _ in range(100):
        qtbot.keyClick(window.ui.lineEdit_maskFile, QtCore.Qt.Key_Backspace)
    qtbot.wait(wait)
    qtbot.keyClicks(window.ui.lineEdit_maskFile, "tests/data/HB2B_Mask_12-18-19.xml")
    qtbot.wait(wait)

    # set calibration
    qtbot.mouseClick(window.ui.checkBox_defaultCalibrationFile, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    for _ in range(100):
        qtbot.keyClick(window.ui.lineEdit_calibrationFile, QtCore.Qt.Key_Backspace)
    qtbot.wait(wait)
    qtbot.keyClicks(window.ui.lineEdit_calibrationFile, "tests/data/HB2B_CAL_Si333.json")
    qtbot.wait(wait)

    # set vanadium
    qtbot.keyClicks(window.ui.lineEdit_vanRunNumber, "tests/data/HB2B_1118.nxs.h5")
    qtbot.wait(wait)

    # set output directory
    qtbot.keyClicks(window.ui.lineEdit_outputDirectory, str(tmpdir))
    qtbot.wait(wait)

    # push button to run manual reduction
    qtbot.mouseClick(window.ui.pushButton_splitConvertSaveProject, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    assert window.ui.progressBar.value() == 100

    # check that the output file was made
    # should actaully check the contents, not just that it exist
    assert os.path.isfile(tmpdir.join("HB2B_938.h5"))

    # plot the detector view
    qtbot.mouseClick(window.ui.pushButton_plotDetView,  QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # change to Reduced Data view, I don't know how to do that with clicking
    window.ui.tabWidget_View.setCurrentIndex(1)
    qtbot.wait(wait)
    # plot the reduction data
    qtbot.mouseClick(window.ui.pushButton_plotDetView,  QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # get the data from the plot canvas and check the data limits and label
    line = window.ui.graphicsView_1DPlot.canvas().get_axis().lines[0]
    assert line.get_label() == 'sub-run: 1, 2theta = 90.00050354003906'
    assert line.get_xdata().min() == pytest.approx(81.88955539289697)
    assert line.get_xdata().max() == pytest.approx(98.07327399000138)
    # first data point is a nan so exclude it
    assert line.get_ydata()[1::].min() == pytest.approx(56.53846153846154)
    assert line.get_ydata()[1::].max() == pytest.approx(580.4936170212766)


def test_manual_reduction_subruns(tmpdir, manual_reduction_window):
    window, qtbot = manual_reduction_window
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # set data
    qtbot.keyClicks(window.ui.lineEdit_runNumber, "tests/data/HB2B_1431.nxs.h5")
    qtbot.wait(wait)

    # set mask
    qtbot.mouseClick(window.ui.checkBox_defaultMaskFile, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    for _ in range(100):
        qtbot.keyClick(window.ui.lineEdit_maskFile, QtCore.Qt.Key_Backspace)

    qtbot.wait(wait)
    qtbot.keyClicks(window.ui.lineEdit_maskFile, "tests/data/HB2B_Mask_12-18-19.xml")
    qtbot.wait(wait)

    # set calibration
    qtbot.mouseClick(window.ui.checkBox_defaultCalibrationFile, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    for _ in range(100):
        qtbot.keyClick(window.ui.lineEdit_calibrationFile, QtCore.Qt.Key_Backspace)
    qtbot.wait(wait)
    qtbot.keyClicks(window.ui.lineEdit_calibrationFile, "tests/data/HB2B_CAL_Si333.json")
    qtbot.wait(wait)

    # set vanadium
    qtbot.keyClicks(window.ui.lineEdit_vanRunNumber, "tests/data/HB2B_1118.nxs.h5")
    qtbot.wait(wait)

    # set output directory
    qtbot.keyClicks(window.ui.lineEdit_outputDirectory, str(tmpdir))
    qtbot.wait(wait)

    # push button to run manual reduction
    qtbot.mouseClick(window.ui.pushButton_splitConvertSaveProject, QtCore.Qt.LeftButton)
    qtbot.wait(wait * 10)

    assert window.ui.progressBar.value() == 100

    # plot the detector view
    qtbot.mouseClick(window.ui.pushButton_plotDetView,  QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # change sub_run
    for _ in range(10):
        qtbot.keyClick(window.ui.comboBox_sub_runs, QtCore.Qt.Key_Down)
        qtbot.wait(wait)

    # change to Reduced Data view, I don't know how to do that with clicking
    window.ui.tabWidget_View.setCurrentIndex(1)
    qtbot.wait(wait)

    # change sub_run
    for _ in range(10):
        qtbot.keyClick(window.ui.comboBox_sub_runs, QtCore.Qt.Key_Up)
        qtbot.wait(wait)

    window.hide()