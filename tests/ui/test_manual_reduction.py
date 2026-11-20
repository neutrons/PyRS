from qtpy import QtCore
import os
import pytest
import matplotlib

matplotlib.use("Agg")
from pyrs.interface.manual_reduction import manualreductionwindow  # noqa E402

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 100


@pytest.fixture
def manual_reduction_window(qtbot):
    """A fresh manual reduction window for each test, closed afterwards (see `tests/ui/conftest.py`)."""
    window = manualreductionwindow.ManualReductionWindow(None)
    qtbot.addWidget(window)
    return window, qtbot


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
    qtbot.mouseClick(window.ui.pushButton_plotDetView, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # change to Reduced Data view, I don't know how to do that with clicking
    window.ui.tabWidget_View.setCurrentIndex(1)
    qtbot.wait(wait)
    # plot the reduction data
    qtbot.mouseClick(window.ui.pushButton_plotDetView, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # get the data from the plot canvas and check the data limits and label
    line = window.ui.graphicsView_1DPlot.canvas().get_axis().lines[0]
    assert line.get_label() == "sub-run: 1, 2theta = 90.00050354003906"
    assert line.get_xdata().min() == pytest.approx(81.88640540875517)
    assert line.get_xdata().max() == pytest.approx(98.06382403754796)
    # first data point is a nan so exclude it
    assert line.get_ydata()[1::].min() == pytest.approx(136.97619047619048)
    assert line.get_ydata()[1::].max() == pytest.approx(1037.9183222958056)


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
    qtbot.mouseClick(window.ui.pushButton_plotDetView, QtCore.Qt.LeftButton)
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


def test_browse_offers_json_and_nexus_calibrations(tmp_path, manual_reduction_window, monkeypatch):
    """The calibration browser shows .json and .nxs calibrations, and the chosen one is reduced with (#1098)"""
    import fnmatch
    import re

    from pyrs.interface import gui_helper

    # Arrange - the file dialog picks a NeXus calibration; a run file to reduce; the reduction is recorded
    window, _ = manual_reduction_window
    window.ui.checkBox_defaultCalibrationFile.setChecked(False)
    assert window.ui.pushButton_browseCalibrationFile.isEnabled()  # unticking the default allows browsing
    chosen = str(tmp_path / "HB2B_CAL.nxs")
    filters = []

    def choose(parent, caption, directory, file_filter, **kwargs):
        filters.append(file_filter)
        return chosen, file_filter

    monkeypatch.setattr(gui_helper.QFileDialog, "getOpenFileName", choose)
    run_file = tmp_path / "HB2B_1017.nxs.h5"
    run_file.write_bytes(b"")  # only its name and existence are checked before the (recorded) reduction
    window.ui.lineEdit_runNumber.setText(str(run_file))
    reduced = []
    monkeypatch.setattr(window._ctrl, "reduce", lambda *args, **kwargs: reduced.append(kwargs) or [])

    # Act
    window.browse_calibration_file()
    window.split_convert_save_nexus()

    # Assert - the dialog's first (default) filter takes both calibration formats, and not raw hdf files
    patterns = re.findall(r"\(([^)]*)\)", filters[0].split(";;")[0])[0].split()
    assert any(fnmatch.fnmatch("HB2B_CAL.nxs", p) for p in patterns)
    assert any(fnmatch.fnmatch("HB2B_Latest.json", p) for p in patterns)
    assert not any(fnmatch.fnmatch("HB2B_1017.hdf", p) for p in patterns)
    assert window.ui.lineEdit_calibrationFile.text() == chosen
    assert reduced[0]["calibration"] == chosen


def test_unusable_calibration_is_reported(tmp_path, manual_reduction_window, monkeypatch):
    """A calibration file reduction refuses (here: no wavelength, #1098) is shown in a pop-up, not raised out of the
    window"""
    from pyrs.interface.manual_reduction import manual_reduction_viewer as viewer_module

    # Arrange
    window, _ = manual_reduction_window
    run_file = tmp_path / "HB2B_1017.nxs.h5"
    run_file.write_bytes(b"")
    window.ui.lineEdit_runNumber.setText(str(run_file))
    messages = []

    def refuse(*args, **kwargs):
        raise ValueError("calibration HB2B_CAL.json gives no wavelength (Lambda); it cannot be applied")

    monkeypatch.setattr(window._ctrl, "reduce", refuse)
    monkeypatch.setattr(viewer_module, "pop_message", lambda parent, title, detail, **kwargs: messages.append(detail))

    # Act
    window.split_convert_save_nexus()

    # Assert
    assert len(messages) == 1
    assert "no wavelength" in messages[0]
