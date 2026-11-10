from pyrs.interface.peak_fitting import fitpeakswindow
from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import json
import pytest
import os
from unittest.mock import MagicMock

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 300


@pytest.fixture(scope="session")
def fit_peaks_window(my_qtbot):
    r"""
    Fixture for the detector calibration window. Creating the window with a session scope and reusing it for all tests.
    This is done to avoid the segmentation fault error that occurs when the window is created with a function scope.
    """
    fit_peak_core = pyrscore.PyRsCore()
    window = fitpeakswindow.FitPeaksWindow(None, fit_peak_core=fit_peak_core)
    return window, my_qtbot


def test_peak_fitting(tmpdir, fit_peaks_window):
    window, qtbot = fit_peaks_window
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
    assert len(file_contents) == 129
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=np.float64, sep=",")) == 35

    # look at 1D results plot
    line = window.ui.graphicsView_fitResult.canvas().get_axis().lines[0]
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

    line = window.ui.graphicsView_fitResult.canvas().get_axis().lines[0]
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
    assert len(file_contents) == 129
    # check number of values in line
    assert len(np.fromstring(file_contents[-1], dtype=np.float64, sep=",")) == 35

    # look at 1D results plot
    line = window.ui.graphicsView_fitResult.canvas().get_axis().lines[0]
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

    line = window.ui.graphicsView_fitResult.canvas().get_axis().lines[0]
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


def test_peak_selection(tmpdir, fit_peaks_window):
    window, qtbot = fit_peaks_window
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

    window.hide()


def test_save_load_peak_range_keeps_table_d0(tmp_path, monkeypatch, fit_peaks_window):
    """The d0 values shown in the peak-range table survive save -> load (#925)."""
    # Arrange
    window, _ = fit_peaks_window
    fit_setup = window._ui_graphicsView_fitSetup
    fit_setup.list_peak_ranges = [[78.0, 85.0], [90.0, 95.0]]
    fit_setup.list_fit_peak_labels = ["Peak0", "Peak1"]
    fit_setup.list_fit_peak_d0 = [1.1, 1.2]
    # No diffraction data is loaded, so the redraw after loading has nothing to plot over.
    monkeypatch.setattr(fit_setup, "plot_data_with_fitting_ranges", lambda: None)
    saved_name = tmp_path / "peak_range"
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_viewer.browse_file", lambda *args, **kwargs: str(saved_name)
    )

    # Act
    window.clicked_save_peak_range()
    fit_setup.list_fit_peak_d0 = []
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_viewer.browse_file",
        lambda *args, **kwargs: str(tmp_path / "peak_range.json"),
    )
    window.clicked_load_peak_range()

    # Assert
    saved = json.loads((tmp_path / "peak_range.json").read_text())
    assert [saved[key]["d0"] for key in ("0", "1")] == [1.1, 1.2]
    assert fit_setup.list_fit_peak_d0 == [1.1, 1.2]
    assert window.ui.peak_range_table.item(1, 3).text() == "1.200000"


def test_remove_peak_range_row_drops_its_d0(monkeypatch, fit_peaks_window):
    """Removing a peak-range row removes that row's d0, so later peaks keep their own (#925)."""
    # Arrange
    window, _ = fit_peaks_window
    fit_setup = window._ui_graphicsView_fitSetup
    fit_setup.list_peak_ranges = [[78.0, 85.0], [90.0, 95.0]]
    fit_setup.list_fit_peak_labels = ["Peak0", "Peak1"]
    fit_setup.list_fit_peak_d0 = [1.1, 1.2]
    fit_setup.list_peak_labels_matplotlib_id = [MagicMock(), MagicMock()]
    monkeypatch.setattr(fit_setup, "plot_data_with_fitting_ranges", lambda: None)
    window.update_peak_ranges_table(
        list_fit_peak_ranges=fit_setup.list_peak_ranges,
        list_fit_peak_labels=fit_setup.list_fit_peak_labels,
        list_fit_peak_d0=fit_setup.list_fit_peak_d0,
    )
    table = window.ui.peak_range_table
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 0, table.columnCount() - 1), True)

    # Act
    window.remove_peak_range_table_row()

    # Assert
    assert fit_setup.list_fit_peak_labels == ["Peak1"]
    assert fit_setup.list_fit_peak_d0 == [1.2]


@pytest.mark.parametrize("answer, expected_calls", [("Yes", 2), ("No", 1)])
def test_save_asks_before_combining_reduction_differences(monkeypatch, fit_peaks_window, answer, expected_calls):
    """A vanadium/mask/binning difference asks the user; Yes saves with them accepted (#1080)."""
    # Arrange
    from pyrs.projectfile.file_object import ReductionMismatchError

    window, _ = fit_peaks_window
    calls = []

    def fake_save(out_file_name, accept_reduction_differences=False):
        calls.append(accept_reduction_differences)
        window._model.save_warnings = ["accepted differences"] if accept_reduction_differences else []
        if not accept_reduction_differences:
            raise ReductionMismatchError("different vanadium runs. Combine them anyway?")

    shown = []
    monkeypatch.setattr(window._model, "save_fit_result", fake_save)
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_viewer.QMessageBox.question",
        lambda *args: getattr(QtWidgets.QMessageBox, answer),
    )
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_viewer.pop_message", lambda *args: shown.append(args)
    )

    # Act
    window.save()

    # Assert
    assert calls == [False, True][:expected_calls]
    assert [args[2] for args in shown] == (["accepted differences"] if answer == "Yes" else [])


def test_load_run_number_never_reduced_reports(tmp_path, monkeypatch, fit_peaks_window):
    """A run whose folders exist but hold no project file shows one error and loads nothing (#931)."""
    # Arrange
    window, _ = fit_peaks_window
    for name in ("manualreduce", "autoreduce"):
        (tmp_path / "shared" / name).mkdir(parents=True)
    monkeypatch.setattr("pyrs.utilities.file_util.get_ipts_dir", lambda run_number: tmp_path)
    shown = []
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_viewer.pop_message", lambda *args, **kwargs: shown.append(args)
    )
    previous = window.current_hidra_file_name
    window.ui.lineEdit_expNumber.setText("1234")

    # Act
    window.load_run_number()

    # Assert
    assert len(shown) == 1
    assert "HB2B_1234.h5" in shown[0][2]
    assert window.current_hidra_file_name == previous
