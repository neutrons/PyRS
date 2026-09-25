from pyrs.interface.peak_fitting import fitpeakswindow
from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import pytest
import os

from pyrs.interface import gui_helper
from pyrs.interface.peak_fitting import peak_fitting_viewer
from pyrs.peaks.peak_fit_engine import FitResult
from pyrs.utilities.config import Config

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


def test_init_widgets_nxstress_and_legacy_action_enablement_matches_config(fit_peaks_window: tuple) -> None:
    r"""
    Confirm the new "Save as NXstress..." action exists, is wired, and its
    enabled state (set in _init_widgets) matches the shipped default config
    (nxstress.enable / legacy_io.enable both True). This intentionally reuses
    the session-scoped `fit_peaks_window` fixture rather than constructing a
    second FitPeaksWindow with a different Config override -- constructing this
    window with function scope is documented (see the fixture above) to cause
    segfaults, so a "flip the config, rebuild the window, assert disabled"
    variant is not attempted here.
    """
    window, _qtbot = fit_peaks_window

    from pyrs.utilities.config import Config

    assert hasattr(window.ui, "actionSaveAsNXstress")
    assert window.ui.actionSaveAsNXstress.isVisible()
    # NXstress additionally needs instrument geometry on the loaded workspace;
    # see `test_nxstress_action_follows_workspace_geometry`.
    ws = window._model.hidra_workspace
    has_geometry = ws is not None and ws.get_instrument_setup() is not None
    assert window.ui.actionSaveAsNXstress.isEnabled() == (Config["nxstress.enable"] and has_geometry)

    assert window.ui.actionSave.isVisible()
    assert window.ui.actionSave.isEnabled() == Config["legacy_io.enable"]
    assert window.ui.actionSaveAs.isVisible()
    assert window.ui.actionSaveAs.isEnabled() == Config["legacy_io.enable"]

    window.hide()


def test_nxstress_action_follows_workspace_geometry(fit_peaks_window: tuple) -> None:
    """*Save as NXstress...* is grayed out after loading a file without instrument
    geometry, and enabled after loading one with it."""
    window, qtbot = fit_peaks_window

    # Act / Assert: no geometry
    window.load_and_plot([os.path.abspath("tests/data/HB2B_1423.h5")])
    qtbot.wait(wait)
    assert window.ui.actionSaveAsNXstress.isVisible()
    assert not window.ui.actionSaveAsNXstress.isEnabled()

    # Act / Assert: geometry
    window.load_and_plot([os.path.abspath("tests/data/HB2B_1017.h5")])
    qtbot.wait(wait)
    assert window.ui.actionSaveAsNXstress.isEnabled() == Config["nxstress.enable"]


@pytest.fixture
def nxstress_ready_window(fit_peaks_window, minimal_HidraWorkspace, minimal_PeakCollection, monkeypatch, tmp_path):
    """The shared window holding a workspace with geometry, with the save dialog
    and message boxes replaced. Call the returned `set_peak_tag` before saving.
    Every change to the window is made through `monkeypatch` and undone on exit."""
    window, _qtbot = fit_peaks_window
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
    monkeypatch.setattr(window._model, "hidra_workspace", ws)
    monkeypatch.setattr(window._model, "fit_result", None)
    window._update_nxstress_action()

    out_path = tmp_path / "saved.nxs"
    messages = []

    def record(parent, message, *args, **kwargs):
        messages.append(message)

    monkeypatch.setattr(peak_fitting_viewer, "browse_file", lambda *args, **kwargs: str(out_path))
    monkeypatch.setattr(peak_fitting_viewer, "pop_message", record)
    monkeypatch.setattr(gui_helper, "pop_message", record)

    def set_peak_tag(peak_tag: str) -> None:
        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()), peak_tag=peak_tag)
        window._model.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

    yield window, out_path, messages, set_peak_tag
    monkeypatch.undo()
    window._update_nxstress_action()


def test_save_as_nxstress_writes_file(nxstress_ready_window) -> None:
    # Arrange
    window, out_path, messages, set_peak_tag = nxstress_ready_window
    set_peak_tag("Fe 110")

    # Act
    window.save_as_nxstress()

    # Assert
    assert out_path.exists()
    assert messages == []


def test_save_as_nxstress_failure_reports_and_cleans_up(nxstress_ready_window) -> None:
    """A tag NXstress cannot parse raises inside the writer; the slot must report
    it (an uncaught slot exception aborts the app) and remove the partial file."""
    # Arrange
    window, out_path, messages, set_peak_tag = nxstress_ready_window
    set_peak_tag("peak0")

    # Act
    window.save_as_nxstress()

    # Assert
    assert not out_path.exists()
    assert messages == ["Failed to save NXstress file"]


def test_save_as_nxstress_cancel_writes_nothing(nxstress_ready_window, monkeypatch) -> None:
    # Arrange
    window, out_path, messages, set_peak_tag = nxstress_ready_window
    set_peak_tag("Fe 110")
    monkeypatch.setattr(peak_fitting_viewer, "browse_file", lambda *args, **kwargs: None)

    # Act
    window.save_as_nxstress()

    # Assert
    assert not out_path.exists()
    assert messages == []
