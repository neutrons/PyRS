from pyrs.interface.texture_fitting.texture_fitting_viewer import TextureFittingViewer
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl

from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import os
import json
import pytest

from pyrs.interface import gui_helper
from pyrs.interface.texture_fitting import texture_fitting_viewer
from pyrs.peaks.peak_fit_engine import FitResult
from pyrs.utilities.config import Config

from tests.conftest import ON_GITHUB_ACTIONS  # set to True when running on build servers

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 200
plot_wait = 100


@pytest.fixture
def texture_fitting_window(qtbot):
    """A fresh texture fitting window for each test, closed afterwards (see `tests/ui/conftest.py`)."""
    model = TextureFittingModel(pyrscore.PyRsCore())
    ctrl = TextureFittingCrtl(model)
    window = TextureFittingViewer(model, ctrl)
    qtbot.addWidget(window)
    return window, qtbot


def test_texture_fitting_viewer(texture_fitting_window):
    window, qtbot = texture_fitting_window

    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # This is to handle modal dialogs
    def handle_dialog(filename):
        qtbot.wait(wait)
        # get a reference to the dialog and handle it here
        dialog = window.findChild(QtWidgets.QFileDialog)
        # get a File Name field
        lineEdit = dialog.findChild(QtWidgets.QLineEdit)
        # Type in file to load and press enter
        qtbot.keyClicks(lineEdit, filename)
        qtbot.wait(wait)
        qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)

    # Browse e11 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "tests/data/HB2B_1599.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_dilg.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)
    assert window._model.runnumber == 1599
    assert window.fit_summary.out_of_plane.isVisible() is False
    qtbot.wait(wait)

    # Browse e11 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "tests/data/HB2B_1599_texture.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_dilg.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)
    assert window._model.runnumber == 1599
    assert window.fit_summary.out_of_plane.isVisible() is True
    qtbot.wait(wait)

    # First get canvas
    canvas = window.fit_window._myCanvas

    # The get start and end mouse points to drag select
    fit_ranges = [[62.864, 66.9115], [71.87344, 76.5544]]

    if ON_GITHUB_ACTIONS:
        rtol = 0.5
    else:
        rtol = 0.1

    for i_loop in range(len(fit_ranges)):
        # Drag select with mouse control
        canvas.figure.canvas.draw()
        start_x, start_y = canvas.figure.axes[0].transData.transform((fit_ranges[i_loop][0], 40))
        end_x, end_y = canvas.figure.axes[0].transData.transform((fit_ranges[i_loop][1], 40))

        qtbot.mousePress(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(start_x), int(start_y)))
        qtbot.wait(wait)
        qtbot.mouseRelease(canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(int(end_x), int(end_y)))
        qtbot.wait(wait)

        np.testing.assert_allclose(
            float(window.fit_setup.fit_range_table.item(i_loop, 0).text()), fit_ranges[i_loop][0], rtol=rtol
        )

        np.testing.assert_allclose(
            float(window.fit_setup.fit_range_table.item(i_loop, 1).text()), fit_ranges[i_loop][1], rtol=rtol
        )

    # load json with fitting range and test that data are loaded
    qtbot.wait(wait)
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "texture_fitting_test.json"))
    qtbot.mouseClick(window.fit_setup.save_fit_info, QtCore.Qt.LeftButton)

    with open("texture_fitting_test.json") as f:
        data = json.load(f)

    assert len(data) == 2
    np.testing.assert_allclose(data["0"]["peak_range"][0], fit_ranges[0][0], rtol=rtol)
    np.testing.assert_allclose(data["0"]["peak_range"][1], fit_ranges[0][1], rtol=rtol)
    np.testing.assert_allclose(data["1"]["peak_range"][0], fit_ranges[1][0], rtol=rtol)
    np.testing.assert_allclose(data["1"]["peak_range"][1], fit_ranges[1][1], rtol=rtol)

    os.remove("texture_fitting_test.json")

    # load json with fitting range and test that data are loaded
    qtbot.wait(wait)
    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "tests/data/texture_fitting.json"))
    qtbot.mouseClick(window.fit_setup.load_fit_info, QtCore.Qt.LeftButton)

    fit_ranges = [[62.864, 66.9115], [71.87344, 76.5544]]

    for i_loop in range(len(fit_ranges)):
        qtbot.wait(wait)
        np.testing.assert_allclose(
            float(window.fit_setup.fit_range_table.item(i_loop, 0).text()), fit_ranges[i_loop][0], rtol=1e-3
        )

        np.testing.assert_allclose(
            float(window.fit_setup.fit_range_table.item(i_loop, 1).text()), fit_ranges[i_loop][1], rtol=1e-3
        )

    # Test fitting data
    qtbot.wait(wait)
    qtbot.mouseClick(window.fit_setup.fit_peaks, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Test data fit visualization
    window.VizSetup.plot_paramX.setCurrentIndex(7)
    qtbot.wait(wait)
    window.VizSetup.plot_paramY.setCurrentIndex(8)
    qtbot.wait(wait)
    window.VizSetup.plot_paramZ.setCurrentIndex(16)
    qtbot.wait(wait)

    qtbot.mouseClick(window.VizSetup.contour_bt, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.VizSetup.scatter_bt, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.VizSetup.lines_bt, QtCore.Qt.LeftButton)
    qtbot.mouseClick(window.VizSetup.polar_bt, QtCore.Qt.LeftButton)

    for _ in range(9):
        qtbot.keyClick(window.plot_select.plot_paramX, QtCore.Qt.Key_Down)
        qtbot.wait(plot_wait)
        window.plot_select.plot_paramY.setCurrentIndex(0)
        for __ in range(18):
            qtbot.keyClick(window.plot_select.plot_paramY, QtCore.Qt.Key_Down)
            qtbot.wait(plot_wait)

    # load json with fitting range and test that data are loaded
    qtbot.wait(wait)
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, ""))
    qtbot.mouseClick(window.fit_setup.export_pole_figs, QtCore.Qt.LeftButton)
    print(np.loadtxt("HB2B_1599_Peak_1.jul", skiprows=3))

    # test that pole figure outputs are equivalent
    np.testing.assert_allclose(
        np.loadtxt("tests/data/HB2B_1599_Peak_1.jul", skiprows=3),
        np.loadtxt("HB2B_1599_Peak_1.jul", skiprows=3),
        rtol=1e-3,
    )

    np.testing.assert_allclose(
        np.loadtxt("tests/data/HB2B_1599_Peak_2.jul", skiprows=3),
        np.loadtxt("HB2B_1599_Peak_2.jul", skiprows=3),
        rtol=1e-3,
    )

    os.remove("HB2B_1599_Peak_2.jul")
    os.remove("HB2B_1599_Peak_1.jul")

    qtbot.keyClick(window.plot_select.out_of_plane, QtCore.Qt.Key_Down)
    # qtbot.wait(plot_wait)
    window.hide()


def test_init_menu_actions_nxstress_enabled_legacy_disabled_matches_config(texture_fitting_window: tuple) -> None:
    r"""
    Confirm the new "Save as NXstress..." action exists, is wired, and its
    enabled state matches the shipped default config (nxstress.enable=True).
    The legacy Save/Save As actions are, by design, left permanently disabled
    here regardless of config -- see texture_fitting_viewer.py's __init__ for
    why. Reuses the session-scoped `texture_fitting_window` fixture rather than
    constructing a second window with a different Config override, for the same
    segfault-avoidance reason documented on the fixture above.
    """
    window, _qtbot = texture_fitting_window

    from pyrs.utilities.config import Config

    assert hasattr(window, "saveAsNXstressAction")
    assert window.saveAsNXstressAction.isVisible()
    # NXstress additionally needs instrument geometry on the loaded workspace;
    # see `test_nxstress_action_follows_workspace_geometry`.
    ws = window._model.ws
    has_geometry = ws is not None and ws.get_instrument_setup() is not None
    assert window.saveAsNXstressAction.isEnabled() == (Config["nxstress.enable"] and has_geometry)

    # Legacy Save/Save As: always disabled here, independent of config.
    assert window.saveAction.isVisible()
    assert window.saveAction.isEnabled() is False
    assert window.saveAsAction.isVisible()
    assert window.saveAsAction.isEnabled() is False


def test_nxstress_action_follows_workspace_geometry(
    texture_fitting_window: tuple, minimal_HidraWorkspace, monkeypatch
) -> None:
    """*Save as NXstress...* is grayed out for a workspace without instrument
    geometry and enabled for one with it."""
    window, _qtbot = texture_fitting_window

    try:
        # Act / Assert: no geometry
        monkeypatch.setattr(window._model, "ws", minimal_HidraWorkspace(with_instrument=False))
        window.update_nxstress_action()
        assert window.saveAsNXstressAction.isVisible()
        assert not window.saveAsNXstressAction.isEnabled()

        # Act / Assert: geometry
        monkeypatch.setattr(window._model, "ws", minimal_HidraWorkspace(with_instrument=True))
        window.update_nxstress_action()
        assert window.saveAsNXstressAction.isEnabled() == Config["nxstress.enable"]
    finally:
        monkeypatch.undo()
        window.update_nxstress_action()


@pytest.fixture
def nxstress_ready_window(
    texture_fitting_window, minimal_HidraWorkspace, minimal_PeakCollection, monkeypatch, tmp_path
):
    """The shared window holding a workspace with geometry, with the save dialog
    and message boxes replaced. Call the returned `set_peak_tag` before saving.
    Every change to the window is made through `monkeypatch` and undone on exit."""
    window, _qtbot = texture_fitting_window
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
    fit_table = window.fit_summary.fit_table_operator
    monkeypatch.setattr(window._model, "ws", ws)
    monkeypatch.setattr(fit_table, "fit_result", None)
    window.update_nxstress_action()

    out_path = tmp_path / "saved.nxs"
    messages = []

    def record(parent, message, *args, **kwargs):
        messages.append(message)

    monkeypatch.setattr(
        texture_fitting_viewer.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out_path), "")),
    )
    monkeypatch.setattr(texture_fitting_viewer, "pop_message", record)
    monkeypatch.setattr(gui_helper, "pop_message", record)

    def set_peak_tag(peak_tag: str) -> None:
        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()), peak_tag=peak_tag)
        fit_table.fit_result = FitResult(peakcollections=[peak], fitted=None, difference=None)

    yield window, out_path, messages, set_peak_tag
    monkeypatch.undo()
    window.update_nxstress_action()


def test_save_as_nxstress_writes_file(nxstress_ready_window) -> None:
    """Also covers the slot reading its fit result from `self.fit_summary`: it
    used `self._parent`, which the main window never defines."""
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
    monkeypatch.setattr(texture_fitting_viewer.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))

    # Act
    window.save_as_nxstress()

    # Assert
    assert not out_path.exists()
    assert messages == []


@pytest.mark.parametrize("kind", ["garbage", "two_entries"])
def test_load_failure_reports_and_keeps_workspace(
    texture_fitting_window, unloadable_nxstress_files, monkeypatch, kind
) -> None:
    """A file that cannot be loaded is reported in a dialog, and the previously
    loaded workspace stays loaded. The model used to swallow the error into an
    unconnected signal, so nothing was reported and, on a first load, plotting
    then aborted the application."""
    # Arrange
    window, _ = texture_fitting_window
    file_load = window.fileLoading.file_load_dilg
    window._project_file = os.path.abspath("tests/data/HB2B_1017.h5")
    file_load.load_project_plot()
    loaded = window._model.ws
    messages = []
    monkeypatch.setattr(
        texture_fitting_viewer, "pop_message", lambda parent, message, *args, **kwargs: messages.append(message)
    )

    # Act
    window._project_file = str(unloadable_nxstress_files[kind])
    file_load.load_project_plot()

    # Assert
    assert messages == [f"Unable to load {unloadable_nxstress_files[kind]}"]
    assert window._model.ws is loaded


class TestParameterPlotsBeforeAnyFit:
    """Choosing what to plot never raises, fitted or not.

    `extract_data` returned `None` for every axis but `sub-runs` until a fit
    existed -- the sample logs the selectors offer from the start included --
    and the `TypeError` that followed escaped the Qt slot and aborted PyRS.
    """

    @pytest.fixture
    def loaded_window(self, texture_fitting_window, monkeypatch):
        """A window with `HB2B_1017.h5` loaded and no fit, recording what the 2D plot draws."""
        window, _ = texture_fitting_window
        window._project_file = os.path.abspath("tests/data/HB2B_1017.h5")
        window.fileLoading.file_load_dilg.load_project_plot()
        drawn, cleared = [], []
        monkeypatch.setattr(window.param_window, "plot_scatter_with_errors", lambda **kwargs: drawn.append(kwargs))
        monkeypatch.setattr(window.param_window, "reset_viewer", lambda: cleared.append(True))
        return window, drawn, cleared

    def test_a_sample_log_axis_is_plotted(self, loaded_window) -> None:
        # Arrange
        window, drawn, _ = loaded_window

        # Act
        window.plot_select.plot_paramY.setCurrentText("vx")

        # Assert
        assert len(drawn) == 1
        np.testing.assert_array_equal(drawn[0]["vec_y"], window.model.ws.get_sample_log_values("vx"))

    def test_loading_with_a_fit_parameter_axis_clears_the_plot(self, loaded_window) -> None:
        """A fit populates the selectors with its parameters; loading a file then
        clears the fit, leaving a selected parameter with nothing to plot."""
        # Arrange -- the state a fit leaves behind
        window, drawn, cleared = loaded_window
        window.plot_select.plot_paramY.addItem("Center")
        window.plot_select.plot_paramY.setCurrentText("Center")
        drawn.clear()
        cleared.clear()

        # Act
        window.fileLoading.file_load_dilg.load_project_plot()

        # Assert -- cleared, not left showing the previous file's plot
        assert drawn == []
        assert cleared
