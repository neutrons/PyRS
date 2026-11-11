from pyrs.interface.texture_fitting.texture_fitting_viewer import TextureFittingViewer
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl

from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import numpy as np
import functools
import os
import shutil
import json
import pytest
from types import SimpleNamespace

from pyrs.interface import gui_helper
from pyrs.interface.texture_fitting import texture_fitting_viewer
from pyrs.peaks.peak_fit_engine import FitResult
from pyrs.utilities.config import Config

from tests.conftest import ON_GITHUB_ACTIONS  # set to True when running on build servers

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 200
plot_wait = 100


@pytest.fixture(scope="session")
def texture_fitting_window(my_qtbot):
    r"""
    Fixture for the detector calibration window. Creating the window with a session scope and reusing it for all tests.
    This is done to avoid the segmentation fault error that occurs when the window is created with a function scope.
    """
    model = TextureFittingModel(pyrscore.PyRsCore())
    ctrl = TextureFittingCrtl(model)
    window = TextureFittingViewer(model, ctrl)
    return window, my_qtbot


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


@pytest.mark.parametrize("typed", [" 1599", "01599"])
def test_load_run_number_padded_text(tmp_path, monkeypatch, texture_fitting_window, typed):
    """A padded run number loads HB2B_1599.h5, the file the lookup checked (#931)."""
    # Arrange
    window, _ = texture_fitting_window
    autoreduce = tmp_path / "shared" / "autoreduce"
    autoreduce.mkdir(parents=True)
    shutil.copy("tests/data/HB2B_1599.h5", autoreduce)
    monkeypatch.setattr("pyrs.utilities.file_util.get_ipts_dir", lambda run_number: tmp_path)
    loader = window.fileLoading.file_load_run_number
    loaded = []
    monkeypatch.setattr(loader, "load_project_plot", lambda: loaded.append(window._project_file))
    loader.lineEdit.setText(typed)

    # Act
    loader.loadRunNumber()

    # Assert
    assert loaded == [str(autoreduce / "HB2B_1599.h5")]


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
    monkeypatch.setattr(fit_table, "fits", None)
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

    def set_peak_tag(peak_tag: str, mask_key: str = "") -> None:
        """As a fit leaves it: `fits` keyed by mask ("" for the default pattern), one shown."""
        peak = minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()), peak_tag=peak_tag)
        fit_table.fits = {mask_key: FitResult(peakcollections=[peak], fitted=None, difference=None)}
        fit_table.fit_result = fit_table.fits[mask_key]

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
    assert messages == ["Failed to save 'saved.nxs'"]


def test_save_as_nxstress_eta_fits_refused(nxstress_ready_window) -> None:
    """Fits of an out-of-plane mask would be filed under the default mask, so
    saving them is refused before anything is written."""
    # Arrange
    window, out_path, messages, set_peak_tag = nxstress_ready_window
    set_peak_tag("Fe 110", mask_key="eta_0.0")

    # Act
    window.save_as_nxstress()

    # Assert
    assert not out_path.exists()
    assert messages == ["Unable to save as NXstress"]


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


def test_fit_unsupported_layout_reported(texture_fitting_window, monkeypatch) -> None:
    """A reduced-data layout `fit_peaks` refuses is reported by the *Fit* slot
    rather than escaping it (an uncaught slot exception aborts PyRS)."""
    # Arrange
    window, _qtbot = texture_fitting_window
    messages = []
    monkeypatch.setattr(
        texture_fitting_viewer, "pop_message", lambda parent, message, *a, **k: messages.append(message)
    )

    def refuse(*args, **kwargs):
        raise NotImplementedError("Texture fitting does not yet support this file's reduced data")

    monkeypatch.setattr(window.controller, "fit_peaks", refuse)

    # Act
    window.fit_setup.fit()

    # Assert
    assert messages == ["Unable to fit these data"]


class _RangeTable:
    """The peak-range table's interface as `fit_peaks` reads it: x_left, x_right, label."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def rowCount(self) -> int:
        return len(self._rows)

    def item(self, row: int, col: int):
        value = self._rows[row][col]
        return SimpleNamespace(text=lambda: str(value))


def test_save_as_nxstress_real_eta_fits_refused(texture_fitting_window, monkeypatch, tmp_path) -> None:
    """End to end on real texture data with geometry
    (`HB2B_1599_texture_with_instrument.h5`, built by
    `tests/scripts/make_texture_fixture_with_instrument.py`): *Save as NXstress...*
    is offered, the eta patterns fit, and saving those fits is refused."""
    # Arrange
    window, _qtbot = texture_fitting_window
    out_path = tmp_path / "saved.nxs"
    messages = []
    monkeypatch.setattr(
        texture_fitting_viewer, "pop_message", lambda parent, message, *a, **k: messages.append(message)
    )
    monkeypatch.setattr(
        texture_fitting_viewer.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), ""))
    )
    fit_table = window.fit_summary.fit_table_operator
    monkeypatch.setattr(fit_table, "fits", None)
    monkeypatch.setattr(fit_table, "fit_result", None)
    # The window is shared: whatever the load replaces is restored on exit.
    for name in ("ws", "sub_runs", "fit_result", "_curr_file_name", "_run_number"):
        monkeypatch.setattr(window._model, name, getattr(window._model, name, None))
    window.controller.load_projectfile("tests/data/HB2B_1599_texture_with_instrument.h5")
    window.update_nxstress_action()
    fits = window.controller.fit_peaks(_RangeTable([(62.864, 66.9115, "Fe 110")]), "PseudoVoigt", "Linear")
    fit_table.fits = fits
    fit_table.fit_result = fits["eta_0.0"]

    try:
        # Act
        enabled = window.saveAsNXstressAction.isEnabled()
        window.save_as_nxstress()

        # Assert
        assert sorted(fits) == ["eta_-5.0", "eta_0.0", "eta_5.0"]
        assert enabled == Config["nxstress.enable"]
        assert not out_path.exists()
        assert messages == (["Unable to save as NXstress"] if enabled else [])
    finally:
        monkeypatch.undo()
        window.update_nxstress_action()
