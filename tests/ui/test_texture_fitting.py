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

from tests.conftest import ON_GITHUB_ACTIONS  # set to True when running on build servers

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

        np.testing.assert_allclose(float(window.fit_setup.fit_range_table.item(i_loop, 0).text()),
                                   fit_ranges[i_loop][0], rtol=rtol)

        np.testing.assert_allclose(float(window.fit_setup.fit_range_table.item(i_loop, 1).text()),
                                   fit_ranges[i_loop][1], rtol=rtol)

    # load json with fitting range and test that data are loaded
    qtbot.wait(wait)
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "texture_fitting_test.json"))
    qtbot.mouseClick(window.fit_setup.save_fit_info, QtCore.Qt.LeftButton)

    with open('texture_fitting_test.json') as f:
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
        np.testing.assert_allclose(float(window.fit_setup.fit_range_table.item(i_loop, 0).text()),
                                   fit_ranges[i_loop][0], rtol=1e-3)

        np.testing.assert_allclose(float(window.fit_setup.fit_range_table.item(i_loop, 1).text()),
                                   fit_ranges[i_loop][1], rtol=1e-3)

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
    print(np.loadtxt('HB2B_1599_Peak_1.jul', skiprows=3))

    # test that pole figure outputs are equivalent
    np.testing.assert_allclose(np.loadtxt('tests/data/HB2B_1599_Peak_1.jul', skiprows=3),
                               np.loadtxt('HB2B_1599_Peak_1.jul', skiprows=3), rtol=1e-3)

    np.testing.assert_allclose(np.loadtxt('tests/data/HB2B_1599_Peak_2.jul', skiprows=3),
                               np.loadtxt('HB2B_1599_Peak_2.jul', skiprows=3), rtol=1e-3)

    os.remove("HB2B_1599_Peak_2.jul")
    os.remove("HB2B_1599_Peak_1.jul")

    qtbot.keyClick(window.plot_select.out_of_plane, QtCore.Qt.Key_Down)
    # qtbot.wait(plot_wait)
    window.hide()
