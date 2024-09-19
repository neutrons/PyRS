from pyrs.interface.detector_calibration.detector_calibration_viewer import DetectorCalibrationViewer  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl  # noqa: E402

from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import functools
import os
# import json
import pytest

wait = 200
plot_wait = 100


@pytest.fixture(scope="session")
def calibration_window(my_qtbot):
    r"""
    Fixture for the detector calibration window. Creating the window with a session scope and reusing it for all tests.
    This is done to avoid the segmentation fault error that occurs when the window is created with a function scope.
    """
    model = DetectorCalibrationModel(pyrscore.PyRsCore())
    ctrl = DetectorCalibrationCrtl(model)
    window = DetectorCalibrationViewer(model, ctrl)
    return window, my_qtbot


def test_detector_calibration(calibration_window):
    window, qtbot = calibration_window

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

    # # Browse e11 Data File ...
    # # wait until dialog is loaded then handle it, this is required
    # # because the dialog is modal
    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog,
                                                    "tests/data/calibration_tests/HB2B_3510.nxs.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_dilg.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)
    assert window._model.nexus_file.split('/')[-1] == 'HB2B_3510.nxs.h5'
    qtbot.wait(wait)

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog,
                                                    "tests/data/calibration_tests/test_ui_recipe_load.json"))
    qtbot.mouseClick(window.peak_lines_setup.load_info, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.wait(wait)
    assert window._model.nexus_file.split('/')[-1] == 'HB2B_3510.nxs.h5'
    qtbot.wait(wait)

    qtbot.mouseClick(window.peak_lines_setup.fit, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.mouseClick(window.peak_lines_setup.calibrate, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.wait(wait)
    while not window.peak_lines_setup.calibrate.isEnabled():
        print(window.peak_lines_setup.calibrate.isEnabled())
        qtbot.wait(wait)

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "HB2B_test_export.json"))
    qtbot.mouseClick(window.peak_lines_setup.export_recipe, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "HB2B_CAL.json"))
    qtbot.mouseClick(window.calib_summary.export_local_calib_bttn, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "HB2B_CAL.json"))
    qtbot.mouseClick(window.fileLoading.file_load_calib.browse_button, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    os.remove("HB2B_test_export.json")
    os.remove("HB2B_CAL.json")

    window.compare_diff_data.sl.setValue(2)
    qtbot.wait(wait)

    window.compare_diff_data.sl.setValue(1)
    qtbot.wait(wait)

    window.compare_diff_data.tabs.setCurrentIndex(1)
    qtbot.wait(wait)

    window.param_window.plot_paramX.setCurrentIndex(1)
    qtbot.wait(wait)

    window.hide()
