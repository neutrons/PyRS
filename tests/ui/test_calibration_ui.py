from pyrs.interface.detector_calibration.detector_calibration_viewer import DetectorCalibrationViewer  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl  # noqa: E402

from pyrs.core import pyrscore
from qtpy import QtCore, QtWidgets
import functools

# import json
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration]

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


def handle_export_dialog(folder, write_shared):
    """Fill in and accept the Export Calibration dialog."""
    dialog = QtWidgets.QApplication.activeModalWidget()
    assert dialog.write_shared.isChecked()  # the shared HB2B calibration is written by default
    dialog.folder_lineEdit.setText(folder)
    dialog.write_shared.setChecked(write_shared)
    dialog.accept()


def test_detector_calibration(calibration_window, tmp_path):
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

    # Browse e11 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, "tests/data/calibration_tests/HB2B_3510.nxs.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_dilg.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)
    assert window._model.nexus_file.split("/")[-1] == "HB2B_3510.nxs.h5"
    qtbot.wait(wait)

    QtCore.QTimer.singleShot(
        300, functools.partial(handle_dialog, "tests/data/calibration_tests/test_ui_recipe_load.json")
    )
    qtbot.mouseClick(window.peak_lines_setup.load_info, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.wait(wait)
    assert window._model.nexus_file.split("/")[-1] == "HB2B_3510.nxs.h5"
    qtbot.wait(wait)

    qtbot.mouseClick(window.peak_lines_setup.fit, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.mouseClick(window.peak_lines_setup.calibrate, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    qtbot.wait(wait)
    while not window.peak_lines_setup.calibrate.isEnabled():
        print(window.peak_lines_setup.calibrate.isEnabled())
        qtbot.wait(wait)

    # Write export/calibration files under pytest's tmp_path
    export_recipe_path = str(tmp_path / "HB2B_test_export.json")
    calib_folder = tmp_path / "calibration"
    calib_folder.mkdir()

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, export_recipe_path))
    qtbot.mouseClick(window.peak_lines_setup.export_recipe, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # Export: the user's folder, with the shared-calibration flag (ticked by default) unticked here
    QtCore.QTimer.singleShot(300, functools.partial(handle_export_dialog, str(calib_folder), False))
    qtbot.mouseClick(window.calib_summary.export_calib_bttn, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    assert not hasattr(window.calib_summary, "export_local_calib_bttn")  # Export replaces it (#1045)
    (calib_path,) = [str(path) for path in calib_folder.iterdir()]

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, calib_path))
    qtbot.mouseClick(window.fileLoading.file_load_calib.browse_button, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    window.compare_diff_data.sl.setValue(2)
    qtbot.wait(wait)

    window.compare_diff_data.sl.setValue(1)
    qtbot.wait(wait)

    # predicted rings on the 2D view (#1039): one contour level per ring, cleared when unticked
    diffraction = window.compare_diff_data
    detector_view = diffraction.tabs.widget(0)
    assert detector_view._ring_overlay is None
    assert diffraction.ring_geometry.currentText() == "Current calibration"  # the default geometry
    assert not diffraction.show_points.isEnabled()  # ring points exist only after the ring-based Fit Peaks
    diffraction.show_rings.setChecked(True)
    qtbot.wait(plot_wait)
    _, rings, _ = window._model.ring_overlay(1, "Current calibration")
    assert rings
    assert len(detector_view._ring_overlay.levels) == len({ring.two_theta for ring in rings})
    diffraction.ring_geometry.setCurrentText("Nominal")
    qtbot.wait(plot_wait)
    assert detector_view._ring_overlay is not None
    diffraction.sl.setValue(2)
    qtbot.wait(plot_wait)
    assert len(detector_view._myCanvas.axes.collections) == 1  # the previous sub-run's rings were removed
    diffraction.show_rings.setChecked(False)
    qtbot.wait(plot_wait)
    assert detector_view._ring_overlay is None and not detector_view._myCanvas.axes.collections
    diffraction.sl.setValue(1)
    qtbot.wait(wait)

    window.compare_diff_data.tabs.setCurrentIndex(1)
    qtbot.wait(wait)

    window.param_window.plot_paramX.setCurrentIndex(1)
    qtbot.wait(wait)

    window.hide()


def test_ring_based_calibration_matches_driver(calibration_window, tmp_path):
    """#1045: the window's ring-based calibration gives exactly the pyrs-pixel-calibration driver's result"""
    import importlib.util
    import json
    from pathlib import Path

    window, qtbot = calibration_window
    window.show()
    nexus = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
    window._nexus_file = nexus
    window.fileLoading.file_load_dilg.load_nexus()
    qtbot.wait(wait)
    setup = window.peak_lines_setup

    # Arrange - the driver's settings; the session window keeps the first test's recipe (max nfev 3,
    # only sub-run 1 kept), which the window rightly applies to the ring-based calibration too
    window.calib_summary.method_combo_box.setCurrentText("trf")
    window.calib_summary.neval_lineEdit.setText("300")
    for row in range(setup.calibrant_table.rowCount()):
        setup.calibrant_table.item(row, 1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    # ... and the calibration it loaded and refined: start from the nominal geometry, as a new run
    table = window.calib_summary._cal_summary_table
    table.setColumnCount(1)
    for row in range(7):
        table.setItem(row, 0, QtWidgets.QTableWidgetItem("0"))
    table.set_wavelength(0, window.controller.get_wavelength())

    # Act - ring-based, default strips, recipe "iterate", as the driver
    setup.method_combo.setCurrentText("Ring-based")
    assert setup.strips_lineEdit.text() == "5" and setup.strips_lineEdit.isEnabled()
    assert not setup.tthbin_lineEdit.isEnabled()
    for i_row in range(8):
        setup.recipe_combos[i_row].setCurrentIndex(0)
    setup.recipe_combos[0].setCurrentText("iterate")
    qtbot.mouseClick(setup.fit, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: window.model._ring_calibration is not None, timeout=120000)
    qtbot.mouseClick(setup.calibrate, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: setup.calibrate.isEnabled() and window.model.rmse is not None, timeout=120000)
    qtbot.wait(wait)

    # picked ring points and rejected strips on the 2D view, with their descriptions on hover
    diffraction = window.compare_diff_data
    detector_view = diffraction.tabs.widget(0)
    diffraction.tabs.setCurrentIndex(0)
    assert diffraction.show_points.isEnabled() and diffraction.show_rejected.isEnabled()
    diffraction.show_points.setChecked(True)
    diffraction.show_rejected.setChecked(True)
    qtbot.wait(plot_wait)
    accepted, rejected = window.model.ring_point_marks(diffraction.sl.value())
    assert len(accepted.text) > 0
    drawn = sum(len(artist.get_offsets()) for artist in detector_view._point_artists)
    assert drawn == 2 * len(accepted.text) + len(rejected.text)  # circles have a dark edge ring
    axes = detector_view._myCanvas.axes
    x, y = axes.transData.transform((accepted.index_0[0] + 0.5, accepted.index_1[0] + 0.5))
    assert detector_view.mark_text_at(x, y) == accepted.text[0]
    diffraction.show_points.setChecked(False)
    diffraction.show_rejected.setChecked(False)
    qtbot.wait(plot_wait)
    assert not detector_view._point_artists
    folder = tmp_path / "ring"
    folder.mkdir()
    written = window.calib_summary.write_calibration(str(folder), write_shared=False)

    # Assert - the same file as the driver, with the window's mask (none without /HFIR) and wavelength
    spec = importlib.util.spec_from_file_location(
        "pyrs_pixel_calibration", Path(__file__).resolve().parents[2] / "scripts" / "pyrs_pixel_calibration.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver_file = str(tmp_path / "driver.json")
    driver.run_calibration(nexus, driver_file, mask_file=window.model.mask_file)
    assert json.load(open(written[0])) == json.load(open(driver_file))

    setup.method_combo.setCurrentText("FitCalibration")
    assert "iterate" not in [setup.recipe_combos[0].itemText(i) for i in range(setup.recipe_combos[0].count())]
    window.hide()
