from pyrs.interface.detector_calibration.detector_calibration_viewer import DetectorCalibrationViewer  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl  # noqa: E402

from pyrs.core import pyrscore
from matplotlib.contour import ContourSet
from qtpy import QtCore, QtWidgets
import functools
import inspect
import json
import os

# import json
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 200
plot_wait = 100


@pytest.fixture(scope="session")
def export_settings(tmp_path_factory):
    """The remembered export folder (#1087) goes to a scratch file, never the user's settings. Redirected before
    the window is built, because the window reads it then."""
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    settings_file = str(tmp_path_factory.mktemp("settings") / "pyrs.ini")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(viewer_module, "_settings", lambda: QtCore.QSettings(settings_file, QtCore.QSettings.IniFormat))
        yield settings_file


@pytest.fixture
def calibration_window(qtbot, export_settings):
    """A fresh detector calibration window for each test, closed afterwards (see `tests/ui/conftest.py`). Built
    after export_settings has redirected the remembered export folder (#1087)."""
    model = DetectorCalibrationModel(pyrscore.PyRsCore())
    ctrl = DetectorCalibrationCrtl(model)
    window = DetectorCalibrationViewer(model, ctrl)
    qtbot.addWidget(window)
    return window, qtbot


def handle_export_dialog(folder):
    """Fill in and accept the Export Calibration dialog for a folder copy (its default, #1087)."""
    dialog = QtWidgets.QApplication.activeModalWidget()
    assert dialog.folder_box.isChecked()
    assert not any(box.isChecked() for box in (dialog.archive_box, dialog.latest_mono_box, dialog.latest_box))
    dialog.folder_lineEdit.setText(folder)
    dialog.accept()


def test_detector_calibration(calibration_window, tmp_path, export_settings):
    window, qtbot = calibration_window
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()
    # the ring-based calibration is the default (#1038), its recipe one click away
    setup = window.peak_lines_setup
    assert setup.method_combo.currentText() == "Ring-based" and window.model.method == "Ring-based"
    assert setup.recipe_combos[0].currentText() == "iterate"

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
    # a recipe saved before the method choice (#1045) is FitCalibration's, despite the ring-based default
    assert setup.method_combo.currentText() == "FitCalibration" and window.model.method == "FitCalibration"
    qtbot.wait(wait)

    qtbot.mouseClick(window.peak_lines_setup.fit, QtCore.Qt.LeftButton)
    qtbot.wait(wait)

    # the recipe's max nfev (3) stops the fit early (status 0), and Export refuses an unconverged
    # calibration; give it enough evaluations to converge
    window.calib_summary.neval_lineEdit.setText("100")
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

    # Export: a folder copy, the dialog's default (#1087)
    QtCore.QTimer.singleShot(300, functools.partial(handle_export_dialog, str(calib_folder)))
    qtbot.mouseClick(window.calib_summary.export_calib_bttn, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    assert not hasattr(window.calib_summary, "export_local_calib_bttn")  # Export replaces it (#1045)
    # the JSON and, ticked by default, its NeXus copy with diagnostics (#1098); the NeXus copy is loaded back below
    json_path, calib_path = sorted(str(path) for path in calib_folder.iterdir())
    assert json_path.endswith(".json") and calib_path == json_path[: -len(".json")] + ".nxs"
    # the folder is remembered for the next export, in the scratch settings
    from pyrs.interface.detector_calibration.detector_calibration_viewer import EXPORT_FOLDER_KEY

    assert QtCore.QSettings(export_settings, QtCore.QSettings.IniFormat).value(EXPORT_FOLDER_KEY) == str(calib_folder)

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
    contours = [c for c in detector_view._myCanvas.axes.collections if isinstance(c, ContourSet)]
    assert contours == [detector_view._ring_overlay]  # the previous sub-run's rings were removed
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


def test_ring_based_calibration_uses_driver_defaults(calibration_window, tmp_path):
    """#1045: the window's ring-based calibration runs with what the pyrs-pixel-calibration driver uses"""
    import json

    window, qtbot = calibration_window
    window.show()
    nexus = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
    window._nexus_file = nexus
    window.fileLoading.file_load_dilg.load_nexus()
    qtbot.wait(wait)
    setup = window.peak_lines_setup

    # Arrange - the driver's settings, set explicitly so the checks below test the window's wiring
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
    assert setup.rounds_lineEdit.text() == "10" and setup.rounds_lineEdit.isEnabled()  # #1038: until converged
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
    assert [marks.text for marks in detector_view._point_marks] == [accepted.text, rejected.text]
    assert detector_view._point_artists
    axes = detector_view._myCanvas.axes
    x, y = axes.transData.transform((accepted.index_0[0] + 0.5, accepted.index_1[0] + 0.5))
    assert detector_view.mark_text_at(x, y) == accepted.text[0]
    diffraction.show_points.setChecked(False)
    diffraction.show_rejected.setChecked(False)
    qtbot.wait(plot_wait)
    assert not detector_view._point_artists
    folder = tmp_path / "ring"
    folder.mkdir()
    written = window.calib_summary.write_calibration(str(folder))

    # Assert - the window ran with what the driver uses (#1101): its command-line defaults (read from its
    # parser) and PixelCalibration's defaults, which the driver does not override. This no longer runs the
    # driver: the bit-for-bit window = driver comparison is test_ring_path_equals_the_driver, at the model
    # level on synthetic data. This test checks the viewer's wiring to the model on real data.
    from pyrs.calibration.pixel_refinement import PixelCalibration
    from pyrs.interface.detector_calibration import detector_calibration_model

    from pyrs.calibration import ring_driver as driver

    defaults = driver.build_parser().parse_args([nexus, "-o", "calibration.json", "--no-mask"])
    refinement_defaults = inspect.signature(PixelCalibration).parameters
    gui_provenance = json.load(open(written[0]))["provenance"]
    inputs = gui_provenance["inputs"]
    assert inputs["extraction"]["strips"] == defaults.strips
    assert inputs["recipes"][-1][:1] == ["iterate"]
    assert detector_calibration_model.ITERATE_PASSES == defaults.passes
    assert inputs["refinement"]["method"] == refinement_defaults["method"].default
    # max nfev: the window's 300 reached the refinement. The driver sets none; on HB2B_3510 the cap does not
    # bind (the files were bit-identical while this test still ran the driver)
    assert inputs["refinement"]["max_nfev"] == 300
    # every sub-run kept, from the nominal geometry at the run's wavelength, as the driver starts
    all_sub_runs = [int(s) for s in window.model._calibration_obj._hidra_ws.get_sub_runs()]
    assert sorted(inputs["extraction"]["sub_runs"] or all_sub_runs) == all_sub_runs
    first_start = gui_provenance["extraction_rounds"][0]["start"]["values"]  # later rounds start where one ended
    assert first_start == pytest.approx([0.0] * 7 + [window.controller.get_wavelength()])
    # iterate ends with its joint polish (no --no-polish by default), then the error estimate
    assert not defaults.no_polish
    routines = [step["routine"] for step in gui_provenance["steps"]]
    assert routines[-2:] == ["FullCalibration", "estimate_errors"]
    # the rings were extracted again until converged, with the driver's round limit
    assert window.model.max_rounds == defaults.max_rounds
    assert 2 <= len(gui_provenance["extraction_rounds"]) <= defaults.max_rounds

    setup.method_combo.setCurrentText("FitCalibration")
    assert not setup.rounds_lineEdit.isEnabled()
    assert "iterate" not in [setup.recipe_combos[0].itemText(i) for i in range(setup.recipe_combos[0].count())]
    # ring marks belong to the ring-based method: switching away disables and clears them
    diffraction.show_points.setChecked(True)
    qtbot.wait(plot_wait)
    assert not diffraction.show_points.isEnabled() and not detector_view._point_artists
    diffraction.show_points.setChecked(False)
    window.hide()


def test_recipe_keeps_method_and_strips(calibration_window, tmp_path, monkeypatch):
    """#1045: a saved recipe brings back the method, the strips, the rounds (#1038) and an "iterate" step"""
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    window, qtbot = calibration_window
    setup = window.peak_lines_setup
    recipe = str(tmp_path / "ring_recipe.json")
    monkeypatch.setattr(viewer_module, "save_json_file", lambda parent: recipe)
    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (recipe, ""))

    # Arrange - data loaded, then Ring-based, 3 strips, two steps (saving kept only the last step
    # before). `load_json` needs the saved "nexus_file", and `save_json` writes it only when data
    # are loaded; a recipe saved without data cannot be loaded back (recorded, not fixed here).
    window._nexus_file = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
    window.fileLoading.file_load_dilg.load_nexus()
    setup.method_combo.setCurrentText("Ring-based")
    setup.strips_lineEdit.setText("3")
    setup.rounds_lineEdit.setText("4")
    for i_row in range(8):
        setup.recipe_combos[i_row].setCurrentIndex(0)
    setup.recipe_combos[0].setCurrentText("shifts")
    setup.recipe_combos[1].setCurrentText("iterate")

    # Act - save, change everything, load
    setup.save_json()
    setup.method_combo.setCurrentText("FitCalibration")
    setup.strips_lineEdit.setText("5")
    setup.rounds_lineEdit.setText("10")
    setup.load_json()
    qtbot.wait(wait)

    # Assert
    assert (setup.method_combo.currentText(), setup.strips_lineEdit.text()) == ("Ring-based", "3")
    assert setup.rounds_lineEdit.text() == "4"
    assert [setup.recipe_combos[i].currentText() for i in range(3)] == ["shifts", "iterate", ""]
    assert json.load(open(recipe))["recipe"] == "shifts,iterate"

    # a recipe from before #1045 (no "calibration_method") with "iterate" selects the ring-based method
    saved = json.load(open(recipe))
    del saved["calibration_method"]
    json.dump(saved, open(recipe, "w"))
    setup.method_combo.setCurrentText("FitCalibration")
    setup.load_json()
    qtbot.wait(wait)
    assert setup.method_combo.currentText() == "Ring-based" and setup.recipe_combos[1].currentText() == "iterate"

    # ... and without "iterate" it is FitCalibration's, even with the ring-based method selected (#1038)
    saved["recipe"] = "shifts,rotations"
    json.dump(saved, open(recipe, "w"))
    setup.load_json()
    qtbot.wait(wait)
    assert setup.method_combo.currentText() == "FitCalibration"
    assert [setup.recipe_combos[i].currentText() for i in range(2)] == ["shifts", "rotations"]

    # an empty ring-based recipe starts with "iterate"; a recipe is never changed by switching method
    for i_row in range(8):
        setup.recipe_combos[i_row].setCurrentIndex(0)
    setup.method_combo.setCurrentText("Ring-based")
    assert setup.recipe_combos[0].currentText() == "iterate"
    setup.method_combo.setCurrentText("FitCalibration")
    setup.recipe_combos[0].setCurrentText("shifts")
    setup.method_combo.setCurrentText("Ring-based")
    assert [setup.recipe_combos[i].currentText() for i in range(2)] == ["shifts", ""]


def test_export_failure_is_reported(calibration_window, tmp_path, monkeypatch):
    """Export to a missing folder writes nothing and tells the user"""
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    window, _ = calibration_window
    messages = []
    monkeypatch.setattr(
        viewer_module,
        "pop_message",
        lambda parent, message, detail=None, kind="error": messages.append((message, kind)),
    )

    # Act
    written = window.calib_summary.write_calibration(str(tmp_path / "nowhere"))

    # Assert
    assert written is None and messages == [("Export failed", "error")]


def test_export_dialog_rules(qtbot):
    """#1087: folder and shared exclusive, a latest file locks the archive, OK needs a target"""
    from pyrs.interface.detector_calibration.detector_calibration_viewer import ExportCalibDialog

    # Arrange
    dialog = ExportCalibDialog(None, folder="/tmp", mono="Si422", description=(True, "ring"))
    ok = dialog.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)

    # Assert - opens with only the folder ticked, ready to export
    assert dialog.folder_box.isChecked() and ok.isEnabled()
    # and with the NeXus copy ticked (#1098)
    assert dialog.nexus_box.isChecked()
    assert dialog.targets() == {
        "folder": "/tmp",
        "latest": False,
        "latest_mono": False,
        "archive": False,
        "nexus": True,
    }

    # a shared target unticks and disables the folder
    dialog.archive_box.setChecked(True)
    assert not dialog.folder_box.isChecked() and not dialog.folder_box.isEnabled()
    assert not dialog.folder_lineEdit.isEnabled()

    # a latest file ticks and locks the archive
    dialog.archive_box.setChecked(False)
    dialog.latest_box.setChecked(True)
    assert dialog.archive_box.isChecked() and not dialog.archive_box.isEnabled()
    assert dialog.targets() == {"folder": None, "latest": True, "latest_mono": False, "archive": True, "nexus": True}
    dialog.latest_mono_box.setChecked(True)
    dialog.latest_box.setChecked(False)
    assert dialog.archive_box.isChecked() and not dialog.archive_box.isEnabled()  # still a latest file

    # unticking every latest file unlocks the archive; unticking it re-enables the folder
    dialog.latest_mono_box.setChecked(False)
    assert dialog.archive_box.isEnabled()
    dialog.archive_box.setChecked(False)
    assert dialog.folder_box.isEnabled()

    # nothing ticked: OK disabled; the folder ticked but empty: OK disabled
    assert not ok.isEnabled()
    dialog.folder_box.setChecked(True)
    dialog.folder_lineEdit.setText("")
    assert not ok.isEnabled()
    dialog.folder_lineEdit.setText("/tmp")
    assert ok.isEnabled()

    # a calibration that cannot be exported keeps OK disabled whatever is ticked
    refused = ExportCalibDialog(None, folder="/tmp", mono="Si422", description=(False, "not refined"))
    assert not refused.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()


def test_removed_export_folder_falls_back(tmp_path):
    """#1087: a remembered folder that has since been removed opens the current folder instead"""
    from pyrs.interface.detector_calibration.detector_calibration_viewer import _remembered_export_folder

    # Arrange
    gone = tmp_path / "gone"

    # Act / Assert
    assert _remembered_export_folder(str(tmp_path)) == str(tmp_path)
    assert _remembered_export_folder(str(gone)) == os.getcwd()
    assert _remembered_export_folder(None) == os.getcwd()


@pytest.mark.parametrize("dialog", ["save", "load"])
def test_recipe_dialogs_filter_json(calibration_window, monkeypatch, dialog):
    """The recipe dialogs pass their filter as the filter, not as the starting folder (DD-26)."""
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    # Arrange - record what the dialog is given; cancel it
    window, _ = calibration_window
    calls = []

    def record(parent, caption, directory="", file_filter="", **kwargs):
        calls.append((directory, file_filter))
        return "", ""

    name = "getSaveFileName" if dialog == "save" else "getOpenFileName"
    monkeypatch.setattr(viewer_module.QFileDialog, name, record)

    # Act
    if dialog == "save":
        viewer_module.save_json_file(window)
    else:
        window.peak_lines_setup.load_json()

    # Assert - one dialog, with no starting folder or file name, and a first (default) filter for .json files
    ((directory, file_filter),) = calls
    assert directory == ""
    assert "*.json" in file_filter.split(";;")[0]


def test_load_calibration_offers_and_reads_nexus(calibration_window, monkeypatch, tmp_path):
    """Load Calibration offers JSON and NeXus calibrations, and a .nxs start fills the Starting column (#1098)"""
    import numpy as np

    from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module
    from pyrs.utilities.calibration_nexus import write_calibration_nexus

    # Arrange - a NeXus calibration file, chosen in the dialog
    window, _ = calibration_window
    path = tmp_path / "start.nxs"
    shifts = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.0)
    geometry = DENEXDetectorGeometry(1024, 1024, 0.3 / 1024, 0.3 / 1024, 0.985, False)
    write_calibration_nexus(str(path), shifts, DENEXDetectorShift(*[0.01] * 7), 1.53, 1e-4, 1, geometry)
    filters = []

    def choose(parent, caption, directory="", file_filter="", **kwargs):
        filters.append(file_filter)
        return [str(path)], file_filter

    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileNames", choose)

    # Act
    window.fileLoading.file_load_calib.openCalibFileDialog()

    # Assert - the default filter shows both formats; the table holds the file's values
    (file_filter,) = filters
    assert "*.json *.nxs" in file_filter.split(";;")[0]
    table = window.calib_summary._cal_summary_table
    values = [float(table.item(row, 0).text()) for row in range(8)]
    np.testing.assert_allclose(values, [0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.0, 1.53])


def _load_good_start(window, monkeypatch, tmp_path):
    """Load a complete start (Lambda 1.53) through Load Calibration; returns its path and the table column after"""
    from pyrs.core.instrument_geometry import DENEXDetectorShift
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module
    from pyrs.utilities.calibration_file_io import write_calibration_to_json

    good = tmp_path / "good.json"
    write_calibration_to_json(
        DENEXDetectorShift(0.002, 0.001, -0.003, 0.2, 0.1, -0.1, 0.0),
        DENEXDetectorShift(*[0.01] * 7),
        1.53,
        1e-4,
        1,
        str(good),
    )
    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(good)], ""))
    window.fileLoading.file_load_calib.openCalibFileDialog()
    table = window.calib_summary._cal_summary_table
    return str(good), [table.item(row, 0).text() for row in range(8)]


@pytest.mark.parametrize("kind", ["missing", "not-a-calibration", "null-value"])
def test_unreadable_calibration_changes_nothing(calibration_window, monkeypatch, tmp_path, kind):
    """A start that cannot be read is reported and changes neither the table nor the remembered file (#1098)"""
    import h5py

    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    # Arrange
    window, _ = calibration_window
    path = tmp_path / "start.nxs"
    if kind == "not-a-calibration":
        with h5py.File(path, "w") as other:
            other.create_group("entry")
    elif kind == "null-value":
        path = tmp_path / "start.json"
        names = ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "TTH_0", "Lambda")
        stored = {name: 0.0 for name in names}
        stored.update({f"error_{name}": -1.0 for name in names}, Status=1, TTH_0=None)
        path.write_text(__import__("json").dumps(stored))
    table = window.calib_summary._cal_summary_table
    good, before = _load_good_start(window, monkeypatch, tmp_path)  # a good start the bad one must not disturb
    messages, asked = [], []
    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(path)], ""))
    monkeypatch.setattr(viewer_module, "pop_message", lambda parent, title, *args: messages.append(title))
    monkeypatch.setattr(viewer_module.QMessageBox, "question", lambda *args: asked.append(args))

    # Act
    window.fileLoading.file_load_calib.openCalibFileDialog()

    # Assert - reported, not asked about; the earlier start is still the table's and the remembered file
    assert messages == ["Cannot load the calibration"]
    assert asked == []
    assert [table.item(row, 0).text() for row in range(8)] == before
    assert window._calibration_input == good


@pytest.mark.parametrize(
    "written, title",
    [(["cal.json", "cal.nxs"], "NeXus copy without diagnostics"), (["cal.json"], "NeXus copy missing")],
)
def test_nexus_warning_says_what_happened(calibration_window, monkeypatch, written, title):
    """The warning tells a NeXus copy written without diagnostics from one not written (#1098 review)"""
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module

    # Arrange
    window, _ = calibration_window
    messages = []
    monkeypatch.setattr(window.controller, "export_calibration", lambda *args, **kwargs: written)
    monkeypatch.setattr(
        viewer_module, "pop_message", lambda parent, message, detail, kind: messages.append((message, detail, kind))
    )
    window.model.nexus_problem = "something went wrong"

    # Act
    window.calib_summary.write_calibration("folder", nexus=True)

    # Assert - the title, the reason as the detail, a warning (not an error); the status bar says it too
    assert messages == [(title, "something went wrong", "warning")]
    assert "something went wrong" in window.statusBar().currentMessage()


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_incomplete_start_asks_to_use_zero(calibration_window, monkeypatch, tmp_path, answer):
    """A start JSON without Lambda: the window says so and asks whether to continue with 0 (#1098)"""
    import json

    from pyrs.core.instrument_geometry import DENEXDetectorShift
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module
    from pyrs.utilities.calibration_file_io import write_calibration_to_json

    # Arrange
    window, _ = calibration_window
    path = tmp_path / "start.json"
    write_calibration_to_json(
        DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.0),
        DENEXDetectorShift(*[0.01] * 7),
        1.5,
        1e-4,
        1,
        str(path),
    )
    stored = json.loads(path.read_text())
    del stored["Lambda"], stored["error_Lambda"]
    path.write_text(json.dumps(stored))
    table = window.calib_summary._cal_summary_table
    good, before = _load_good_start(window, monkeypatch, tmp_path)  # Lambda 1.53, so a written 0 shows
    asked, messages = [], []
    reply = viewer_module.QMessageBox.Yes if answer == "yes" else viewer_module.QMessageBox.No
    monkeypatch.setattr(viewer_module, "pop_message", lambda parent, title, *args: messages.append(title))

    def ask(parent, title, text, *args):
        asked.append(text)
        return reply

    monkeypatch.setattr(viewer_module.QMessageBox, "question", ask)
    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(path)], ""))

    # Act
    window.fileLoading.file_load_calib.openCalibFileDialog()

    # Assert - asked once, naming Lambda; Yes fills the table with Lambda 0, No changes nothing
    (question,) = asked
    assert "Lambda" in question
    assert "Continue with 0" in question
    assert messages == []  # asked, not reported as unreadable
    values = [table.item(row, 0).text() for row in range(8)]
    if answer == "yes":
        assert float(before[7]) == pytest.approx(1.53)
        assert float(values[7]) == 0.0
        assert float(values[0]) == pytest.approx(0.001)
        assert window._calibration_input == str(path)
    else:
        assert values == before
        assert window._calibration_input == good


@pytest.mark.parametrize("kind", ["good", "unreadable", "incomplete-declined"])
def test_recipe_start_file_is_read_like_load_calibration(calibration_window, monkeypatch, tmp_path, kind):
    """A recipe's input_calibration goes through the same reader and rules as Load Calibration (#1098)"""
    import json

    from pyrs.core.instrument_geometry import DENEXDetectorShift
    from pyrs.interface.detector_calibration import detector_calibration_viewer as viewer_module
    from pyrs.interface.detector_calibration.detector_calibration_model import METHOD_RING
    from pyrs.utilities.calibration_file_io import write_calibration_to_json

    # Arrange - a loaded run and an earlier good start; the recipe names another start file
    window, _ = calibration_window
    window._nexus_file = "HB2B_0001.nxs.h5"  # the recipe's run: not loaded again
    earlier, before = _load_good_start(window, monkeypatch, tmp_path)
    start = tmp_path / "recipe_start.json"
    if kind != "unreadable":
        write_calibration_to_json(
            DENEXDetectorShift(0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            DENEXDetectorShift(*[0.01] * 7),
            1.54,
            1e-4,
            1,
            str(start),
        )
    if kind == "incomplete-declined":
        stored = json.loads(start.read_text())
        del stored["Rot_x"], stored["error_Rot_x"]
        start.write_text(json.dumps(stored))
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {"nexus_file": "HB2B_0001.nxs.h5", "calibration_method": METHOD_RING, "input_calibration": str(start)}
        )
    )
    messages, asked = [], []
    monkeypatch.setattr(viewer_module.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(recipe), ""))
    monkeypatch.setattr(viewer_module, "pop_message", lambda parent, title, *args: messages.append(title))
    monkeypatch.setattr(
        viewer_module.QMessageBox, "question", lambda *args: asked.append(args) or viewer_module.QMessageBox.No
    )

    # Act
    window.peak_lines_setup.load_json()

    # Assert
    table = window.calib_summary._cal_summary_table
    values = [table.item(row, 0).text() for row in range(8)]
    if kind == "good":
        assert float(values[0]) == pytest.approx(0.004) and float(values[7]) == pytest.approx(1.54)
        assert window._calibration_input == str(start)
        assert messages == [] and asked == []
    else:
        assert values == before
        assert window._calibration_input == earlier
        assert (messages, len(asked)) == ((["Cannot load the calibration"], 0) if kind == "unreadable" else ([], 1))
