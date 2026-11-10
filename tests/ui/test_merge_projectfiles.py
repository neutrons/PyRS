from pyrs.interface.combine_runs import combine_runs_viewer
from pyrs.interface.combine_runs.combine_runs_viewer import CombineRunsViewer
from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel
from pyrs.interface.combine_runs.combine_runs_crtl import CombineRunsCrtl

from qtpy import QtCore, QtWidgets
import functools
import os
import shutil
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 500
plot_wait = 100


@pytest.fixture(scope="session")
def combine_projects_window(my_qtbot):
    r"""
    Fixture for the detector calibration window. Creating the window with a session scope and reusing it for all tests.
    This is done to avoid the segmentation fault error that occurs when the window is created with a function scope.
    """
    model = CombineRunsModel()
    ctrl = CombineRunsCrtl(model)
    window = CombineRunsViewer(model, ctrl)
    return window, my_qtbot


def test_merged_projectfile_viewer(combine_projects_window, monkeypatch):
    window, qtbot = combine_projects_window
    export_file = "test_export.h5"
    # the test files are historic (no calibration record): combining them flags it in a modal dialog
    # (#1063), which would block the test; record the messages instead
    messages = []
    monkeypatch.setattr(
        combine_runs_viewer,
        "pop_message",
        lambda parent, message, detail="", kind="info": messages.append((message, kind)),
    )

    try:
        window.show()
        qtbot.wait(wait)

        assert window.isVisible()

        files_list = '"tests/data/HB2B_1327.h5", "tests/data/HB2B_1328.h5", "tests/data/HB2B_1331.h5", "tests/data/HB2B_1332.h5"'

        # This is to handle modal dialogs
        def handle_dialog(text):
            dialog = window.findChild(QtWidgets.QFileDialog)
            print(type(dialog))
            # get a File Name field
            lineEdit = dialog.findChild(QtWidgets.QLineEdit)
            # Type in file to load and press enter
            qtbot.keyClicks(lineEdit, text)
            qtbot.wait(wait)
            qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)
            qtbot.wait(wait)

        window.fileLoading.file_load_dilg._auto_prompt_export = False

        QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, files_list))
        qtbot.mouseClick(window.fileLoading.file_load_dilg.browse_button, QtCore.Qt.LeftButton)

        qtbot.wait(wait)
        assert window.model._hidra_ws.get_sub_runs().size == 362
        assert messages == [("Reduction not fully checked", "warning")]
        qtbot.wait(wait)

        QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, export_file))
        window.fileLoading.file_load_dilg.saveFileDialog()
    finally:
        window.hide()
        if os.path.exists(export_file):
            os.remove(export_file)


def _run_number_loader(window, monkeypatch, ipts):
    """The run-number loader, with the IPTS lookup pointed at ``ipts`` and pop-ups recorded."""
    messages = []
    monkeypatch.setattr(
        combine_runs_viewer,
        "pop_message",
        lambda parent, message, detail="", kind="info": messages.append((message, detail, kind)),
    )
    monkeypatch.setattr("pyrs.utilities.file_util.get_ipts_dir", lambda run_number: ipts)
    loader = window.fileLoading.file_load_run_number
    monkeypatch.setattr(loader, "_auto_prompt_export", False)  # no blocking save dialog
    return loader, messages


def test_load_missing_run_number_reports(combine_projects_window, monkeypatch, tmp_path):
    """A run that was never reduced shows the lookup's message; nothing is combined (#931)."""
    # Arrange
    window, _ = combine_projects_window
    (tmp_path / "shared" / "manualreduce").mkdir(parents=True)
    loader, messages = _run_number_loader(window, monkeypatch, tmp_path)
    window._project_files = None
    loader.lineEdit.setText("1327")

    # Act
    loader.loadRunNumbers()

    # Assert
    assert len(messages) == 1
    _, detail, kind = messages[0]
    assert kind == "error" and "HB2B_1327.h5" in detail and "reduced yet" in detail
    assert window._project_files is None


def test_combine_by_run_number(combine_projects_window, monkeypatch, tmp_path):
    """Run numbers and a range load the runs' files and combine them (#931: this never worked)."""
    # Arrange
    window, _ = combine_projects_window
    manualreduce = tmp_path / "shared" / "manualreduce"
    manualreduce.mkdir(parents=True)
    for run in (1327, 1328, 1331, 1332):
        shutil.copy(f"tests/data/HB2B_{run}.h5", manualreduce)
    loader, messages = _run_number_loader(window, monkeypatch, tmp_path)
    loader.lineEdit.setText("1327:1328, 1331, 1332")

    # Act
    loader.loadRunNumbers()

    # Assert
    assert all(kind != "error" for _, _, kind in messages)
    assert window.model._hidra_ws.get_sub_runs().size == 362


def test_single_run_does_not_offer_export(combine_projects_window, monkeypatch, tmp_path):
    """One run combines nothing: the user is told, and no export dialog is offered."""
    # Arrange
    window, _ = combine_projects_window
    manualreduce = tmp_path / "shared" / "manualreduce"
    manualreduce.mkdir(parents=True)
    shutil.copy("tests/data/HB2B_1327.h5", manualreduce)
    loader, messages = _run_number_loader(window, monkeypatch, tmp_path)
    monkeypatch.setattr(loader, "_auto_prompt_export", True)
    exports = []
    monkeypatch.setattr(loader, "saveFileDialog", lambda: exports.append(True))
    loader.lineEdit.setText("1327")

    # Act
    loader.loadRunNumbers()

    # Assert
    assert [(message, kind) for message, _, kind in messages] == [("Select at least two runs to combine", "error")]
    assert exports == []
    assert window._project_files is None


def test_browse_cancel_does_nothing(combine_projects_window, monkeypatch):
    """Cancelling the Browse dialog (an empty list, not None) shows nothing and loads nothing (#931)."""
    # Arrange
    window, _ = combine_projects_window
    messages = []
    monkeypatch.setattr(combine_runs_viewer, "pop_message", lambda *args, **kwargs: messages.append(args))
    monkeypatch.setattr(combine_runs_viewer.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([], ""))
    window._project_files = ["previous.h5"]

    # Act
    window.fileLoading.file_load_dilg.openFileDialog()

    # Assert
    assert messages == []
    assert window._project_files == ["previous.h5"]
