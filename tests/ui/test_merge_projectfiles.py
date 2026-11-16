from pathlib import Path
from typing import Callable, List

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface import gui_helper
from pyrs.interface.combine_runs import combine_runs_viewer
from pyrs.interface.combine_runs.combine_runs_viewer import CombineRunsViewer
from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel
from pyrs.interface.combine_runs.combine_runs_crtl import CombineRunsCrtl

from qtpy import QtCore, QtWidgets
import functools
import os
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.integration]

wait = 500
plot_wait = 100


@pytest.fixture
def combine_projects_window(qtbot):
    """A fresh combine-projects window for each test, closed afterwards (see `tests/ui/conftest.py`)."""
    model = CombineRunsModel()
    ctrl = CombineRunsCrtl(model)
    window = CombineRunsViewer(model, ctrl)
    qtbot.addWidget(window)
    return window, qtbot


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


def test_save_file_dialog_nxstress_writes_valid_nxs_file(
    combine_projects_window: tuple, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
) -> None:
    r"""
    Exercises saveFileDialogNXstress()'s own mechanics (dialog -> extension
    imposition -> export_project_files(".nxs")) directly, using a synthetic
    in-memory workspace with instrument geometry -- NXstress requires geometry
    unconditionally, and (as found while implementing this spec)
    tests/data/HB2B_1327.h5 (the real file this suite's other tests combine)
    genuinely has none. Data-correctness of the .nxs round trip itself is
    already covered at the model level by
    tests/integration/test_nxstress_viewer_roundtrip.py's CombineRuns tests.
    """
    window, qtbot = combine_projects_window
    export_file = "test_export.nxs"

    window.model._hidra_ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)
    window._project_files = ["placeholder"]

    try:
        window.show()
        qtbot.wait(wait)

        def handle_dialog(text: str) -> None:
            dialog = window.findChild(QtWidgets.QFileDialog)
            lineEdit = dialog.findChild(QtWidgets.QLineEdit)
            qtbot.keyClicks(lineEdit, text)
            qtbot.wait(wait)
            qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)  # type: ignore[attr-defined]
            qtbot.wait(wait)

        QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, export_file))  # type: ignore[attr-defined]
        window.fileLoading.file_load_dilg.saveFileDialogNXstress()
        qtbot.wait(wait)

        assert os.path.exists(export_file)
    finally:
        window.hide()
        if os.path.exists(export_file):
            os.remove(export_file)


def test_load_project_files_auto_prompt_calls_both_save_dialogs_in_order(
    combine_projects_window: tuple, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
) -> None:
    r"""
    Verify load_project_files() sequences the .h5 then the NXstress auto-prompt
    dialog, each independently gated by its own config flag, without driving two
    full modal dialogs (which would be fragile to stage in sequence) -- spies on
    the two save-dialog methods instead. combine_project_files() is stubbed out
    (rather than combining real files) so this test's outcome only depends on
    the sequencing logic itself, not on whether some particular real project
    file happens to carry instrument geometry.
    """
    window, qtbot = combine_projects_window
    file_load = window.fileLoading.file_load_dilg

    call_order: List[str] = []
    file_load.saveFileDialog = lambda: call_order.append("h5")
    file_load.saveFileDialogNXstress = lambda: call_order.append("nxstress")
    file_load._auto_prompt_export = True

    original_load_combine_projects = window.controller.load_combine_projects

    def fake_load_combine_projects(project_files: List[str]) -> None:
        window.model._hidra_ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        # no warnings: a combine's warning would open a modal dialog
        window.model.warnings = []

    window.controller.load_combine_projects = fake_load_combine_projects

    try:
        window._project_files = ["placeholder"]
        file_load.load_project_files()
    finally:
        del file_load.saveFileDialog
        del file_load.saveFileDialogNXstress
        file_load._auto_prompt_export = False
        window.controller.load_combine_projects = original_load_combine_projects

    assert call_order == ["h5", "nxstress"]


def _refused_export_setup(window, *, minimal_HidraWorkspace, monkeypatch, chosen: Path) -> List[tuple]:
    """The window holding a workspace with geometry, its save dialogs
    answering `chosen`, and message boxes recorded instead of shown."""
    messages: List[tuple] = []

    def record(parent, message, detail="", kind="info"):
        messages.append((message, kind))

    monkeypatch.setattr(combine_runs_viewer, "pop_message", record)
    monkeypatch.setattr(gui_helper, "pop_message", record)
    monkeypatch.setattr(
        combine_runs_viewer.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(chosen), ""))
    )
    monkeypatch.setattr(
        window.model, "_hidra_ws", minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)
    )
    monkeypatch.setattr(window, "_project_files", ["placeholder"])
    return messages


def test_save_file_dialog_reports_a_failed_legacy_export(
    combine_projects_window: tuple, minimal_HidraWorkspace: Callable[..., HidraWorkspace], monkeypatch, tmp_path
) -> None:
    r"""
    A legacy export that cannot be written -- here, into a directory that does
    not exist -- is reported, and does not escape the Qt slot, where an uncaught
    exception aborts PyRS. Before this PR, every legacy export failure did.
    """
    # Arrange
    window, _qtbot = combine_projects_window
    chosen = tmp_path / "missing" / "out.h5"
    messages = _refused_export_setup(
        window, minimal_HidraWorkspace=minimal_HidraWorkspace, monkeypatch=monkeypatch, chosen=chosen
    )

    # Act
    window.fileLoading.file_load_dilg.saveFileDialog()

    # Assert
    assert messages == [("Failed to save 'out.h5'", "error")]
    assert not chosen.exists()


def test_save_file_dialog_nxstress_failure_keeps_existing_file(
    combine_projects_window: tuple, minimal_HidraWorkspace: Callable[..., HidraWorkspace], monkeypatch, tmp_path
) -> None:
    r"""
    An NXstress export goes through `write_nxstress_or_report`: a failed write is
    reported, and an existing file at the destination is left untouched.
    """
    # Arrange
    window, _qtbot = combine_projects_window
    chosen = tmp_path / "out.nxs"
    chosen.write_text("keep")
    messages = _refused_export_setup(
        window, minimal_HidraWorkspace=minimal_HidraWorkspace, monkeypatch=monkeypatch, chosen=chosen
    )

    def refuse(fileout: str) -> None:
        Path(fileout).write_text("trunc")
        raise ValueError("refused by the writer")

    monkeypatch.setattr(window.controller, "export_combined_projectfile", refuse)

    # Act
    window.fileLoading.file_load_dilg.saveFileDialogNXstress()

    # Assert
    assert messages == [("Failed to save 'out.nxs'", "error")]
    assert chosen.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nxs"]
