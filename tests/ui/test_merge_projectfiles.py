from typing import Callable, List

from pyrs.core.workspaces import HidraWorkspace
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


def test_merged_projectfile_viewer(combine_projects_window):
    window, qtbot = combine_projects_window
    export_file = "test_export.h5"

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

    window.model._hidra_ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
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
        window.model._hidra_ws = minimal_HidraWorkspace(with_instrument=True)

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
