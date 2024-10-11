from pyrs.interface.combine_runs.combine_runs_viewer import CombineRunsViewer
from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel
from pyrs.interface.combine_runs.combine_runs_crtl import CombineRunsCrtl

from qtpy import QtCore, QtWidgets
import functools
import pytest

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

    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    files_list = '"tests/data/HB2B_1327.h5", '\
        '"tests/data/HB2B_1328.h5", '\
        '"tests/data/HB2B_1331.h5", '\
        '"tests/data/HB2B_1332.h5"'

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

    QtCore.QTimer.singleShot(300, functools.partial(handle_dialog, 'test_export.h5'))
    window.fileLoading.file_load_dilg.saveFileDialog()

    window.hide()
