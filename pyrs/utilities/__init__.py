# flake8: noqa
# type: ignore
import os
from qtpy.uic import loadUi
from pyrs.interface import designer
from .file_util import *

__all__ = ['load_ui'] + file_util.__all__


def load_ui(ui_filename, baseinstance):
    ui_filename = os.path.split(ui_filename)[-1]
    ui_path = os.path.dirname(designer.__file__)

    # get the location of the ui directory
    # this function assumes that all ui files are there
    filename = os.path.join(ui_path, ui_filename)

    return loadUi(filename, baseinstance=baseinstance)
