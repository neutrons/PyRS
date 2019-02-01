try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_sscalvizwindow
from pyrs.utilities import checkdatatypes
import pyrs.core.pyrscore
import os
import gui_helper
import numpy
import platform
import ui.ui_calibrationwindow
import dialogs
import datetime

# setup of constants
SLICE_VIEW_RESOLUTION = 0.0001


class InstrumentCalibrationWindow(QMainWindow):
    """
    GUI window to calculate strain and stress with simple visualization
    """
    def __init__(self, parent, pyrs_core):
        """
        initialization
        :param parent:
        :param pyrs_core:
        """
        super(InstrumentCalibrationWindow, self).__init__(parent)

        # check
        assert isinstance(pyrs_core, pyrs.core.pyrscore.PyRsCore), 'PyRS core {0} of type {1} must be a PyRsCore ' \
                                                                   'instance.'.format(pyrs_core, type(pyrs_core))

        self._core = pyrs_core

        # class variables for calculation (not GUI)
        self._default_dir = None

        # set up UI
        self.ui = ui.ui_calibrationwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        return
