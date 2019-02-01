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

        # TODO - NIGHT - Define link to methods
        self.ui.pushButton_decreaseCenterX.clicked.connect(self.decrease_value)

        return

    def _promote_widgets(self):

        # frame_detector2DView
        # frame_multiplePlotsView
        # frame_reducedDataView

        # TODO - NIGHT - Implement UI to promote widgets - NIGHT

        # TODO - NIGHT - In UI, better name shall be given to widgets - NIGHT


        return

    def decrease_value(self):

        sender = self.sender()

        print ('[DB...BAT] Sender:  {}'.format(self.sender()))
        print ('[DB...BAT] Methods: \n'.format(dir(sender)))
