try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_manualreductionwindow
from pyrs.core.pyrscore import PyRsCore
from pyrs.utilities import hb2b_utilities
import os
import gui_helper
import numpy


class ManualReductionWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(ManualReductionWindow, self).__init__(parent)

        # class variables
        self._core = None
        self._currIPTSNumber = None

        # set up UI
        self.ui = ui.ui_manualreductionwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        # set up the event handling
        self.ui.pushButton_setIPTSNumber.clicked.connect(self.do_set_ipts_exp)
        self.ui.pushButton_batchReduction.clicked.connect(self.do_reduce_batch_runs)


        return

    def setup_window(self, pyrs_core):
        """
        set up the manual reduction window from its parent
        :param pyrs_core:
        :return:
        """
        # check
        assert isinstance(pyrs_core, PyRsCore), 'Controller core {0} must be a PyRSCore instance but not a {1}.' \
                                                ''.format(pyrs_core, pyrs_core.__class__.__name__)

        self._core = pyrs_core

        return

    def do_reduce_batch_runs(self):
        """
        (simply) reduce a list of runs in same experiment in a batch
        :return:
        """
        ... ...


    def do_set_ipts_exp(self):
        """
        set IPTS number
        :return:
        """
        try:
            ipts_number = gui_helper.parse_integer(str(self.ui.lineEdit_iptsNumber.text()))
            exp_number = gui_helper.parse_integer(str(self.ui.lineEdit_expNumberToSlice.text()))
            self._currIPTSNumber = ipts_number
            self._core.set_ipts_number(ipts_number)
        except RuntimeError:
            gui_helper.pop_message(self, 'IPTS number shall be set to an integer.', message_type='error')

        return

