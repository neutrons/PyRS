from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from qtpy import QtCore
from qtpy.QtWidgets import QMainWindow
from pyrs.utilities import load_ui

from pyrs.core import pyrscore
from pyrs.interface.peak_fitting import fitpeakswindow
from pyrs.interface.manual_reduction import manualreductionwindow

# include this try/except block to remap QString needed when using IPython
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except (AttributeError, ImportError):
    def _fromUtf8(s): return s


class PyRSLauncher(QMainWindow):
    """
    The main window launched for PyRS
    """

    def __init__(self):
        """
        initialization
        """
        super(PyRSLauncher, self).__init__(None)

        # set up UI
        # ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'pyrsmain.ui'))
        # self.ui = load_ui(ui_path, baseinstance=self)
        self.ui = load_ui('pyrsmain.ui', baseinstance=self)

        # define
        self.ui.pushButton_manualReduction.clicked.connect(self.do_launch_manual_reduction)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_launch_fit_peak_window)

        self.ui.actionQuit.triggered.connect(self.do_quit)

        # child windows
        self.peak_fit_window = None
        self.manual_reduction_window = None

    @property
    def core(self):
        """
        offer the access of the reduction core
        :return:
        """
        return self._reduction_core

    def do_launch_fit_peak_window(self):
        """
        launch peak fit window
        :return:
        """
        # core
        fit_peak_core = pyrscore.PyRsCore()

        # set up interface object
        # if self.peak_fit_window is None:
        self.peak_fit_window = fitpeakswindow.FitPeaksWindow(self, fit_peak_core=fit_peak_core)
        self.peak_fit_window.show()

        # # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.hide()

    def do_launch_manual_reduction(self):
        """
        launch manual data reduction window
        :return:
        """
        if self.manual_reduction_window is None:
            self.manual_reduction_window = manualreductionwindow.ManualReductionWindow(self)
            # self.manual_reduction_window.setup_window()

        # show
        self.manual_reduction_window.show()

        # # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.close()

    def do_quit(self):
        """
        close window
        :return:
        """
        # close all 5 child windows
        if self.peak_fit_window is not None:
            self.peak_fit_window.close()

        if self.manual_reduction_window is not None:
            self.manual_reduction_window.close()

        self.close()
