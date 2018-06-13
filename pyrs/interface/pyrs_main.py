try:
    from PyQt5.QtWidgets import QMainWindow
except ImportError:
    from PyQt4.QtGui import QMainWindow
from ui import ui_pyrsmain as ui_pyrsmain
from pyrs.core import pyrscore
import fitpeakswindow


class PyRSLauncher(QMainWindow):
    """
    The main window launched for PyRS
    """
    def __init__(self):
        """
        initialization
        """
        super(PyRSLauncher, self).__init__(None)

        # core
        self._reduction_core = pyrscore.PyRsCore()

        # set up main UI widgets
        self.ui = ui_pyrsmain.Ui_MainWindow()
        self.ui.setupUi(self)

        # define
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_launch_fit_peak_window)

        self.ui.actionQuit.triggered.connect(self.do_quit)

        # child windows
        self.peak_fit_window = None

        return
    
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
        self.peak_fit_window = fitpeakswindow.FitPeaksWindow(self)
        self.peak_fit_window.setup_window(self._reduction_core)
        self.peak_fit_window.show()

        return

    def do_quit(self):
        """
        close window
        :return:
        """
        if self.peak_fit_window is not None:
            self.peak_fit_window.close()

        self.close()

        return
