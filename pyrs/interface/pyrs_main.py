try:
    from PyQt5.QtWidgets import QMainWindow
except ImportError:
    from PyQt4.QtGui import QMainWindow
from ui import ui_pyrsmain as ui_pyrsmain
from pyrs.core import pyrscore
import fitpeakswindow


class PyRSLauncher(QMainWindow):
    """
    blabla
    """
    def __init__(self):
        """
        blabla
        """
        super(PyRSLauncher, self).__init__(None)

        # core
        self._reduction_core = pyrscore.PyRsCore()

        # blabla
        self.ui = ui_pyrsmain.Ui_MainWindow()
        self.ui.setupUi(self)

        # define
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_launch_fit_peak_window)

        # TODO actionQuit

        # child windows
        self.peak_fit_window = None

        return
    
    @property
    def core(self):
        return self._reduction_core

    def do_launch_fit_peak_window(self):
        """

        :return:
        """
        self.peak_fit_window = fitpeakswindow.FitPeaksWindow(self)
        self.peak_fit_window.setup_window(self._reduction_core)
        self.peak_fit_window.show()

        return


