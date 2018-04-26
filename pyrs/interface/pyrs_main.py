from PyQt4.QtGui import QMainWindow
from ui import ui_pyrsmain as ui_pyrsmain
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

        # blabla
        self.ui = ui_pyrsmain.Ui_MainWindow()
        self.ui.setupUi(self)

        # define
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_launch_fit_peak_window)

        return

    def do_launch_fit_peak_window(self):
        """

        :return:
        """
        print ('Launch fit peak window')

        self.peak_fit_window = fitpeakswindow.FitPeaksWindow(self)
        self.peak_fit_window.show()


        return

