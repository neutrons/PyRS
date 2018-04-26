from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_peakfitwindow


class FitPeaksWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(FitPeaksWindow, self).__init__(parent)

        # class variables
        self._core = None

        # set up UI
        self.ui = ui.ui_peakfitwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        # set up handling
        self.ui.pushButton_loadScans.clicked.connect(self.do_load_scans)

        return

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self._core is None:
            raise RuntimeError('Not set up yet!')

    def do_load_scans(self):
        """
        load scan's reduced files
        :return:
        """
        self._check_core()

        # get file name from working directory

        # blabla

        return


    def setup_window(self, pyrs_core):
        """

        :param pyrs_core:
        :return:
        """
        # check
        # blabla

        self._core = pyrs_core

        return
