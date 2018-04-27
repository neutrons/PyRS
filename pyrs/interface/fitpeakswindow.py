from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_peakfitwindow
import os
from pyrs.core import scandataio as scandataio


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
        self.ui.pushButton_loadHDF.clicked.connect(self.do_load_scans)
        self.ui.pushButton_browseHDF.clicked.connect(self.do_browse_hdf)

        return

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self._core is None:
            raise RuntimeError('Not set up yet!')

    def do_browse_hdf(self):
        """
        browse HDF file
        :return:
        """
        ipts_number = None
        exp_number = None
        scan_number = None

        default_dir = self._get_default_hdf(ipts_number, exp_number, scan_number)
        if default_dir is None:
            default_dir = self._core.get_working_dir()

        hdf_name = str(QFileDialog.getOpenFileName(self, 'HB2B Raw HDF File', default_dir, 'HDF(*.h5);;All Files(*.*)'))
        if os.path.exists(hdf_name):
            setText(hdf_name)

        return

    def do_load_scans(self):
        """
        load scan's reduced files
        :return:
        """
        scan_file = scandataio.DiffractionDataFile()
        scan_file.load_rs_file(None)

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
