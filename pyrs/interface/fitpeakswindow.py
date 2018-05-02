try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_peakfitwindow
import os


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

        # TODO actionQuit

        return

    def _check_core(self):
        """
        check whether PyRs.Core has been set to this window
        :return:
        """
        if self._core is None:
            raise RuntimeError('Not set up yet!')

    def _get_default_hdf(self):
        """
        use IPTS and Exp to determine
        :return:
        """
        # TODO
        ipts_number = None
        exp_number = None

        return None

    def do_browse_hdf(self):
        """
        browse HDF file
        :return:
        """
        self._check_core()

        default_dir = self._get_default_hdf()
        if default_dir is None:
            default_dir = self._core.working_dir

        file_filter = 'HDF(*.h5);;All Files(*.*)'
        open_value = QFileDialog.getOpenFileName(self, 'HB2B Raw HDF File', default_dir, file_filter)
        print open_value

        if isinstance(open_value, tuple):
            # PyQt5
            hdf_name = str(open_value[0])
        else:
            hdf_name = str(open_value)

        if len(hdf_name) == 0:
            # use cancel
            return

        if os.path.exists(hdf_name):
            self.ui.lineEdit_expFileName.setText(hdf_name)
        else:
            # pass
            raise RuntimeError('File {0} does not exist.'.format(hdf_name))

        return

    def do_load_scans(self):
        """
        load scan's reduced files
        :return:
        """
        self._check_core()

        # get file
        rs_file_name = str(self.ui.lineEdit_expFileName.text())

        # load file
        data_key, message = self._core.load_rs_raw(rs_file_name)

        # edit information
        self.ui.label_loadedFileInfo.setText(message)

        # get the range of log indexes
        log_range = self._core.data_center.get_scan_range(data_key)
        self.ui.label_logIndexMin.setText(str(log_range[0]))
        self.ui.label_logIndexMax.setText(str(log_range[-1]))

        # get the sample logs
        sample_log_names = self._core.data_center.get_sample_logs_list(data_key)

        self.ui.comboBox_xaxisNames.clear()
        self.ui.comboBox_yaxisNames.clear()
        self.ui.comboBox_xaxisNames.addItem('Log Index')

        # TODO continue to set all the sample log names to combos!


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
