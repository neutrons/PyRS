try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_strainstresscalwindow
from pyrs.utilities import checkdatatypes
import pyrs.core.pyrscore
import os
import gui_helper
import numpy
import platform


class StrainStressCalculationWindow(QMainWindow):
    """
    GUI window to calculate strain and stress with simple visualization
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(StrainStressCalculationWindow, self).__init__(parent)

        # class variables
        self._core = None

        # set up UI
        self.ui = ui.ui_strainstresscalwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self._init_widgets()

        # set up event handling

        # current data/states
        self._curr_data_key = None

        return

    def do_browse_ld_file(self):
        """

        :return:
        """

    def do_browse_nd_file(self):
        """

        :return:
        """

    def do_browse_td_file(self):
        """

        :return:
        """

    def do_load_peak_info_files(self):
        """
        load peak information files
        :return:
        """
        td_file_name = str(self.ui.lineEdit.text())
        ld_file_name = str(self.ui.lineEdit.text())
        nd_file_name = str(self.ui.lineEdit.text())

        data_key, message = self._core.load_stain_stress_source_file(td_data_file=td_file_name,
                                                                     nd_data_file=nd_file_name,
                                                                     ld_data_file=ld_file_name)
        if data_key is None:
            gui_helper.pop_message(self, message, message_type='error')
        else:
            self._curr_data_key = data_key

        return

    def do_load_strain_file(self):
        """

        :return:
        """

        return

    def browse_file(self, caption, filter):




