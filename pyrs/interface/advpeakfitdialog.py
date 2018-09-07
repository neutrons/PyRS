try:
    from PyQt5.QtWidgets import QDialog, QFileDialog, QMainWindow
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    from PyQt4.QtGui import QDialog, QFileDialog, QMainWindow
    from PyQt4.QtCore import pyqtSignal
import pyrs.utilities.hb2b_utilities as hb2b
import os
import gui_helper
import numpy
from ui import ui_peakfitadvsetting


class SmartPeakFitControlDialog(QDialog):
    """
    GUI window for user to fit peaks
    """

    FitPeakSignal = pyqtSignal(list, name='Smart Peak Fit Signal')

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(SmartPeakFitControlDialog, self).__init__(parent)

        # set up parent window
        assert isinstance(parent, QMainWindow), 'Parent window {} must be a QMainWindow but not a {}' \
                                                ''.format(parent, type(parent))
        self._main_window = parent

        # link signal
        self.FitPeakSignal.connect(self._main_window.fit_peaks_smart)

        # set up UI
        self.ui = ui_peakfitadvsetting.Ui_Dialog()
        self.ui.setupUi(self)

        # init UI
        self._init_widgets()

        # define event handling methods
        self.ui.pushButton_smartFitPeaks.clicked.connect(self.do_smart_peaks_fitting)
        self.ui.pushButton_close.clicked.connect(self.do_close)

        return

    def _init_widgets(self):
        """
        initialize widgets
        :return:
        """
        for combo_box_i in [self.ui.comboBox_lorentzian, self.ui.comboBox_peudovoigt,
                            self.ui.comboBox_voigt, self.ui.comboBox_guassian]:
            combo_box_i.addItem('Not Used')
            for fit_order in range(1, 4+1):
                combo_box_i.addItem('{}'.format(fit_order))
            # END-FOR
        # END-FOR

        return

    def do_close(self):
        """
        close the window
        :return:
        """
        self.close()

    def do_smart_peaks_fitting(self):
        """
        fit peaks in a "smart" algorithm such that the algorithm will fit each peak by multiple peak profile
        and select the best fit
        :return:
        """
        order_peak_list = list()
        order_set = set()

        for peak_name, combo_box_i in [('Gaussian', self.ui.comboBox_guassian),
                                       ('PseudoVoigt', self.ui.comboBox_peudovoigt),
                                       ('Voigt', self.ui.comboBox_voigt),
                                        ('Lorentzian', self.ui.comboBox_lorentzian)]:
            order_i_str = str(combo_box_i.currentText())
            if order_i_str.lower() == 'not used':
                continue

            fit_order_i = int(order_i_str)

            # check whether a fit-order cannot be used twice
            if fit_order_i in order_set:
                gui_helper.pop_message(self, message='Fit order {} appear twice.'.format(fit_order_i),
                                       message_type='error')
                return
            else:
                order_set.add(fit_order_i)

            order_peak_list.append((fit_order_i, peak_name))
        # END-FOR

        # emit signal to fit
        self.FitPeakSignal.emit(order_peak_list)

        return
