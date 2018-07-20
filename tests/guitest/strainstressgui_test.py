#!/usr/bin/python
# In order to test the strain/stress calculation/visualization GUI (GUI)
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import strainstresscalwindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main():
    """
    test main
    """
    fit_window = fitpeakswindow.FitPeaksWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    fit_window.setup_window(pyrs_core)

    fit_window.show()

    fit_window.ui.lineEdit_expFileName.setText('tests/testdata/BD_Data_Log.hdf5')
    fit_window.ui.comboBox_peakType.setCurrentIndex(1)
    fit_window.do_load_scans()
    if False:
        fit_window.ui.lineEdit_scanNUmbers.setText('4')
        fit_window.do_plot_diff_data()
        fit_window.do_plot_meta_data()
        fit_window.do_fit_peaks()
    if True:
        fit_window.do_fit_peaks()
        fit_window.save_data_for_mantid(None, 'peaks.nxs')

    return fit_window


def main(argv):
    """
    """
    if QApplication.instance():
        _app = QApplication.instance()
    else:
        _app = QApplication(sys.argv)
    return _app


if __name__ == '__main__':
    # Main application
    print ('Test Peak Fit GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
