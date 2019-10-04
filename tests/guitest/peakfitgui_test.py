#!/usr/bin/python
# In order to test the peak fit window (GUI)
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import fitpeakswindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main():
    """
    Test main for fitting peaks
    """
    fit_window = fitpeakswindow.FitPeaksWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    fit_window.setup_window(pyrs_core)

    fit_window.show()

    # 3 candidate files
    fit_window.ui.lineEdit_expFileName.setText('tests/testdata/Hydra_16-1_cor_log.hdf5')
    fit_window.ui.comboBox_peakType.setCurrentIndex(0)  # Test pseudo voigt
    fit_window.do_load_hydra_file()
    if False:
        fit_window.do_plot_meta_data()

    if False:
        fit_window.do_fit_peaks()
        fit_window.ui.lineEdit_scanNumbers.setText('0')
        fit_window.do_plot_diff_data()
        fit_window.save_data_for_mantid(None, 'peaks.nxs')
        out_file_name = '16-1_TD.cor_Log.fit.hdf5'
        fit_window.save_fit_result(out_file_name)

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
    print('Test Peak Fit GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
