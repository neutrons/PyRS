#!/usr/bin/python
# In order to test the peak fit window (GUI)
import sys
import pyrs.interface
from pyrs.interface import fitpeakswindow
import pyrs.core
from pyrs.core import pyrscore
from PyQt4.QtGui import QApplication


def test_main():
    """
    test main
    """
    fit_window = fitpeakswindow.FitPeaksWindow(None)
    # fit_window.show()
    pyrs_core = pyrscore.PyRsCore()

    fit_window.ui.lineEdit_expFileName.setText('tests/testdata/BD_Data_Log.hdf5')

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
    test_window.show()
    # I cannot close it!  test_window.close()

    app.exec_()
