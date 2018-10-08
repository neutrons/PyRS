#!/usr/bin/python
# In order to test GUI for manual_reduction analysis
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import manualreductionwindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main():
    """
    test main
    """
    manual_reduction_window = manualreductionwindow.ManualReductionWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    manual_reduction_window.setup_window(pyrs_core)

    manual_reduction_window.show()

    return manual_reduction_window


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
    print ('Test manual_reduction Analysis GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
