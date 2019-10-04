#!/usr/bin/python
import sys
import imp
from qtpy import QtCore
from qtpy.QtWidgets import QDialog, QApplication
import pyrs.interface.pyrs_main


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
    app = main(sys.argv)

    main_window = pyrs.interface.pyrs_main.PyRSLauncher()  # .FourCircleMainWindow()
    main_window.show()

    app.exec_()
