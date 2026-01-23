#!/usr/bin/python
import sys
from qtpy.QtWidgets import QApplication
import pyrs.interface.pyrs_main


def main(argv=None):
    """ """
    if argv is None:
        argv = sys.argv
    if QApplication.instance():
        _app = QApplication.instance()
    else:
        _app = QApplication(argv)
    main_window = pyrs.interface.pyrs_main.PyRSLauncher()  # .FourCircleMainWindow()
    main_window.show()

    _app.exec_()
    return _app


if __name__ == "__main__":
    # Main application
    app = main()
