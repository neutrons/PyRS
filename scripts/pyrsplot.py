#!/usr/bin/python
import sys
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QApplication
import pyrs.interface.pyrs_main


def main(argv=None):
    """ """
    if argv is None:
        argv = sys.argv
    # Without an explicit org/app name, Qt falls back to a generic "QtProject"
    # identity for QSettings -- used implicitly by every QFileDialog (browse/export
    # dialogs) to persist sidebar/history state. That fallback file is shared by
    # *every* unnamed Qt/PyQt application on the machine (e.g. Mantid Workbench,
    # Qt Designer, another PyRS window), so two such apps writing to it at the same
    # time can race over its QLockFile and produce
    # 'Could not remove our own lock file ...QtProject.conf.lock.rmlock'. Giving
    # PyRS its own identity isolates its dialog state in ~/.config/PyRS/PyRS.conf.
    QCoreApplication.setOrganizationName("PyRS")
    QCoreApplication.setOrganizationDomain("ornl.gov")
    QCoreApplication.setApplicationName("PyRS")
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
