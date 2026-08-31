#!/usr/bin/python
import getpass
import os
import sys
import tempfile
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QApplication
import pyrs.interface.pyrs_main


def main(argv=None):
    """ """
    if argv is None:
        argv = sys.argv
    # QFileDialog (used by every browse/export dialog) hardcodes
    # QSettings(QSettings::UserScope, "QtProject") in its own C++ implementation to
    # persist sidebar/history state -- this is NOT influenced by
    # setOrganizationName()/setApplicationName() below. That fixed "QtProject.conf"
    # file, under XDG_CONFIG_HOME, must not live on an NFS-mounted home directory
    # (e.g. ORNL analysis nodes' /SNS/users/<user>): QLockFile's stale-lock
    # recovery depends on atomic rename()/link(), which NFS does not guarantee, so
    # a crashed/killed PyRS process leaves a lock that every subsequent launch
    # fails to clean up ("Could not remove our own lock file ...", "Device or
    # resource busy"). Redirecting XDG_CONFIG_HOME to a PyRS-private directory on
    # local disk (per-user, so it can't collide with other users on a shared
    # login/analysis node; not per-machine-persistent, since local disk is
    # node-specific -- losing dialog history when switching nodes is harmless)
    # sidesteps this entirely, since local filesystems support the atomic
    # operations QLockFile's recovery needs. Must happen before any QFileDialog is
    # constructed. See docs/ground_truths.md for how this was diagnosed.
    pyrs_config_home = os.path.join(tempfile.gettempdir(), f"pyrs-qt-config-{getpass.getuser()}")
    os.makedirs(pyrs_config_home, mode=0o700, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = pyrs_config_home
    # Kept for any future explicit QSettings/QStandardPaths use -- does not affect
    # QFileDialog's own settings, which are isolated by the XDG_CONFIG_HOME
    # redirect above instead.
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
