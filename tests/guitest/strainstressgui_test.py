#!/usr/bin/python
# In order to test the strain/stress calculation/visualization GUI (GUI)
from pyrs.core import pyrscore
import sys
import os
import pyrs.interface
from pyrs.interface import strainstresscalwindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main(test_dir):
    """
    test main
    """
    # set up window and link to call
    ss_window = strainstresscalwindow.StrainStressCalculationWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    pyrs_core.working_dir = test_dir

    ss_window.setup_window(pyrs_core)

    # show the window
    ss_window.show()

    # start a new project
    ss_window.create_new_project('test_strain_stress')

    # load files
    ss_window.load_raw_file(file_name='LD', direction='e11')
    ss_window.load_raw_file(file_name='TD', direction='e22')
    ss_window.load_raw_file(file_name='RD', direction='e33')

    # constrained stress/strain
    ss_window.align_loaded_data(e33=False, sample_resolution=(0.01, 0.01, 0.01))

    # plane strain and save
    ss_window.calcualte_plane_strain()
    ss_window.save_present_project('test_strain_stress_plane_stress.h5')

    # plain stress and save
    ss_window.calcualte_plane_stress()
    ss_window.save_present_project('test_strain_stress.h5')

    # unconstrained stress/strain
    ss_window.align_loaded_data(e33=True, sample_resolution=(0.01, 0.01, 0.01))

    # calculate and save project
    ss_window.calcualte_unconstrained_stress()
    ss_window.save_present_project('test_strain_stress.h5')

    return ss_window


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
    test_dir = '/tmp/pyrs_test_ss'
    if os.path.exists(test_dir) is False:
        os.mkdir(test_dir)
    test_window = test_main(test_dir=test_dir)

    # I cannot close it!  test_window.close()
    app.exec_()
