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
    ss_window.create_new_session('test_strain_stress', False, False)

    # load files
    ss_window.ui.lineEdit_e11ScanFile.setText('tests/temp/LD_Data_Log.hdf5')
    ss_window.ui.lineEdit_e22ScanFile.setText('tests/temp/BD_Data_Log.hdf5')
    ss_window.ui.lineEdit_e33ScanFile.setText('tests/temp/ND_Data_Log.hdf5')
    ss_window.do_load_strain_files()

    # pre-grid-alignment
    ss_window.ui.comboBox_sampleLogNameX.insertItem(0, 'vx')
    ss_window.ui.comboBox_sampleLogNameX.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameY.insertItem(0, 'vy')
    ss_window.ui.comboBox_sampleLogNameY.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameZ.insertItem(0, 'vz')
    ss_window.ui.comboBox_sampleLogNameZ.setCurrentIndex(0)
    ss_window.do_get_grid_alignment_info()   # e33=False, sample_resolution=(0.01, 0.01, 0.01))

    # align grid

    # constrained stress/strain

    # # plane strain and save
    # ss_window.calcualte_plane_strain()
    # ss_window.save_present_project('test_strain_stress_plane_stress.h5')

    # # plain stress and save
    # ss_window.calcualte_plane_stress()
    # ss_window.save_present_project('test_strain_stress.h5')

    # # unconstrained stress/strain
    # ss_window.align_loaded_data(e33=True, sample_resolution=(0.01, 0.01, 0.01))

    # # calculate and save project
    # ss_window.calcualte_unconstrained_stress()
    # ss_window.save_present_project('test_strain_stress.h5')

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
