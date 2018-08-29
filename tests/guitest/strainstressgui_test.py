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

    # set the uer alignment
    setup_dict = {'Max': {'Y': None, 'X': 140.0, 'Z': None},
                  'Resolution': {'Y': None, 'X': None, 'Z': None},
                  'Min': {'Y': None, 'X': -140.0, 'Z': None}}

    grid_info_window = ss_window.align_user_grids(direction='e22', user_define_flag=False, grids_setup_dict=setup_dict,
                                                  show_aligned_grid=True)

    # work on grid information window
    grid_info_window.ui.comboBox_parameterNamesAnalysis.setCurrentIndex(2)  # set to center_d
    grid_info_window.do_load_params_raw_grid()
    grid_info_window.ui.tabWidget_alignedParams.setCurrentIndex(2)

    ss_window.core.strain_stress_calculator.export_2d_slice('center_d', True, 'e11', 1,
                                                            slice_pos=0.0, slice_resolution=0.001,
                                                            file_name='/tmp/pyrs_test_ss/test.hdf5')

    # pop out the window for grid check
    ss_window.do_show_aligned_grid()

    # try to export the aligned the grids
    grid_info_window.ui.tabWidget_alignedParams.setCurrentIndex(1)
    ss_window.core.strain_stress_calculator.export_aligned_2d_slice('center_d', True, 'e11', 1,
                                                                    slice_pos=0.0, slice_resolution=0.001,
                                                                    file_name='/tmp/pyrs_test_ss/test_aligned.hdf5')

    # set value
    ss_window.ui.lineEdit_youngModulus.setText('3.0')
    ss_window.ui.lineEdit_poissonRatio.setText('0.5')
    ss_window.ui.lineEdit_d0.setText('1.22')

    # calculate unconstrained strain and stress
    ss_window.do_calculate_strain_stress()
    ss_window.save_present_project('test_strain_stress_plane_stress.h5')
    ss_window.plot_strain_stress_slice(param='epsilon', index=[0, 0], dir=1, position=0.0)

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
