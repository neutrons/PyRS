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


def test_main_2d(test_dir):
    """
    test main for the old 2D data in order to test other feature
    :param test_dir:
    :return:
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

    # work on grid (alignment) window
    # go to tab-2 (raw and aligned peak parameter value)
    grid_info_window.ui.comboBox_parameterNamesAnalysis.setCurrentIndex(2)  # set to center_d
    grid_info_window.do_load_params_raw_grid()
    # show plot on raw grids and then mapped to output grids
    grid_info_window.ui.tabWidget_alignedParams.setCurrentIndex(2)
    # set to d_center
    grid_info_window.ui.comboBox_parameterNamesAnalysis.setCurrentIndex(2)
    assert str(grid_info_window.ui.comboBox_parameterNamesAnalysis.currentText()) == 'center_d',\
        'Shall be center_d but not {}'.format(str(grid_info_window.ui.comboBox_parameterNamesAnalysis.currentText()))
    # load raw, plot, map and plot
    grid_info_window.do_load_params_raw_grid()
    grid_info_window.do_load_params_raw_grid()
    grid_info_window.do_load_mapped_values()
    grid_info_window.do_load_params_raw_grid()

    ss_window.core.strain_stress_calculator.export_2d_slice('center_d', True, 'e11', 1,
                                                            slice_pos=0.0, slice_resolution=0.001,
                                                            file_name='/tmp/pyrs_test_ss/test.hdf5')

    # # set value
    ss_window.ui.lineEdit_youngModulus.setText('207.')  # GPA
    ss_window.ui.lineEdit_poissonRatio.setText('0.3')
    ss_window.ui.lineEdit_d0.setText('1.1698')
    ss_window.ui.radioButton_uniformD0.setChecked(True)

    # calculate unconstrained strain and stress
    ss_window.do_calculate_strain_stress()
    ss_window.do_show_strain_stress_table()
    # ss_window.save_present_project('test_strain_stress_plane_stress.h5')
    # ss_window.plot_strain_stress_slice(param='epsilon', index=[0, 0], dir=1, position=0.0)

    return ss_window


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
    ss_window.ui.lineEdit_e11ScanFile.setText('tests/temp/16-1_LD.cor_Log.gaussian.hdf5')
    ss_window.ui.lineEdit_e22ScanFile.setText('tests/temp/16-1_ND.cor_Log.gaussian.hdf5')
    ss_window.ui.lineEdit_e33ScanFile.setText('tests/temp/16-1_TD.cor_Log.gaussian.hdf5')
    ss_window.do_load_strain_files()

    # pre-grid-alignment
    ss_window.ui.comboBox_sampleLogNameX_E11.insertItem(0, 'sx')
    ss_window.ui.comboBox_sampleLogNameX_E11.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameX_E22.insertItem(0, 'sx')
    ss_window.ui.comboBox_sampleLogNameX_E22.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameX_E33.insertItem(0, 'sx')
    ss_window.ui.comboBox_sampleLogNameX_E33.setCurrentIndex(0)

    ss_window.ui.comboBox_sampleLogNameY_E11.insertItem(0, 'sy')
    ss_window.ui.comboBox_sampleLogNameY_E11.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameY_E22.insertItem(0, 'sy')
    ss_window.ui.comboBox_sampleLogNameY_E22.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameY_E33.insertItem(0, 'sy')
    ss_window.ui.comboBox_sampleLogNameY_E33.setCurrentIndex(0)

    ss_window.ui.comboBox_sampleLogNameZ_E11.insertItem(0, 'szs')
    ss_window.ui.comboBox_sampleLogNameZ_E11.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameZ_E22.insertItem(0, 'szs')
    ss_window.ui.comboBox_sampleLogNameZ_E22.setCurrentIndex(0)
    ss_window.ui.comboBox_sampleLogNameZ_E33.insertItem(0, 'sz')
    ss_window.ui.comboBox_sampleLogNameZ_E33.setCurrentIndex(0)

    ss_window.do_get_grid_alignment_info()   # e33=False, sample_resolution=(0.01, 0.01, 0.01))

    # set the uer alignment
    setup_dict = {'Max': {'Y': None, 'X': 140.0, 'Z': None},
                  'Resolution': {'Y': None, 'X': None, 'Z': None},
                  'Min': {'Y': None, 'X': -140.0, 'Z': None}}

    grid_info_window = ss_window.align_user_grids(direction='e22', user_define_flag=False, grids_setup_dict=setup_dict,
                                                  show_aligned_grid=True)

    # work on grid (alignment) window
    # go to tab-2 (raw and aligned peak parameter value)
    grid_info_window.ui.comboBox_parameterNamesAnalysis.setCurrentIndex(2)  # set to center_d
    grid_info_window.do_load_params_raw_grid()
    # show plot on raw grids and then mapped to output grids
    grid_info_window.ui.tabWidget_alignedParams.setCurrentIndex(2)
    # set to d_center
    grid_info_window.ui.comboBox_parameterNamesAnalysis.setCurrentIndex(2)
    assert str(grid_info_window.ui.comboBox_parameterNamesAnalysis.currentText()) == 'center_d',\
        'Shall be center_d but not {}'.format(str(grid_info_window.ui.comboBox_parameterNamesAnalysis.currentText()))
    # load raw, plot, map and plot
    grid_info_window.do_load_params_raw_grid()
    grid_info_window.do_load_params_raw_grid()
    grid_info_window.do_load_mapped_values()
    grid_info_window.do_load_params_raw_grid()

    ss_window.core.strain_stress_calculator.export_2d_slice('center_d', True, 'e11', 1,
                                                            slice_pos=0.0, slice_resolution=0.001,
                                                            file_name='/tmp/pyrs_test_ss/test.hdf5')

    # pop out the window for grid check
    # ss_window.do_show_aligned_grid()
    #
    # # try to export the aligned the grids
    # grid_info_window.ui.tabWidget_alignedParams.setCurrentIndex(1)
    # ss_window.core.strain_stress_calculator.export_2d_slice('center_d', False, 'e11', 1,
    #                                                         slice_pos=0.0, slice_resolution=0.001,
    #                                                         file_name='/tmp/pyrs_test_ss/test_aligned.hdf5')
    #
    # # set value
    ss_window.ui.lineEdit_youngModulus.setText('207.')  # GPA
    ss_window.ui.lineEdit_poissonRatio.setText('0.3')   
    ss_window.ui.lineEdit_d0.setText('1.1698')
    ss_window.ui.radioButton_uniformD0.setChecked(True)

    # calculate unconstrained strain and stress
    ss_window.do_calculate_strain_stress()
    ss_window.do_show_strain_stress_table()
    # ss_window.save_present_project('test_strain_stress_plane_stress.h5')
    # ss_window.plot_strain_stress_slice(param='epsilon', index=[0, 0], dir=1, position=0.0)

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
    case = 0
    if case == 0:
        # full 3D grid
        test_window = test_main(test_dir=test_dir)
    elif case == 1:
        # simplified 2D grid
        test_window = test_main_2d(test_dir=test_dir)
    else:
        raise NotImplementedError('No test main')

    # I cannot close it!  test_window.close()
    app.exec_()
