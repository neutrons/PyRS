#!/usr/bin/python
# In order to test GUI for texture analysis
from pyrs.core import pyrscore
import sys
from pyrs.interface import textureanalysiswindow
from qtpy.QtWidgets import QApplication


def test_main():
    """
    test main
    """
    texture_window = textureanalysiswindow.TextureAnalysisWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    texture_window.setup_window(pyrs_core)

    texture_window.show()
    texture_window.ui.comboBox_peakType.setCurrentIndex(0)

    test_data_set = [(1, 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5'),
                     (2, 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5'),
                     (3, 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'),
                     (4, 'tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5'),
                     (5, 'tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5'),
                     (6, 'tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5'),
                     (7, 'tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5')]

    # load data
    texture_window.load_h5_scans(test_data_set)
    # texture_window.ui.lineEdit_scanNumbers.setText('0:2')

    # fit and calculate pole figure

    # fit peaks
    texture_window.do_fit_peaks()
    # calculate pole figure
    texture_window.do_cal_pole_figure()
    # test plot the sample logs or values
    texture_window.do_plot_meta_data()
    # plot pole figure
    texture_window.do_plot_pole_figure()

    # texture_window.save_data_for_mantid(None, 'peaks.nxs')

    return texture_window


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
    print('Test Texture Analysis GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
