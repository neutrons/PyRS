#!/usr/bin/python
# In order to test GUI for texture analysis
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import textureanalysiswindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main():
    """
    test main
    """
    texture_window = textureanalysiswindow.TextureAnalysisWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    texture_window.setup_window(pyrs_core)

    texture_window.show()
    texture_window.ui.comboBox_peakType.setCurrentIndex(1)
    texture_window.load_h5_scans('tests/testdata/BD_Data_Log.hdf5')
    if False:
        texture_window.ui.lineEdit_scanNUmbers.setText('4')
        texture_window.do_plot_diff_data()
        texture_window.do_plot_meta_data()
        texture_window.do_fit_peaks()
    if True:
        texture_window.do_fit_peaks()
        texture_window.save_data_for_mantid(None, 'peaks.nxs')

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
    print ('Test Texture Analysis GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
