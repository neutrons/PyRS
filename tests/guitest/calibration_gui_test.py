#!/usr/bin/python
# In order to test the peak fit window (GUI)
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import calibrationwindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_main():
    """
    test main
    """
    pyrs_core = pyrscore.PyRsCore()
    geom_cal_window = calibrationwindow.InstrumentCalibrationWindow(None, pyrs_core)

    geom_cal_window.show()

    # # 3 candidate files
    # # tests/testdata/16-1_TD.cor_Log.hdf5
    # # tests/testdata/16-1_ND.cor_Log.hdf5
    # # tests/testdata/16-1_LD.cor_Log.hdf5
    # geom_cal_window.ui.lineEdit_expFileName.setText('tests/testdata/16-1_TD.cor_Log.hdf5')
    # geom_cal_window.ui.comboBox_peakType.setCurrentIndex(2)
    # geom_cal_window.do_load_scans()
    # if False:
    #     geom_cal_window.do_plot_meta_data()
    #
    # if True:
    #     geom_cal_window.do_fit_peaks()
    #     geom_cal_window.ui.lineEdit_scanNumbers.setText('0')
    #     geom_cal_window.do_plot_diff_data()
    #     geom_cal_window.save_data_for_mantid(None, 'peaks.nxs')
    #     out_file_name = '16-1_TD.cor_Log.fit.hdf5'
    #     geom_cal_window.save_fit_result(out_file_name)

    return geom_cal_window


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
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
