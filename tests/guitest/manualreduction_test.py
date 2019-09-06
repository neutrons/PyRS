#!/usr/bin/python
# In order to test GUI for manual_reduction analysis
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import manualreductionwindow
import pyrs.core
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def test_hzb_data(reduction_window):
    """

    :param reduction_window:
    :return:
    """
    # TODO - #84 - Implement
    # 1. Load HZB data

    # 2. Plot raw data

    # 3. Plot reduced data

    # 4. Reduce HZB data

    # 5. Plot re-reduced data

    return


def test_main():
    """
    test main
    """
    manual_reduction_window = manualreductionwindow.ManualReductionWindow(None)
    pyrs_core = pyrscore.PyRsCore()
    manual_reduction_window.setup_window(pyrs_core)

    manual_reduction_window.show()

    # Test 1
    test_hzb_data(manual_reduction_window)

    #
    # # set up IPTS and run number
    # manual_reduction_window.ui.lineEdit_iptsNumber.setText('12345')
    # manual_reduction_window.ui.lineEdit_calibratonFile.setText('test/whatever.hdf5')
    #
    # # test the strip view
    # if False:
    #     slice_setup_window = manual_reduction_window.do_launch_slice_setup()
    #     assert slice_setup_window.get_nexus_name() == 'whatever', 'NeXus file name is not set up right'
    #     slice_setup_window.set_nexus_file('RELF')
    #     slice_setup_window.plot_sample_log(log_name='blabla')
    # # END-IF
    #
    # # test batch reduction
    # manual_reduction_window.ui.lineEdit_runNumbersList.setText('98124:98130, 98114')
    # manual_reduction_window.do_reduce_batch_runs()
    #
    # manual_reduction_window.ui.graphicsView_detectorView.plot_detector_view('shall be a numpy array')

    return manual_reduction_window


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
    print ('Test manual_reduction Analysis GUI')
    app = main(sys.argv)

    # this must be here!
    test_window = test_main()
    # I cannot close it!  test_window.close()

    app.exec_()
