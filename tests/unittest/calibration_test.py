#!/usr/bin/python
# System (library) test for classes and methods that will be used in instrument geometry calibration

import os
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import fitpeakswindow
import pyrs.core
# try:
#     from PyQt5.QtWidgets import QApplication
# except ImportError:
#     from PyQt4.QtGui import QApplication

# default testing directory is ..../PyRS/
print (os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def test_xray_geometry_calibration():
    """ System test on classes and methods to calibrate XRAY instrument geometry
    :return:
    """
    # Initialize core/controller
    controller = pyrscore.PyRsCore()

    # Load data
    try:
        assert os.path.exists('tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf')
    except AssertionError as err:
        raise AssertionError('File tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf does not exist. '
                             'Current directory: {}'.format(os.getcwd()))
    controller.load_hidra_project(hidra_h5_name='tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf',
                                  project_name='system_test_xray')

    # Calibrate
    # (1) Reduce data
    controller.reduce_diffraction_data('system_test_xray', two_theta_step=0.01, pyrs_engine=True)

    # (2) Calibrate
    # TODO - To be implemented soon

    return


if __name__ == '__main__':
    test_xray_geometry_calibration()
