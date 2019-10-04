#!/usr/bin/python
# System (library) test for classes and methods that will be used in instrument geometry calibration

import os
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import fitpeakswindow
import pyrs.core
import pytest


def test_xray_geometry_calibration():
    """ System test on classes and methods to calibrate XRAY instrument geometry
    :return:
    """
    # Initialize core/controller
    controller = pyrscore.PyRsCore()

    # pre-requisite is that the data file exists
    test_data = os.path.join(os.getcwd(), 'tests', 'data', 'BD_Data_Log.hdf5')
    assert os.path.exists('tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf'), 'File does not exist'

    # Load data
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
