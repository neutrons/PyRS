#!/usr/bin/python
# In order to test the core methods for strain calculation
import os
from pyrs.core import pyrscore
import sys
import pyrs.core

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

# default testing directory is ..../PyRS/
print (os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def test_strain_calculation():
    """
    main testing body to test the workflow to calculate strain
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # import data file: detector ID and file name
    test_data_set = {'TD': 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5',
                     'ND': 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5',
                     'LD': 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'}

    # load data
    data_key, message = rs_core.load_stain_stress_source_file(td_data_file=test_data_set['TD'],
                                                              nd_data_file=test_data_set['ND'],
                                                              ld_data_file=test_data_set['LD'])

    # peak fitting for detector - ALL
    rs_core.calculate_strain(data_key)

    # export
    rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


if __name__ == '__main__':
    """ main
    """
    test_strain_calculation()
