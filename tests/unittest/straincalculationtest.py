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
    test_data_set = {'e11': 'tests/testdata/LD_Data_Log.hdf5',
                     'e22': 'tests/testdata/BD_Data_Log.hdf5',
                     'e33': 'tests/testdata/ND_Data_Log.hdf55'}

    # start a session
    rs_core.stress_calculator.new_session('test strain/stress module')

    # load data
    rs_core.stress_calculator.load_stain_stress_source_file(file_name=test_data_set['e11'], direction='e11')
    rs_core.stress_calculator.load_stain_stress_source_file(file_name=test_data_set['e22'], direction='e22')
    rs_core.stress_calculator.load_stain_stress_source_file(file_name=test_data_set['e33'], direction='e33')

    # check and align measurement points around
    rs_core.stress_calculator.align_measuring_points(['e11', 'e22', 'e33'])

    # peak fitting for detector - ALL
    for direction in ['e11', 'e22', 'e33']:
        assert rs_core.stress_calculator.are_peaks_fitted(direction=direction),\
            'Peaks must be fitted as a pre-requisite'

    rs_core.calculate_strain(data_key)

    # export
    rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


if __name__ == '__main__':
    """ main
    """
    test_strain_calculation()
