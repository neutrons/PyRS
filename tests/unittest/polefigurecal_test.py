#!/usr/bin/python
# In order to test the peak fit window (GUI)
import os
from pyrs.core import pyrscore
import sys
import pyrs.interface
from pyrs.interface import fitpeakswindow
import pyrs.core
from pyrs.core.polefigurecalculator import PoleFigureCalculator
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

# default testing directory is ..../PyRS/
print (os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def test_pole_figure_calculation():
    """
    main testing body to test the workflow to calculate pole figure
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # import data file: detector ID and file name
    test_data_set = [(1, 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5'),
                     (2, 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5'),
                     (3, 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'),
                     (4, 'tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5'),
                     (5, 'tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5'),
                     (6, 'tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5'),
                     (7, 'tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5')]

    data_key, message = rs_core.load_rs_raw_set(test_data_set)

    # peak fitting for detector 1
    scan_range = rs_core.data_center.get_scan_range(data_key, 1)
    rs_core.fit_peaks((data_key, 1), scan_index=scan_range, peak_type='Gaussian',
                      background_type='Linear', fit_range=(80, 85))
    rs_core.save_nexus((data_key, 1), '/tmp/matrix.nxs')

    peak_intensities = rs_core.get_peak_intensities((data_key, 1))

    # initialize pole figure
    pole_figure_calculator = PoleFigureCalculator()

    log_names = [('2theta', '2theta'),
                 ('omega', 'omega'),
                 ('chi', 'chi'),
                 ('phi', 'phi')]

    pole_figure_calculator.set_experiment_logs(rs_core.data_center.get_scan_index_logs_values((data_key, 1),
                                                                                              log_names))
    pole_figure_calculator.calculate_pole_figure(peak_intensity_dict=peak_intensities)
    pole_figure_calculator.export_pole_figure('/tmp/test_polefigure.dat')


if __name__ == '__main__':
    test_pole_figure_calculation()
