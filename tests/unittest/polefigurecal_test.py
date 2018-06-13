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

    # import data file
    data_key, message = rs_core.load_rs_raw(test_data)

    # peak fitting
    scan_range = rs_core.data_center.get_scan_range(data_key)
    rs_core.fit_peaks(data_key, scan_index=scan_range, peak_type='Gaussian',
                      background_type='Linear', fit_range=(80, 85))
    peak_intensities = rs_core.get_peak_intensities(data_key)

    # initialize pole figure
    pole_figure_calculator = PoleFigureCalculator()

    log_names = [('2theta', '2theta'),
                 ('omega', 'omega'),
                 ('chi', 'mrot'),
                 ('phi', 'mtilt')]

    pole_figure_calculator.set_experiment_logs(rs_core.data_center.get_scan_index_logs_values(data_key,
                                                                                              log_names))
    pole_figure_calculator.calculate_pole_figure(peak_intensity_dict=peak_intensities)
    pole_figure_calculator.export_pole_figure('/tmp/test_polefigure.dat')


if __name__ == '__main__':
    test_pole_figure_calculation()
