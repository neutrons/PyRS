#!/user/bin/python
# Test class and methods implemented for peak fitting
import os
import numpy
from pyrs.core import pyrscore
from pyrs.core import instrument_geometry
from pyrs.utilities import script_helper
from matplotlib import pyplot as plt


class PeakFittingTest(object):
    """
    Class to test peak fitting related classes and methods
    """
    def __init__(self, input_file_name):
        """ Initialization
        :param input_file_name: name of testing HIDRA project file
        """
        # Create calibration control
        self._reduction_controller = pyrscore.PyRsCore()

        # Load data
        self._project_name = 'NFRS2 Peaks'
        self._reduction_controller.load_hidra_project(input_file_name, project_name=self._project_name,
                                                      load_detector_counts=False,
                                                      load_diffraction=True)

        return

    def fit_pseudo_voigt(self):
        """
        Fit pseudo-voigt peaks
        :return:
        """
        # Fit peaks
        self._reduction_controller.fit_peaks(self._project_name, sub_run_list=None,
                                             peak_type='PseudoVoigt', background_type='Linear',
                                             fit_range=(70., 95.))

        return

    def save_fit_result(self, out_file_name, peak_tag):

        # Save result with default value on file name to import from and export to
        self._reduction_controller.save_peak_fit_result(self._project_name, out_file_name, peak_tag)

        return


def main():
    """
    Test main
    :return:
    """
    test_project_file_name = 'tests/testdata/Hydra_16-1_cor_log.hdf5'

    # TODO - #81 NOW - Find wave length and put to a proper place in both Hidra project file and Hidra workspace

    # Create tester
    tester = PeakFittingTest(test_project_file_name)
    # fit
    tester.fit_pseudo_voigt()
    # save
    tester.save_fit_result(test_project_file_name, 'Si111')

    # TODO - #81 NOW - More tests
    # 1. get the best fit and plot
    # 2. get the worst fit and plot
    # 3. plot all peak center
    # 4. plot all mixing factor
    # 5. plot all peak width

    return


if __name__ == '__main__':
    main()

