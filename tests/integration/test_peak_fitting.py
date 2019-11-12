#!/user/bin/python
# Test class and methods implemented for peak fitting
import numpy
from pyrs.core import pyrscore
from matplotlib import pyplot as plt
import pytest
from collections import namedtuple
import os
import shutil


# Named tuple for peak information
PeakInfo = namedtuple('PeakInfo', 'center left_bound right_bound tag')


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
        hd_ws = self._reduction_controller.load_hidra_project(input_file_name, project_name=self._project_name,
                                                              load_detector_counts=False,
                                                              load_diffraction=True)

        # set wave length
        hd_ws.set_wavelength(1.071, False)

        return

    def fit_peak(self, peak_profile, peak_info):
        """Fit peak

        Parameters
        ----------
        peak_profile
        peak_info : ~collections.namedtuple

        Returns
        -------

        """
        # Convert input named tuple to pyrs dictionary
        peak_info_dict = {peak_info.tag: {'Center': peak_info.center,
                                          'Range': (peak_info.left_bound, peak_info.right_bound)}}

        # Fit peak
        self._reduction_controller.fit_peaks(self._project_name, sub_run_list=None,
                                             peak_type=peak_profile, background_type='Linear',
                                             peaks_fitting_setup=peak_info_dict)

        return

    def fit_pseudo_voigt(self):
        """
        Fit pseudo-voigt peaks
        :return:
        """
        peak_info_dict = {'Fe111': {'Center': 94.5, 'Range': [91., 97]}}

        # Fit peaks
        self._reduction_controller.fit_peaks(self._project_name, sub_run_list=None,
                                             peak_type='PseudoVoigt', background_type='Linear',
                                             peaks_fitting_setup=peak_info_dict)

        return

    def fit_gaussian(self):
        """ Test fitting with Gaussian
        :return:
        """
        peak_info_dict = {'Fe111': {'Center': 84., 'Range': [70., 95]}}

        # Fit peaks
        self._reduction_controller.fit_peaks(self._project_name, sub_run_list=None,
                                             peak_type='Gaussian', background_type='Linear',
                                             peaks_fitting_setup=peak_info_dict)

        return

    def show_fit_result(self, show_effective_params):
        """ Get the fit results and plot
        :return:
        """
        # TODO - #81 NOW - Implement get_peak_fitting_result()
        peak_name_list, peak_params_matrix = \
            self._reduction_controller.get_peak_fitting_result(self._project_name, return_format=numpy.ndarray,
                                                               effective_parameter=show_effective_params)
        # Plot peak width
        sub_run_vec = peak_params_matrix[:, 0]
        peak_width = peak_params_matrix[:, 5]  # rs_project_file.HidraConstants.Peak_FWHM for pandas output
        plt.plot(sub_run_vec, peak_width, color='red', label='FWHM')

        # plt.show()

        # Test to get the original data and fitted data from
        exp_data_set = self._reduction_controller.get_diffraction_data(self._project_name, sub_run=3, mask=None)
        fit_data_set = self._reduction_controller.get_modeled_data(self._project_name, sub_run=3)

        plt.plot(exp_data_set[0], exp_data_set[1], color='black', label='Experiment')
        plt.plot(fit_data_set[0], fit_data_set[1], color='green', label='Fitted')
        plt.show()

        # Effective parameters
        peak_params_dict = self._reduction_controller.get_peak_fitting_result(self._project_name,
                                                                              return_format=dict,
                                                                              effective_parameter=True)

        # Print parameters
        print('sub-run = {}, peak width = {}'.format(3, peak_params_dict[3]['PeakWidth']))

        return

    def save_fit_result(self, src_project_file, out_file_name, peak_tag):
        """Save peak fitting result to project file with previous

        Parameters
        ----------
        src_project_file
        out_file_name
        peak_tag

        Returns
        -------

        """
        # Copy the source file to output file
        if src_project_file is not None:
            shutil.copy(src_project_file, out_file_name)

        # Save result with default value on file name to import from and export to
        self._reduction_controller.save_peak_fit_result(self._project_name, out_file_name, peak_tag, overwrite=False)

        return


# @pytest.mark.parametrize('source_project_file, output_project_file, peak_type, peak_info',
#                          [('data/HB2B_1017.h5', 'HB2B_1017_2Peaks.h5', 'PseudoVoigt',
#                            [PeakInfo(81., 78., 83., 'LeftPeak'), PeakInfo(85., 83., 87., 'RightPeak')])],
#                          ids=['HB2B1017PeakFit'])
def broken_test_fit_2peaks(source_project_file, output_project_file, peak_type, peak_info_list):
    """Performance test on data with multiple peaks on multiple sub runs

    Also on the 'real' situation that some peaks do not even exist on some sub runs

    Parameters
    ----------
    source_project_file
    output_project_file
    peak_type
    peak_info_list

    Returns
    -------

    """
    # Test only 1
    if source_project_file != 'data/HB2B_1017.h5':
        return

    # Create tester
    tester = PeakFittingTest(source_project_file)

    # Fit peak
    tester.fit_peak(peak_type, peak_info_list[0])
    tester.fit_peak(peak_type, peak_info_list[1])

    # save to project file
    tester.save_fit_result(source_project_file, output_project_file, peak_info_list[0].tag)

    return


@pytest.mark.parametrize('source_project_file, output_project_file, peak_type, peak_info',
                         [('/HFIR/HB2B/IPTS-22731/shared/ProjectFiles/HB2B_1065.h5', 'HB2B_1065_Peak.h5',
                           'PseudoVoigt', PeakInfo(90.5, 89.9, 91.6, '311'))],
                         ids=['HB2B1065PeakExport'])
def skip_test_fit_2peaks(source_project_file, output_project_file, peak_type, peak_info):
    """Performance test on data with multiple peaks and/or sub runs

    This also tends to ddd a new test for strain/stress data (goal is to generate light-weight HiDRA file)

    Parameters
    ----------
    source_project_file
    output_project_file
    peak_type
    peak_info

    Returns
    -------

    """
    # Test only 1
    if source_project_file != 'data/HB2B_1017.h5':
        return

    # Create tester
    tester = PeakFittingTest(source_project_file)

    # Fit peak
    tester.fit_peak(peak_type, peak_info)

    # save to project file
    tester.save_fit_result(source_project_file, output_project_file, peak_info.tag)

    return


@pytest.mark.parametrize('project_file_name, peak_file_name, peak_type, peak_info',
                         [('data/Hidra_16-1_cor_log.h5', 'Hidra_16-1_cor_log_peak.h5', 'Gaussian',
                           PeakInfo(94.5, 91, 97, 'Fe111')),  # NSFR2 peak
                          ('data/HB2B_938.h5', 'HB2B_938_peak.h5', 'PseudoVoigt',
                           PeakInfo(95.5, 91, 97, 'Si111'))],
                         ids=('FakeHB2B', 'HB2B_938'))
def test_main(project_file_name, peak_file_name, peak_type, peak_info):
    """Test peak fitting

    Parameters
    ----------
    project_file_name : str
        Hidra project file containing reduced diffraction 2theta pattern
    peak_file_name : str
        Hidra project file containing peaks
    peak_type : str
        Peak profile type: [gaussian, pseudovoigt]
    peak_info: namedtuple
        center, left_bound, right_bound

    Returns
    -------
    None

    """
    # Create tester
    tester = PeakFittingTest(project_file_name)

    # Fit peak
    tester.fit_peak(peak_type, peak_info)

    # save to project file
    tester.save_fit_result(project_file_name, peak_file_name, peak_info.tag)

    return


# TODO - MAKE IT WORK!
def test_calculating_com():
    # calculate_center_of_mass(self, peak_tag, peak_range):
    pass


def test_convert_peaks_centers_to_dspacing():
    #
    pass


@pytest.mark.parametrize('project_file_name, csv_file_name',
                         [('data/HB2B_938_peak.h5', 'HB2B_938.h5')],
                         ids=['HB2B_938CSV'])
def test_write_csv(project_file_name, csv_file_name):
    """Test the method to export CSV file

    Returns
    -------

    """
    # Load project file
    assert os.path.exists(project_file_name), 'Project file {} does not exist'.format(project_file_name)

    # Create calibration control
    controller = pyrscore.PyRsCore()

    controller.load_hidra_project(project_file_name, project_name='csv.{}'.format(project_file_name),
                                  load_detector_counts=False, load_diffraction=False, load_peaks=True)

    # Check tag
    peak_tags = controller.get_peak_tags()
    assert True

    # Output CSV file
    controller.export_summary(peak_tags[0], csv_file_name)

    return


if __name__ == '__main__':
    pytest.main([__file__])
