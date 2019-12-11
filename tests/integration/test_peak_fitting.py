#!/user/bin/python
# Test class and methods implemented for peak fitting
import numpy
from pyrs.core import pyrscore
from pyrs.core.peak_collection import PeakCollection
from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction
from pyrs.core.summary_generator import SummaryGenerator
from pyrs.dataobjects import SampleLogs
from pyrs.utilities.rs_project_file import HidraProjectFile
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

    def fit_gaussian(self):
        """ Test fitting with Gaussian
        :return:
        """
        peak_info_dict = {'Fe111': {'Center': 84., 'Range': [70., 95]}}

        # Fit peaks
        self._reduction_controller.fit_peaks(self._project_name, sub_run_list=None,
                                             peak_type='Gaussian', background_type='Linear',
                                             peaks_fitting_setup=peak_info_dict)

    def show_fit_result(self, show_effective_params):
        """ Get the fit results and plot
        :return:
        """
        # TODO - #81 NOW - Implement get_peak_fitting_result()
        # first return is `peak_name_list`
        _, peak_params_matrix = \
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


@pytest.mark.parametrize('source_project_file, output_project_file, peak_type, peak_info',
                         [('/HFIR/HB2B/IPTS-22731/shared/ProjectFiles/HB2B_1065.h5', 'HB2B_1065_Peak.h5',
                           'PseudoVoigt', PeakInfo(90.5, 89.9, 91.6, '311'))],
                         ids=['HB2B1065PeakExport'])
def test_retrieve_fit_metadata(source_project_file, output_project_file, peak_type, peak_info):
    pass
    # Create tester
    # tester = PeakFittingTest(source_project_file)

    # Fit peak
    # tester.fit_peak(peak_type, peak_info)

    # retrieve Center

    # retrieve Height

    # retrieve intensity

    # retrieve FWHM

    # retrieve d_spacing


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


# TODO - MAKE IT WORK!
def test_calculating_com():
    # calculate_center_of_mass(self, peak_tag, peak_range):
    pass


def test_convert_peaks_centers_to_dspacing():
    #
    pass


def test_write_csv():
    csv_filename = 'test_write_single.csv'
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    # create a PeakCollection
    gaussian = PeakShape.GAUSSIAN
    linear = BackgroundFunction.LINEAR

    subruns = [1, 2, 3]
    data_type = [(name, numpy.float32) for name in gaussian.native_parameters + linear.native_parameters]
    data = numpy.zeros(len(subruns), dtype=data_type)  # doesn't matter what the values are
    error = numpy.zeros(len(subruns), dtype=data_type)  # doesn't matter what the values are

    peaks = PeakCollection('fake', gaussian, linear)
    peaks.set_peak_fitting_values(subruns, data, error, [10., 20., 30.])

    # create a SampleLog
    sample = SampleLogs()
    sample.subruns = subruns
    sample['variable1'] = numpy.linspace(0., 100., len(subruns))
    sample['variable2'] = numpy.linspace(100., 200., len(subruns))  # not to be found in the output
    sample['constant1'] = numpy.linspace(1., 1.+5E-11, len(subruns))  # values less than tolerance

    # write things out to disk
    generator = SummaryGenerator(csv_filename,
                                 log_list=['variable1', 'constant1', 'missing1'])
    generator.setHeaderInformation(dict())  # means empty header

    generator.write_csv(sample, [peaks])

    assert os.path.exists(csv_filename), '{} was not created'.format(csv_filename)

    EXPECTED_HEADER = '''# IPTS number
# Run
# Scan title
# Sample name
# Item number
# HKL phase
# Strain direction
# Monochromator setting
# Calibration file
# Hidra project file
# Manual vs auto reduction
# missing: missing1
# constant1 = 1 +/- 2e-11'''.split('\n')

    # verify the file contents
    with open(csv_filename, 'r') as handle:
        # read in the file and remove whitespace
        contents = [line.strip() for line in handle.readlines()]

    # verify exact match on the header
    for exp, obs in zip(contents[:len(EXPECTED_HEADER)], EXPECTED_HEADER):
        assert exp == obs

    # verify the column headers
    assert contents[len(EXPECTED_HEADER)].startswith('sub-run,variable1,fake_Center,')
    assert contents[len(EXPECTED_HEADER)].endswith(',fake_chisq')

    assert len(contents) == len(EXPECTED_HEADER) + 1 + len(subruns), 'Does not have full body'
    # verify that the number of columns is correct
    # columns are (subruns, one log, parameter values, uncertainties, chisq)
    for line in contents[len(EXPECTED_HEADER) + 1:]:  # skip past header and constant log
        assert len(line.split(',')) == 1 + 1 + 7 * 2 + 1

    # cleanup
    os.remove(csv_filename)


EXPECTED_HEADER_1065 = '''# IPTS number = 22731
# Run = 1065
# Scan title = A boring axis move
# Sample name
# Item number
# HKL phase
# Strain direction = Powder
# Monochromator setting
# Calibration file
# Hidra project file = /some/place/random.h5
# Manual vs auto reduction
# missing: S1width, S1height, S1distance, RadialDistance
# chi = 0 +/- 0
# phi = 0 +/- 0
# omega = 135 +/- 0'''.split('\n')

EXPECTED_HEADER_938 = '''# IPTS number = 22731
# Run = 938
# Scan title = Testing
# Sample name
# Item number
# HKL phase
# Strain direction = ND
# Monochromator setting
# Calibration file
# Hidra project file = /some/place/random.h5
# Manual vs auto reduction
# missing: S1width, S1height, S1distance, RadialDistance
# sx = 0.00073674 +/- 0
# chi = 0 +/- 0
# phi = 0 +/- 0
# omega = 0 +/- 0
# sz = 3.8844 +/- 0
# sy = 0.00057072 +/- 0
# 2theta = 90.001 +/- 0'''.split('\n')


@pytest.mark.parametrize('project_file_name, csv_filename, expected_header, num_subruns, num_logs,'
                         ' startswith, endswith',
                         [('/HFIR/HB2B/shared/PyRS/HB2B_1065_Peak.h5', 'HB2B_1065.csv', EXPECTED_HEADER_1065, 99, 7,
                           'sub-run,vx,vy,vz,', ',311_chisq'),
                          ('data/HB2B_938_peak.h5', 'HB2B_938.csv', EXPECTED_HEADER_938, 1, 3,
                           'sub-run,vx,vy,vz,Si111_Center', ',Si111_chisq')],
                         ids=['HB2B_1065_CSV', 'HB2B_938_CSV'])
def test_write_csv_from_project(project_file_name, csv_filename, expected_header, num_subruns, num_logs,
                                startswith, endswith):
    """Test the method to export CSV file
    """
    # load project file
    if not os.path.exists(project_file_name):
        pytest.skip('Project file {} does not exist'.format(project_file_name))
    project = HidraProjectFile(project_file_name)

    # get information from the project file
    peak_tags = project.read_peak_tags()
    peak_collections = [project.read_peak_parameters(tag) for tag in peak_tags]  # all tags
    sample_logs = project.read_sample_logs()
    assert sample_logs.subruns.size == num_subruns  # just as a quick check

    # write out the csv file
    generator = SummaryGenerator(csv_filename)
    generator.setHeaderInformation({'project': '/some/place/random.h5'})  # only set one value
    generator.write_csv(sample_logs, peak_collections)

    # testing
    assert os.path.exists(csv_filename), '{} was not created'.format(csv_filename)

    # verify the file contents
    with open(csv_filename, 'r') as handle:
        # read in the file and remove whitespace
        contents = [line.strip() for line in handle.readlines()]

    # verify exact match on the header
    for exp, obs in zip(contents[:len(expected_header)], expected_header):
        assert exp == obs

    # verify the column headers
    assert contents[len(expected_header)].startswith(startswith)
    assert contents[len(expected_header)].endswith(endswith)

    assert len(contents) == len(expected_header) + 1 + num_subruns, 'Does not have full body'
    # verify that the number of columns is correct
    # columns are (subruns, seven logs, parameter values, uncertainties, chisq)
    for line in contents[len(expected_header) + 1:]:  # skip past header and constant log
        assert len(line.split(',')) == 1 + num_logs + 7 * 2 + 1

    # cleanup
    os.remove(csv_filename)


if __name__ == '__main__':
    pytest.main([__file__])
