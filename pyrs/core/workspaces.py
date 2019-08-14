# Data manager
import numpy
import os
from pyrs.utilities import checkdatatypes
from pyrs.utilities import rs_project_file
from pyrs.core import instrument_geometry


class HidraWorkspace(object):
    """
    This workspace is the central data structure to manage all the raw and/or processed data.
    It provides
    - container for raw counts
    - container for reduced diffraction data
    - container for fitted peaks' parameters
    - container for instrument information
    """
    def __init__(self):
        """
        initialization
        """
        # raw counts
        self._raw_counts = dict()  # dict [sub-run] = count vector

        # spectra-sub run mapper
        self._sub_run_to_spectrum = None  # [sub-run] = spectrum, spectrum: 0 - ... continuous
        self._spectrum_to_sub_run = None  # [spectrum] = sub-run

        # diffraction
        self._2theta_vec = None  # ndarray.  shape = (m, ) m = number of 2theta
        self._diff_data_set = dict()  # [mask id] = ndarray: shape=(n, m), n: number of sub-run, m: number of of 2theta

        # instrument
        self._instrument_setup = None
        self._instrument_geometry_shift = None  # geometry shift

        # sample logs
        self._sample_log_dict = dict()  # sample logs

        # self._data_dict = dict()  # key = data key, data = data class
        # self._file_ref_dict = dict()  # key = file name, value = data key / reference ID

        return

    def _create_subrun_spectrum_map(self, sub_run_list):
        """
        Set up the sub-run/spectrum maps
        :param sub_run_list:
        :return:
        """
        # this is the only place _sub_run_to_spectrum and _spectrum_to_sub_run that appear at the left of '='
        self._sub_run_to_spectrum = dict()
        self._spectrum_to_sub_run = dict()

        # besides the dictionaries are created
        print ('L214: sub runs:', sub_run_list)
        for spec_id, sub_run in enumerate(sorted(sub_run_list)):
            self._sub_run_to_spectrum[sub_run] = spec_id
            self._spectrum_to_sub_run[spec_id] = sub_run

        return

    def _load_raw_counts(self, hidra_file):
        """ Load raw detector counts from HIDRA file
        :param hidra_file:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        for sub_run_i in sorted(self._sub_run_to_spectrum.keys()):
            counts_vec_i = hidra_file.get_raw_counts(sub_run_i)
            self._raw_counts[sub_run_i] = counts_vec_i
        # END-FOR

        return

    # TODO FIXME - NOW TONIGHT - Fix this!!! or combine with other methods
    def _load_reduced_diffraction_data(self, hidra_file):
        """
        Load reduced diffraction data from HIDRA file
        :param hidra_file:
        :return:
        """
        # TODO #79 - #74 - Clean the whole method!!!
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # get X value
        vec_2theta = hidra_file.get_reduced_diff_2theta_vec()  # TODO #79 - #74,65 - Not implemented
        self._2theta_vec = vec_2theta[:]

        # initialize data set for reduced diffraction patterns
        self._diff_data_set = numpy.ndarray(shape=(num_spec, vec_2theta.shape[0]),
                                            dtype='float')

        # check whether there is diffraction data reduced with mask
        diff_mask_list = hidra_file.get_reduced_data_masks()
        for mask_name in diff_mask_list:
            # init masks
            self._diff_data_mask_set[mask_name] = numpy.ndarray(shape=self._diff_data_set.shape,
                                                                dtype=self._diff_data_set.dtype)
        # END-FOR

        for sub_run_i in sorted(self._sub_run_to_spectrum.keys()):
            # get spectrum ID
            spec_i = self._sub_run_to_spectrum[sub_run_i]

            # main
            diff_main_vec_i = hidra_file.get_reduced_diff_intensity(sub_run_i)
            self._diff_data_set[spec_i] = diff_main_vec_i

            # masks
            for mask_name in diff_mask_list:
                diff_mask_vec_i = hidra_file.get_reduced_diff_intensity(sub_run_i, mask_name)
                self._diff_data_mask_set[mask_name][spec_i] = diff_main_vec_i
            # END-FOR (mask)
        # END-FOR (sub-run)

        return

    def _load_instrument(self,  hidra_file):
        """
        Load instrument setup from HIDRA file
        :param hidra_file:
        :return:
        """
        # Check
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # Get values
        self._instrument_setup = hidra_file.get_instrument_geometry()

        return

    def _load_sample_logs(self, hidra_file):
        """

        :param hidra_file:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # Get special values
        self._sample_log_dict = hidra_file.get_logs()

        return


    def get_2theta(self, sub_run):
        """ Get 2theta value from sample log
        This is a special one
        :param sub_run:
        :return: float number as 2theta
        """
        try:
            two_theta = self._sample_log_dict['2Theta'][sub_run]
        except KeyError as key_err:
            raise RuntimeError('Unable to retrieve 2theta value from {} due to {}'
                               .format(sub_run, key_err))

        return two_theta

    def get_instrument_setup(self):
        """ Get the handler to instrument setup
        :return:
        """
        return self._instrument_setup

    def get_detector_counts(self, sub_run):
        """
        Get raw detector counts in the order of pixel IDs by a given sub run
        :param sub_run:
        :return:
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if sub_run not in self._raw_counts:
            raise RuntimeError('Sub run {} does not exist in loaded raw counts. FYI loaded '
                               'sub runs are {}'.format(sub_run, self._raw_counts.keys()))

        return self._raw_counts[sub_run]

    def get_subruns(self):
        """ Get sub runs that loaded to this workspace
        :return:
        """
        sub_runs = sorted(self._sub_run_to_spectrum.keys())
        return sub_runs

    def load_hidra_project(self, hidra_file, load_raw_counts, load_reduced_diffraction):
        """
        Load HIDRA project file
        :param hidra_file: HIDRA project file instance (not file name)
        :param load_raw_counts: Flag to load raw counts
        :param load_reduced_diffraction: Flag to load reduced diffraction data
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # create the spectrum map
        sub_run_list = hidra_file.get_sub_runs()
        self._create_subrun_spectrum_map(sub_run_list)

        # load raw detector counts
        if load_raw_counts:
            self._load_raw_counts(hidra_file)

        # load reduced diffraction
        if load_reduced_diffraction:
            self._load_reduced_diffraction_data(hidra_file)

        # load instrument
        self._load_instrument(hidra_file)

        # load sample logs
        self._load_sample_logs(hidra_file)

        return

    def get_detector_shift(self):
        """
        Get detector geometry shift
        :return: AnglerDetectorShift instance
        """
        return self._instrument_geometry_shift

    def get_reduced_diffraction_data(self, sub_run, mask_id=None):
        """
        get data set of a single diffraction pattern
        :param sub_run:
        :param mask_id: None (as default main) or ID as a String
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if mask_id is None:
            # mask_id = 'main'
            pass
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id)

        if sub_run not in self._sub_run_to_spectrum:
            raise RuntimeError('Sub run {} does not exist. FYI: available sub runs are {}'
                               ''.format(sub_run, self._sub_run_to_spectrum.keys()))
        spec_index = self._sub_run_to_spectrum[sub_run]

        # Vector 2theta
        vec_2theta = self._2theta_vec.copy()

        # Vector intensity
        try:
            vec_intensity = self._diff_data_set[mask_id][spec_index].copy()
        except KeyError as run_err:
            raise RuntimeError('Mask ID {} does not exist in reduced diffraction pattern. '
                               'The available masks are {}'
                               ''.format(mask_id, self._diff_data_set.keys()))
        return vec_2theta, vec_intensity

    def get_sample_log_values(self, sample_log_name):
        """
        Get ONE INDIVIDUAL sample log's values as a vector
        :param sample_log_name:
        :return: vector of integer or float in the same order as sub run number
        """
        checkdatatypes.check_string_variable('Sample log name', sample_log_name,
                                             self._sample_log_dict.keys())

        return self._sample_log_dict[sample_log_name].copy()

    def has_raw_data(self, sub_run):
        """ Check whether a raw file that has been loaded
        :param sub_run:
        :return:
        """
        checkdatatypes.check_int_variable('Sub run', sub_run, (1, None))

        return sub_run in self._raw_counts

    # TODO FIXME - NOW TONIGHT #72 - Make it work!
    def has_sample_log(self, data_reference_id, sample_log_name):
        """
        check whether a certain sample log exists in a loaded data file
        :param data_reference_id:
        :param sample_log_name:
        :return:
        """
        self._check_data_key(data_reference_id)

        # separate main key and sub key
        if isinstance(data_reference_id, tuple):
            main_key = data_reference_id[0]
            sub_key = data_reference_id[1]
            has_log = sample_log_name in self._data_dict[main_key][sub_key].sample_log_names
        else:
            main_key = data_reference_id
            has_log = sample_log_name in self._data_dict[main_key].sample_log_names

        return has_log

    def set_reduced_diffraction_data(self, sub_run, mask_id, bin_edges, hist):
        """ Set reduced diffraction data
        :param sub_run:
        :param mask_id: None (no mask) or String (with mask indexed by this string)
        :param bin_edges:
        :param hist:
        :return:
        """
        # TODO - TONIGHT NOW - Check & Doc

        # Set 2-theta (X)
        if self._2theta_vec is None:
            # First time set up
            # Set X
            self._2theta_vec = bin_edges.copy()

        elif self._2theta_vec.shape != bin_edges.shape:
            # Need to check if previously set
            raise RuntimeError('2theta vector are different between parent method set {} and '
                               'reduction engine returned {}'.format(self._2theta_vec.shape, bin_edges.shape))
        # END-IF-ELSE

        # Initialize Y with mask
        print ('L667: Mask ID: "{}"'.format(mask_id))

        if mask_id not in self._diff_data_set:
            num_sub_runs = len(self._sub_run_to_spectrum)
            self._diff_data_set[mask_id] = numpy.ndarray(shape=(num_sub_runs, hist.shape[0]), dtype=hist.dtype)

        # Check array shape
        if self._diff_data_set[mask_id].shape[1] != hist.shape[0]:
            raise RuntimeError('blabla')  # TODO - TONGIHT NOW #72 - Better error message

        # Set Y
        spec_id = self._sub_run_to_spectrum[sub_run]
        self._diff_data_set[mask_id][spec_id] = hist

        return

    def save_reduced_diffraction_data(self, hidra_project):
        """ Export reduced diffraction data to project
        :param hidra_project:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_project, rs_project_file.HydraProjectFile)

        hidra_project.set_reduced_diffraction_dataset(self._2theta_vec, self._diff_data_set)

        return

    @property
    def sample_log_names(self):
        """
        return the sample log names
        :return:
        """
        return self._sample_log_dict.keys()

    def sample_log_values(self, sample_log_name):
        """
        get sample log value
        :param sample_log_name:
        :return:
        """
        checkdatatypes.check_string_variable('Sample log name', sample_log_name)
        if sample_log_name not in self._sample_log_dict:
            raise RuntimeError('Sample log {0} cannot be found.'.format(sample_log_name))

        return self._sample_log_dict[sample_log_name]

    @property
    def sample_logs_for_plot(self):
        """ Get names of sample logs that can be plotted, i.e., the log values are integer or float
        :return:
        """
        sample_logs = list()
        for sample_log_name in self._sample_log_dict.keys():
            sample_log_value = self._sample_log_dict[sample_log_name]
            if sample_log_value.dtype != object:
                sample_logs.append(sample_log_name)

        return sorted(sample_logs)
