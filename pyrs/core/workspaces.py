# Data manager
import numpy
import os
from pyrs.utilities import checkdatatypes
from pyrs.utilities import rs_project_file
from pyrs.core import instrument_geometry

class ScanDataHolder(object):
    """
    holder for a single scan data which contains diffraction data and sample logs
    """
    def __init__(self, file_name, diff_data_dict, sample_log_dict):
        """

        :param file_name:
        :param diff_data_dict: dict (key: int/scan log index; value: tuple: vec_2theta, vec_intensity
        :param sample_log_dict:
        """
        # check
        checkdatatypes.check_file_name(file_name)
        checkdatatypes.check_dict('Diffraction data dictionary', diff_data_dict)
        checkdatatypes.check_dict('Sample log dictionary', sample_log_dict)

        # check diffraction data dictionary
        for log_index in sorted(diff_data_dict.keys()):
            checkdatatypes.check_int_variable('Diffraction data log index', log_index, value_range=[0, None])
            diff_tup = diff_data_dict[log_index]
            checkdatatypes.check_tuple('Diffraction data set', diff_tup, 2)
            vec_2theta = diff_tup[0]
            vec_intensity = diff_tup[1]
            checkdatatypes.check_numpy_arrays('Vector for 2theta and intensity', [vec_2theta, vec_intensity],
                                              dimension=1, check_same_shape=True)

        # store a list of all existing scan (log) indexes in ascending order
        self._scan_log_indexes = diff_data_dict.keys()
        self._scan_log_indexes.sort()
        self._scan_log_index_vec = numpy.array(self._scan_log_indexes)

        # check sample log dictionary
        for log_name in sample_log_dict:
            # skip peak fit part
            if log_name == 'peak_fit':
                continue
            checkdatatypes.check_string_variable('Sample log name', log_name)
            log_value_vec = sample_log_dict[log_name]
            checkdatatypes.check_numpy_arrays('Sample log {0} value vector'.format(log_name),
                                              log_value_vec, 1, False)
            if len(log_value_vec) != len(self._scan_log_indexes):
                raise RuntimeError('Number of log values of {0} {1} is not equal to number of scan logs {2}'
                                   ''.format(log_name, len(log_value_vec), len(self._scan_log_indexes)))

        # set
        self._file_name = file_name
        self._diff_data_dict = diff_data_dict
        self._sample_log_dict = sample_log_dict

        return

    @property
    def sample_log_names(self):
        """
        return the sample log names
        :return:
        """
        return self._sample_log_dict.keys()

    @property
    def raw_file_name(self):
        """
        raw data file (HDF5)'s name
        :return:
        """
        return self._file_name

    def get_diff_data(self, scan_log_index):
        """
        get diffraction data
        :param scan_log_index:
        :return:
        """
        checkdatatypes.check_int_variable('Scan (log) index', scan_log_index, [0, None])
        if scan_log_index not in self._scan_log_indexes:
            raise RuntimeError('User specified scan log with index {0} is not found with range [{1}, {2}]'
                               ''.format(scan_log_index, self._scan_log_indexes[0], self._scan_log_indexes[-1]))

        return self._diff_data_dict[scan_log_index]

    def get_sample_log_names(self, can_plot):
        """
        get sample log names
        :param can_plot: if True, only return the sample logs that can be plotted (i.e., float or integer)
        :return:
        """
        if can_plot:
            sample_logs = list()
            for sample_log_name in self._sample_log_dict.keys():
                # TODO FIXME - 20180930 - skip peak fitting
                if sample_log_name == 'peak_fit':
                    continue
                sample_log_value = self._sample_log_dict[sample_log_name]
                if sample_log_value.dtype != object:
                    sample_logs.append(sample_log_name)
        else:
            sample_logs = self._sample_log_dict.keys()

        return sorted(sample_logs)

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

    def get_scan_log_index_range(self):
        """
        get the list of all the log indexes
        :return:
        """
        return self._scan_log_indexes[:]


class HidraWorkspace(object):
    """
    Raw diffraction data (could be corrected) manager for all loaded and (possibly) processed data
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

        # sample logs
        self._sample_log_dict = dict()  # sample logs

        # self._data_dict = dict()  # key = data key, data = data class
        # self._file_ref_dict = dict()  # key = file name, value = data key / reference ID

        return

    def get_2theta(self, sub_run):
        """
        Get 2theta log
        This is a special one
        :param sub_run:
        :return:
        """
        return self._sample_log_dict['2Theta'][sub_run]

    def get_instrument_setup(self):
        """
        Get the handler to instrument setup
        :return:
        """
        return self._instrument_setup

    def get_raw_data(self, sub_run):
        # TODO - TONIGHT NOW - doc & check
        print ('L159: Raw counts keys: ', self._raw_counts.keys())
        print ('L159: Raw counts: ', self._raw_counts[sub_run])
        return self._raw_counts[sub_run]

    def get_subruns(self):
        """ Get sub runs that loaded to this workspace
        :return:
        """
        sub_runs = sorted(self._sub_run_to_spectrum.keys())
        print ('L163: sub runs', sub_runs)
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

    def _load_reduced_diffraction_data(self, hidra_file):
        """
        Load reduced diffraction data from HIDRA file
        :param hidra_file:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # get X value
        vec_2theta = hidra_file.get_reduced_diff_2theta_vec()
        self._diff_2theta_vec = vec_2theta[:]

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

    # TODO - Need to evaluate
    def _check_data_key(self, data_key):
        """
        check whether a data key is valid and exist
        :param data_key:
        :return:
        """
        # 2 cases
        if isinstance(data_key, tuple):
            # it could be main-key/sub-key pair
            assert len(data_key) == 2, 'If data key is tuple, then it must have 2 elements.'
            main_key = data_key[0]
            sub_key = data_key[1]
        else:
            main_key = data_key
            sub_key = None

        # check main keys
        checkdatatypes.check_string_variable('Data reference ID', main_key)
        if main_key not in self._data_dict:
            raise RuntimeError('Data reference ID (key) {0} does not exist. Existing keys are {1}'
                               ''.format(data_key, self._data_dict.keys()))

        # check possibly sub-key
        if sub_key is not None and sub_key not in self._data_dict[main_key]:
            raise RuntimeError('Sub key {0} does not exist in data with main-key {1}. Existing '
                               'sub-keys are {2}'.format(sub_key, main_key, self._data_dict[main_key].keys()))

        return

    # TODO - Need to evaluate
    def _load_raw_counts_old(self, diff_data_dict, sample_log_dict, h5file, replace=True):
        """
        add a loaded raw data set
        :param diff_data_dict:
        :param sample_log_dict:
        :param h5file:
        :param replace:
        :return:
        """
        data_key = self.generate_data_key(h5file)

        if data_key in self._data_dict and replace:
            raise RuntimeError('Data file {0} has been loaded and not allowed to be replaced.'.format(h5file))

        self._data_dict[data_key] = ScanDataHolder(h5file, diff_data_dict, sample_log_dict)
        self._file_ref_dict[h5file] = data_key

        return data_key

    def add_raw_data_set(self, diff_data_dict_set, sample_log_dict, det_h5_list, replace=True):
        """
        add a loaded raw data set
        :param diff_data_dict_set:
        :param sample_log_dict:
        :param det_h5_list:
        :param replace:
        :return:
        """
        data_key = self.generate_data_set_key(det_h5_list)

        if data_key in self._data_dict and not replace:
            raise RuntimeError('Data file {0} has been loaded and not allowed to be replaced.'.format(det_h5_list))
        else:
            self._data_dict[data_key] = dict()

        for det_id, h5file in det_h5_list:
            sub_key = self.generate_sub_data_key(det_id, h5file)
            self._data_dict[data_key][sub_key] = ScanDataHolder(h5file, diff_data_dict_set[det_id],
                                                                sample_log_dict[det_id])
            self._file_ref_dict[h5file] = data_key, sub_key
        # END-FOR

        return data_key

    def delete_data(self, reference_id):
        """
        delete a data set data key/reference ID
        :param reference_id:
        :return: boolean (False if reference ID is not in this instance)
        """
        # check input: correct type and existed
        self._check_data_key(reference_id)

        # remove from both site
        file_name = self._data_dict[reference_id].raw_file_name
        del self._data_dict[reference_id]
        del self._file_ref_dict[file_name]

        print ('[INFO] Data from file {0} with reference ID {1} is removed from project.'
               ''.format(file_name, reference_id))

        return True

    @staticmethod
    def generate_data_key(file_name):
        """
        generate a quasi-unique data (reference) ID for a file and unique within 2^8 occurance with same file name
        :param file_name:
        :return:
        """
        checkdatatypes.check_string_variable('Data file name for data reference ID', file_name)

        base_name = os.path.basename(file_name)
        dir_name = os.path.dirname(file_name)

        data_key = base_name + '_' + str(abs(hash(dir_name) % 256))

        return data_key

    @staticmethod
    def generate_data_set_key(det_h5_list):
        """
        generate a quasi-unique data (reference) ID for a file and unique within 2^8 occurance
        with same file name:  the name will be based on the first file's name
        :param det_h5_list:
        :return:
        """
        # check input and sort the files
        checkdatatypes.check_list('HDF5 file list for each detector', det_h5_list)
        det_h5_list.sort()

        # construct the name from the first file
        file_name = det_h5_list[0][1]
        checkdatatypes.check_string_variable('Data file name for data reference ID', file_name)

        base_name = os.path.basename(file_name)
        dir_name = os.path.dirname(file_name)

        data_key = base_name + '_{0}_'.format(len(det_h5_list)) + str(abs(hash(dir_name) % 256))

        return data_key

    @staticmethod
    def generate_sub_data_key(det_id, file_name):
        """
        generate a quasi-unique data (reference) ID for a file under a unique data key
        :param det_id
        :param file_name:
        :return:
        """
        # FUTURE: in this stage, detector ID as integer is good enough to be a sub key
        checkdatatypes.check_string_variable('Data file name for data reference ID', file_name)
        checkdatatypes.check_int_variable('Detector ID', det_id, (0, None))
        data_key = det_id

        return data_key

    def get_data_set(self, data_key_set, scan_index):
        """
        get data set of a single diffraction pattern
        :param data_key_set:
        :param scan_index:
        :return:
        """
        if isinstance(data_key_set, tuple) and len(data_key_set) == 2:
            data_ref_id, sub_key = data_key_set
        elif isinstance(data_key_set, tuple):
            raise RuntimeError('Wrong!')
        else:
            data_ref_id = data_key_set
            checkdatatypes.check_string_variable('Data reference ID', data_ref_id)
            sub_key = None

        # check input
        self._check_data_key(data_ref_id)
        try:
            if sub_key is None:
                data_set = self._data_dict[data_ref_id].get_diff_data(scan_index)
            else:
                data_set = self._data_dict[data_ref_id][sub_key].get_diff_data(scan_index)
        except ValueError as value_err:
            raise RuntimeError('Unable to get data from scan log index {} due to {}'.format(scan_index, value_err))

        return data_set

    def get_sample_logs_list(self, data_key_set, can_plot):
        """
        get the list of sample logs' names
        :param data_key_set:
        :param can_plot: True for log that can be plotted (no object type); Otherwise, all sample logs
        :return: list of strings
        """
        # data key set can be a tuple for multiple sample rotations (pole figure)
        if isinstance(data_key_set, tuple):
            data_key, sub_key = data_key_set
        else:
            data_key = data_key_set
            sub_key = None

        self._check_data_key(data_key)

        if sub_key is None:
            names = self._data_dict[data_key].get_sample_log_names(can_plot)
        else:
            names = self._data_dict[data_key][sub_key].get_sample_log_names(can_plot)

        return names

    def get_sample_log_values(self, data_key, sample_log_name):
        """
        Get ONE INDIVIDUAL sample log's values as a vector
        :param data_key:
        :param sample_log_name:
        :return:
        """
        self._check_data_key(data_key)

        print ('[DB...BAT] Data key: {} is of type {}'.format(data_key, type(data_key)))

        # return log values in 2 style
        if isinstance(data_key, tuple):
            # with sub keys
            main_key = data_key[0]
            sub_key = data_key[1]
            log_value_vec = self._data_dict[main_key][sub_key].sample_log_values(sample_log_name)
        else:
            # without sub keys
            log_value_vec = self._data_dict[data_key].sample_log_values(sample_log_name)

        return log_value_vec

    def get_scan_index_logs_values(self, data_key_set, log_name_pair_list):
        """
        Get a set of sample logs' values and return with scan indexes
        :param data_key_set:
        :param log_name_pair_list:
        :return: dictionary (key = scan log index) of dictionary (key = sample log name)
        """
        # check input
        if isinstance(data_key_set, tuple):
            data_key, sub_key = data_key_set
        else:
            data_key = data_key_set
            sub_key = None
        sample_log_list = self.get_sample_logs_list(data_key_set, True)

        checkdatatypes.check_list('Sample logs names', log_name_pair_list)  #, sample_log_list)
        for target_name, log_name in log_name_pair_list:
            # need  more check
            if log_name not in sample_log_list:
                raise RuntimeError('Log {0} not in {1}'.format(log_name, sample_log_list))

        # go through scan index
        scan_logs_dict = dict()
        if sub_key is None:
            scan_index_range = self._data_dict[data_key].get_scan_log_index_range()
        else:
            scan_index_range = self._data_dict[data_key][sub_key].get_scan_log_index_range()
        for scan_index in scan_index_range:
            entry_dict = dict()
            for target_name, log_name in log_name_pair_list:
                if sub_key is None:
                    log_value = self._data_dict[data_key].sample_log_values(log_name)
                else:
                    log_value = self._data_dict[data_key][sub_key].sample_log_values(log_name)
                # print ('[DB...INFO] Log value = {0} of type {1}'.format(log_value, type(log_value)))
                entry_dict[log_name] = log_value[scan_index]
            # END-FOR

            scan_logs_dict[scan_index] = entry_dict
        # END-FOR

        return scan_logs_dict

    def get_scan_range(self, data_key, sub_key=None):
        """
        get the range of scan log indexes
        :param data_key:
        :param sub_key:
        :return: list of scan log indexes
        """
        self._check_data_key(data_key)

        if isinstance(self._data_dict[data_key], dict):
            if sub_key is None:
                raise RuntimeError('Sub-key must be given by user/caller for data set case')
            ret_range = self._data_dict[data_key][sub_key].get_scan_log_index_range()
        else:
            if sub_key is not None:
                raise RuntimeError('Not-data-set mode.  Sub key does not make sense')
            ret_range = self._data_dict[data_key].get_scan_log_index_range()

        return ret_range

    def get_sub_keys(self, data_key):
        """
        get sub key for a dta
        :param data_key:
        :return:
        """
        if isinstance(self._data_dict[data_key], dict):
            return self._data_dict[data_key].keys()

        return None

    def has_data(self, reference_id):
        """
        check whether a data key/reference ID exists
        :param reference_id:
        :return:
        """
        checkdatatypes.check_string_variable('Reference ID', reference_id)

        return reference_id in self._data_dict

    def has_raw_data(self, file_name):
        """
        check whether a raw file that has been loaded
        :param file_name:
        :return:
        """
        checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=False, is_dir=False)

        return file_name in self._file_ref_dict

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
            raise RuntimeError('2theta vector are different to set: {} vs {}'.format(self._2theta_vec.shape,
                                                                                     bin_edges.shape))
        # END-IF-ELSE

        # Initialize Y with mask
        print ('L667: Mask ID: "{}"'.format(mask_id))

        if mask_id not in self._diff_data_set:
            num_sub_runs = len(self._sub_run_to_spectrum)
            self._diff_data_set[mask_id] = numpy.ndarray(shape=(num_sub_runs, hist.shape[0]), dtype=hist.dtype)

        # Check array shape
        if self._diff_data_set[mask_id].shape[1] != hist.shape[0]:
            raise RuntimeError('blabla')

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
