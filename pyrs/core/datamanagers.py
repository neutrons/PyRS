# Data manager
import numpy
import os
from pyrs.utilities import checkdatatypes


class ScanDataHolder(object):
    """
    holder for a single scan data which contains diffraction data and sample logs
    """
    def __init__(self, file_name, diff_data_dict, sample_log_dict):
        """

        :param file_name:
        :param diff_data_dict:
        :param sample_log_dict:
        """
        # check
        checkdatatypes.check_file_name(file_name)
        checkdatatypes.check_dict('Diffraction data dictionary', diff_data_dict)
        checkdatatypes.check_dict('Sample log dictionary', sample_log_dict)

        # check diffraction data dictionary
        for log_index in diff_data_dict:
            checkdatatypes.check_int_variable('Diffraction data log index', log_index, value_range=[0, None])
            diff_tup = diff_data_dict[log_index]
            checkdatatypes.check_tuple('Diffraction data set', diff_tup, 2)
            vec_2theta = diff_tup[0]
            vec_intensity = diff_tup[1]
            checkdatatypes.check_numpy_arrays('Vector for 2theta and intensity', [vec_2theta, vec_intensity], dimension=1,
                                        check_same_shape=True)

        # store a list of all existing scan (log) indexes in ascending order
        self._scan_log_indexes = diff_data_dict.keys()
        self._scan_log_indexes.sort()
        self._scan_log_index_vec = numpy.array(self._scan_log_indexes)

        # check sample log dictionary
        for log_name in sample_log_dict:
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


class RawDataManager(object):
    """
    Raw diffraction data (could be corrected) manager for all loaded and (possibly) processed data
    """
    def __init__(self):
        """
        initialization
        """
        self._data_dict = dict()  # key = data key, data = data class
        self._file_ref_dict = dict()  # key = file name, value = data key / reference ID

        return

    def _check_data_key(self, data_key):
        """
        check whether a data key is valid and exist
        :param data_key:
        :return:
        """
        checkdatatypes.check_string_variable('Data reference ID', data_key)
        if data_key not in self._data_dict:
            raise RuntimeError('Data reference ID (key) {0} does not exist.'.format(data_key))

        return

    def add_raw_data(self, diff_data_dict, sample_log_dict, h5file, replace=True):
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

    def get_data_set(self, data_ref_id, scan_index):
        """
        get data set of a single diffraction pattern
        :param data_ref_id:
        :param scan_index:
        :return:
        """
        # check input
        self._check_data_key(data_ref_id)
        data_set = self._data_dict[data_ref_id].get_diff_data(scan_index)

        return data_set

    def get_sample_logs_list(self, data_key, can_plot):
        """
        get the list of sample logs' names
        :param data_key:
        :param can_plot: True for log that can be plotted (no object type); Otherwise, all sample logs
        :return: list of strings
        """
        self._check_data_key(data_key)

        return self._data_dict[data_key].get_sample_log_names(can_plot)

    def get_sample_log_values(self, data_key, sample_log_name):
        """
        Get ONE INDIVIDUAL sample log's values as a vector
        :param data_key:
        :param sample_log_name:
        :return:
        """
        self._check_data_key(data_key)

        return self._data_dict[data_key].sample_log_values(sample_log_name)

    def get_scan_index_logs_values(self, data_key, log_name_list):
        """
        Get a set of sample logs' values and return with scan indexes
        :param data_key:
        :param log_name_list:
        :return:
        """
        # check input
        sample_log_list = self.get_sample_logs_list(data_key, True)
        checkdatatypes.check_list('Sample logs names', log_name_list, sample_log_list)
        for target_name, log_name in log_name_list:
            # need  more check
            if log_name not in sample_log_list:
                raise RuntimeError('Log {0} not in {1}'.format(log_name, sample_log_list))

        # go through scan index
        scan_logs_dict = dict()
        for scan_index in self._data_dict[data_key].get_sample_log_index_range():
            entry_dict = dict()
            for log_name in log_name_list:
                log_value = self._data_dict[data_key].sample_log_values(log_name)
                print ('[DB...INFO] Log value = {0} of type {1}'.format(log_value, type(log_value)))
                entry_dict[log_name] = log_value[0]
            # END-FOR

            scan_logs_dict[scan_index] = entry_dict
        # END-FOR

        return scan_logs_dict

    def get_scan_range(self, data_key):
        """
        get the range of scan log indexes
        :param data_key:
        :return: list of scan log indexes
        """
        self._check_data_key(data_key)
        return self._data_dict[data_key].get_scan_log_index_range()

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

        return sample_log_name in self._data_dict[data_reference_id].sample_log_names
