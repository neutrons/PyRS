# Data manager
import numpy
import os
import rshelper


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
        rshelper.check_file_name(file_name)
        rshelper.check_dict('Diffraction data dictionary', diff_data_dict)
        rshelper.check_dict('Sample log dictionary', sample_log_dict)

        # check diffraction data dictionary
        for log_index in diff_data_dict:
            rshelper.check_int_variable('Diffraction data log index', log_index, value_range=[0, None])
            diff_tup = diff_data_dict[log_index]
            rshelper.check_tuple('Diffraction data set', diff_tup, 2)
            vec_2theta = diff_tup[0]
            vec_intensity = diff_tup[1]
            rshelper.check_numpy_arrays('Vector for 2theta and intensity', [vec_2theta, vec_intensity], dimension=1,
                                        check_same_shape=True)

        # store a list of all existing scan (log) indexes in ascending order
        self._scan_log_indexes = diff_data_dict.keys()
        self._scan_log_indexes.sort()
        self._scan_log_index_vec = numpy.array(self._scan_log_indexes)

        # check sample log dictionary
        for log_name in sample_log_dict:
            rshelper.check_string_variable('Sample log name', log_name)
            log_value_vec = sample_log_dict[log_name]
            rshelper.check_numpy_arrays('Sample log {0} value vector'.format(log_name),
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
        rshelper.check_int_variable('Scan (log) index', scan_log_index, [0, None])
        if scan_log_index not in self._scan_log_indexes:
            raise RuntimeError('User specified scan log with index {0} is not found with range [{1}, {2}]'
                               ''.format(scan_log_index, self._scan_log_indexes[0], self._scan_log_indexes[-1]))

        return self._diff_data_dict[scan_log_index]

    def get_sample_log_value(self, sample_log_name):
        """
        get sample log value
        :param sample_log_name:
        :return:
        """
        rshelper.check_string_variable('Sample log name', sample_log_name)
        if sample_log_name not in self._sample_log_dict:
            raise RuntimeError('Sample log {0} cannot be found.'.format(sample_log_name))

        return self._scan_log_index_vec, self._sample_log_dict[sample_log_name]


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

        return

    def delete_data(self, reference_id):
        """
        delete a data set data key/reference ID
        :param reference_id:
        :return: boolean (False if reference ID is not in this instance)
        """
        # check input: correct type and existed
        rshelper.check_string_variable('Data reference ID', reference_id)
        if reference_id not in self._data_dict:
            return False

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
        rshelper.check_string_variable('Data file name for data reference ID', file_name)

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
        if self.has_data(data_ref_id) is False:
            raise RuntimeError('Data reference ID {0} is not found in data center.'.format(data_ref_id))

        data_set = self._data_dict[data_ref_id].get_diff_data(scan_index)

        return data_set

    def has_data(self, reference_id):
        """
        check whether a data key/reference ID exists
        :param reference_id:
        :return:
        """
        rshelper.check_string_variable('Reference ID', reference_id)

        return reference_id in self._data_dict

    def has_raw_data(self, file_name):
        """
        check whether
        :param file_name:
        :return:
        """