# Data manager
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

        # check each dictionary
        for log_index in diff_data_dict:
            rshelper.check_int_variable('Diffraction data log index', log_index, value_range=[0, None])
            diff_tup = diff_data_dict[log_index]
            rshelper.check_tuple('Diffraction data set', diff_tup, 2)
            vec_2theta = diff_tup[0]
            vec_intensity = diff_tup[1]
            rshelper.check_numpy_arrays('Vector for 2theta and intensity', [vec_2theta, vec_intensity], dimension=1,
                                        check_same_shape=True)

        # set
        self._file_name = file_name

        return

    @property
    def raw_file_name(self):
        """
        raw data file (HDF5)'s name
        :return:
        """
        return self._file_name

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

        # remove
        file_name = self._data_dict[reference_id].raw_file_name



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