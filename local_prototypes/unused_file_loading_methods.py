
# From reduction_manager.py
def load_data(self, data_file_name, sub_run=None, target_dimension=None, load_to_workspace=True):
    """
    Load data set and
    - determine the instrument size (for PyHB2BReduction and initialize the right one if not created)
    :param data_file_name:
    :param sub_run: integer for sun run number
    :param target_dimension: if TIFF, target dimension will be used to bin the data
    :param load_to_workspace: if TIFF, option to create a workspace
    :return: data ID to look up, 2-theta (None if NOT recorded)
    """
    # check inputs
    checkdatatypes.check_file_name(data_file_name, True, False, False, 'Data file to load')

    # check file type
    if data_file_name.endswith('.nxs.h5'):
        file_type = 'nxs.h5'
    else:
        file_type = data_file_name.split('.')[-1].lower()

    # load
    two_theta = None

    if file_type == 'tif' or file_type == 'tiff':
        # TIFF
        data_id = self._load_tif_image(data_file_name, target_dimension, rotate=True,
                                       load_to_workspace=load_to_workspace)

    elif file_type == 'bin':
        # SPICE binary
        data_id = self._load_spice_binary(data_file_name)

    elif file_type == 'hdf5' or file_type == 'h5':
        # PyRS HDF5
        data_id, two_theta = self._load_pyrs_h5(data_file_name, sub_run, load_to_workspace)

    elif file_type == 'nxs.h5' or file_type == 'nxs':
        # Event NeXus
        data_id = self._load_nexus(data_file_name)

    else:
        # not supported
        raise RuntimeError('File type {} from input {} is not supported.'.format(file_type, data_file_name))

    self._last_loaded_data_id = data_id

    return data_id, two_theta


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


class HidraWorkspace:
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
