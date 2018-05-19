# This is the core of PyRS serving as the controller of PyRS and hub for all the data
import scandataio
import datamanagers
import peakfitengine
import rshelper
import numpy as np
import mantid_fit_peak


class PyRsCore(object):
    """
    PyRS core
    """
    def __init__(self):
        """
        initialization
        """
        # declaration of class members
        self._file_io_controller = scandataio.DiffractionDataFile()  # a I/O instance for standard HB2B file
        self._data_manager = datamanagers.RawDataManager()

        # working environment
        self._working_dir = 'tests/testdata/'

        # current/default status
        self._curr_data_key = None
        self._last_optimizer = None

        return

    @property
    def file_controller(self):
        """
        return handler to data loader and saver
        :return:
        """
        return self._file_io_controller

    @property
    def peak_fitting_controller(self):
        """
        return handler to peak fitting manager
        :return:
        """
        return self._peak_fitting_controller

    @property
    def current_data_reference_id(self):
        # TODO
        return self._curr_data_key

    @property
    def data_center(self):
        """
        return handler to data center which stores and manages all the data loaded and processed
        :return:
        """
        return self._data_manager

    @property
    def working_dir(self):
        """
        get working directory
        :return:
        """
        return self._working_dir

    @working_dir.setter
    def working_dir(self, user_dir):
        """
        set working directory
        :param user_dir:
        :return:
        """
        rshelper.check_file_name('Working directory', user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    def fit_peaks(self, data_key, scan_index, peak_type, background_type, fit_range):
        """
        fit a single peak of a measurement in a multiple-log scan
        :param data_key:
        :param scan_index:
        :param peak_type:
        :param background_type:
        :param fit_range
        :return:
        """
        # TODO check inputs
        rshelper.check_string_variable('Peak type', peak_type)
        rshelper.check_string_variable('Background type', background_type)

        # get scan indexes
        if scan_index is None:
            scan_index_list = self._data_manager.get_scan_range(data_key)
        elif isinstance(scan_index, int):
            # check range
            scan_index_list = [scan_index]
        elif isinstance(scan_index, list):
            scan_index_list = scan_index
        else:
            raise RuntimeError('Scan index ({0}) is not supported.'.format(scan_index))

        # get data
        diff_data_list = list()
        for log_index in scan_index_list:
            diff_data = self._data_manager.get_data_set(data_key, log_index)
            diff_data_list.append(diff_data)
        # END-FOR

        ref_id = 'TODO FIND A GOOD NAMING CONVENTION'
        peak_optimizer = mantid_fit_peak.MantidPeakFitEngine(diff_data_list, ref_id=ref_id)

        # TODO FIXME: A quick observe?  Use center of mass!
        peak_center_vec = peak_optimizer.calculate_center_of_mass()

        peak_optimizer.fit_peaks(peak_type, background_type, peak_center_vec, fit_range, None)

        self._last_optimizer = peak_optimizer

        return ref_id

    def get_fit_parameters(self, data_key):
        # TODO
        return self._last_optimizer.get_function_parameter_names()

    def get_peak_fit_param_value(self, data_key, param_name):
        # TODO
        return self._last_optimizer.get_fitted_params(param_name)

    def get_diff_data(self, data_key, scan_log_index):
        """
        get diffraction data of a certain
        :param data_key:
        :param scan_log_index:
        :return:
        """
        # get data key
        if data_key is None:
            data_key = self._curr_data_key
            if data_key is None:
                raise RuntimeError('There is no current loaded data.')
        # END-IF

        # get data
        diff_data_set = self._data_manager.get_data_set(data_key, scan_log_index)

        return diff_data_set

    def get_modeled_data(self, data_key, scan_log_index):
        # TODO
        # get data key
        if data_key is None:
            data_key = self._curr_data_key
            if data_key is None:
                raise RuntimeError('There is no current loaded data.')
        # END-IF

        # TODO : better data manager!
        if self._last_optimizer is not None:
            data_set = self._last_optimizer.get_calculated_peak(scan_log_index)
        else:
            data_set = None
            print ('[LAST OPTIMIZER IS NONE')

        return data_set

    def load_rs_raw(self, h5file):
        """
        load HB2B raw h5 file
        :param h5file:
        :return: str as message
        """
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file(h5file)

        data_key = self.data_center.add_raw_data(diff_data_dict, sample_log_dict, h5file, replace=True)
        message = 'Load {0} (Ref ID {1})'.format(h5file, data_key)

        # set to current key
        self._curr_data_key = data_key

        return data_key, message

    def save_nexus(self, data_key, file_name):
        # TODO
        data_key = self._curr_data_key
        # TODO check data key with last optimizer
        matrix_name = self._last_optimizer.get_data_workspace_name()

        import scandataio
        scandataio.save_mantid_nexus(matrix_name, file_name)
