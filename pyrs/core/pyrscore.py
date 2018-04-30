# This is the core of PyRS serving as the controller of PyRS and hub for all the data
import scandataio
import datamanagers


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
    def data_center(self):
        """
        return handler to data center which stores and manages all the data loaded and processed
        :return:
        """
        return self._data_manager

    def load_rs_raw(self, h5file):
        """
        load HB2B raw h5 file
        :param h5file:
        :return: str as message
        """
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file(h5file)

        data_key = self.data_center.add_raw_data(diff_data_dict, sample_log_dict, h5file, replace=True)
        message = 'Load {0} with reference ID {1}'.format(h5file, data_key)

        return message
