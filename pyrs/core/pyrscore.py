# This is the core of PyRS serving as the controller of PyRS and hub for all the data


class PyRsCore(object):
    """
    PyRS core
    """
    def __init__(self):
        """
        initialization
        """

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
        return self._data_center