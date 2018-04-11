# This module serves as the file I/O controller
import scandataio


class FileReadWriteController(object):
    """
    File IO controller
    """
    def __index__(self, rs_core):
        """

        :param rs_core:
        :return:
        """
        assert rs_core is not None, 'PyRS core cannot be a None.'

        self._rs_core = rs_core

        return

    def import_scan_data(self, file_name):
        """

        :param file_name:
        :return:
        """

        return


    def save_scan_data(self, scan_data_set, file_name):
        """
        save scan data
        :param scan_data_set:
        :param file_name:
        :return:
        """