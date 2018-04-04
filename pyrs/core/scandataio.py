import os
import rshelper as helper
import h5py


class DiffractionDataFile(object):
    """
    class to read and write diffraction data file
    """
    def __init__(self):
        """
        initialization
        """
        return

    def import_diffraction_data(self, data_key, data_set, description):
        """

        :param data_key:
        :param data_set:
        :param description:
        :return:
        """
        return

    def load_rs_file(self, file_name):
        """ parse h5 file
        :param file_name:
        :return:
        """
        helper.check_file_name(file_name, check_exist=True)



        return

    def save_rs_file(self, file_name):
        """

        :param file_name:
        :return:
        """

        return
