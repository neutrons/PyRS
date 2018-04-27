import os
import rshelper as helper
import h5py
import numpy


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

        # access sub tree
        scan_h5 = h5py.File(file_name, 'r')
        diff_data_dir = scan_h5['Diffraction Data']

        # loop through the Logs
        num_logs = diff_data_dir.size()
        for log_index in range(num_logs):
            log_name_i = 'Log {0}'.format(log_index)
            log_i = diff_data_dir[log_name_i]

            for item_name in log_i.keys():
                if item_name == 'Corrected 2theta':
                    vec_2theta = log_i[item_name].value
                elif item_name == 'Corrected Intensity':
                    vec_y = log_i[item_name]
                else:
                    # 1 dimensional (single data point)
                    if item_name not in sample_logs:
                        # create entry as ndarray if it does not exist
                        log_type = log_i[item_name].data.type()
                        sample_logs[item_name] = numpy.ndarray(shape=(num_logs,), dtype=log_type)

                    # add the log
                    sample_logs[item_name][log_index] = log_i[item_name].value
                # END-IF
            # END-FOR
        # END-FOR

        return

    def save_rs_file(self, file_name):
        """

        :param file_name:
        :return:
        """

        return
