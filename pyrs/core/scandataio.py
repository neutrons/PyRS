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
        scan_h5 = h5py.File(file_name)
        if 'Diffraction Data' not in scan_h5.keys():
            raise RuntimeError(scan_h5.keys())
        diff_data_group = scan_h5['Diffraction Data']

        # loop through the Logs
        num_logs = len(diff_data_group)
        sample_logs = dict()

        for log_index in range(num_logs):
            log_name_i = 'Log {0}'.format(log_index)
            log_i = diff_data_group[log_name_i]

            for item_name in log_i.keys():
                item_i = log_i[item_name].value

                if isinstance(item_i, numpy.ndarray):
                    if item_name == 'Corrected 2theta':

                        len(item_i.shape) == 1 or log_i[item_name].value.shape[1] == 1
                        vec_2theta = log_i[item_name].value.flatten('F')
                    elif item_name == 'Corrected Intensity':
                        vec_y = log_i[item_name].value.flatten('F')
                else:
                    # 1 dimensional (single data point)
                    if item_name not in sample_logs:
                        # create entry as ndarray if it does not exist
                        if isinstance(item_i, str):
                            sample_logs[item_name] = numpy.ndarray(shape=(num_logs,), dtype=object)
                        else:
                            sample_logs[item_name] = numpy.ndarray(shape=(num_logs,), dtype=item_i.dtype)

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
