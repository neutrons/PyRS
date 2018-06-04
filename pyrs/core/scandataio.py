import os
import rshelper as helper
import h5py
import numpy
import sys
home_dir = os.path.expanduser('~')
if home_dir.startswith('/SNS/'):
    # analysis cluster
    # nightly: sys.path.insert(1, '/opt/mantidnightly/bin/')
    # local build
    sys.path.insert(1, '/SNS/users/wzz/Mantid_Project/builds/debug/bin/')
elif home_dir.startswith('/home/wzz'):
    # VZ's workstation
    sys.path.insert(1, '/home/wzz/Mantid_Project/builds/debug-master/bin')
from mantid.simpleapi import SaveNexusProcessed


class DiffractionDataFile(object):
    """
    class to read and write diffraction data file
    """
    def __init__(self):
        """
        initialization
        """
        return

    @staticmethod
    def find_changing_logs(sample_logs):
        """
        find the sample logs with value changed
        :param sample_logs: dict of sample log vector as an outcome from method load_rs_file
        :return:
        """
        dev_log_list = list()
        for log_name in sample_logs:
            # get vector
            log_value_vector = sample_logs[log_name]
            assert isinstance(log_value_vector, numpy.ndarray) and len(log_value_vector.shape) == 1,\
                'Log {0} value'

            # check data type
            if log_value_vector.dtype == object:
                continue

            try:
                # print (log_name, log_value_vector.dtype)
                # print (log_value_vector)
                std_dev = log_value_vector.std()
                dev_log_list.append((std_dev, log_name))
            except ValueError:
                pass
        # END-FOR

        dev_log_list.sort(reverse=True)

        return dev_log_list

    def import_diffraction_data(self, data_key, data_set, description):
        """

        :param data_key:
        :param data_set:
        :param description:
        :return:
        """
        return

    @staticmethod
    def load_rs_file(file_name):
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
        diff_data_dict = dict()

        for log_index in range(num_logs):
            log_name_i = 'Log {0}'.format(log_index)
            h5_log_i = diff_data_group[log_name_i]

            vec_2theta = None
            vec_y = None

            for item_name in h5_log_i.keys():
                item_i = h5_log_i[item_name].value

                if isinstance(item_i, numpy.ndarray):
                    if item_name == 'Corrected 2theta':
                        # corrected 2theta
                        if not (len(item_i.shape) == 1 or h5_log_i[item_name].value.shape[1] == 1):
                            raise RuntimeError('Unable to support a non-1D corrected 2theta entry')
                        vec_2theta = h5_log_i[item_name].value.flatten('F')
                    elif item_name == 'Corrected Intensity':
                        if not (len(item_i.shape) == 1 or h5_log_i[item_name].value.shape[1] == 1):
                            raise RuntimeError('Unable to support a non-1D corrected intensity entry')
                        vec_y = h5_log_i[item_name].value.flatten('F')
                else:
                    # 1 dimensional (single data point)
                    item_name_str = str(item_name)
                    if item_name_str not in sample_logs:
                        # create entry as ndarray if it does not exist
                        if isinstance(item_i, str):
                            # string can only be object type
                            sample_logs[item_name_str] = numpy.ndarray(shape=(num_logs,), dtype=object)
                        else:
                            # raw type
                            sample_logs[item_name_str] = numpy.ndarray(shape=(num_logs,), dtype=item_i.dtype)

                    # add the log
                    sample_logs[item_name_str][log_index] = h5_log_i[item_name].value
                # END-IF
            # END-FOR

            # record 2theta-intensity
            if vec_2theta is None or vec_y is None:
                raise RuntimeError('Log {0} does not have either Corrected 2theta or Corrected Intensity'
                                   ''.format(log_index))
            else:
                diff_data_dict[log_index] = vec_2theta, vec_y

        # END-FOR

        return diff_data_dict, sample_logs

    def save_rs_file(self, file_name):
        """

        :param file_name:
        :return:
        """

        return


def save_mantid_nexus(workspace_name, file_name):
    # TODO
    SaveNexusProcessed(InputWorkspace=workspace_name,
                       Filename=file_name,
                       Title='blabla')
