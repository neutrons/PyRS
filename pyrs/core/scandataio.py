import os
from pyrs.utilities import checkdatatypes
import h5py
import numpy
from shutil import copyfile
from mantid.simpleapi import SaveNexusProcessed
from mantid.api import AnalysisDataService


class DiffractionDataFile(object):
    """
    class to read and write diffraction data file
    """
    def __init__(self):
        """
        initialization
        """
        return

    # TESTME - Recently implemented
    @staticmethod
    def export_peak_fit(src_rs_file_name, target_rs_file_name, peak_fit_dict):
        """
        export peak fitting result to a RS (residual stress) intermediate file
        :param src_rs_file_name:
        :param target_rs_file_name:
        :param peak_fit_dict:
        :return:
        """
        # check
        checkdatatypes.check_file_name(src_rs_file_name, check_exist=True)
        checkdatatypes.check_file_name(target_rs_file_name, check_writable=True, check_exist=False)

        # copy the file?
        if src_rs_file_name != target_rs_file_name:
            copyfile(src_rs_file_name, target_rs_file_name)

        # open file
        target_file = h5py.File(target_rs_file_name, 'r+')
        diff_entry = target_file['Diffraction Data']
        for scan_log_key in diff_entry.keys():
            scan_log_index = int(scan_log_key.split()[1])
            fit_info_i = peak_fit_dict[scan_log_index]
            # add an entry
            diff_entry[scan_log_key].create_group('peak_fit')
            for key in fit_info_i:
                print ('[db...bat] fit info: {}'.format(fit_info_i))
                diff_entry[scan_log_key]['peak_fit'][key] = fit_info_i[key]
            # END-FOR
        # END-FOR

        target_file.close()

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
        :return: 2-tuple as diff_data_dict, sample_logs
        """
        def add_peak_fit_parameters(sample_log_dict, h5_group, log_index, total_scans):
            """
            add peak fitted parameters value to sample log dictionary
            :param sample_log_dict:
            :param h5_group:
            :param log_index:
            :param total_scans
            :return:
            """
            # need it?
            need_init = False
            if 'peak_fit' not in sample_logs:
                sample_logs['peak_fit'] = dict()
                need_init = True

            for par_name in h5_group.keys():
                # get value
                par_value = h5_group[par_name].value
                # init sample logs vector if needed
                if need_init:
                    sample_logs['peak_fit'][par_name] = numpy.ndarray(shape=(total_scans,), dtype=par_value.dtype)
                # set value
                sample_logs['peak_fit'][par_name][log_index] = par_value
            # END-FOR

            return

        checkdatatypes.check_file_name(file_name, check_exist=True)

        # access sub tree
        scan_h5 = h5py.File(file_name)
        if 'Diffraction Data' not in scan_h5.keys():
            raise RuntimeError(scan_h5.keys())
        diff_data_group = scan_h5['Diffraction Data']

        # loop through the Logs
        num_scan_logs = len(diff_data_group)
        sample_logs = dict()
        diff_data_dict = dict()

        for scan_log_index in range(num_scan_logs):
            log_name_i = 'Log {0}'.format(scan_log_index)
            h5_log_i = diff_data_group[log_name_i]

            vec_2theta = None
            vec_y = None

            for item_name in h5_log_i.keys():
                # special peak fit
                if item_name == 'peak_fit':
                    add_peak_fit_parameters(sample_logs, h5_log_i[item_name], scan_log_index, total_scans=num_scan_logs)
                    continue

                # get value
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
                            sample_logs[item_name_str] = numpy.ndarray(shape=(num_scan_logs,), dtype=object)
                        else:
                            # raw type
                            sample_logs[item_name_str] = numpy.ndarray(shape=(num_scan_logs,), dtype=item_i.dtype)

                    # add the log
                    sample_logs[item_name_str][scan_log_index] = h5_log_i[item_name].value
                # END-IF
            # END-FOR

            # record 2theta-intensity
            if vec_2theta is None or vec_y is None:
                raise RuntimeError('Log {0} does not have either Corrected 2theta or Corrected Intensity'
                                   ''.format(scan_log_index))
            else:
                diff_data_dict[scan_log_index] = vec_2theta, vec_y

        # END-FOR

        return diff_data_dict, sample_logs

    def load_rs_file_set(self, file_name_list):
        """ Load a set of Residual Stress's intermediate data files
        :param file_name_list:
        :return:
        """
        # sort file name by order
        file_name_list.sort()

        # prepare the data structures
        num_logs = len(file_name_list)
        sample_logs_set = dict()
        diff_data_dict_set = dict()

        for det_id, file_name in file_name_list:
            checkdatatypes.check_file_name(file_name, check_exist=True)

            # define single file dictionary
            sample_logs = dict()
            diff_data_dict = dict()

            # access sub tree
            scan_h5 = h5py.File(file_name)
            if 'Diffraction Data' not in scan_h5.keys():
                raise RuntimeError(scan_h5.keys())
            diff_data_group = scan_h5['Diffraction Data']
            print ('File: {0}'.format(file_name))

            # loop through the Logs
            h5_log_i = diff_data_group

            # get 'Log #'
            log_index_vec = h5_log_i['Log #'].value[0, 0].astype(int)
            # print ('Log #: Shape = {0}. Value = {1}'.format(log_index_vec.shape, log_index_vec))

            for item_name in h5_log_i.keys():
                # skip log index
                if item_name == 'Log #':
                    continue

                item_i = h5_log_i[item_name].value
                if isinstance(item_i, numpy.ndarray):
                    # case for diffraction data
                    if item_name == 'Corrected Diffraction Data':
                        print ('Item {0}: shape = {1}'.format(item_name, item_i.shape))
                        # corrected 2theta and diffraction
                        if item_i.shape[2] != len(log_index_vec):
                            raise RuntimeError('File {0}: Corrected Diffraction Data ({1}) has different '
                                               'number of entries than log indexes ({2})'
                                               ''.format(file_name, item_i.shape[2], len(log_index_vec)))
                        for i_log_index in range(len(log_index_vec)):
                            vec_2theta = item_i[:, 0, i_log_index]
                            vec_intensity = item_i[:, 1, i_log_index]
                            diff_data_dict[log_index_vec[i_log_index]] = vec_2theta, vec_intensity
                        # END-FOR

                    elif item_name == 'Corrected Intensity':
                        raise NotImplementedError('Not supposed to be here!')
                    else:
                        # sample log data
                        vec_sample_i = item_i[0, 0].astype(float)
                        # dictionary = dict(zip(log_index_vec, vec_sample_i))
                        # sample_logs[str(item_name)] = dictionary  # make sure the log name is a string
                        sample_logs[str(item_name)] = vec_sample_i
                    # END-IF-ELSE
                else:
                    # 1 dimensional (single data point)
                    raise RuntimeError('There is no use case for single-value item so far. '
                                       '{0} of value {1} is not supported to parse in.'
                                       ''.format(item_i, item_i.value))
                # END-IF
            # END-FOR

            # conclude for single file
            sample_logs_set[det_id] = sample_logs
            diff_data_dict_set[det_id] = diff_data_dict
        # END-FOR (log_index, file_name)

        return diff_data_dict_set, sample_logs_set

    def save_rs_file(self, file_name):
        """

        :param file_name:
        :return:
        """

        return


def get_temp_directory():
    """
    get a temporary directory to write files
    :return:
    """
    # current workspace first
    temp_dir = os.getcwd()
    if os.access(temp_dir, os.W_OK):
        return temp_dir

    # /tmp/ second
    temp_dir = '/tmp/'
    if os.path.exists(temp_dir):
        return temp_dir

    # last solution: home directory
    temp_dir = os.path.expanduser('~')

    return temp_dir


def save_mantid_nexus(workspace_name, file_name, title=''):
    """
    save workspace to NeXus for Mantid to import
    :param workspace_name:
    :param file_name:
    :param title:
    :return:
    """
    # check input
    checkdatatypes.check_file_name(file_name, check_exist=False,
                                   check_writable=True, is_dir=False)
    checkdatatypes.check_string_variable('Workspace title', title)

    # check workspace
    checkdatatypes.check_string_variable('Workspace name', workspace_name)
    if AnalysisDataService.doesExist(workspace_name):
        SaveNexusProcessed(InputWorkspace=workspace_name,
                           Filename=file_name,
                           Title=title)
    else:
        raise RuntimeError('Workspace {0} does not exist in Analysis data service. Available '
                           'workspaces are {1}.'
                           ''.format(workspace_name, AnalysisDataService.getObjectNames()))

    # END-IF-ELSE

    return
