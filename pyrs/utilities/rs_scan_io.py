from pyrs.utilities import checkdatatypes
import h5py
import numpy
import six


def load_rs_file(file_name):
    """ parse h5 file
    Note: peak_fit data are in sample log dictionary
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
        if 'peak_fit' not in sample_log_dict:
            sample_log_dict['peak_fit'] = dict()
            need_init = True

        for par_name in h5_group.keys():
            # get value
            par_value = h5_group[par_name].value
            # init sample logs vector if needed
            if need_init:
                sample_log_dict['peak_fit'][par_name] = numpy.ndarray(shape=(total_scans,), dtype=par_value.dtype)
            # set value
            sample_log_dict['peak_fit'][par_name][log_index] = par_value

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

    try:
        for scan_log_index in range(num_scan_logs):
            log_name_i = 'Log {0}'.format(scan_log_index)
            h5_log_i = diff_data_group[log_name_i]

            vec_2theta = None
            vec_y = None

            for item_name in h5_log_i.keys():
                # special peak fit
                if item_name == 'peak_fit':
                    add_peak_fit_parameters(sample_logs, h5_log_i[item_name], scan_log_index,
                                            total_scans=num_scan_logs)
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
                        if isinstance(item_i, six.string_types):
                            # string can only be object type
                            sample_logs[item_name_str] = numpy.ndarray(shape=(num_scan_logs,), dtype=object)
                        else:
                            # raw type
                            try:
                                sample_logs[item_name_str] = numpy.ndarray(shape=(num_scan_logs,),
                                                                           dtype=item_i.dtype)
                            except AttributeError as att_err:
                                err_msg = 'Item {} with value {} is a unicode object and cannot be converted to ' \
                                          'ndarray due to \n{}'.format(item_name_str, item_i, att_err)
                                raise AttributeError(err_msg)

                    # add the log
                    sample_logs[item_name_str][scan_log_index] = h5_log_i[item_name].value

            # record 2theta-intensity
            if vec_2theta is None or vec_y is None:
                raise RuntimeError('Log {0} does not have either Corrected 2theta or Corrected Intensity'
                                   ''.format(scan_log_index))
            else:
                diff_data_dict[scan_log_index] = vec_2theta, vec_y
    except KeyError as key_error:
        raise RuntimeError('Failed to load {} due to {}'.format(file_name, key_error))

    return diff_data_dict, sample_logs
