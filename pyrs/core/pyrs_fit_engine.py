# This is the virtual base class as the fitting frame
import rshelper


class RsPeakFitEngine(object):
    """
    virtual peak fit engine
    """
    def __init__(self, data_set_list, ref_id):
        """
        initialization
        :param data_set_list:
        :param ref_id:
        """
        # check
        rshelper.check_list('Data set list', data_set_list)
        rshelper.check_string_variable('Peak fitting reference ID', ref_id)

        # for scipy: keep the numpy array will be good enough
        self._data_set = data_set_list

        return

    def fit_peaks(self, peak_function_name, background_function_name, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :param background_function_name:
        :param scan_index:
        :return:
        """
        raise NotImplementedError('Virtual base class member method fit_peaks()')

    def get_calculated_peak(self, scan_index):
        """
        get the calculated peak's value
        :param scan_index:
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')

    def get_fitted_params(self, param_name):
        """
        get the value of a fitted parameter
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')

    def get_number_scans(self):
        """
        get number of scans in input data to fit
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')



