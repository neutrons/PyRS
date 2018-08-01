# This is the virtual base class as the fitting frame
from pyrs.utilities import checkdatatypes


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
        checkdatatypes.check_list('Data set list', data_set_list)
        checkdatatypes.check_string_variable('Peak fitting reference ID', ref_id)

        # for scipy: keep the numpy array will be good enough
        self._data_set = data_set_list
        self._reference_id = ref_id

        # for fitted result
        self._peak_center_vec = None  # 2D vector for observed center of mass and highest data point

        return

    def export_fit_result(self):
        """
        export fit result for all the peaks
        :return: a dictionary of fitted peak information
        """
        raise NotImplementedError('Virtual base class member method export_fit_result()')

    def fit_peaks(self, peak_function_name, background_function_name, fit_range, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :param background_function_name:
        :param fit_range:
        :param scan_index:
        :return:
        """
        raise NotImplementedError('Virtual base class member method fit_peaks()')

    def get_calculated_peak(self, scan_log_index):
        """
        get the calculated peak's value
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')

    def get_number_scans(self, param_name):
        """
        get the value of a fitted parameter
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_number_scans()')

    def get_number_scans(self):
        """
        get number of scans in input data to fit
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_number_scans()')





