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

    def write_result(self):
        """
        write (append) the peak fitting result to input HDF5 for further data reduction such as
        calculating stress/strain.
        The file format shall be documented as a standard
        :return:
        """
        # TODO - 20180727 - Implement!

# In [17]: log97entry.create_group('peak_fit')
# Out[17]: <HDF5 group "/Diffraction Data/Log 97/peak_fit" (0 members)>
#
# In [18]: peak_fit_97 = log
# %logoff     %logon      %logstart   %logstate   %logstop    log97entry  log98entry
#
# In [18]: peak_fit_97 = log97entry['peak_fit']
#
# In [19]: peak_fit_97['type'
#    ....: ] = 'Gaussian'
#
# In [20]: peak_fit_97['Height'] = 45.0
#
# In [21]: peak_fit_97['Chi2'] = 56.3
#
# In [22]: rwfile.close()



