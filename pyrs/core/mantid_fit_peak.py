# Peak fitting engine by calling mantid
from mantid.simpleapi import FitPeaks
import rshelper


class MantidPeakFitEngine(object):
    """
    peak fitting engine class for mantid
    """
    def __init__(self, data_set_list, ref_id):
        """
        initialization
        :param data_set_list:
        :param ref_id:
        :param
        """
        # check
        rshelper.check_numpy_arrays('Data set list', data_set_list, 2, check_same_shape=False)
        rshelper.check_string_variable('Peak fitting reference ID', ref_id)

        self._data_workspace = None
        self._workspace_name = self._get_matrix_name(ref_id)
        self.generate_matrix_workspace(data_set_list)

        return

    def fit_peaks(self, peak_function_name, background_function_name, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :return:
        """
        rshelper.check_string_variable('Peak function name', peak_function_name)
        rshelper.check_string_variable('Background function name', background_function_name)
        if scan_index is not None:
            rshelper.check_int_variable('Scan (log) index', scan_index, value_range=[0, self.get_number_scans()])
            start = scan_index
            stop = scan_index
        else:
            start = 0
            stop = self.get_number_scans() - 1

        # fit
        results = FitPeaks(InputWorkspace=self._workspace_name, OutputWorkspace=self._output_name,
                           StartWorkspaceIndex=start, StopWorkspaceIndex=stop,
                           PeakFunction=peak_function_name, BackgroundType=background_function_name,
                           PeakCenters=observed_peak_centers)

        # process output
        raise NotImplementedError('ASAP')

        return

    def get_number_scans(self):
        """
        get number of scans in input data to fit
        :return:
        """
        if self._data_workspace is None:
            raise RuntimeError('No data is set up!')

        return self._data_workspace.getNumberHistograms()




