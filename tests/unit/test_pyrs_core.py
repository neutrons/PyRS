#!/usr/bin/python
from pyrs.core import pyrscore
import os


def test_main():
    """
    test main
    :return:
    """
    rs_core = pyrscore.PyRsCore()

    # pre-requisite is that the data file exists
    test_data = os.path.join(os.getcwd(), 'tests', 'data', 'BD_Data_Log.hdf5')
    assert os.path.exists(test_data), 'File does not exist'

    # load data
    data_key, message = rs_core.load_hidra_project(test_data)
    print('Data reference ID: {0}'.format(data_key))
    # scan log data range
    print('Scan log index range: {0}'.format(rs_core.data_center.get_scan_range(data_key)))
    # sample logs
    print('Sample logs: {0}'.format(rs_core.data_center.get_sample_logs_names(data_key, can_plot=True)))
    # fit peaks
    rs_core.fit_peaks(data_key, None, 'Gaussian', 'Linear', [80, 90])
    vec_index = rs_core.data_center.get_scan_range(data_key)
    print(type(vec_index))
    vec_chi2 = rs_core.get_peak_fit_param_value(data_key, 'chi2')
    print(len(vec_chi2))

    return


if __name__ == '__main__':
    test_main()
