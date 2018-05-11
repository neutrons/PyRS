#!/usr/bin/python
from pyrs.core import pyrscore
import os

# default testing directory is ..../PyRS/
print (os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def test_main():
    """
    test main
    :return:
    """
    rs_core = pyrscore.PyRsCore()

    # load data
    data_key, message = rs_core.load_rs_raw(test_data)
    print ('Data reference ID: {0}'.format(data_key))
    # scan log data range
    print ('Scan log index range: {0}'.format(rs_core.data_center.get_scan_range(data_key)))
    # sample logs
    print ('Sample logs: {0}'.format(rs_core.data_center.get_sample_logs_list(data_key, can_plot=True)))
    # fit peaks
    rs_core.fit_peaks(data_key, None, 'Gaussian', 'Linear', [80, 90])
    vec_index = rs_core.data_center.get_scan_range(data_key)
    print (type(vec_index))
    vec_chi2 = rs_core.get_peak_fit_param_value(data_key, 'chi2')
    print (len(vec_chi2))

    return


if __name__ == '__main__':
    test_main()
