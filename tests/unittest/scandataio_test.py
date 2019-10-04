#!/usr/bin/python
from pyrs.core import rs_scan_io
import os
import matplotlib.pyplot as plt
import numpy

# default testing directory is ..../PyRS/
print(os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))

# Load data:
diff_data_file = rs_scan_io.DiffractionDataFile()
diff_data_dict, sample_log_dict = diff_data_file.load_rs_file(test_data)

# Look at the diffraction data
if True:
    vec_2theta, vec_intensity = diff_data_dict[4]
    plt.plot(vec_2theta, vec_intensity)

    # prototype on fitting
    import scipy
    import scipy.optimize
    import numpy

    # def func(x, a, b, c, x0):
    #     return a * numpy.exp(-b * (x-x0)**2) + c

    def func1(x, a, b, x0):
        return a * numpy.exp(-b * (x-x0)**2)

    def func2(x, c):
        return c

    p0 = [300, 1, 82, 40]
    fit_results = scipy.optimize.curve_fit(func1+func2, vec_2theta, vec_intensity, p0=p0)
    print(fit_results)

    fit_result = fit_results[0]
    vec_model = func(vec_2theta, fit_result[0], fit_result[1], fit_result[2], fit_result[3])
    plt.plot(vec_2theta, vec_model)

    # scipy.optimize.curve_fit(func, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True,
    #                          bounds=(-inf, inf), method=None, jac=None, **kwargs)


# Look at the log data
if False:
    for log_name in sample_log_dict:
        print log_name

    dev_log_list = diff_data_file.find_changing_logs(sample_log_dict)
    for tup in dev_log_list:
        print('Log {0}: Dev = {1}'.format(tup[1], tup[0]))

    omega_vector = sample_log_dict['omega']
    log_index_vector = numpy.arange(len(omega_vector))
    log_vector = sample_log_dict['vx']
    plt.plot(log_index_vector, log_vector)
    plt.show()

    # plt.plot(log_index_vector, omega_vector)
    #
    # time_stamp_vector = sample_log_dict['Time Stamp']
    # plt.plot(log_index_vector, time_stamp_vector)
    #
    # mrot_vector = sample_log_dict['mrot']
    # plt.plot(log_index_vector, mrot_vector)

plt.show()
