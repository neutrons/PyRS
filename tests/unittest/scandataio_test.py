#!/usr/bin/python
from pyrs.core import scandataio
import os
import matplotlib.pyplot as plt
import numpy

# default testing directory is ..../PyRS/
print (os.getcwd())

# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print (os.path.exists(test_data))

diff_data_file = scandataio.DiffractionDataFile()
diff_data_dict, sample_log_dict = diff_data_file.load_rs_file(test_data)

for log_name in sample_log_dict:
    print log_name

dev_log_list = diff_data_file.find_changing_logs(sample_log_dict)
for tup in dev_log_list:
    print ('Log {0}: Dev = {1}'.format(tup[1], tup[0]))

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

# vec_2theta, vec_intensity = diff_data_dict[4]
# plt.plot(vec_2theta, vec_intensity)
#
# plt.show()



