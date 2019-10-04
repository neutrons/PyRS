#!/usr/bin/python
# System (library) test for classes and methods that will be used to calculate strain and stress
import os
import sys
from pyrs.core import pyrscore
from pyrs.utilities import rs_project_file
from pyrs.core import strain_stress_calculator
from pyrs.utilities import rs_scan_io


def contain_strain_stress_main():
    """
    Main method to test class and methods to calculate
    :return:
    """
    # Source data
    dir_file = dict()
    dir_file[1] = 'tests/temp/16-1_LD.cor_Log.gaussian.hdf5'
    dir_file[2] = 'tests/temp/16-1_ND.cor_Log.gaussian.hdf5'
    dir_file[3] = 'tests/temp/16-1_TD.cor_Log.gaussian.hdf5'
    for dir_i in [1, 2, 3]:
        if not os.path.exists(dir_file[dir_i]):
            print('[ERROR] File {} does not exist.  Current working directory: {}'
                  ''.format(dir_file[dir_i], os.getcwd()))
            sys.exit(-1)
    # END-FOR

    # Start a new project
    hidra_project_file = rs_project_file.HydraProjectFileMode('tests/testdata/strain_stress_test.hdf5',
                                                              rs_project_file.HydraProjectFileMode.READONLY)

    # Get parameters of sample logs
    for dir_i in [1, 2, 3]:
        diff_data_dict, sample_log_list = rs_scan_io.load_rs_file(dir_file[1])
        # TODO - FUTURE - Continue from here

    # set up the combo box for 3 directions
    sample_logs_list = self._core.strain_stress_calculator.get_sample_logs_names(direction, to_set=False)

    self._setup_sample_logs_combo_box(sample_logs_list, direction)

    return


if __name__ == '__main__':
    contain_strain_stress_main()
