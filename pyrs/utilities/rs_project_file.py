# This is rs_scan_io.DiffractionFile's 2.0 version
import os
import  h5py
import checkdatatypes


class HydraProjectFile(object):
    """ Read and/or write an HB2B project to an HDF5 with entries for detector counts, sample logs, reduced data,
    fitted peaks and etc.
    All the import/export information will be buffered in order to avoid exception during operation

    File structure:
    - experiment
        - scans (raw counts)
        - logs
    - instrument
        - calibration
    -

    """
    def __init__(self, project_file_name, mode):
        """ initialization
        """
        # check
        checkdatatypes.check_string_variable('Project file name', project_file_name)
        checkdatatypes.check_string_variable('Project I/O mode', mode, ['r', 'w', 'a'])  # read/write/append

        # open file for H5
        self._project_h5 = None
        self._is_writable = False

        if mode == 'r':
            # read: check file existing?
            checkdatatypes.check_file_name(project_file_name, True, False, False, 'Read-only Project file')
            self._project_h5 = h5py.File(project_file_name, mode='r')

        elif mode == 'w':
            # write
            checkdatatypes.check_file_name(project_file_name, False, True, False, 'Write-only project file')
            self._is_writable = True
            self._project_h5 = h5py.File(project_file_name, mode='w')
            self._init_file()

        else:
            # append
            checkdatatypes.check_file_name(project_file_name, True, True, False, '(Append-mode) project file')
            self._is_writable = True
            self._project_h5 = h5py.File(project_file_name, mode='a')

        return

    def _init_file(self):
        """
        initialize the current opened project file from scratch by opening it
        :return:
        """
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'

        # data
        exp_entry = self._project_h5.create_group('experiment')
        exp_entry.create_group('sub-runs')
        exp_entry.create_group('logs')

        # instrument
        instrument = self._project_h5.create_group('instrument')
        instrument.create_group('calibration')

        # reduced data
        reduced_data = self._project_h5.create_group('reduced data')
        reduced_data.create_group('2theta')
        reduced_data.create_group('d-spacing')

        return

    def add_scan_counts(self, scan_index, counts_array):
        """ add raw detector counts collected in a single scan/Pt
        :return:
        """
        # check
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'
        checkdatatypes.check_int_variable('Sub-run index', scan_index, (0, None))

        # create group
        scan_i_group = self._project_h5['experiment']['sub-runs'].create_group('{:04}'.format(scan_index))
        scan_i_group.create_dataset('counts', data=counts_array)

        return

    def add_diffraction_data(self, sub_run_index, vec_x, vec_y, unit):
        """ add reduced and corrected diffraction data in specified unit
        :param unit:
        :return:
        """
        # TODO - TONIGHT 0 - Many checks...
        checkdatatypes.check_string_variable('Unit', unit, ['2theta', 'd-spacing'])

        if 'reduced data' not in self._project_h5.keys():
            raise RuntimeError('No reduced data')
        elif unit not in self._project_h5['reduced data']:
            raise RuntimeError('{} missing'.format(unit))

        # create group
        reduced_data = self._project_h5['reduced data'][unit].create_group('{:04}'.format(sub_run_index))
        reduced_data.create_dataset('x', data=vec_x)
        reduced_data.create_dataset('y', data=vec_y)

        return

    def add_experiment_information(self, log_name, log_value_array):
        """ add information about the experiment including scan indexes, sample logs, 2theta and etc
        :param log_name: name of the sample log
        :param log_value_array:
        :return:
        """
        # check
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'
        checkdatatypes.check_string_variable('Log name', log_name)

        self._project_h5['experiment']['logs'].create_dataset(log_name, data=log_value_array)

        return

    def close(self):
        """
        close file
        :return:
        """
        assert self._project_h5 is not None, 'cannot be None'

        self._project_h5.close()

        return

    def get_log_value(self, log_name, sub_run):
        assert self._project_h5 is not None, 'blabla'

        sub_run_index = sub_run - 1  # FIXME TODO - TONIGHT - correct shall be : sub_run -> sub run index -> log[scan index]
        log_value = self._project_h5['experiment']['logs'][log_name][sub_run_index]

        return log_value

    def get_scan_counts(self, sub_run):
        """
        get the raw detector counts
        :return:
        """
        assert self._project_h5 is not None, 'blabla'
        checkdatatypes.check_int_variable('sun run', sub_run, (0, None))

        counts = self._project_h5['experiment']['sub-runs']['{:04}'.format(sub_run)]['counts'].value

        return counts

    def get_sub_runs(self):
        """
        get list of the sub runs
        :return:
        """
        sub_runs_str_list = self._project_h5['experiment']['sub-runs']

        sub_run_list = [None] * len(sub_runs_str_list)
        for index, sub_run_str in enumerate(sub_runs_str_list):
            sub_run_list[index] = int(sub_run_str)

        return sub_run_list

    def set_geometry(self):
        """
        set the instrument geometry information with calibration
        :return:
        """
        return

    def save_hydra_project(self):
        """
        convert all the information about project to HDF file
        :return:
        """
        return


def test_main():
    project_h5 = h5py.File(project_file_name, 'r')
    scan_index = project_h5['experiment']['logs']['Scan Index'][0]
    assert scan_index == 1
    print project_h5['experiment'].keys()
    counts_vec_index001 = project_h5['experiment']['sub-runs']['{:04}'.format(scan_index)]['counts'].value
    print (counts_vec_index001.max())
    two_theta_value = project_h5['experiment']['logs']['2Theta'][0]
