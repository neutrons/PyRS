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
        exp_entry.create_group('scans')
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
        checkdatatypes.check_int_variable('Scan index', scan_index, (0, None))

        # create group
        scan_i_group = self._project_h5['experiment']['scans'].create_group('{}'.format(scan_index))
        scan_i_group.create_dataset('counts', counts_array)

        return

    def add_diffraction_data(self, unit):
        """ add reduced and corrected diffraction data in specified unit
        :param unit:
        :return:
        """
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

        self._project_h5['experiment'].create_dataset(log_name, log_value_array)

        return

    def close(self):
        """
        close file
        :return:
        """
        assert self._project_h5 is not None, 'cannot be None'

        self._project_h5.close()

        return

    def get_scan_counts(self):
        """
        get the raw detector counts
        :return:
        """
        return

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
