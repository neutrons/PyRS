# This is rs_scan_io.DiffractionFile's 2.0 version
import os
import h5py
import checkdatatypes
from pyrs.core import instrument_geometry
from enum import Enum
import numpy


class HydraProjectFileMode(Enum):
    """
    Enumerate for file access mode
    """
    READONLY = 1   # read-only
    READWRITE = 2  # read and write
    OVERWRITE = 3  # new file


class DiffractionUnit(Enum):
    """
    Enumerate for diffraction data's unit (2theta or d-spacing)
    """
    TwoTheta = '2theta'
    DSpacing = 'dSpacing'

    @classmethod
    def unit(cls, unit):
        """
        Get the unit in String
        :return:
        """
        if unit == DiffractionUnit.TwoTheta:
            return '2theta'

        return 'dSpacing'


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
        """
        Initialization
        :param project_file_name: project file name
        :param mode: I/O mode
        """
        # check
        checkdatatypes.check_string_variable('Project file name', project_file_name)
        checkdatatypes.check_type('Project I/O mode', mode, HydraProjectFileMode)

        # open file for H5
        self._project_h5 = None
        self._is_writable = False

        if mode == HydraProjectFileMode.READONLY:
            # read: check file existing?
            checkdatatypes.check_file_name(project_file_name, True, False, False, 'Read-only Project file')
            self._project_h5 = h5py.File(project_file_name, mode='r')

        elif mode == HydraProjectFileMode.OVERWRITE:
            # write
            checkdatatypes.check_file_name(project_file_name, False, True, False, 'Write-only project file')
            self._is_writable = True
            self._project_h5 = h5py.File(project_file_name, mode='w')
            self._init_project()

        elif mode == HydraProjectFileMode.READWRITE:
            # append (read and write)
            checkdatatypes.check_file_name(project_file_name, True, True, False, '(Append-mode) project file')
            self._is_writable = True
            self._project_h5 = h5py.File(project_file_name, mode='a')

        else:
            # not supported
            raise RuntimeError('Hydra project file I/O mode {} is not supported'.format(HydraProjectFileMode))

        # more class variables
        self._io_mode = mode

        return

    def _init_project(self):
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
        geometry_group = instrument.create_group('geometry setup')
        geometry_group.create_group('detector')

        # reduced data
        reduced_data = self._project_h5.create_group('diffraction')
        # reduced_data.create_group('2theta')
        # reduced_data.create_group('d-spacing')

        return

    def get_instrument_geometry(self, calibrated):
        """
        Get instrument geometry parameters
        :param calibrated:
        :return: an instance of instrument_geometry.InstrumentSetup
        """
        return

    def set_instrument_geometry(self, instrument_setup):
        """
        Add instrument geometry and wave length information to project file
        :param instrument_setup:
        :return:
        """
        # check inputs
        self._validate_write_operation()
        checkdatatypes.check_type('Instrument geometry setup', instrument_setup, instrument_geometry.HydraSetup)

        # write value to instrument
        instrument_group = self._project_h5['instrument']

        # write attributes
        instrument_group.attrs['name'] = instrument_setup.name

        # get the entry for raw instrument setup
        detector_group = instrument_group['geometry setup']['detector']
        raw_geometry = instrument_setup.get_instrument_geometry(False)
        detector_group.create_dataset('L2', data=numpy.array(raw_geometry.arm_length))
        detector_group.create_dataset('detector size',
                                      numpy.array(instrument_setup.get_instrument_geometry(False).detector_size))
        detector_group.create_dataset('pixel dimension',
                                      numpy.array(instrument_setup.get_instrument_geometry(False).pixel_dimension))

        # wave length
        # wavelength_group = 
        # TODO - NEXT TONIGHT - Implement wave length group

        return

    def set_instrument_calibration(self):
        return

    def set_peak_fit_result(self, sub_run_number, peak_tag, peak_profile, peak_params):

        self._validate_write_operation()

        # access or create node for peak data
        fit_entry = self._project_h5['Peaks']
        peak_entry = self._create_peak_node(sub_run_number, peak_tag, {'profile': peak_profile})

        # add value
        for param_name_i in peak_params:
            peak_entry[param_name_i] = peak_params[param_name_i]
        # END-FOR

        return

    def get_wave_length(self):
        return

    def set_wave_length(self):
        return

    def add_raw_counts(self, sub_run_number, counts_array):
        """ add raw detector counts collected in a single scan/Pt
        :return:
        """
        # check
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'
        checkdatatypes.check_int_variable('Sub-run index', sub_run_number, (0, None))

        # create group
        scan_i_group = self._project_h5['experiment']['sub-runs'].create_group('{:04}'.format(sub_run_number))
        scan_i_group.create_dataset('counts', data=counts_array)

        return

    def add_diffraction_data(self, sub_run_index, vec_x, vec_y, unit):
        """ add reduced and corrected diffraction data in specified unit
        :param unit:
        :return:
        """
        raise RuntimeError('This method is deprecated by set_2theta_diffraction_data and set_d_diffraction_data')
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

        try:
            self._project_h5['experiment']['logs'].create_dataset(log_name, data=log_value_array)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to add log {} due to {}'.format(log_name, run_err))

        return

    def close(self):
        """
        Close file without checking whether the file can be written or not
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

    def set_2theta_diffraction_data(self, sub_run, two_theta_vector, intensity_vector):
        """
        Set the 2theta-intensity (reduced) to file
        :param sub_run:
        :param two_theta_vector:
        :param intensity_vector:
        :return:
        """
        # check inputs and state
        self._validate_write_operation()
        checkdatatypes.check_int_variable('Sub run', sub_run, (0, None))
        checkdatatypes.check_numpy_arrays('2theta vector and intensity vector',
                                          [two_theta_vector, intensity_vector],
                                          dimension=None,
                                          check_same_shape=True)

        diff_group = self._create_diffraction_node(sub_run)
        diff_2t_group = diff_group[DiffractionUnit.unit(DiffractionUnit.TwoTheta)]

        # set value
        diff_2t_group.create_dataset('2theta', data=two_theta_vector)
        diff_2t_group.create_dataset('intensity', data=intensity_vector)

        return

    def set_d_spacing_diffraction_data(self, sub_run, d_vector, intensity_vector):
        """
        Set the dSpacing-intensity (reduced diffraction) to file
        :param sub_run:
        :param d_vector:
        :param intensity_vector:
        :return:
        """
        # check inputs and state
        self._validate_write_operation()
        checkdatatypes.check_int_variable('Sub run', sub_run, (0, None))
        checkdatatypes.check_numpy_arrays('d-spacing vector and intensity vector',
                                          [d_vector, intensity_vector],
                                          dimension=None,
                                          check_same_shape=True)

        # check or create new node/entry for this sub run
        diff_group = self._create_diffraction_node(sub_run)
        diff_d_group = diff_group[DiffractionUnit.DSpacing]

        # set value
        diff_d_group.create_dataset('2theta', data=d_vector)
        diff_d_group.create_dataset('intensity', data=intensity_vector)

        return

    def save_hydra_project(self):
        """
        convert all the information about project to HDF file.
        As the data has been written to h5.File instance already, the only thing left is to close the file
        :return:
        """
        self._validate_write_operation()

        self._project_h5.close()

        return

    def set_information(self, info_dict):
        """
        set project information to attributes
        :param info_dict:
        :return:
        """
        # check and validate
        checkdatatypes.check_dict('Project file general information', info_dict)
        self._validate_write_operation()

        for info_name in info_dict:
            self._project_h5.attrs[info_name] = info_dict[info_name]

        return

    def _create_diffraction_node(self, sub_run_number):
        """ Create a node to record diffraction data
        It will check if such node already exists
        :exception: RuntimeError is raised if such 'sub run' node exists but not correct
        :param sub_run_number:
        :return:
        """
        # create a new node if it does not exist
        sub_run_group_name = '{0:04}'.format(sub_run_number)

        print ('[DB...BAT] sub group entry name in hdf: {}'.format(sub_run_group_name))

        # check existing node or create a new node
        if sub_run_group_name in self._project_h5['diffraction']:
            # sub-run node exist and check
            diff_group = self._project_h5['diffraction'][sub_run_group_name]
            if not (DiffractionUnit.TwoTheta in diff_group and DiffractionUnit.DSpacing in diff_group):
                raise RuntimeError('Diffraction node for sub run {} exists but is not complete'.format(sub_run_number))
        else:
            # create new node: parent, child-2theta, child-dspacing
            diff_group = self._project_h5['diffraction'].create_group(sub_run_group_name)
            diff_group.create_group(DiffractionUnit.unit(DiffractionUnit.TwoTheta))
            diff_group.create_group(DiffractionUnit.unit(DiffractionUnit.DSpacing))

        return diff_group

    def _create_peak_node(self, sub_run_number, peak_tag, attrib_dict):
        """
        Create a node to record the peak fitting parameters
        :param sub_run_number:
        :param peak_tag:
        :param attrib_dict:
        :return:
        """
        # check inputs
        checkdatatypes.check_int_variable('Sub-run number', sub_run_number, (0, None))
        checkdatatypes.check_string_variable('Peak tag (example: 111)', peak_tag)
        checkdatatypes.check_dict('Attributions', attrib_dict)

        # create a new node if it does not exist
        if peak_tag not in self._project_h5['peak']:
            peak_tag_entry = self._project_h5['peak'].create_group(peak_tag)
        else:
            peak_tag_entry = self._project_h5['peak'][peak_tag]

        # create a new node for this sub run or access it
        sub_run_node_name = '{0:04}'.format(sub_run_number)
        if sub_run_node_name in peak_tag_entry:
            sub_run_node = peak_tag_entry[sub_run_node_name]
        else:
            sub_run_node = peak_tag_entry.create_group(sub_run_node_name)

        # add attributes
        for attrib_name in attrib_dict:
            sub_run_node.attrs[attrib_name] = attrib_dict[attrib_name]

        return sub_run_node

    def _validate_write_operation(self):
        """
        Validate whether a writing operation is allowed for this file
        :exception: run time exception
        :return:
        """
        if self._io_mode == HydraProjectFileMode.READONLY:
            raise RuntimeError('Project file {} is set to read-only by user'.format(self._project_h5.name))

        return


def test_main():
    project_h5 = h5py.File(project_file_name, 'r')
    scan_index = project_h5['experiment']['logs']['Scan Index'][0]
    assert scan_index == 1
    print project_h5['experiment'].keys()
    counts_vec_index001 = project_h5['experiment']['sub-runs']['{:04}'.format(scan_index)]['counts'].value
    print (counts_vec_index001.max())
    two_theta_value = project_h5['experiment']['logs']['2Theta'][0]
