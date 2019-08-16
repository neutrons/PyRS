# This is rs_scan_io.DiffractionFile's 2.0 version
import os
import h5py
import checkdatatypes
from pyrs.core import instrument_geometry
from enum import Enum
import numpy

# TODO - NOW TONIGHT #72 - Use more enumerate and constant for HDF groups' names


# TODO - NOW TONIGHT #72 - Apply HidraConstants to HydraProjectFile
class HidraConstants(object):
    RAW_DATA = 'raw data'
    REDUCED_DATA = 'reduced diffraction data'
    REDUCED_MAIN = 'main'   # default reduced data
    SUB_RUNS = 'sub-runs'
    CALIBRATION = 'calibration'
    SAMPLE_LOGS = 'logs'
    INSTRUMENT = 'instrument'
    GEOMETRY_SETUP = 'geometry setup'
    DETECTOR_PARAMS = 'detector'
    TWO_THETA = '2Theta'


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


# TODO - #80 - Clean up the codes for all the methods and docs...
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
    - reduced diffraction data
        - main
          - sub-run
          - ...
        - mask_A
          - sub-run
          - ...
        - mask_B
          - sub-run
          - ...

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
        exp_entry = self._project_h5.create_group(HidraConstants.RAW_DATA)
        exp_entry.create_group(HidraConstants.SUB_RUNS)
        exp_entry.create_group(HidraConstants.SAMPLE_LOGS)

        # instrument
        instrument = self._project_h5.create_group(HidraConstants.INSTRUMENT)
        instrument.create_group(HidraConstants.CALIBRATION)
        geometry_group = instrument.create_group('geometry setup')
        geometry_group.create_group('detector')
        geometry_group.create_group('wave length')

        # reduced data
        reduced_data = self._project_h5.create_group(HidraConstants.REDUCED_DATA)
        # reduced_data.create_group('2theta')
        # reduced_data.create_group('d-spacing')

        return

    def get_diffraction_2theta_vector(self):
        # TODO - NOW TONIGHT #80 #72 - Doc

        print (self._project_h5.name)
        two_theta_vec = self._project_h5[HidraConstants.REDUCED_DATA][HidraConstants.TWO_THETA].value

        return two_theta_vec

    def get_reduced_diffraction_data(self, mask_id, sub_run):
        """

        :param mask_id:
        :param sub_run: If sub run = None: ...
        :return: 1D array or 2D array depending on sub ru
        """
        # Mask
        if mask_id is None:
            mask_id = HidraConstants.REDUCED_MAIN

        if sub_run is None:
            # all the sub runs
            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id].value
        else:
            # specific one sub run
            sub_run_list = self.get_sub_runs()
            sub_run_index = sub_run_list.index(sub_run)

            if mask_id is None:
                mask_id = HidraConstants.REDUCED_MAIN

            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id].value[sub_run_index]
        # END-IF-ELSE

        return reduced_diff_hist

    def get_instrument_geometry(self):
        """
        Get instrument geometry parameters
        :return: an instance of instrument_geometry.InstrumentSetup
        """
        # Get group
        geometry_group = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.GEOMETRY_SETUP]
        detector_group = geometry_group[HidraConstants.DETECTOR_PARAMS]

        # Get value
        num_rows, num_cols = detector_group['detector size'].value
        pixel_size_x, pixel_size_y = detector_group['pixel dimension'].value
        arm_length = detector_group['L2'].value

        # Initialize
        instrument_setup = instrument_geometry.AnglerCameraDetectorGeometry(num_rows=num_rows,
                                                                            num_columns=num_cols,
                                                                            pixel_size_x=pixel_size_x,
                                                                            pixel_size_y=pixel_size_y,
                                                                            arm_length=arm_length,
                                                                            calibrated=False)

        return instrument_setup

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
        instrument_group = self._project_h5[HidraConstants.INSTRUMENT]

        # write attributes
        instrument_group.attrs['name'] = instrument_setup.name

        # get the entry for raw instrument setup
        detector_group = instrument_group['geometry setup']['detector']
        raw_geometry = instrument_setup.get_instrument_geometry(False)
        detector_group.create_dataset('L2', data=numpy.array(raw_geometry.arm_length))
        det_size = numpy.array(instrument_setup.get_instrument_geometry(False).detector_size)
        detector_group.create_dataset('detector size', data=det_size)
        pixel_dimension = list(instrument_setup.get_instrument_geometry(False).pixel_dimension)
        detector_group.create_dataset('pixel dimension', data=numpy.array(pixel_dimension))

        # wave length
        wavelength_group = instrument_group['geometry setup']['wave length']
        wavelength_group.create_dataset('wavelength', data=numpy.array([instrument_setup.get_wavelength(None, False)]))

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
        scan_i_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS].create_group(
            '{:04}'.format(sub_run_number))
        scan_i_group.create_dataset('counts', data=counts_array)

        return

    # TODO FIXME - #80 - Anymore?
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
        reduced_data = self._project_h5[HidraConstants.REDUCED_DATA][unit].create_group('{:04}'.format(sub_run_index))
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
            self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS].create_dataset(
                log_name, data=log_value_array)
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

    def get_logs(self):
        """
        Retrieve all the (sample) logs from Hidra project file
        :return:
        """
        # Get the group
        logs_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]

        # Get the sub run numbers
        sub_runs = logs_group[HidraConstants.SUB_RUNS].value

        # Get 2theta and others
        logs_value_set = dict()
        for log_name in logs_group.keys():
            # no sub runs
            if log_name == HidraConstants.SUB_RUNS:
                continue

            # get array
            log_value_vec = logs_group[log_name].value
            if log_value_vec.shape != sub_runs.shape:
                raise RuntimeError('Sample log {} does not match sub runs'.format(log_name))

            log_value_dict = dict()
            for s_index in range(sub_runs.shape[0]):
                log_value_dict[sub_runs[s_index]] = log_value_vec[s_index]
            # END-FOR

            logs_value_set[log_name] = log_value_dict
        # END-FOR

        return logs_value_set

    def get_log_value(self, log_name):
        assert self._project_h5 is not None, 'blabla'

        log_value = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS][log_name]

        return log_value

    def get_raw_counts(self, sub_run):
        """
        get the raw detector counts
        :return:
        """
        assert self._project_h5 is not None, 'blabla'
        checkdatatypes.check_int_variable('sun run', sub_run, (0, None))

        sub_run_str = '{:04}'.format(sub_run)
        counts = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS][sub_run_str]['counts'].value

        return counts

    def get_sub_runs(self):
        """
        get list of the sub runs
        :return:
        """
        sub_runs_str_list = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS]

        sub_run_list = [None] * len(sub_runs_str_list)
        for index, sub_run_str in enumerate(sub_runs_str_list):
            sub_run_list[index] = int(sub_run_str)

        return sub_run_list

    # TODO FIXME - #80 - This is not a correct method anymore.  Find one to merge to
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
        diff_2t_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_vector)
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

    def set_reduced_diffraction_dataset(self, two_theta_vec, diff_data_set):
        # TODO - TONIGHT NOW #72 - Check & Doc
        diff_group = self._project_h5[HidraConstants.REDUCED_DATA]

        # Add 2theta vector
        if HidraConstants.TWO_THETA in diff_group.keys():
            # over write data
            try:
                diff_group[HidraConstants.TWO_THETA][...] = two_theta_vec
            except TypeError:
                # usually two theta vector size changed
                del diff_group[HidraConstants.TWO_THETA]
                diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_vec)
        else:
            # new data
            diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_vec)

        # Add Diffraction data
        for mask_id in diff_data_set:
            if mask_id is None:
                data_name = HidraConstants.REDUCED_MAIN
            else:
                data_name = mask_id

            if data_name in diff_group.keys():
                # overwrite
                diff_h5_data = diff_group[data_name]
                try:
                    diff_h5_data[...] = diff_data_set[mask_id]
                except TypeError:
                    # usually two theta vector size changed
                    del diff_group[data_name]
                    diff_group.create_dataset(data_name, data=diff_data_set[mask_id])
            else:
                # new
                diff_group.create_dataset(data_name, data=diff_data_set[mask_id])
        # END-FOR

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
        print ('[DB...BAT] Diffraction node sub group/entries: {}'
               ''.format( self._project_h5[HidraConstants.REDUCED_DATA].keys()))
        if sub_run_group_name in self._project_h5[HidraConstants.REDUCED_DATA]:
            # sub-run node exist and check
            print ('[DB...BAT] sub-group: {}'.format(sub_run_group_name))
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA][sub_run_group_name]
            if not (DiffractionUnit.TwoTheta in diff_group and DiffractionUnit.DSpacing in diff_group):
                raise RuntimeError('Diffraction node for sub run {} exists but is not complete'.format(sub_run_number))
        else:
            # create new node: parent, child-2theta, child-dspacing
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA].create_group(sub_run_group_name)
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
