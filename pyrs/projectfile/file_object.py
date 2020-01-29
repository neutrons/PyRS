# This is rs_scan_io.DiffractionFile's 2.0 version
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from enum import Enum
import h5py
from mantid.kernel import Logger
import numpy
import os
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry, HidraSetup
from pyrs.peaks import PeakCollection
from pyrs.dataobjects import HidraConstants, SampleLogs
from pyrs.projectfile import HidraProjectFileMode

__all__ = ['HidraProjectFile']


class DiffractionUnit(Enum):
    '''Enumeration for diffraction data's unit (2theta or d-spacing)'''
    TwoTheta = '2theta'
    DSpacing = 'dSpacing'

    def __str__(self):
        return self.value


class HidraProjectFile(object):
    '''Read and/or write an HB2B project to an HDF5 with entries for detector counts, sample logs, reduced data,
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
    '''

    def __init__(self, project_file_name, mode=HidraProjectFileMode.READONLY):
        """
        Initialization
        :param project_file_name: project file name
        :param mode: I/O mode
        """
        # configure logging for this class
        self._log = Logger(__name__)

        # convert the mode to the enum
        self._io_mode = HidraProjectFileMode.getMode(mode)

        # check the file
        if not project_file_name:
            raise RuntimeError('Must supply a filename')
        self._file_name = str(project_file_name)  # force it to be a string
        self._checkFileAccess()

        # open the file using h5py
        self._project_h5 = h5py.File(self._file_name, mode=str(self._io_mode))
        if self._io_mode == HidraProjectFileMode.OVERWRITE:
            self._init_project()

    def _checkFileAccess(self):
        '''Verify the file has the correct acces permissions and set the value of ``self._is_writable``
        '''
        # prepare the call to check the file permissions
        check_exist = ((self._io_mode == HidraProjectFileMode.READONLY) or
                       (self._io_mode == HidraProjectFileMode.READWRITE))
        self._is_writable = (not self._io_mode == HidraProjectFileMode.READONLY)

        # create a custom message based on requested access mode
        if self._io_mode == HidraProjectFileMode.READONLY:
            description = 'Read-only project file'
        elif self._io_mode == HidraProjectFileMode.OVERWRITE:
            description = 'Write-only project file'
        elif self._io_mode == HidraProjectFileMode.READWRITE:
            description = 'Append-mode project file'
        else:  # this should never happen
            raise RuntimeError('Hidra project file I/O mode {} is not supported'.format(HidraProjectFileMode))

        # convert the filename to an absolute path so error messages are clearer
        self._file_name = os.path.abspath(self._file_name)

        # do the check
        checkdatatypes.check_file_name(self._file_name, check_exist, self._is_writable, is_dir=False,
                                       description=description)

    def _init_project(self):
        """
        initialize the current opened project file from scratch by opening it
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
        # geometry
        geometry_group = instrument.create_group('geometry setup')
        geometry_group.create_group('detector')
        geometry_group.create_group('wave length')
        # detector (pixel) efficiency
        instrument.create_group(HidraConstants.DETECTOR_EFF)

        # mask entry and 2 sub entries
        mask_entry = self._project_h5.create_group(HidraConstants.MASK)
        mask_entry.create_group(HidraConstants.DETECTOR_MASK)
        mask_entry.create_group(HidraConstants.SOLID_ANGLE_MASK)

        # peaks
        self._project_h5.create_group('peaks')

        # reduced data
        self._project_h5.create_group(HidraConstants.REDUCED_DATA)

    def __del__(self):
        self.close()

    @property
    def name(self):
        """
        File name on HDD
        """
        return self._project_h5.name

    def append_raw_counts(self, sub_run_number, counts_array):
        """Add raw detector counts collected in a single scan/Pt

        Parameters
        ----------
        sub_run_number : int
            sub run number
        counts_array : ~numpy.ndarray
            detector counts
        """
        # check
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'
        checkdatatypes.check_int_variable('Sub-run index', sub_run_number, (0, None))

        # create group
        scan_i_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS].create_group(
            '{:04}'.format(sub_run_number))
        scan_i_group.create_dataset('counts', data=counts_array.reshape(-1))

    def append_experiment_log(self, log_name, log_value_array):
        """ add information about the experiment including scan indexes, sample logs, 2theta and etc
        :param log_name: name of the sample log
        :param log_value_array:
        """
        # check
        assert self._project_h5 is not None, 'cannot be None'
        assert self._is_writable, 'must be writable'
        checkdatatypes.check_string_variable('Log name', log_name)

        try:
            self._log.debug('Add sample log: {}'.format(log_name))
            self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS].create_dataset(
                log_name, data=log_value_array)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to add log {} due to {}'.format(log_name, run_err))
        except TypeError as type_err:
            raise RuntimeError('Failed to add log {} with value {} of type {}: {}'
                               ''.format(log_name, log_value_array, type(log_value_array), type_err))

    def read_default_masks(self):
        """Read default mask, i.e., for pixels at the edges

        Returns
        -------
        numpy.ndarray
            array for mask.  None for no mask

        """
        try:
            mask_name = HidraConstants.DEFAULT_MASK
            default_mask = self.read_mask_detector_array(mask_name)
        except RuntimeError:
            default_mask = None

        return default_mask

    def read_user_masks(self, mask_dict):
        """Read user-specified masks

        Parameters
        ----------
        mask_dict : dict
            dictionary to store masks (array)

        Returns
        -------
        None

        """
        # Get mask names except default mask
        try:
            mask_names = sorted(self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK].keys())
        except KeyError:
            # return if the file has an old format
            return
        if HidraConstants.DEFAULT_MASK in mask_names:
            mask_names.remove(HidraConstants.DEFAULT_MASK)

        # Read mask one by one
        for mask_name in mask_names:
            mask_dict[mask_name] = self.read_mask_detector_array(mask_name)

    def read_mask_detector_array(self, mask_name):
        """Get the mask from hidra project file (.h5) in the form of numpy array

        Location
          root
            - mask
                - detector
                     - mask_name

        Parameters
        ----------
        mask_name : str
            name of mask

        Returns
        -------
        numpy.ndarray
            mask array

        """
        try:
            mask_array = self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK][mask_name].value
        except KeyError as key_err:
            if HidraConstants.MASK not in self._project_h5.keys():
                err_msg = 'Project file {} does not have "{}" entry.  Its format is not up-to-date.' \
                          ''.format(self._file_name, HidraConstants.MASK)
            elif HidraConstants.DETECTOR_MASK not in self._project_h5[HidraConstants.MASK]:
                err_msg = 'Project file {} does not have "{}" entry under {}. ' \
                          'Its format is not up-to-date.' \
                          ''.format(self._file_name, HidraConstants.DETECTOR_MASK, HidraConstants.MASK)
            else:
                err_msg = 'Detector mask {} does not exist.  Available masks are {}.' \
                          ''.format(mask_name, self._project_h5[HidraConstants.MASK].keys())
            raise RuntimeError('{}\nFYI: {}'.format(err_msg,  key_err))

        return mask_array

    def write_mask_detector_array(self, mask_name, mask_array):
        """Write detector mask

        Structure:
          root
            - mask
                - detector
                     - default/universal
                     - mask_name


        Parameters
        ----------
        mask_name : str or None
            mask name.  None for default/universal detector mask
        mask_array : numpy.ndarray
            (N, ), masks, 0 for masking, 1 for ROI

        Returns
        -------
        None

        """
        # Set the default case
        if mask_name is None:
            mask_name = HidraConstants.DEFAULT_MASK

        if mask_name in self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK]:
            # delete the existing mask
            del self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK][mask_name]

        # add new detector mask (array)
        self._project_h5[HidraConstants.MASK][HidraConstants.DETECTOR_MASK].create_dataset(mask_name,
                                                                                           data=mask_array)

    def write_mask_solid_angle(self, mask_name, solid_angle_bin_edges):
        """
        Add mask in the form of solid angle
        Location: ..../main entry/mask/solid angle/
        data will be a range of solid angles and number of patterns to generate.
        example solid angle range = -8, 8, number of pattern = 3

        :param solid_angle_bin_edges: numpy 1D array as s0, s1, s2, ...
        """
        # Clean previously set if name exists
        if mask_name in self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK]:
            del self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK][mask_name]

        # Add new mask in
        solid_angle_entry = self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK]
        solid_angle_entry.create_dataset(mask_name, data=solid_angle_bin_edges)

    def read_mask_solid_angle(self, mask_name):
        """Get the masks in the form of solid angle bin edges
        """
        try:
            mask_array = self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK][mask_name]
        except KeyError as key_err:
            raise RuntimeError('Detector mask {} does not exist.  Available masks are {}. FYI: {}'
                               ''.format(mask_name,
                                         self._project_h5[HidraConstants.MASK][HidraConstants.SOLID_ANGLE_MASK].keys(),
                                         key_err))

        return mask_array

    def close(self):
        '''
        Close the file without checking whether the file can be written or not. This can
        be called multiple times without issue.
        '''
        if self._project_h5 is not None:
            self._project_h5.close()
            self._project_h5 = None  #
            self._log.information('File {} is closed'.format(self._file_name))

    def save(self, verbose=False):
        """
        convert all the information about project to HDF file.
        As the data has been written to h5.File instance already, the only thing left is to close the file
        """
        self._validate_write_operation()

        if verbose:
            self._log.information('Changes are saved to {0}. File is now closed.'.format(self._project_h5.filename))

        self.close()

    def read_diffraction_2theta_array(self):
        """Get the (reduced) diffraction data's 2-theta vector

        Returns
        -------
        numpy.ndarray
            1D vector for unified 2theta vector for all sub runs
            2D array for possibly various 2theta vector for each

        """
        if HidraConstants.TWO_THETA not in self._project_h5[HidraConstants.REDUCED_DATA]:
            # FIXME - This is a patch for 'legacy' data.  It will be removed after codes are stable
            tth_key = '2Theta'
        else:
            tth_key = HidraConstants.TWO_THETA

        two_theta_vec = self._project_h5[HidraConstants.REDUCED_DATA][tth_key].value

        return two_theta_vec

    def read_diffraction_intensity_vector(self, mask_id, sub_run):
        """ Get the (reduced) diffraction data's intensity
        :param mask_id:
        :param sub_run: If sub run = None: ...
        :return: 1D array or 2D array depending on sub ru
        """
        # Get default for mask/main
        if mask_id is None:
            mask_id = HidraConstants.REDUCED_MAIN

        checkdatatypes.check_string_variable('Mask ID', mask_id,
                                             list(self._project_h5[HidraConstants.REDUCED_DATA].keys()))

        # Get data to return
        if sub_run is None:
            # all the sub runs
            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id].value
        else:
            # specific one sub run
            sub_run_list = self.read_sub_runs()
            sub_run_index = sub_run_list.index(sub_run)

            if mask_id is None:
                mask_id = HidraConstants.REDUCED_MAIN

            reduced_diff_hist = self._project_h5[HidraConstants.REDUCED_DATA][mask_id].value[sub_run_index]
        # END-IF-ELSE

        return reduced_diff_hist

    def read_diffraction_masks(self):
        """
        Get the list of masks
        """
        masks = list(self._project_h5[HidraConstants.REDUCED_DATA].keys())

        # Clean up data entry '2theta' (or '2Theta')
        if HidraConstants.TWO_THETA in masks:
            masks.remove(HidraConstants.TWO_THETA)

        # FIXME - Remove when Hidra-16_Log.h5 is fixed with correction entry name as '2theta'
        # (aka HidraConstants.TWO_THETA)
        if '2Theta' in masks:
            masks.remove('2Theta')

        return masks

    def read_instrument_geometry(self):
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
        instrument_setup = AnglerCameraDetectorGeometry(num_rows=num_rows,
                                                        num_columns=num_cols,
                                                        pixel_size_x=pixel_size_x,
                                                        pixel_size_y=pixel_size_y,
                                                        arm_length=arm_length,
                                                        calibrated=False)

        return instrument_setup

    def read_sample_logs(self):
        """Get sample logs

        Retrieve all the (sample) logs from Hidra project file.
        Raw information retrieved from rs project file is numpy arrays

        Returns
        -------
        ndarray, dict
            ndarray : 1D array for sub runs
            dict : dict[sample log name] for sample logs in ndarray
        """
        # Get the group
        logs_group = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]

        if HidraConstants.SUB_RUNS not in logs_group.keys():
            raise RuntimeError('Failed to find {} in {} group of the file'.format(HidraConstants.SUB_RUNS,
                                                                                  HidraConstants.SAMPLE_LOGS))

        # Get 2theta and others
        samplelogs = SampleLogs()
        # first set subruns
        samplelogs[HidraConstants.SUB_RUNS] = logs_group[HidraConstants.SUB_RUNS].value
        for log_name in logs_group.keys():
            samplelogs[log_name] = logs_group[log_name].value

        return samplelogs

    def read_log_value(self, log_name):
        """Get a log's value

        Parameters
        ----------
        log_name

        Returns
        -------
        ndarray or single value
        """
        assert self._project_h5 is not None, 'Project HDF5 is not loaded yet'

        log_value = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS][log_name]

        return log_value

    def read_raw_counts(self, sub_run):
        """
        get the raw detector counts
        """
        assert self._project_h5 is not None, 'blabla'
        checkdatatypes.check_int_variable('sun run', sub_run, (0, None))

        sub_run_str = '{:04}'.format(sub_run)
        try:
            counts = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS][sub_run_str]['counts'].value
        except KeyError as key_error:
            err_msg = 'Unable to access sub run {} with key {}: {}\nAvailable runs are: {}' \
                      ''.format(sub_run, sub_run_str, key_error,
                                self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SUB_RUNS].keys())
            raise KeyError(err_msg)

        return counts

    def read_sub_runs(self):
        """
        get list of the sub runs
        """
        self._log.debug(str(self._project_h5.keys()))
        self._log.debug(self._file_name)
        # coded a little wacky to be less than 120 characters across
        sub_runs_str_list = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        if HidraConstants.SUB_RUNS in sub_runs_str_list:
            sub_runs_str_list = sub_runs_str_list[HidraConstants.SUB_RUNS].value
        else:
            sub_runs_str_list = []

        self._log.debug('.... Sub runs: {}'.format(sub_runs_str_list))

        sub_run_list = [None] * len(sub_runs_str_list)
        for index, sub_run_str in enumerate(sub_runs_str_list):
            sub_run_list[index] = int(sub_run_str)

        self._log.debug('.... Sub runs: {}'.format(sub_run_list))

        return sub_run_list

    def write_instrument_geometry(self, instrument_setup):
        """
        Add instrument geometry and wave length information to project file
        """
        # check inputs
        self._validate_write_operation()
        checkdatatypes.check_type('Instrument geometry setup', instrument_setup, HidraSetup)

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
        wavelength_group = instrument_group[HidraConstants.GEOMETRY_SETUP][HidraConstants.WAVELENGTH]
        try:
            wl = instrument_setup.get_wavelength(None)
        except (NotImplementedError, RuntimeError) as run_err:
            # No wave length from workspace: do nothing
            self._log.error(str(run_err))
            wl = None

        # Set wave length
        if wl is not None:
            wavelength_group.create_dataset('Calibrated', data=numpy.array([wl]))

    def read_peak_tags(self):
        """Get all the tags of peaks with parameters stored in HiDRA project

        Returns
        -------
        list
            list of string for all the peak tags

        """
        # Get main group
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        return peak_main_group.keys()

    def read_peak_parameters(self, peak_tag):
        """Get the parameters related to a peak

        The parameters including
        (1) peak profile (2) sub runs (3) chi2 (4) parameter names (5) parameter values

        Returns
        -------
        ~pyrs.core.peak_collection.PeakCollection
            All of the information from fitting a peak across subruns
        """
        # Get main group
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        # Get peak entry
        if peak_tag not in peak_main_group.keys():
            raise RuntimeError('Peak tag {} cannot be found'.format(peak_tag))
        peak_entry = peak_main_group[peak_tag]

        # Get all the attribute and data
        profile = peak_entry.attrs[HidraConstants.PEAK_PROFILE]
        background = peak_entry.attrs[HidraConstants.BACKGROUND_TYPE]
        sub_run_array = peak_entry[HidraConstants.SUB_RUNS].value
        chi2_array = peak_entry[HidraConstants.PEAK_FIT_CHI2].value
        param_values = peak_entry[HidraConstants.PEAK_PARAMS].value
        error_values = peak_entry[HidraConstants.PEAK_PARAMS_ERROR].value

        # validate the information makes sense
        if param_values.shape != error_values.shape:
            raise RuntimeError('Parameters[{}] and Errors[{}] have different shape'.format(param_values.shape,
                                                                                           error_values.shape))
        peak_collection = PeakCollection(peak_tag, profile, background)
        peak_collection.set_peak_fitting_values(subruns=sub_run_array, parameter_values=param_values,
                                                parameter_errors=error_values, fit_costs=chi2_array)
        return peak_collection

    def write_peak_fit_result(self, fitted_peaks):
        """Set the peak fitting results to project file.

         The tree structure for fitted peak in all sub runs is defined as
        - peaks
            - [peak-tag]
                - attr/'peak profile'
                - sub runs
                - parameter values
                - parameter fitting error

        Parameters
        ----------
        fitted_peaks : pyrs.core.peak_collection.PeakCollection

        Returns
        -------

        """
        # Check inputs and file status
        self._validate_write_operation()

        # Get value from peak collection
        peak_tag = fitted_peaks.peak_tag
        peak_profile = str(fitted_peaks.peak_profile)
        background_type = str(fitted_peaks.background_type)

        checkdatatypes.check_string_variable('Peak tag', peak_tag)
        checkdatatypes.check_string_variable('Peak profile', peak_profile)
        checkdatatypes.check_string_variable('Background type', background_type)

        # access or create node for peak with given tag
        peak_main_group = self._project_h5[HidraConstants.PEAKS]

        if peak_tag not in peak_main_group:
            # create peak-tag entry if it does not exist
            single_peak_entry = peak_main_group.create_group(peak_tag)
        else:
            # if peak-tag entry, get the reference to the entry
            single_peak_entry = peak_main_group[peak_tag]

        # Attributes
        self.set_attributes(single_peak_entry, HidraConstants.PEAK_PROFILE, peak_profile)
        self.set_attributes(single_peak_entry, HidraConstants.BACKGROUND_TYPE, background_type)

        single_peak_entry.create_dataset(HidraConstants.SUB_RUNS, data=fitted_peaks.sub_runs)
        single_peak_entry.create_dataset(HidraConstants.PEAK_FIT_CHI2, data=fitted_peaks.fitting_costs)
        peak_values, peak_errors = fitted_peaks.get_native_params()
        single_peak_entry.create_dataset(HidraConstants.PEAK_PARAMS, data=peak_values)
        single_peak_entry.create_dataset(HidraConstants.PEAK_PARAMS_ERROR, data=peak_errors)

    def read_wavelengths(self):
        """Get calibrated wave length

        Returns
        -------
        Float
            Calibrated wave length.  NaN for wave length is not ever set
        """
        # Init wave length
        wl = numpy.nan

        # Get the node
        try:
            mono_node = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO]
            if HidraConstants.WAVELENGTH in mono_node:
                wl = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO][HidraConstants.WAVELENGTH].value
                if wl.shape[0] == 0:
                    # empty numpy array: no data. keep as nan
                    pass
                elif wl.shape[0] == 1:
                    # 1 calibrated wave length
                    wl = wl[0]
                else:
                    # not supported
                    raise RuntimeError('There are more than 1 wave length registered')
                    # END-IF
        except KeyError:
            # monochromator node does not exist
            self._log.error('Node {} does not exist in HiDRA project file {}'
                            ''.format(HidraConstants.MONO, self._file_name))
        # END

        return wl

    def write_wavelength(self, wave_length):
        """ Set the calibrated wave length
        Location:
          .../instrument/monochromator setting/ ... .../
        Note:
        - same wave length to all sub runs
        - only calibrated wave length in project file
        - raw wave length comes from a table with setting
        :param wave_length: wave length in A
        :return: None
        """
        checkdatatypes.check_float_variable('Wave length', wave_length, (0, 1000))

        # Create 'monochromator setting' node if it does not exist
        if HidraConstants.MONO not in list(self._project_h5[HidraConstants.INSTRUMENT].keys()):
            self._project_h5[HidraConstants.INSTRUMENT].create_group(HidraConstants.MONO)

        # Get node and write value
        wl_entry = self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.MONO]
        wl_entry.create_dataset(HidraConstants.WAVELENGTH, data=numpy.array([wave_length]))

    def read_efficiency_correction(self):
        """
        Set detector efficiency correction measured from vanadium (efficiency correction)
        Returns
        -------
        numpy ndarray
            Efficiency array
        """
        calib_run_number = \
            self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].attrs[HidraConstants.RUN]

        det_eff_array =\
            self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF]['{}'.format(calib_run_number)]

        return det_eff_array

    def write_efficiency_correction(self, calib_run_number, eff_array):
        """ Set detector efficiency correction measured from vanadium (efficiency correction)
        Location: ... /main entry/calibration/efficiency:
        Data: numpy array with 1024**2...
        Attribute: add the run number created from to the attribute
        Parameters
        ----------
        calib_run_number : integer
            Run number where the efficiency calibration comes from
        eff_array : numpy ndarray (1D)
            Detector (pixel) efficiency
        """
        # Add attribute
        self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].attrs[HidraConstants.RUN] = \
            calib_run_number

        # Set data
        self._project_h5[HidraConstants.INSTRUMENT][HidraConstants.DETECTOR_EFF].create_dataset(
            '{}'.format(calib_run_number), data=eff_array)

    def write_information(self, info_dict):
        """
        set project information to attributes
        """
        # check and validate
        checkdatatypes.check_dict('Project file general information', info_dict)
        self._validate_write_operation()

        for info_name in info_dict:
            self._project_h5.attrs[info_name] = info_dict[info_name]

    def write_reduced_diffraction_data_set(self, two_theta_array, diff_data_set):
        """Set the reduced diffraction data (set)

        Parameters
        ----------
        two_theta_array : numppy.ndarray
            2D array for 2-theta vector, which could be various to each other among sub runs
        diff_data_set : dict
            dictionary of 2D arrays for reduced diffraction patterns' intensities
        """
        # Check input
        checkdatatypes.check_numpy_arrays('Two theta vector', [two_theta_array], 2, False)
        checkdatatypes.check_dict('Diffraction data set', diff_data_set)

        # Retrieve diffraction group
        diff_group = self._project_h5[HidraConstants.REDUCED_DATA]

        # Add 2theta vector
        if HidraConstants.TWO_THETA in diff_group.keys():
            # over write data
            try:
                diff_group[HidraConstants.TWO_THETA][...] = two_theta_array
            except TypeError:
                # usually two theta vector size changed
                del diff_group[HidraConstants.TWO_THETA]
                diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_array)
        else:
            # new data
            diff_group.create_dataset(HidraConstants.TWO_THETA, data=two_theta_array)

        # Add Diffraction data
        for mask_id in diff_data_set:
            # Get data
            diff_data_matrix_i = diff_data_set[mask_id]
            self._log.information('Mask {} data set shape: {}'.format(mask_id, diff_data_matrix_i.shape))
            # Check
            checkdatatypes.check_numpy_arrays('Diffraction data (matrix)', [diff_data_matrix_i], None, False)
            if two_theta_array.shape != diff_data_matrix_i.shape:
                raise RuntimeError('Length of 2theta vector ({}) is different from intensities ({})'
                                   ''.format(two_theta_array.shape, diff_data_matrix_i.shape))
            # Set name for default mask
            if mask_id is None:
                data_name = HidraConstants.REDUCED_MAIN
            else:
                data_name = mask_id

            # Write
            if data_name in diff_group.keys():
                # overwrite
                diff_h5_data = diff_group[data_name]
                try:
                    diff_h5_data[...] = diff_data_matrix_i
                except TypeError:
                    # usually two theta vector size changed
                    del diff_group[data_name]
                    diff_group.create_dataset(data_name, data=diff_data_matrix_i)
            else:
                # new
                diff_group.create_dataset(data_name, data=diff_data_matrix_i)

    def write_sub_runs(self, sub_runs):
        """ Set sub runs to sample log entry
        """
        if isinstance(sub_runs, list):
            sub_runs = numpy.array(sub_runs)
        else:
            checkdatatypes.check_numpy_arrays('Sub run numbers', [sub_runs], 1, False)

        sample_log_entry = self._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS]
        sample_log_entry.create_dataset(HidraConstants.SUB_RUNS, data=sub_runs)

    def _create_diffraction_node(self, sub_run_number):
        """ Create a node to record diffraction data
        It will check if such node already exists
        :exception: RuntimeError is raised if such 'sub run' node exists but not correct
        """
        # create a new node if it does not exist
        sub_run_group_name = '{0:04}'.format(sub_run_number)

        self._log.debug('sub group entry name in hdf: {}'.format(sub_run_group_name))

        # check existing node or create a new node
        self._log.debug('Diffraction node sub group/entries: {}'
                        ''.format(self._project_h5[HidraConstants.REDUCED_DATA].keys()))
        if sub_run_group_name in self._project_h5[HidraConstants.REDUCED_DATA]:
            # sub-run node exist and check
            self._log('sub-group: {}'.format(sub_run_group_name))
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA][sub_run_group_name]
            if not (DiffractionUnit.TwoTheta in diff_group and DiffractionUnit.DSpacing in diff_group):
                raise RuntimeError('Diffraction node for sub run {} exists but is not complete'.format(sub_run_number))
        else:
            # create new node: parent, child-2theta, child-dspacing
            diff_group = self._project_h5[HidraConstants.REDUCED_DATA].create_group(sub_run_group_name)
            diff_group.create_group(str(DiffractionUnit.TwoTheta))
            diff_group.create_group(str(DiffractionUnit.DSpacing))

        return diff_group

    def _validate_write_operation(self):
        """
        Validate whether a writing operation is allowed for this file
        :exception: RuntimeError
        """
        if not self._is_writable:
            raise RuntimeError('Project file {} is set to read-only by user'.format(self._project_h5.name))

    @staticmethod
    def set_attributes(h5_group, attribute_name, attribute_value):
        """
        Set attribute to a group
        """
        checkdatatypes.check_string_variable('Attribute name', attribute_name)

        h5_group.attrs[attribute_name] = attribute_value
