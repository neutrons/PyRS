# Data manager
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import numpy
from pyrs.dataobjects import HidraConstants, SampleLogs
from pyrs.projectfile import HidraProjectFile
from pyrs.utilities import checkdatatypes


class HidraWorkspace(object):
    """
    This workspace is the central data structure to manage all the raw and/or processed data.
    It provides
    - container for raw counts
    - container for reduced diffraction data
    - container for fitted peaks' parameters
    - container for instrument information
    """

    def __init__(self, name='hidradata'):
        """
        initialization
        """
        # workspace name
        self._name = name

        # raw counts
        self._raw_counts = dict()  # dict [sub-run] = count vector

        # wave length
        self._wave_length = None  # single wave length for all sub runs
        self._wave_length_dict = None
        self._wave_length_calibrated_dict = None

        # diffraction
        self._2theta_matrix = None  # ndarray.  shape = (m, ) m = number of 2theta
        self._diff_data_set = dict()  # [mask id] = ndarray: shape=(n, m), n: number of sub-run, m: number of of 2theta

        # instrument
        self._instrument_setup = None
        self._instrument_geometry_shift = None  # geometry shift

        # sample logs
        self._sample_logs = SampleLogs()  # sample logs: [log name, sub run] = value

        # raw Hidra project file
        self._project_file_name = None
        self._project_file = None

        # Masks
        self._default_mask = None
        self._mask_dict = dict()

    @property
    def name(self):
        """
        Workspace name
        :return:
        """
        return self._name

    @property
    def hidra_project_file(self):
        """Name of the associated HiDRA project file

        Returns
        -------

        """
        return self._project_file_name

    def _load_raw_counts(self, hidra_file):
        """ Load raw detector counts from HIDRA file
        :param hidra_file:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        for sub_run_i in self._sample_logs.subruns:
            counts_vec_i = hidra_file.read_raw_counts(sub_run_i)
            self._raw_counts[sub_run_i] = counts_vec_i
        # END-FOR

        return

    def _load_reduced_diffraction_data(self, hidra_file):
        """ Load reduced diffraction data from HIDRA file
        :param hidra_file: HidraProjectFile instance
        :return:
        """
        # Check inputs
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        # get 2theta value
        try:
            vec_2theta = hidra_file.read_diffraction_2theta_array()
        except KeyError as key_err:
            print('[INFO] Unable to load 2theta vector from HidraProject file due to {}.'
                  'It is very likely that no reduced data is recorded.'
                  ''.format(key_err))
            return
        # TRY-CATCH

        # Get number of spectra
        num_spec = len(hidra_file.read_sub_runs())

        # Promote to 2theta from vector to array
        if len(vec_2theta.shape) == 1:
            # convert from 1D array to 2D
            tth_size = vec_2theta.shape[0]
            matrix_2theta = numpy.repeat(vec_2theta.reshape(1, tth_size), num_spec, axis=0)
        else:
            matrix_2theta = vec_2theta

        # Set value
        self._2theta_matrix = numpy.copy(matrix_2theta)

        # initialize data set for reduced diffraction patterns
        diff_mask_list = hidra_file.read_diffraction_masks()
        for mask_name in diff_mask_list:
            if mask_name == 'main':
                mask_name = None
            self._diff_data_set[mask_name] = numpy.ndarray(shape=(num_spec, vec_2theta.shape[0]), dtype='float')
        # END-FOR

        # Load data: all including masks / ROI
        for mask_name in diff_mask_list:
            # force to None
            if mask_name == 'main':
                mask_name = None
            self._diff_data_set[mask_name] = hidra_file.read_diffraction_intensity_vector(mask_id=mask_name,
                                                                                          sub_run=None)
        print('[INFO] Loaded diffraction data from {} includes : {}'
              ''.format(self._project_file_name, self._diff_data_set.keys()))

    def _load_instrument(self, hidra_file):
        """ Load instrument setup from HIDRA file
        :param hidra_file: HIDRA project file instance
        :return:
        """
        # Check
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        # Get values
        self._instrument_setup = hidra_file.read_instrument_geometry()

    def _load_masks(self, hidra_file):
        """Load masks from Hidra project file

        Parameters
        ----------
        hidra_file :  pyrs.projectfile.file_object.HidraProjectFile
            Hidra project file instance

        Returns
        -------

        """
        # Check
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        # Default mask: get value and set
        default_mask = hidra_file.read_default_masks()
        if default_mask is not None:
            self.set_detector_mask(default_mask, True)

        # User specified mask
        mask_dict = dict()
        hidra_file.read_user_masks(mask_dict)
        for mask_name in mask_dict:
            self.set_detector_mask(mask_dict[mask_name], False, mask_name)

    def _load_sample_logs(self, hidra_file):
        """ Load sample logs.
        Note: this method can clear all the sample logs added previously. But it is not
            an issue in the real use cases.
        :param hidra_file:  HIDRA project file instance
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        # overwrite the existing sample logs
        self._sample_logs = hidra_file.read_sample_logs()

    def _load_wave_length(self, hidra_file):
        """Load wave length from HidraProject file

        Parameters
        ----------
        hidra_file : pyrs.projectfile.file_object.HidraProjectFile
            Project file (instance)

        Returns
        -------
        None

        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)

        # reset the wave length (dictionary) from HIDRA project file
        self._wave_length = hidra_file.read_wavelengths()

    def get_detector_2theta(self, sub_run):
        """ Get 2theta value from sample log
        This is a special one
        :param sub_run: sub run number (integer)
        :return: float number as 2theta
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))
        try:
            two_theta = self._sample_logs[HidraConstants.TWO_THETA, sub_run]
        except KeyError as key_err:
            raise RuntimeError('Unable to retrieve 2theta value ({}) from sub run {} due to missing key {}.'
                               'Available sample logs are {}'
                               .format(HidraConstants.TWO_THETA,
                                       sub_run, key_err, self._sample_logs.keys()))
        return two_theta[0]  # convert from numpy array of length 1 to a scalar

    def get_l2(self, sub_run):
        """ Get L2 for a specific sub run
        :param sub_run: sub run number (integer)
        :return: L2 or None (i.e., using default L2)
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))

        if HidraConstants.L2 in self._sample_logs:
            # L2 is a valid sample log: get L2
            try:
                # convert from numpy array of length 1 to a scalar
                l2 = self._sample_logs[HidraConstants.L2, sub_run][0]
            except KeyError as key_err:
                raise RuntimeError('Unable to retrieve L2 value for {} due to {}. Available sun runs are {}'
                                   .format(sub_run, key_err, self._sample_logs[HidraConstants.L2]))
        else:
            # L2 might be unchanged
            l2 = None

        return l2

    def get_instrument_setup(self):
        """ Get the handler to instrument setup
        :return:
        """
        return self._instrument_setup

    def get_detector_counts(self, sub_run):
        """Get the detector counts of a sub run (split)

        Parameters
        ----------
        sub_run : int
            sub run number

        Returns
        -------
        numpy.ndarray

        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))  # consider 0 as a single sub run
        if int(sub_run) not in self._raw_counts:
            raise RuntimeError('Sub run {} does not exist in loaded raw counts. FYI loaded '
                               'sub runs are {}'.format(sub_run, self._raw_counts.keys()))

        return self._raw_counts[sub_run]

    def get_sub_runs(self):
        """Get sub runs that loaded to this workspace

        Returns
        -------
        numpy.ndarray
            1D array for sorted sub runs

        """
        if len(self._sample_logs.subruns) == 0:
            raise RuntimeError('Sub run - spectrum map has not been built')

        return self._sample_logs.subruns

    def get_wavelength(self, calibrated, throw_if_not_set, sub_run=None):
        """Get wave length

        Parameters
        ----------
        calibrated : bool
            whether the wave length is calibrated or raw
        throw_if_not_set : bool
            throw an exception if wave length is not set to workspace
        sub_run : None or int
            sub run number for the wave length associated with

        Returns
        -------
        float or dict

        """
        # Return the universal wave length if it is set
        if sub_run is None and self._wave_length is not None:
            return self._wave_length

        if calibrated:
            # calibrated wave length
            if self._wave_length_calibrated_dict is None:
                if throw_if_not_set:
                    raise RuntimeError('There is no calibrated wave length in HidraWorkspace {}'.format(self._name))
                else:
                    wave_length_dict = None
            else:
                wave_length_dict = self._wave_length_calibrated_dict
        else:
            # native wave length
            if self._wave_length_dict is None:
                if throw_if_not_set:
                    raise RuntimeError('There is no original/native wave length in HidraWorkspace {}'
                                       ''.format(self._name))
                else:
                    wave_length_dict = None
            else:
                wave_length_dict = self._wave_length_dict

        # Return the wave length of the sub run
        if sub_run is not None:
            return wave_length_dict[sub_run]

        return wave_length_dict

    def load_hidra_project(self, hidra_file, load_raw_counts, load_reduced_diffraction):
        """
        Load HIDRA project file
        :param hidra_file: HIDRA project file instance (not file name)
        :param load_raw_counts: Flag to load raw counts
        :param load_reduced_diffraction: Flag to load reduced diffraction data
        :return:
        """
        # Check input
        checkdatatypes.check_type('HIDRA project file', hidra_file, HidraProjectFile)
        self._project_file_name = hidra_file.name
        self._project_file = hidra_file

        # create the spectrum map - must exist before loading the counts array
        self._sample_logs.subruns = hidra_file.read_sub_runs()

        # load raw detector counts and load instrument
        if load_raw_counts:
            self._load_raw_counts(hidra_file)
            self._load_instrument(hidra_file)

        # load reduced diffraction
        if load_reduced_diffraction:
            self._load_reduced_diffraction_data(hidra_file)

        # load sample logs
        self._load_sample_logs(hidra_file)

        # load masks
        self._load_masks(hidra_file)

        # load the wave length
        self._load_wave_length(hidra_file)

    def get_detector_mask(self, is_default, mask_id=None):
        """Get detector mask

        Parameters
        ----------
        is_default : bool
            If True, get the default detector mask
        mask_id : str
            with is_default is False, get the user-specified mask/ROI

        Returns
        -------
        numpy.ndarray or None
            detector mask.  None in case no default detector mask

        """
        # Default mask
        if is_default:
            return self._default_mask

        # User-specific mask
        if mask_id not in self._mask_dict:
            raise RuntimeError('Mask ID {} does not exist in HidraWorkspace {}.  Available masks are '
                               '{}'.format(mask_id, self._name, self._mask_dict.keys()))

        return self._mask_dict[mask_id]

    def get_detector_shift(self):
        """
        Get detector geometry shift
        :return: AnglerDetectorShift instance
        """
        return self._instrument_geometry_shift

    def get_reduced_diffraction_data_set(self, mask_id=None):
        """Get reduced diffraction data set including 2theta and intensities

        Get the full data set (matrix) of reduced diffraction pattern in 2theta unit

        Parameters
        ----------
        mask_id : str or None
            None (as default main) or ID as a String

        Returns
        -------
        ndarray, ndarray
            2theta in 2D array
            intensities in 2D array

        """
        # Check
        if mask_id is None:
            # mask_id is 'main'
            pass
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id)

        # Vector 2theta
        matrix_2theta = self._2theta_matrix.copy()

        try:
            intensity_matrix = self._diff_data_set[mask_id].copy()
        except KeyError:
            raise RuntimeError('Mask ID {} does not exist in reduced diffraction pattern. '
                               'The available masks are {}'
                               ''.format(mask_id, self._diff_data_set.keys()))

        return matrix_2theta, intensity_matrix

    def get_reduced_diffraction_data_2theta(self, sub_run):
        """Get 2theta vector of reduced diffraction data

        Parameters
        ----------
        sub_run : int
            sub run number

        Returns
        -------
        numpy.ndarray
            vector of 2theta

        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        # Get spectrum index
        spec_index = self._sample_logs.get_subrun_indices(sub_run)[0]
        # Vector 2theta
        vec_2theta = self._2theta_matrix[spec_index][:]

        return vec_2theta

    def get_reduced_diffraction_data(self, sub_run, mask_id=None):
        """Get data set of a single diffraction pattern

        Parameters
        ----------
        sub_run: int
            sub run number (integer)
        mask_id : str or None
            None (as default main) or ID as a String
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            vector 2theta, vector intensity

        """
        # Check inputs
        # sub run number might start from 0
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))
        if mask_id is None:
            # mask_id = 'main'
            pass
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id)

        spec_index = self._sample_logs.get_subrun_indices(sub_run)[0]

        # Vector 2theta
        vec_2theta = self._2theta_matrix[spec_index][:]

        # Vector intensity
        try:
            vec_intensity = self._diff_data_set[mask_id][spec_index].copy()
        except KeyError:
            raise RuntimeError('Mask ID {} does not exist in reduced diffraction pattern. '
                               'The available masks are {}'
                               ''.format(mask_id, self._diff_data_set.keys()))
        return vec_2theta, vec_intensity

    def get_sample_log_names(self):
        return sorted(self._sample_logs.keys())

    def get_sample_log_value(self, sample_log_name, sub_run=None):
        """

        Parameters
        ----------
        sample_log_name
        sub_run

        Returns
        -------
        float
            time-averaged sample log value for this sub run

        """
        checkdatatypes.check_string_variable('Sample log name', sample_log_name,
                                             list(self._sample_logs.keys()))

        log_value = self._sample_logs[sample_log_name, sub_run]

        if isinstance(log_value, numpy.ndarray):
            assert log_value.shape == (1, ), 'Single log {} (= {}) is a numpy array with multiple items' \
                                             '(shape = {})'.format(sample_log_name, log_value, log_value.shape)
            log_value = log_value[0]

        return log_value

    def get_sample_log_values(self, sample_log_name, sub_runs=None):
        """Get ONE INDIVIDUAL sample log's values as a vector

        Exceptions
        ----------
        RuntimeError : if sample log name not in sample_log_dict

        Parameters
        ----------
        sample_log_name : str
            sample_log_name
        sub_runs : list or ndarray or None
            None for all log values, List/ndarray for selected sub runs
        Returns
        -------
        ndarray
            sample log values ordered by sub run numbers with given sub runs or all sub runs

        """
        if sample_log_name == HidraConstants.SUB_RUNS and \
                sample_log_name not in self._sample_logs.keys():
            return self.get_sub_runs()

        checkdatatypes.check_string_variable('Sample log name', sample_log_name,
                                             list(self._sample_logs.keys()))

        return self._sample_logs[sample_log_name, sub_runs]

    def get_spectrum_index(self, sub_run):
        """
        Get spectrum (index) from sub run number
        :param sub_run: sub run number (integer)
        :return:
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))

        return self._sample_logs.get_subrun_indices(sub_run)[0]

    def get_sub_runs_from_spectrum(self, spectra):
        """ Get sub runs corresponding to spectra (same as ws index)
        :param spectra: list/vector/array of spectra (workspace indexes)
        :return:
        """
        if not (isinstance(spectra, list) or isinstance(spectra, numpy.ndarray)):
            raise AssertionError('{} must be list or array'.format(type(spectra)))

        return self._sample_logs.subruns[spectra]

    def has_raw_data(self, sub_run):
        """ Check whether a raw file that has been loaded
        :param sub_run: sub run number (integer)
        :return:
        """
        checkdatatypes.check_int_variable('Sub run', sub_run, (1, None))

        return sub_run in self._raw_counts

    def has_sample_log(self, sample_log_name):
        """
        check whether a certain sample log exists in the workspace (very likely loaded from file)
        :param sample_log_name: sample log name
        :return:
        """
        # Check inputs
        checkdatatypes.check_string_variable('Sample log name', sample_log_name)

        has_log = sample_log_name in self._sample_logs

        return has_log

    def set_instrument_geometry(self, instrument):
        self._instrument_setup = instrument

    def set_detector_mask(self, mask_array, is_default, mask_id=None):
        """Set mask array to HidraWorkspace

        Record the mask to HidraWorkspace future reference

        Parameters
        ----------
        mask_array : numpy.darray
            mask bit for each pixel
        is_default : bool
            whether this mask is the default mask from beginning
        mask_id : str
            ID for mask

        Returns
        -------

        """
        checkdatatypes.check_numpy_arrays('Detector mask', [mask_array], None, False)

        # Convert mask to 1D array
        if len(mask_array.shape) == 2:
            # rule out unexpected shape
            if mask_array.shape[1] != 1:
                raise RuntimeError('Mask array with shape {} is not acceptable'.format(mask_array.shape))
            # convert from (N, 1) to (N,)
            num_pixels = mask_array.shape[0]
            mask_array = mask_array.reshape((num_pixels,))
        # END-IF

        if is_default:
            self._default_mask = mask_array
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id, allow_empty=False)
            self._mask_dict[mask_id] = mask_array

    def set_raw_counts(self, sub_run_number, counts):
        """
        Set the raw counts to
        :param sub_run_number: integer for sub run number
        :param counts: ndarray of detector counts
        :return:
        """
        checkdatatypes.check_numpy_arrays('Counts', [counts], dimension=None, check_same_shape=False)

        if len(counts.shape) == 2 and counts.shape[1] == 1:
            # 1D array in 2D format: set to 1D array
            counts = counts.reshape((counts.shape[0],))

        self._raw_counts[int(sub_run_number)] = counts

    def set_reduced_diffraction_data(self, sub_run, mask_id, two_theta_array, intensity_array):
        """Set reduced diffraction data to workspace

        Parameters
        ----------
        sub_run : int
            sub run number
        mask_id : None or str
            mask ID.  None for no-mask or masked by default/universal detector masks on edges
        two_theta_array : numpy.ndarray
            2theta bins (center)
        intensity_array : numpy.ndarray
            histogrammed intensities

        Returns
        -------
        None

        """
        # Check status of reducer whether sub run number and spectrum are initialized
        if len(self._sample_logs.subruns) == 0:
            raise RuntimeError('Sub run - spectrum map has not been set up yet!')

        # Check inputs
        # sub run number valid or not
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if mask_id is not None:
            checkdatatypes.check_string_variable('Mask ID', mask_id)
        # two theta array and intensity array shall match on size
        if two_theta_array.shape != intensity_array.shape:
            raise RuntimeError('Two theta array (bin centers) must have same dimension as intensity array. '
                               'Now they are {} and {}'.format(two_theta_array.shape, intensity_array.shape))

        # Set 2-theta 2D array
        if self._2theta_matrix is None or len(self._2theta_matrix.shape) != 2:
            # First time set up or legacy from input file: create the 2D array
            num_sub_runs = len(self._sample_logs.subruns)
            self._2theta_matrix = numpy.ndarray(shape=(num_sub_runs, two_theta_array.shape[0]),
                                                dtype=intensity_array.dtype)
            # set the diffraction data (2D) array with new dimension
            num_sub_runs = len(self._sample_logs.subruns)
            self._diff_data_set[mask_id] = numpy.ndarray(shape=(num_sub_runs, intensity_array.shape[0]),
                                                         dtype=intensity_array.dtype)
        elif mask_id not in self._diff_data_set:
            # A new mask: reset the diff_data_set again
            num_sub_runs = len(self._sample_logs.subruns)
            self._diff_data_set[mask_id] = numpy.ndarray(shape=(num_sub_runs, intensity_array.shape[0]),
                                                         dtype=intensity_array.dtype)
        # END-IF

        # Get spectrum index from sub run number
        spec_id = self._sample_logs.get_subrun_indices(sub_run)[0]

        # Another sanity check on the size of 2theta and intensity
        if self._2theta_matrix.shape[1] != two_theta_array.shape[0] \
                or self._diff_data_set[mask_id].shape[1] != intensity_array.shape[0]:
            # Need to check if previously set
            raise RuntimeError('2theta vector are different between parent method set {} and '
                               'reduction engine returned {} OR '
                               'Histogram (shape: {}) to set does not match data diffraction data set defined in '
                               'worksapce (shape: {})'.format(self._2theta_matrix.shape, two_theta_array.shape,
                                                              intensity_array.shape[0],
                                                              self._diff_data_set[mask_id].shape[1]))
        # END-IF-ELSE

        # Set 2theta array
        self._2theta_matrix[spec_id] = two_theta_array
        # Set intensity
        self._diff_data_set[mask_id][spec_id] = intensity_array

    def set_sample_log(self, log_name, sub_runs, log_value_array):
        """Set sample log value for each sub run, i.e., average value in each sub run

        Parameters
        ----------
        log_name : str
            sample log name
        sub_runs: ndarray
            sub runs with same shape as log_value_array
        log_value_array : ndarray
            log values

        Returns
        -------
        None
        """
        # Check inputs
        checkdatatypes.check_string_variable('Log name', log_name)
        checkdatatypes.check_numpy_arrays('Sub runs and log values', [sub_runs, log_value_array], 1, True)
        if len(self._sample_logs) > 0:
            self._sample_logs.matching_subruns(sub_runs)
        else:
            self._sample_logs.subruns = numpy.atleast_1d(sub_runs)

        # Set sub runs and log value to dictionary
        self._sample_logs[log_name] = numpy.atleast_1d(log_value_array)

    def set_sub_runs(self, sub_runs):
        """Set sub runs to this workspace

        Including create the sub run and spectrum map

        Parameters
        ----------
        sub_runs: list
            list of integers as sub runs
        Returns
        -------

        """
        self._sample_logs.subruns = sorted(sub_runs)

    def save_experimental_data(self, hidra_project, sub_runs=None, ignore_raw_counts=False):
        """Save experimental data including raw counts and sample logs to HiDRA project file

        Export (aka save) raw detector counts and sample logs from this HidraWorkspace to a HiDRA project file

        Exporting sub run's counts is an option

        Parameters
        ----------
        hidra_project: HidraProjectFile
            reference to a HyDra project file
        sub_runs: None or list/ndarray(1D)
            None for exporting all or the specified sub runs
        ignore_raw_counts : bool
            flag to not to export raw counts to file

        Returns
        -------
        None
        """
        # Add raw counts if it is specified to save
        if not ignore_raw_counts:
            for sub_run_i in self._raw_counts.keys():
                if sub_runs is None or sub_run_i in sub_runs:
                    hidra_project.append_raw_counts(sub_run_i, self._raw_counts[sub_run_i])
                else:
                    print('[WARNING] sub run {} is not exported to {}'
                          ''.format(sub_run_i, hidra_project.name))
                # END-IF-ELSE
            # END-FOR

        # Add entry for sub runs (first)
        if sub_runs is None:
            # all sub runs
            sub_runs_array = numpy.array(sorted(self._raw_counts.keys()))
        elif isinstance(sub_runs, list):
            # convert to ndarray
            sub_runs_array = numpy.array(sub_runs)
        else:
            # same thing
            sub_runs_array = sub_runs
        hidra_project.append_experiment_log(HidraConstants.SUB_RUNS, sub_runs_array)

        # Add regular sample logs
        for log_name in self._sample_logs.keys():
            # no operation on 'sub run': skip
            if log_name == HidraConstants.SUB_RUNS:
                continue

            # Convert each sample log to a numpy array
            sample_log_value = self.get_sample_log_values(sample_log_name=log_name,
                                                          sub_runs=sub_runs)

            # Add log value to project file
            hidra_project.append_experiment_log(log_name, sample_log_value)
        # END-FOR

        # Save default mask
        if self._default_mask is not None:
            hidra_project.write_mask_detector_array(HidraConstants.DEFAULT_MASK, self._default_mask)
        # Save other masks
        for mask_id in self._mask_dict:
            hidra_project.write_mask_detector_array(mask_id, self._mask_dict[mask_id])

        # Save wave length
        if self._wave_length is not None:
            hidra_project.write_wavelength(self._wave_length)

    def save_reduced_diffraction_data(self, hidra_project):
        """ Export reduced diffraction data to project
        :param hidra_project: HidraProjectFile instance
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_project, HidraProjectFile)

        hidra_project.write_reduced_diffraction_data_set(self._2theta_matrix, self._diff_data_set)

    @property
    def sample_log_names(self):
        """
        return the sample log names
        :return:
        """
        return sorted(self._sample_logs.keys())

    @property
    def sample_logs_for_plot(self):
        """ Get names of sample logs that can be plotted, i.e., the log values are integer or float
        """
        return sorted(self._sample_logs.plottable_logs)

    def set_wavelength(self, wave_length, calibrated):
        """ Set wave length which could be either a float (uniform) or a dictionary
        :param wave_length:
        :param calibrated: Flag for calibrated wave length
        :return:
        """
        # Set universal wave length
        self._wave_length = wave_length

        # Get the sub runs
        sub_runs = self.get_sub_runs()

        if isinstance(wave_length, float):
            # single wave length value
            wl_dict = dict()
            for sub_run in sub_runs:
                wl_dict[sub_run] = wave_length
        elif isinstance(wave_length, dict):
            # already in the dictionary format: check the sub runs
            dict_keys = sorted(wave_length.keys())
            if dict_keys != sub_runs:
                raise RuntimeError('Input wave length dictionary has different set of sub runs')
            wl_dict = wave_length
        else:
            # unsupported format
            raise RuntimeError('Wave length {} in format {} is not supported.'
                               ''.format(wave_length, type(wave_length)))

        # Set to desired target
        if calibrated:
            self._wave_length_calibrated_dict = wl_dict
        else:
            self._wave_length_dict = wl_dict

    def reset_diffraction_data(self):
        """Reset the data structures to store the diffraction data set

        Returns
        -------
        None

        """
        self._2theta_matrix = None
