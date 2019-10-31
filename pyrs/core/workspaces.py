# Data manager
import numpy
from pyrs.utilities import checkdatatypes
from pyrs.utilities import rs_project_file

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

        # spectra-sub run mapper
        self._sub_run_to_spectrum = None  # [sub-run] = spectrum, spectrum: 0 - ... continuous
        self._spectrum_to_sub_run = None  # [spectrum] = sub-run

        # wave length
        self._wave_length_dict = None
        self._wave_length_calibrated_dict = None

        # diffraction
        self._2theta_vec = None  # ndarray.  shape = (m, ) m = number of 2theta
        self._diff_data_set = dict()  # [mask id] = ndarray: shape=(n, m), n: number of sub-run, m: number of of 2theta

        # instrument
        self._instrument_setup = None 
        self._instrument_geometry_shift = None  # geometry shift

        # sample logs
        self._sample_log_dict = dict()  # sample logs: [log name][sub run] = value

        # raw Hidra project file
        self._project_file_name = None

        return

    @property
    def name(self):
        """
        Workspace name
        :return:
        """
        return self._name

    def _create_subrun_spectrum_map(self, sub_run_list):
        """
        Set up the sub-run/spectrum maps: This is the only place in this code to write to _sub_run_to_spectrum
        and _spectrum_to_sub_run
        :param sub_run_list:
        :return:
        """
        # this is the only place _sub_run_to_spectrum and _spectrum_to_sub_run that appear at the left of '='
        self._sub_run_to_spectrum = dict()
        self._spectrum_to_sub_run = dict()

        # besides the dictionaries are created
        for spec_id, sub_run in enumerate(sorted(sub_run_list)):
            self._sub_run_to_spectrum[sub_run] = spec_id
            self._spectrum_to_sub_run[spec_id] = sub_run

        return

    def _load_raw_counts(self, hidra_file):
        """ Load raw detector counts from HIDRA file
        :param hidra_file:
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        for sub_run_i in sorted(self._sub_run_to_spectrum.keys()):
            counts_vec_i = hidra_file.get_raw_counts(sub_run_i)
            self._raw_counts[sub_run_i] = counts_vec_i
        # END-FOR

        return

    def _load_reduced_diffraction_data(self, hidra_file):
        """ Load reduced diffraction data from HIDRA file
        :param hidra_file: HidraProjectFile instance
        :return:
        """
        # Check inputs
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # get 2theta value
        try:
            vec_2theta = hidra_file.get_diffraction_2theta_vector()
        except KeyError as key_err:
            print('[INFO] Unable to load 2theta vector from HidraProject file due to {}.'
                  'It is very likely that no reduced data is recorded.'
                  ''.format(key_err))
            return
        # TRY-CATCH
        self._2theta_vec = vec_2theta[:]

        # initialize data set for reduced diffraction patterns
        num_spec = len(hidra_file.get_sub_runs())
        diff_mask_list = hidra_file.get_diffraction_masks()
        print('[DB...BAT...TEST#81] Masks of diffraction data in HidraProjectFile: {}'.format(diff_mask_list))
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
            self._diff_data_set[mask_name] = hidra_file.get_diffraction_intensity_vector(mask_id=mask_name,
                                                                                         sub_run=None)
        # END-FOR (mask)

        print('[INFO] Loaded diffraction data from {} includes : {}'
              ''.format(self._project_file_name, self._diff_data_set.keys()))

        return

    def _load_instrument(self, hidra_file):
        """ Load instrument setup from HIDRA file
        :param hidra_file: HIDRA project file instance
        :return:
        """
        # Check
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # Get values
        self._instrument_setup = hidra_file.get_instrument_geometry()

        return

    def _load_sample_logs(self, hidra_file):
        """ Load sample logs.
        Note: this method can clear all the sample logs added previously. But it is not
            an issue in the real use cases.
        :param hidra_file:  HIDRA project file instance
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # Get special values
        self._sample_log_dict = hidra_file.get_logs()

        return

    def _load_wave_length(self, hidra_file):
        """ Load wave length
        :param hidra_file:  HIDRA project file instance
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)

        # reset the wave length (dictionary) from HIDRA project file
        self._wave_length_dict = hidra_file.get_wave_lengths()

        return

    def get_2theta(self, sub_run):
        """ Get 2theta value from sample log
        This is a special one
        :param sub_run: sub run number (integer)
        :return: float number as 2theta
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))
        try:
            two_theta = self._sample_log_dict[rs_project_file.HidraConstants.TWO_THETA][sub_run]
        except KeyError as key_err:
            raise RuntimeError('Unable to retrieve 2theta value from {} due to {}'
                               .format(sub_run, key_err))

        return two_theta

    def get_l2(self, sub_run):
        """ Get L2 for a specific sub run
        :param sub_run: sub run number (integer)
        :return: L2 or None (i.e., using default L2)
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))

        if rs_project_file.HidraConstants.L2 in self._sample_log_dict:
            # L2 is a valid sample log: get L2
            try:
                l2 = self._sample_log_dict[rs_project_file.HidraConstants.L2][sub_run]
            except KeyError as key_err:
                raise RuntimeError('Unable to retrieve L2 value for {} due to {}. Available sun runs are {}'
                                   .format(sub_run, key_err, self._sample_log_dict[rs_project_file.HidraConstants.L2]))
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
        """
        Get raw detector counts in the order of pixel IDs by a given sub run
        :param sub_run:
        :return:
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if sub_run not in self._raw_counts:
            raise RuntimeError('Sub run {} does not exist in loaded raw counts. FYI loaded '
                               'sub runs are {}'.format(sub_run, self._raw_counts.keys()))

        return self._raw_counts[sub_run]

    def get_sub_runs(self):
        """ Get sub runs that loaded to this workspace
        :return: list of sorted sub runs
        """
        sub_runs = sorted(self._sub_run_to_spectrum.keys())
        if len(sub_runs) == 0:
            raise RuntimeError('Sub run - spectrum map has not been built')

        return sub_runs

    def get_wave_length(self, calibrated, throw_if_not_set):
        """ Get the wave length from the workspace
        :param calibrated: Flag for returning calibrated wave length
        :param throw_if_not_set: Flag to throw an exception if relative wave length (dict) is not set
        :return:
        """
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
        checkdatatypes.check_type('HIDRA project file', hidra_file, rs_project_file.HydraProjectFile)
        self._project_file_name = hidra_file.name

        # create the spectrum map
        sub_run_list = hidra_file.get_sub_runs()
        self._create_subrun_spectrum_map(sub_run_list)

        # load raw detector counts and load instrument
        if load_raw_counts:
            self._load_raw_counts(hidra_file)
            self._load_instrument(hidra_file)

        # load reduced diffraction
        if load_reduced_diffraction:
            self._load_reduced_diffraction_data(hidra_file)

        # load sample logs
        self._load_sample_logs(hidra_file)

        # load the wave length
        self._load_wave_length(hidra_file)

        return

    def get_detector_shift(self):
        """
        Get detector geometry shift
        :return: AnglerDetectorShift instance
        """
        return self._instrument_geometry_shift

    def get_reduced_diffraction_data_set(self, mask_id=None):
        """ Get the full data set (matrix) of reduced diffraction pattern in 2theta unit
        :param mask_id: None (as default main) or ID as a String
        :return:
        """
        # Check
        if mask_id is None:
            # mask_id is 'main'
            pass
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id)

        # Vector 2theta
        vec_2theta = self._2theta_vec.copy()

        try:
            intensity_matrix = self._diff_data_set[mask_id].copy()
        except KeyError:
            raise RuntimeError('Mask ID {} does not exist in reduced diffraction pattern. '
                               'The available masks are {}'
                               ''.format(mask_id, self._diff_data_set.keys()))

        return vec_2theta, intensity_matrix

    def get_reduced_diffraction_data(self, sub_run, mask_id=None):
        """
        get data set of a single diffraction pattern
        :param sub_run: sub run number (integer)
        :param mask_id: None (as default main) or ID as a String
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if mask_id is None:
            # mask_id = 'main'
            pass
        else:
            checkdatatypes.check_string_variable('Mask ID', mask_id)

        if sub_run not in self._sub_run_to_spectrum:
            raise RuntimeError('Sub run {} does not exist. FYI: available sub runs are {}'
                               ''.format(sub_run, self._sub_run_to_spectrum.keys()))
        spec_index = self._sub_run_to_spectrum[sub_run]

        # Vector 2theta
        vec_2theta = self._2theta_vec.copy()

        # Vector intensity
        try:
            vec_intensity = self._diff_data_set[mask_id][spec_index].copy()
        except KeyError:
            raise RuntimeError('Mask ID {} does not exist in reduced diffraction pattern. '
                               'The available masks are {}'
                               ''.format(mask_id, self._diff_data_set.keys()))
        return vec_2theta, vec_intensity

    def get_sample_log_values(self, sample_log_name):
        """
        Get ONE INDIVIDUAL sample log's values as a vector
        :param sample_log_name:
        :return: vector of integer or float in the same order as sub run number
        """
        if sample_log_name == rs_project_file.HidraConstants.SUB_RUNS and \
                sample_log_name not in self._sample_log_dict.keys():
            return self.get_sub_runs()

        checkdatatypes.check_string_variable('Sample log name', sample_log_name,
                                             self._sample_log_dict.keys())

        return self._sample_log_dict[sample_log_name].copy()

    def get_spectrum_index(self, sub_run):
        """
        Get spectrum (index) from sub run number
        :param sub_run: sub run number (integer)
        :return:
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))

        if sub_run not in self._sub_run_to_spectrum:
            raise KeyError('Sub run {} does not exist in spectrum/sub run map.  Available sub runs are {}'
                           ''.format(sub_run, self._sub_run_to_spectrum.keys()))

        return self._sub_run_to_spectrum[sub_run]

    def get_sub_runs_from_spectrum(self, spectra):
        """ Get sub runs corresponding to spectra (same as ws index)
        :param spectra: list/vector/array of spectra (workspace indexes)
        :return:
        """
        if not (isinstance(spectra, list) or isinstance(spectra, numpy.ndarray)):
            raise AssertionError('{} must be list or array'.format(type(spectra)))

        sub_run_vec = numpy.ndarray(shape=(len(spectra), ), dtype='int')
        for i, ws_index in enumerate(spectra):
            sub_run_vec[i] = self._spectrum_to_sub_run[ws_index]

        return sub_run_vec

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

        has_log = sample_log_name in self._sample_log_dict

        return has_log

    def set_raw_counts(self, sub_run_number, counts):
        """
        Set the raw counts to
        :param sub_run_number: integer for sub run number
        :param counts: ndarray of detector counts
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run_number, (1, None))
        checkdatatypes.check_numpy_arrays('Counts', [counts], dimension=None,
                                          check_same_shape=False)

        # Set
        self._raw_counts[sub_run_number] = counts

        return

    def set_reduced_diffraction_data(self, sub_run, mask_id, bin_edges, hist):
        """ Set reduced diffraction data to workspace
        :param sub_run:
        :param mask_id: None (no mask) or String (with mask indexed by this string)
        :param bin_edges:
        :param hist:
        :return:
        """
        # Check inputs
        checkdatatypes.check_int_variable('Sub run number', sub_run, (1, None))
        if mask_id is not None:
            checkdatatypes.check_string_variable('Mask ID', mask_id)
            print('L667: Mask ID: "{}"'.format(mask_id))

        # check status
        if self._sub_run_to_spectrum is None:
            raise RuntimeError('Sub run - spectrum map has not been set up yet!')

        # Set 2-theta (X)
        if self._2theta_vec is None:
            # First time set up
            # Set X
            self._2theta_vec = bin_edges.copy()

        elif self._2theta_vec.shape != bin_edges.shape:
            # Need to check if previously set
            raise RuntimeError('2theta vector are different between parent method set {} and '
                               'reduction engine returned {}'.format(self._2theta_vec.shape, bin_edges.shape))
        # END-IF-ELSE

        # Initialize Y with mask

        if mask_id not in self._diff_data_set:
            num_sub_runs = len(self._sub_run_to_spectrum)
            self._diff_data_set[mask_id] = numpy.ndarray(shape=(num_sub_runs, hist.shape[0]), dtype=hist.dtype)

        # Check array shape
        if self._diff_data_set[mask_id].shape[1] != hist.shape[0]:
            raise RuntimeError('Histogram (shape: {}) to set does not match data diffraction data set defined in'
                               'worksapce (shape: {})'.format(hist.shape[0], self._diff_data_set[mask_id].shape[1]))

        # Set Y
        spec_id = self._sub_run_to_spectrum[sub_run]
        self._diff_data_set[mask_id][spec_id] = hist

        return

    def set_sample_log(self, log_name, log_value_array):
        """
        Set sample log value for each sub run, i.e., average value in each sub run
        :param log_name:
        :param log_value_array:
        :return:
        """
        # Check inputs
        checkdatatypes.check_string_variable('Log name', log_name)
        checkdatatypes.check_numpy_arrays('Log value ', [log_value_array], 1, False)

        # Set
        self._sample_log_dict[log_name] = log_value_array

        return

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
        sub_runs = sorted(sub_runs)

        self._create_subrun_spectrum_map(sub_runs)

        return


    def save_experimental_data(self, hidra_project, sub_runs=None):
        """Save experimental data including raw counts and sample logs to HiDRA project file
        Export (aka save) raw detector counts and sample logs from this HidraWorkspace to a HiDRA project file
        Parameters
        ----------
        hidra_project: HydraProjectFile
            reference to a HyDra project file
        sub_runs: None or list/ndarray(1D)
            None for exporting all or the specified sub runs
        Returns
        -------
        None
        """

        # Raw counts
        for sub_run_i in self._raw_counts.keys():
            if sub_runs is None or sub_run_i in sub_runs:
                hidra_project.add_raw_counts(sub_run_i, self._raw_counts[sub_run_i])
            else:
                print('[WARNING] sub run {} is not exported to {}'
                      ''.format(sub_run_i, hidra_project.name))

            # END-IF-ELSE
        # END-FOR

        # Add sub runs first
        if sub_runs is None:
            # all sub runs
            sub_runs_array = numpy.array(sorted(self._raw_counts.keys()))
        elif isinstance(sub_runs, list):
            # convert to ndarray
            sub_runs_array = numpy.array(sub_runs)
        else:
            # same thing
            sub_runs_array = sub_runs

        hidra_project.add_experiment_log(rs_project_file.HidraConstants.SUB_RUNS, sub_runs_array)

        # Add regular ample logs
        for log_name in self._sample_log_dict.keys():
            # no operation on 'sub run': skip
            if log_name == rs_project_file.HidraConstants.SUB_RUNS:
                continue

            # Convert each sample log to a numpy array
            sample_log_value = self.get_sample_log_values(sample_log_name=log_name,
                                                          sub_runs=sub_runs)

            # Add log value to project file
            hidra_project.add_experiment_log(log_name, sample_log_value)

        # END-FOR

        return

    def save_reduced_diffraction_data(self, hidra_project):
        """ Export reduced diffraction data to project
        :param hidra_project: HidraProjectFile instance
        :return:
        """
        checkdatatypes.check_type('HIDRA project file', hidra_project, rs_project_file.HydraProjectFile)

        hidra_project.set_reduced_diffraction_data_set(self._2theta_vec, self._diff_data_set)

        return

    @property
    def sample_log_names(self):
        """
        return the sample log names
        :return:
        """
        return self._sample_log_dict.keys()

    def sample_log_values(self, sample_log_name):
        """
        get sample log value
        :param sample_log_name:
        :return:
        """
        checkdatatypes.check_string_variable('Sample log name', sample_log_name)
        if sample_log_name not in self._sample_log_dict:
            raise RuntimeError('Sample log {0} cannot be found.'.format(sample_log_name))

        return self._sample_log_dict[sample_log_name]

    @property
    def sample_logs_for_plot(self):
        """ Get names of sample logs that can be plotted, i.e., the log values are integer or float
        :return:
        """
        sample_logs = list()
        for sample_log_name in self._sample_log_dict.keys():
            sample_log_value = self._sample_log_dict[sample_log_name]
            if sample_log_value.dtype != object:
                sample_logs.append(sample_log_name)

        return sorted(sample_logs)

    def set_wave_length(self, wave_length, calibrated):
        """ Set wave length which could be either a float (uniform) or a dictionary
        :param wave_length:
        :param calibrated: Flag for calibrated wave length
        :return:
        """
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
        # END-IF-ELSE

        # Set to desired target
        if calibrated:
            self._wave_length_calibrated_dict = wl_dict
        else:
            self._wave_length_dict = wl_dict

        return
