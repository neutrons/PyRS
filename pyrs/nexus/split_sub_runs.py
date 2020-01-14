# This is a numpy version for prototyping to load NeXus and split events for sub runs
# by numpy and hdf5
import h5py
import numpy as np
from pyrs.utilities import checkdatatypes
from pyrs.dataobjects.constants import HidraConstants
import datetime
import os
from mantid.simpleapi import mtd, Load


HIDRA_PIXEL_NUMBER = 1024**2


def load_split_nexus_python(nexus_name):
    """Wrapping method to load and split event NeXus by sub runs

    Parameters
    ----------
    nexus_name

    Returns
    -------
    dict, dict
        counts, sample logs

    """
    # Init processor
    nexus_processor = NexusProcessor(nexus_name)

    # Get splitters
    sub_run_times, sub_runs = nexus_processor.get_sub_run_times_value()

    # Split counts
    sub_run_counts = nexus_processor.split_events_sub_runs(sub_run_times, sub_runs)

    # Split logs
    sample_logs = nexus_processor.split_sample_logs(sub_run_times, sub_runs)

    return sub_run_counts, sample_logs


class NexusProcessor(object):
    """
    Class to process NeXus files in PyRS
    """
    def __init__(self, nexus_file_name):
        """Init

        Parameters
        ----------
        nexus_file_name : str
            HB2B Event NeXus file name
        """
        self._nexus_name = nexus_file_name

        # check and load
        checkdatatypes.check_file_name(nexus_file_name, True, False, False, 'HB2B event NeXus file name')

        # Load
        self._nexus_h5 = h5py.File(nexus_file_name, 'r')

        # Check number of neutron events.  Raise exception if there is no neutron event
        if self._nexus_h5['entry']['bank1_events']['total_counts'].value[0] < 0.1:
            # no counts
            raise RuntimeError('Run {} does to have counts'.format(self._nexus_name))

    def __del__(self):
        """Destructor

        Close h5py.File instance

        Returns
        -------
        None

        """
        self._nexus_h5.close()

    def get_sub_run_times_value(self):
        """Get the sample log (time and value) of sub run (aka scan indexes)

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            sub run (scan index) times

        """
        # Get the time and value of 'scan_index' (entry) in H5
        scan_index_times = self._nexus_h5['entry']['DASlogs']['scan_index']['time'].value
        scan_index_value = self._nexus_h5['entry']['DASlogs']['scan_index']['value'].value

        if scan_index_times.shape[0] <= 1:
            raise RuntimeError('Sub scan (time = {}, value = {}) is not valid'
                               ''.format(scan_index_times, scan_index_value))

        sub_run_times, sub_runs = self.generate_sub_run_splitter(scan_index_times, scan_index_value)

        # Correct start scan_index time
        corrected_value = self.correct_starting_scan_index_time()
        if corrected_value is not None:
            sub_run_times[0] = corrected_value

        return sub_run_times, sub_runs

    @staticmethod
    def generate_sub_run_splitter(scan_index_times, scan_index_value):
        """Generate event splitters according to sub runs

        """
        # Init
        sub_run_time_list = list()
        sub_run_value_list = list()
        num_scan_index = scan_index_times.shape[0]

        # Loop through all scan indexes to get the correct splitters
        curr_sub_run = 0
        for i_scan in range(num_scan_index):
            if scan_index_value[i_scan] != curr_sub_run:
                #  New run no same as old one: There will be some change!
                if curr_sub_run > 0:
                    # previous run shall be saved: it is ending: record the ending time/current time
                    sub_run_time_list.append(scan_index_times[i_scan])

                if scan_index_value[i_scan] > 0:
                    # new scan index is valid: a new run shall start: record the starting time and sub run value
                    sub_run_time_list.append(scan_index_times[i_scan])
                    sub_run_value_list.append(scan_index_value[i_scan])

                # Update the curr_sub_run
                curr_sub_run = scan_index_value[i_scan]

                # Note: there is one scenario to append 2 and same time stamp: scan index change from i to j, where
                # both i and j are larger than 0
            # END-IF
        # END-FOR

        # Check the ending
        if curr_sub_run > 0:
            # In case the stop (scan_index = 0) is not recorded
            sub_run_time_list.append(np.nan)

        # Convert from list to array
        sub_run_times = np.array(sub_run_time_list)
        sub_run_numbers = np.array(sub_run_value_list)

        # Sanity check
        if sub_run_times.shape[0] % 2 == 1 or sub_run_times.shape[0] == 0:
            raise RuntimeError('Algorithm error: Failed to parse\nTime: {}\nValue: {}.\n'
                               'Current resulted time ({}) is incorrect as odd/even'
                               ''.format(scan_index_times, scan_index_value, sub_run_times))

        if sub_run_times.shape[0] != sub_run_numbers.shape[0] * 2:
            raise RuntimeError('Sub run number {} and sub run times do not match (as twice)'
                               ''.format(sub_run_numbers, sub_run_times))

        return sub_run_times, sub_run_numbers

    def correct_starting_scan_index_time(self, abs_tolerance=0.05):
        """Correct the DAS-issue for mis-record the first scan_index/sub run before the motor is in position

        This goes through a subset of logs and compares when they actually
        get to their specified setpoint, updating the start time for
        event filtering. When this is done ``self._starttime`` will have been updated.

        Parameters
        ----------
        abs_tolerance: float
            When then log is within this absolute tolerance of the setpoint, it is correct

        Returns
        -------
        float
            Corrected value or None

        """
        # loop through the 'special' logs
        start_time = -1E-9

        for log_name in ['sx', 'sy', 'sz', '2theta', 'omega', 'chi', 'phi']:
            if log_name not in self._nexus_h5['entry']['DASlogs'].keys():
                continue  # log doesn't exist - not a good one to look at
            if log_name + 'Setpoint' not in self._nexus_h5['entry']['DASlogs'].keys():
                continue  # log doesn't have a setpoint - not a good one to look at

            # get the observed values of the log
            observed = self._nexus_h5['entry']['DASlogs'][log_name]['value'].value
            if len(observed) <= 1 or observed.std() <= .5 * abs_tolerance:
                continue  # don't bother if the log is constant within half of the tolerance

            # look for the setpoint and find when the log first got there
            # only look at first setpoint
            set_point = self._nexus_h5['entry']['DASlogs'][log_name + 'Setpoint']['value'].value[0]
            for i, value in enumerate(observed):
                if abs(value - set_point) < abs_tolerance:
                    # pick the larger of what was found and the previous largest value
                    start_time = max(self._nexus_h5['entry']['DASlogs'][log_name]['time'].value[i], 0.)
                    break

        # unset the start time if it is before the actual start of the run
        if start_time <= 0:
            start_time = None
        else:
            print('[DEBUG] Shift from start_time = {}'
                  ''.format(start_time))

        return start_time

    def split_events_sub_runs(self, sub_run_times, sub_run_values):
        """Split events by sub runs

        Note: this filters events in the resolution of pulse time.  It is same as Mantid.FilterByLogValue

        Parameters
        ----------
        sub_run_times: numpy.ndarray
            sub run times.  T[2n] = sub run start time, T[2n + 1] = sub run stop time
        sub_run_values: numpy.ndarray
            sub run value: V[n] = sub run number
            V.shape[0] = T.shape[0] / 2

        Returns
        -------
        dict
            Dictionary of split counts for each sub runs.  key = sub run number, value = numpy.ndarray

        """
        # Get pulse times
        pulse_time_array = self._nexus_h5['entry']['bank1_events']['event_time_zero'].value

        # Search index of sub runs' boundaries (start/stop time) in pulse time array
        subrun_pulseindex_array = np.searchsorted(pulse_time_array, sub_run_times)

        # get event index array: same size as pulse times
        event_index_array = self._nexus_h5['entry']['bank1_events']['event_index'].value
        event_id_array = self._nexus_h5['entry']['bank1_events']['event_id'].value

        # split data
        num_sub_runs = sub_run_values.shape[0]
        sub_run_counts_dict = dict()

        for i_sub_run in range(num_sub_runs):
            # get the start and stop index in pulse array
            start_pulse_index = subrun_pulseindex_array[2 * i_sub_run]
            stop_pulse_index = subrun_pulseindex_array[2 * i_sub_run + 1]

            # In case of start
            if start_pulse_index >= event_index_array.size:
                # event ID out of boundary
                start_event_id = event_id_array.shape[0]
            else:
                # get start andn stop event ID from event index array
                start_event_id = event_index_array[start_pulse_index]
            if stop_pulse_index >= event_index_array.size:
                print('[WARNING] for sub run {} out of {}, stop pulse index {} is out of boundary of {}'
                      ''.format(i_sub_run, num_sub_runs, stop_pulse_index, event_index_array.shape))
                # stop_pulse_index = event_index_array.size - 1
                # supposed to be the last pulse and thus use the last + 1 event ID's index
                stop_event_id = event_id_array.shape[0]
            else:
                # natural one
                stop_event_id = event_index_array[stop_pulse_index]

            # get sub set of the events falling into this range
            sub_run_events = event_id_array[start_event_id:stop_event_id]

            # Count the occurrence of each event ID (aka detector ID) as counts on each detector pixel
            hist = np.bincount(sub_run_events, minlength=HIDRA_PIXEL_NUMBER)
            sub_run_counts_dict[int(sub_run_values[i_sub_run])] = hist

        return sub_run_counts_dict

    def split_sample_logs(self, sub_run_times, sub_run_value):
        """Split sample logs according to sub runs

        Parameters
        ----------
        sub_run_times : numpy.ndarray
            sub run times
        sub_run_value : numpy.ndarray
            sub run numbers

        Returns
        -------
        dict
            split sample logs. key = sub run, value = numpy.ndarray

        """
        # Log entry
        das_log_entry = self._nexus_h5['entry']['DASlogs']
        log_names = list(das_log_entry.keys())  # some version of h5py returns KeyMap instead of list
        # move scan index
        log_names.remove('scan_index')

        # make sure sub runs are integer
        sub_run_value = sub_run_value.astype(int)

        # Import sample logs in Time-Series
        split_log_dict = dict()
        irregular_log_names = list()
        for log_name in log_names:
            single_log_entry = das_log_entry[log_name]
            try:
                times = single_log_entry['time']
                value = single_log_entry['value']

                # check type
                value_type = str(value.dtype)

                if value_type.count('float') == 0 and value_type.count('int') == 0:
                    print('[WARNING] Log {} has dtype {}.  No split'.format(log_name, value_type))
                    continue

                split_log, time_i = split_das_log(times, value, sub_run_times, sub_run_value)
                split_log_dict[log_name] = split_log
            except KeyError:
                irregular_log_names.append(log_name)
        # END-FOR

        # Warning output
        if len(irregular_log_names) > 0:
            print('[WARNING] DAS logs: {} are not time series'.format(irregular_log_names))

        # Add back scan index
        split_log_dict['scan_index'] = sub_run_value

        # Duration
        duration_logs = sub_run_times[1::2] - sub_run_times[::2]
        split_log_dict['duration'] = duration_logs

        return split_log_dict

    def create_sample_log_dict(self, sub_run_times, sub_run_numbers):
        """Create dictionary for sample log of a sub run

        Goal:
            1. set self._sample_log_dict[log_name][sub_run_index] with log value (single or time-averaged)
            2. set self._sample_log_dict[HidraConstants.SUB_RUN_DURATION][sub_run_index] with duration

        Parameters
        ----------
        sub_run_times : numpy.ndarray
            sub run times as the relative time to 'start_time'
        sub_run_numbers : numpy.ndarray
            sub run values

        Returns
        -------

        """
        # Check
        if sub_run_numbers.shape[0] * 2 != sub_run_times.shape[0]:
            raise RuntimeError('....')

        # Load file
        ws_name = os.path.basename(self._nexus_name).split('.')[0]
        workspace = Load(Filename=ws_name, MetaDataOnly=True)
        run_obj = workspace.run()

        # Get 'start_time' and create a new sub_run_times in datetime64
        run_start_abs = np.datetime64(run_obj['start_time'].value)
        delta_times = (sub_run_times * 1E9).astype(np.timedelta64)  # convert to nanosecond then to timedetal64
        sub_run_splitter_times = run_start_abs + delta_times

        # this contains all of the sample logs
        sample_log_dict = dict()
        log_array_size = sub_run_numbers.shape[0]
        # loop through all available logs
        for log_name in run_obj.keys():
            # create and calculate the sample log
            sample_log_dict[log_name] = self.split_property(run_obj.getProperty(log_name), sub_run_splitter_times,
                                                            log_array_size)
        # END-FOR

        # create a fictional log for duration
        if HidraConstants.SUB_RUN_DURATION not in sample_log_dict:
            sample_log_dict[HidraConstants.SUB_RUN_DURATION] = sub_run_times[1::2] - sub_run_times[::2]

        return sample_log_dict

    @staticmethod
    def split_property(log_property, splitter_times, log_array_size):
        """
        Calculate the mean value of the sample log "within" the sub run time range
        :param name: Mantid run property's name
        :return:
        """
        # Init split sample logs
        log_dtype = log_property.dtype()
        split_log = np.ndarray(shape=(log_array_size,), dtype=log_dtype)

        if isinstance(log_property.value, np.ndarray) and str(log_dtype) in ['f', 'i']:
            # Float or integer time series property
            split_log[:] = log_property.value
            for i_sb in range(log_array_size):
                time_average_value = log_property.filter
            raise NotImplementedError('From here!')

        elif isinstance(log_property.value, np.ndarray) and str(log_dtype) in ['f', 'i']:
            # ndarray. but not float or integer: get the first value
            split_log[:] = log_property.value[0]
        elif isinstance(log_property.value, list):
            # list, but not time series property: get the first value
            split_log[:] = log_property.value[0]
        else:
            # single value log
            split_log[:] = log_property.value

        return split_log


def split_das_log(log_times, log_values, sub_run_times, sub_run_numbers):
    """Split a time series property to sub runs

    Parameters
    ----------
    log_times : numpy.ndarray
        relative sample log time in second to run start
    log_values : numpy.ndarray
        sample log value
    sub_run_times: numpy.ndarray
        relative sub run start and stop time in second to run start.
        It is twice size of sub runs. 2n index for run start and (2n + 1) index for run stop
    sub_run_numbers : numpy.ndarray
        sub run number, dtype as integer

    Returns
    -------
    numpy.nddarry, float
        split das log in time average for each sub run, duration (second) to split log

    """
    # Profile
    start_time = datetime.datetime.now()

    # Initialize the output numpy array
    split_log_values = np.ndarray(shape=sub_run_numbers.shape, dtype=log_values.dtype)

    # Two cases: single and multiple value
    if log_values.shape[0] == 1:
        # single value: no need to split.  all the sub runs will have same value
        split_log_values[:] = log_values[0]

    else:
        # multiple values: split
        split_bound_indexes = np.searchsorted(log_times, sub_run_times)

        # then calculate time average for each sub run
        for i_sr in range(sub_run_numbers.shape[0]):
            # the starting value of the log in a sub run shall be the last change before the sub run start,
            # so the searched index shall be subtracted by 1
            # avoid (1) breaking lower boundary and (2) breaking the upper boundary
            start_index = min(max(0, split_bound_indexes[2 * i_sr] - 1), log_times.shape[0] - 1)

            # the stopped value of the log shall be the log value 1 prior to the searched result
            stop_index = split_bound_indexes[2 * i_sr + 1]

            # re-define the range of the log time
            sub_times = log_times[start_index:stop_index]
            if sub_times.shape[0] == 0:
                raise RuntimeError('Algorithm error!')
            # change times at the start and append the stop time
            sub_times[0] = sub_run_times[2 * i_sr]
            sub_times = np.append(sub_times, sub_run_times[2 * i_sr + 1])

            # get the range value
            sub_values = log_values[start_index:stop_index]

            # calculate the time average
            try:
                weighted_sum = np.sum(sub_values[:] * (sub_times[1:] - sub_times[:-1]))
            except TypeError as type_err:
                print('Sub values: {}\nSub times: {}'.format(sub_values, sub_times))
                print('Sub value type: {}'.format(sub_values.dtype))
                raise type_err
            time_averaged = weighted_sum / (sub_times[-1] - sub_times[0])

            # record
            split_log_values[i_sr] = time_averaged
        # END-FOR (sub runs)
    # END-IF

    stop_time = datetime.datetime.now()

    return split_log_values, (stop_time - start_time).total_seconds()
