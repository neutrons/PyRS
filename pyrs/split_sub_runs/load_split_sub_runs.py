# This is a numpy version for prototyping to load NeXus and split events for sub runs
# by numpy and hdf5
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import h5py
import numpy as np
from pyrs.utilities import checkdatatypes
from pyrs.dataobjects.constants import HidraConstants
import datetime
import os
from mantid.simpleapi import mtd, DeleteWorkspace, LoadEventNexus, LoadMask
from mantid.kernel import BoolTimeSeriesProperty, FloatFilteredTimeSeriesProperty, FloatTimeSeriesProperty
from mantid.kernel import Int32TimeSeriesProperty, Int64TimeSeriesProperty, Int32FilteredTimeSeriesProperty,\
    Int64FilteredTimeSeriesProperty


HIDRA_PIXEL_NUMBER = 1024**2


def load_split_nexus_python(nexus_name, mask_file_name):
    """Wrapping method to load and split event NeXus by sub runs

    Parameters
    ----------
    nexus_name : str
        NeXus file name
    mask_file_name: str or None
        Mantid mask file in XML

    Returns
    -------
    dict, dict
        counts, sample logs

    """
    # Init processor
    nexus_processor = NexusProcessor(nexus_name)

    # Mask detector
    if mask_file_name:
        mask_array = nexus_processor.process_mask(mask_file_name)
    else:
        mask_array = None

    # Get splitters
    sub_run_times, sub_runs = nexus_processor.get_sub_run_times_value()

    # Split counts
    time_split_start = datetime.datetime.now()
    sub_run_counts = nexus_processor.split_events_sub_runs(sub_run_times, sub_runs, mask_array)
    time_split_end = datetime.datetime.now()
    print('[INFO] Sub run splitting duration = {} second from {} to {}'
          ''.format((time_split_end - time_split_start).total_seconds(), time_split_start, time_split_end))

    # Split logs
    sample_logs = nexus_processor.split_sample_logs(sub_run_times, sub_runs)
    log_split_end = datetime.datetime.now()
    print('[INFO] Sub run splitting duration = {} second from {} to {}'
          ''.format((log_split_end - time_split_end).total_seconds(), time_split_end, log_split_end))
    del nexus_processor

    return sub_run_counts, sample_logs, mask_array


def convert_pulses_to_datetime64(h5obj):
    '''The h5object is the h5py handle to ``event_time_zero``. This only supports pulsetimes in seconds'''
    if h5obj.attrs['units'].decode() != 'second':
        raise RuntimeError('Do not understand time units "{}"'.format(h5obj.attrs['units']))

    # the value is number of seconds as a float
    pulse_time = h5obj.value

    # Convert deltas to times with units. This has to be done through
    # nanoseconds because numpy truncates things to integers during the conversion
    pulse_time = pulse_time * 1.e9 * np.timedelta64(1, 'ns')

    # get absolute offset and convert to absolute time
    start_time = np.datetime64(h5obj.attrs['offset'])

    return pulse_time + start_time


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

        # Create workspace for sample logs and optionally mask
        # Load file
        self._ws_name = os.path.basename(self._nexus_name).split('.')[0]
        self._workspace = LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._ws_name,
                                         MetaDataOnly=True, LoadMonitors=False)
        # raise an exception if there is only one scan index entry
        # this is an underlying assumption of the rest of the code
        if self._workspace.run()['scan_index'].size == 1:
            # Get the time and value of 'scan_index' (entry) in H5
            scan_index_times = self._workspace.run()['scan_index'].times
            scan_index_value = self._workspace.run()['scan_index'].value
            raise RuntimeError('Sub scan (time = {}, value = {}) is not valid'
                               ''.format(scan_index_times, scan_index_value))

    def __del__(self):
        if self._ws_name in mtd:
            DeleteWorkspace(Workspace=self._ws_name)

    def process_mask(self, mask_file_name):
        """

        Parameters
        ----------
        mask_file_name

        Returns
        -------

        """
        # Check input
        checkdatatypes.check_file_name(mask_file_name, True, False, False, 'Mask XML file')
        if self._workspace is None:
            raise RuntimeError('Meta data only workspace {} does not exist'.format(self._ws_name))

        # Load mask XML to workspace
        mask_ws_name = os.path.basename(mask_file_name.split('.')[0])
        mask_ws = LoadMask(Instrument='nrsf2', InputFile=mask_file_name, RefWorkspace=self._workspace,
                           OutputWorkspace=mask_ws_name)

        # Extract mask out
        # get the Y array from mask workspace: shape = (1048576, 1)
        mask_array = mask_ws.extractY().flatten()
        # in Mantid's mask workspace, 1 stands for mask (value cleared), 0 stands for non-mask (value kept)
        mask_array = 1 - mask_array.astype(int)

        return mask_array

    def get_sub_run_times_value(self):
        """Get the sample log (time and value) of sub run (aka scan indexes)

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            sub run (scan index) times

        """
        # Get the time and value from the mantid workspace
        scan_index_times = self._workspace.run()['scan_index'].times   # absolute times
        scan_index_value = self._workspace.run()['scan_index'].value

        if scan_index_times.shape[0] <= 1:
            raise RuntimeError('Sub scan (time = {}, value = {}) is not valid'
                               ''.format(scan_index_times, scan_index_value))

        sub_run_times, sub_runs = self.generate_sub_run_splitter(scan_index_times, scan_index_value)

        # Move start time to when interesting logs stop moving
        corrected_value = self.correct_starting_scan_index_time(sub_run_times[0])
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

    def correct_starting_scan_index_time(self, start_time, abs_tolerance=0.05):
        """Correct the DAS-issue for mis-record the first scan_index/sub run before the motor is in position

        This goes through a subset of logs and compares when they actually
        get to their specified setpoint, updating the start time for
        event filtering. When this is done ``self._starttime`` will have been updated.

        Parameters
        ----------
        start_time: numpy.datetime64
            The start time according to the scan_index log
        abs_tolerance: float
            When then log is within this absolute tolerance of the setpoint, it is correct

        Returns
        -------
        float
            Corrected value or None

        """
        run_obj = self._workspace.run()
        # loop through the 'special' logs
        for log_name in ['sx', 'sy', 'sz', '2theta', 'omega', 'chi', 'phi']:
            if log_name not in run_obj:
                continue  # log doesn't exist - not a good one to look at
            if log_name + 'Setpoint' not in run_obj:
                continue  # log doesn't have a setpoint - not a good one to look at
            if run_obj[log_name].size() == 1:
                continue  # there is only one value

            # get the observed values of the log
            observed = run_obj[log_name].value
            if observed.std() <= .5 * abs_tolerance:
                continue  # don't bother if the log is constant within half of the tolerance

            # look for the setpoint and find when the log first got there
            # only look at first setpoint
            set_point = run_obj[log_name + 'Setpoint'].value[0]
            for log_time, value in zip(run_obj[log_name].times, observed):
                if abs(value - set_point) < abs_tolerance:
                    # pick the larger of what was found and the previous largest value
                    if log_time > start_time:
                        start_time = log_time
                    break

        print('[DEBUG] Shift from start_time = {}'.format(np.datetime_as_string(start_time)))

        return start_time

    def split_events_sub_runs(self, sub_run_times, sub_run_values, mask_array):
        """Split events by sub runs

        Note: this filters events in the resolution of pulse time.  It is same as Mantid.FilterByLogValue

        Parameters
        ----------
        sub_run_times: numpy.ndarray
            sub run times.  T[2n] = sub run start time, T[2n + 1] = sub run stop time
        sub_run_values: numpy.ndarray
            sub run value: V[n] = sub run number
            V.shape[0] = T.shape[0] / 2
        mask_array : numpy.ndarray or None
            array of 1 or 0 for masking

        Returns
        -------
        dict
            Dictionary of split counts for each sub runs.  key = sub run number, value = numpy.ndarray

        """

        # Load: this h5 will be opened all the time
        with h5py.File(self._nexus_name, 'r') as nexus_h5:
            bank1_events = nexus_h5['entry']['bank1_events']
            # Check number of neutron events.  Raise exception if there is no neutron event
            if bank1_events['total_counts'].value[0] < 0.1:
                # no counts
                raise RuntimeError('Run {} has no count.  Proper reduction requires the run to have count'
                                   ''.format(self._nexus_name))

            # get event index array: same size as pulse times
            event_index_array = bank1_events['event_index'].value
            # detector id for the events
            event_id_array = bank1_events['event_id'].value

            # get pulse times
            pulse_time_array = convert_pulses_to_datetime64(bank1_events['event_time_zero'])

        # Search index of sub runs' boundaries (start/stop time) in pulse time array
        subrun_pulseindex_array = np.searchsorted(pulse_time_array, sub_run_times)

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

            # Mask
            if mask_array is not None:
                assert hist.shape == mask_array.shape
                hist *= mask_array

            sub_run_counts_dict[int(sub_run_values[i_sub_run])] = hist

        return sub_run_counts_dict

    def split_sample_logs(self, sub_run_times, sub_run_numbers):
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
            raise RuntimeError('Should have twice as many times as values')

        run_obj = self._workspace.run()

        # this contains all of the sample logs
        sample_log_dict = dict()
        log_array_size = sub_run_numbers.shape[0]
        # loop through all available logs
        for log_name in run_obj.keys():
            # create and calculate the sample log
            sample_log_dict[log_name] = self.split_property(run_obj.getProperty(log_name), sub_run_times,
                                                            log_array_size)
        # END-FOR

        # create a fictional log for duration
        if HidraConstants.SUB_RUN_DURATION not in sample_log_dict:
            durations = (sub_run_times[1::2] - sub_run_times[::2]) / np.timedelta64(1, 's')
            sample_log_dict[HidraConstants.SUB_RUN_DURATION] = durations

        return sample_log_dict

    def split_property(self, log_property, splitter_times, log_array_size):
        """Calculate the mean value of the sample log "within" the sub run time range

        Parameters
        ----------
        log_property
        splitter_times
        log_array_size

        Returns
        -------
        numpy.ndarray
            split logs

        """
        # Init split sample logs
        log_dtype = log_property.dtype()
        split_log = np.ndarray(shape=(log_array_size,), dtype=log_dtype)

        if isinstance(log_property.value, np.ndarray) and str(log_dtype) in ['f', 'i']:
            # Float or integer time series property: split and get time average
            for i_sb in range(log_array_size):
                split_log[i_sb] = self._calculate_sub_run_time_average(log_property, splitter_times[2 * i_sb],
                                                                       splitter_times[2 * i_sb + 1])
            # END-FOR
        elif isinstance(log_property.value, np.ndarray) and str(log_dtype) in ['f', 'i']:
            # value is ndarray. but not float or integer: get the first value
            split_log[:] = log_property.value[0]
        elif isinstance(log_property.value, list):
            # list, but not time series property: get the first value
            split_log[:] = log_property.value[0]
        else:
            # single value log
            split_log[:] = log_property.value

        return split_log

    @staticmethod
    def _calculate_sub_run_time_average(log_property, sub_run_start_time, sub_run_stop_time):

        # create a Boolean time series property as the filter
        time_filter = BoolTimeSeriesProperty('filter')
        time_filter.addValue(sub_run_start_time, True)
        time_filter.addValue(sub_run_stop_time, False)

        # filter and get time average value
        if isinstance(log_property, FloatTimeSeriesProperty):
            filtered_tsp = FloatFilteredTimeSeriesProperty(log_property, time_filter)
        elif isinstance(log_property, Int32TimeSeriesProperty):
            filtered_tsp = Int32FilteredTimeSeriesProperty(log_property, time_filter)
        elif isinstance(log_property, Int64TimeSeriesProperty):
            filtered_tsp = Int64FilteredTimeSeriesProperty(log_property, time_filter)
        else:
            raise NotImplementedError('TSP log property {} of type {} is not supported'
                                      ''.format(log_property.name, type(log_property)))
        # END-IF

        time_average_value = filtered_tsp.timeAverageValue()
        del filtered_tsp

        return time_average_value
