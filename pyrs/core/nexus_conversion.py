"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
import h5py
from mantid.kernel import Logger, BoolTimeSeriesProperty, FloatFilteredTimeSeriesProperty, FloatTimeSeriesProperty
from mantid.kernel import Int32TimeSeriesProperty, Int64TimeSeriesProperty, Int32FilteredTimeSeriesProperty,\
    Int64FilteredTimeSeriesProperty
from mantid.simpleapi import mtd, DeleteWorkspace, LoadEventNexus, LoadMask, RemoveLogs
import numpy as np
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, HidraSetup
from pyrs.core import MonoSetting  # type: ignore
from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities import checkdatatypes

SUBRUN_LOGNAME = 'scan_index'
NUM_PIXEL_1D = 1024
HIDRA_PIXEL_NUMBER = NUM_PIXEL_1D * NUM_PIXEL_1D
PIXEL_SIZE = 0.3 / NUM_PIXEL_1D
ARM_LENGTH = 0.985

DEFAULT_KEEP_LOGS = ['experiment_identifier', 'run_number', 'run_title', 'file_notes', 'start_time', 'end_time',
                     'SampleId', 'SampleName', 'SampleDescription', 'StrainDirection', 'hklPhase', 'Wavelength',
                     'Filename', 'sub-run', 'duration', 'mrot', 'mtilt', 'mb220', 'mb511', 'ISD', 'ISR:X:Gap',
                     'ISR:Y:Gap', 'DOX', 'DOY', 'DOR', 'omega', '2theta', 'phi', 'chi', 'sx', 'sy', 'sz',
                     'vx', 'vy', 'vz', 'omegaSetpoint', '2thetaSetpoint', 'phiSetpoint', 'chiSetpoint', 'sxSetpoint',
                     'sySetpoint', 'szSetpoint', 'scan_index', 'duration']


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


def calculate_sub_run_time_average(log_property, time_filter) -> float:
    '''Determine the time average value of the supplied log'''
    if log_property.size() == 1:  # single value property just copy
        time_average_value = log_property.value
    elif time_filter is None:  # no filtering means use all values
        time_average_value = log_property.timeAverageValue()
    else:
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

        time_average_value = filtered_tsp.timeAverageValue()
        del filtered_tsp

    return time_average_value


class Splitter:
    r"""
    Time splitter

    Parameters
    ----------
    runObj: ~mantid.
    """
    def __init__(self, runObj):
        self._log = Logger(__name__)

        # verify the scan index exists
        try:
            if runObj['scan_index'].size() == 0:
                raise RuntimeError('"scan_index" is empty')
        except KeyError as e:
            raise RuntimeError('"scan_index" does not exist') from e

        # Get the time and value from the run object
        scan_index_times = runObj['scan_index'].times   # absolute times
        scan_index_value = runObj['scan_index'].value
        # TODO add final time from pcharge logs + 1s with scan_index=0

        if np.unique(scan_index_value).size == 1:
            raise RuntimeError('WARNING: only one scan_index value')  # TODO should be something else

        self.times = None
        self.subruns = None
        self.propertyFilters = list()

        self.__generate_sub_run_splitter(scan_index_times, scan_index_value)
        self.__correct_starting_scan_index_time(runObj)
        self._createPropertyFilters()

    def __generate_sub_run_splitter(self, scan_index_times, scan_index_value) -> None:
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
            # In case the stop (scan_index = 0) is not recorded - add end time one day into the future
            sub_run_time_list.append(sub_run_time_list[-1] + np.timedelta64(1, 'D'))

        # Convert from list to array
        self.times = np.array(sub_run_time_list)
        self.subruns = np.array(sub_run_value_list)

        # Sanity check
        if self.times.shape[0] % 2 == 1 or self.times.shape[0] == 0:
            raise RuntimeError('Algorithm error: Failed to parse\nTime: {}\nValue: {}.\n'
                               'Current resulted time ({}) is incorrect as odd/even'
                               ''.format(scan_index_times, scan_index_value, self.times))

        if self.times.shape[0] != self.subruns.shape[0] * 2:
            raise RuntimeError('Sub run number {} and sub run times {} do not match (as twice)'
                               ''.format(self.subruns, self.times))

    def __correct_starting_scan_index_time(self, runObj, abs_tolerance: float = 0.05) -> None:
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
        start_time = self.times[0]
        # loop through the 'special' logs
        for log_name in ['sx', 'sy', 'sz', '2theta', 'omega', 'chi', 'phi']:
            if log_name not in runObj:
                continue  # log doesn't exist - not a good one to look at
            if log_name + 'Setpoint' not in runObj:
                continue  # log doesn't have a setpoint - not a good one to look at
            if runObj[log_name].size() == 1:
                continue  # there is only one value

            # get the observed values of the log
            observed = runObj[log_name].value
            if observed.std() <= .5 * abs_tolerance:
                continue  # don't bother if the log is constant within half of the tolerance

            # look for the setpoint and find when the log first got there
            # only look at first setpoint
            set_point = runObj[log_name + 'Setpoint'].value[0]
            for log_time, value in zip(runObj[log_name].times, observed):
                if abs(value - set_point) < abs_tolerance:
                    # pick the larger of what was found and the previous largest value
                    if log_time > start_time:
                        start_time = log_time
                    break

        self._log.debug('Shift from start_time {} to {}'.format(np.datetime_as_string(self.times[0]),
                        np.datetime_as_string(start_time)))
        self.times[0] = start_time

    @property
    def durations(self):
        return (self.times[1::2] - self.times[::2]) / np.timedelta64(1, 's')

    @property
    def size(self) -> int:
        return self.subruns.size

    def _createPropertyFilters(self) -> None:
        self.propertyFilters = list()
        if self.subruns.size == 1:
            self.propertyFilters.append(None)
        else:
            for subrun_index in range(self.subruns.size):
                subrun_start_time = self.times[2 * subrun_index]
                subrun_stop_time = self.times[2 * subrun_index + 1]

                # create a Boolean time series property as the filter
                time_filter = BoolTimeSeriesProperty('filter')
                time_filter.addValue(subrun_start_time, True)
                time_filter.addValue(subrun_stop_time, False)

                self.propertyFilters.append(time_filter)


class NeXusConvertingApp:
    """
    Convert NeXus file to Hidra project file
    """
    def __init__(self, nexus_file_name, mask_file_name=None, extra_logs=list()):
        """Initialization

        Parameters
        ----------
        nexus_file_name : str
            Name of NeXus file
        mask_file_name : str
            Name of masking file
        extra_logs : list, tuple
            list of string with no default logs to keep in project file
        """
        # configure logging for this class
        self._log = Logger(__name__)

        # validate NeXus file exists
        checkdatatypes.check_file_name(nexus_file_name, True, False, False, 'NeXus file')
        self._nexus_name = nexus_file_name

        # validate mask file exists
        if mask_file_name is None:
            self._mask_file_name = None
        else:
            checkdatatypes.check_file_name(mask_file_name, True, False, False, 'Mask file')
            self._mask_file_name = mask_file_name
            if not mask_file_name.lower().endswith('.xml'):
                raise NotImplementedError('Only Mantid mask in XML format is supported now.  File '
                                          '{} with type {} is not supported yet.'
                                          ''.format(mask_file_name, mask_file_name.split('.')[-1]))

        # workspaces
        self._event_ws_name = os.path.basename(nexus_file_name).split('.')[0]

        logs_to_keep = list(extra_logs)
        logs_to_keep.extend(DEFAULT_KEEP_LOGS)

        self.__load_logs(logs_to_keep)

        # load the mask
        self.mask_array = None  # TODO to promote direct access
        if mask_file_name:
            self.__load_mask(mask_file_name)

        # create the hidra workspace
        self._hidra_workspace = workspaces.HidraWorkspace(self._nexus_name)

        # Set a default instrument with this workspace
        # set up instrument
        # initialize instrument with hard coded values
        instrument = DENEXDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D, PIXEL_SIZE, PIXEL_SIZE,
                                           ARM_LENGTH, False)

        self._hidra_workspace.set_instrument_geometry(instrument)

        # project file
        self._project_file = None

    def __del__(self):
        if self._event_ws_name in mtd:
            DeleteWorkspace(Workspace=self._event_ws_name, EnableLogging=False)

    def __load_logs(self, logs_to_keep):
        '''Use mantid to load the logs then set up the Splitters object'''
        self._event_wksp = LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name,
                                          MetaDataOnly=True, LoadMonitors=False)

        # remove unwanted sample logs
        RemoveLogs(self._event_wksp, KeepLogs=logs_to_keep)

        # raise an exception if there is only one scan index entry
        # this is an underlying assumption of the rest of the code
        if self._event_wksp.run()['scan_index'].size() == 1 \
                or np.unique(self._event_wksp.run()['scan_index'].value).size == 1:
            self._splitter = None
        else:
            # object to be used for splitting times
            self._splitter = Splitter(self._event_wksp.run())

    def __load_mask(self, mask_file_name):
        # Check input
        checkdatatypes.check_file_name(mask_file_name, True, False, False, 'Mask XML file')
        if self._event_wksp is None:
            raise RuntimeError('Meta data only workspace {} does not exist'.format(self._event_ws_name))

        # Load mask XML to workspace
        mask_ws_name = os.path.basename(mask_file_name.split('.')[0])
        mask_ws = LoadMask(Instrument='nrsf2', InputFile=mask_file_name, RefWorkspace=self._event_wksp,
                           OutputWorkspace=mask_ws_name)

        # Extract mask out
        # get the Y array from mask workspace: shape = (1048576, 1)
        self.mask_array = mask_ws.extractY().flatten()
        # in Mantid's mask workspace: one stands delete, zero stands for keep
        # we multiply by the value: zero is delete, one is keep
        self.mask_array = 1 - self.mask_array.astype(int)

        # clean up
        DeleteWorkspace(Workspace=mask_ws_name)

    def _generate_subrun_event_indices(self, pulse_time_array, event_index_array, num_events):
        # convert times to array indices - a[i-1] < v <= a[i]
        subrun_pulseindex_array = np.searchsorted(pulse_time_array, self._splitter.times)

        # locations that are greater than the number of pixels
        mask = subrun_pulseindex_array < event_index_array.size

        # it doesn't matter what the initial values are
        subrun_event_index = np.empty(subrun_pulseindex_array.size, dtype=subrun_pulseindex_array.dtype)
        # standard method is mappping
        subrun_event_index[mask] = event_index_array[subrun_pulseindex_array[mask]]
        # things off the end should be set to consume the rest of the events
        subrun_event_index[np.logical_not(mask)] = num_events + 1

        # make sure filter is sorted
        if not np.all(subrun_event_index[:-1] <= subrun_event_index[1:]):
            raise RuntimeError('Filter indices are not ordered: {}'.format(subrun_event_index))

        return subrun_event_index

    def split_events_sub_runs(self):
        '''Filter the data by ``scan_index`` and set counts array in the hidra_workspace'''
        # Load: this h5 will be opened all the time
        with h5py.File(self._nexus_name, 'r') as nexus_h5:
            bank1_events = nexus_h5['entry']['bank1_events']
            # Check number of neutron events.  Raise exception if there is no neutron event
            if bank1_events['total_counts'].value[0] < 0.1:
                # no counts
                raise RuntimeError('Run {} has no count.  Proper reduction requires the run to have count'
                                   ''.format(self._nexus_name))

            # detector id for the events
            event_id_array = bank1_events['event_id'].value

            if self._splitter:
                # get event index array: same size as pulse times
                event_index_array = bank1_events['event_index'].value
                # get pulse times
                pulse_time_array = convert_pulses_to_datetime64(bank1_events['event_time_zero'])
                subrun_eventindex_array = self._generate_subrun_event_indices(pulse_time_array, event_index_array,
                                                                              event_id_array.size)
                # reduce memory foot print
                del pulse_time_array, event_index_array

        # split data
        subruns = list()
        if self._splitter:
            for subrun, start_event_index, stop_event_index in zip(self._splitter.subruns.tolist(),
                                                                   subrun_eventindex_array[::2].tolist(),
                                                                   subrun_eventindex_array[1::2].tolist()):
                subruns.append(subrun)
                # get sub set of the events falling into this range
                # and count the occurrence of each event ID (aka detector ID) as counts on each detector pixel
                hist = np.bincount(event_id_array[start_event_index:stop_event_index], minlength=HIDRA_PIXEL_NUMBER)

                # mask (set to zero) the pixels that are not wanted
                if self.mask_array is not None:
                    assert hist.shape == self.mask_array.shape
                    hist *= self.mask_array

                # set it in the workspace
                self._hidra_workspace.set_raw_counts(int(subrun), hist)
        else:  # or histogram everything
            subruns.append(1)
            hist = np.bincount(event_id_array, minlength=HIDRA_PIXEL_NUMBER)

            # mask (set to zero) the pixels that are not wanted
            if self.mask_array is not None:
                assert hist.shape == self.mask_array.shape
                hist *= self.mask_array

            # set it in the workspace
            self._hidra_workspace.set_raw_counts(1, hist)

        return np.array(subruns)

    def split_sample_logs(self, subruns):
        r"""
        Partition each log entry according to the subruns

        Goal:
            1. set sample logs on the hidra workspace
            2. set duration on the hidra worksapce

        Returns
        -------
        dict
            Each key corresponds to one log name, and each value corresponds to an array of log values. Each item
            in this array corresponds to the average value of the log within a particular subrun
        """
        run_obj = self._event_wksp.run()

        # Example: if we have three subruns and the average value of log entry 'vx` for each subrun
        # is 0.1, 0.3, and 0.5, then we have  ample_log_dict['vx'] == np.array([0.1, 0.3, 0.5])
        sample_log_dict = dict()

        if self._splitter:
            log_array_size = self._splitter.subruns.shape[0]
        else:
            log_array_size = 1

        # loop through all available logs
        for log_name in run_obj.keys():
            # create and calculate the sample log
            sample_log_dict[log_name] = self.__split_property(run_obj, log_name, log_array_size)
        # END-FOR

        # create a fictional log for duration
        if HidraConstants.SUB_RUN_DURATION not in sample_log_dict:
            if self._splitter:
                sample_log_dict[HidraConstants.SUB_RUN_DURATION] = self._splitter.durations
            else:
                duration = np.ndarray(shape=(log_array_size,), dtype=float)
                duration[0] = run_obj.getPropertyAsSingleValue('duration')
                sample_log_dict[HidraConstants.SUB_RUN_DURATION] = duration

        # set the logs on the hidra workspace
        for log_name, log_value in sample_log_dict.items():
            if log_name in ['scan_index', HidraConstants.SUB_RUNS]:
                continue  # skip 'SUB_RUNS'
            # find the units of the log
            if log_name == HidraConstants.SUB_RUN_DURATION:
                log_units = 'second'
            else:
                log_units = run_obj.getProperty(log_name).units
            self._hidra_workspace.set_sample_log(log_name, subruns, log_value, units=log_units)

        return sample_log_dict  # needed for testing

    def __split_property(self, runObj, log_name, log_array_size):
        """Calculate the mean value of the sample log "within" the sub run time range

        Parameters
        ----------
        runObj
        log_name
        log_array_size

        Returns
        -------
        numpy.ndarray
            split logs
        """
        # Init split sample logs
        log_property = runObj[log_name]
        log_dtype = log_property.dtype()
        split_log = np.ndarray(shape=(log_array_size,), dtype=log_dtype)

        if self._splitter and isinstance(log_property.value, np.ndarray) and str(log_dtype) in ['f', 'i']:
            # Float or integer time series property: split and get time average
            for i_sb in range(log_array_size):
                split_log[i_sb] = calculate_sub_run_time_average(log_property,
                                                                 self._splitter.propertyFilters[i_sb])
        else:
            try:
                split_log[:] = runObj.getPropertyAsSingleValue(log_name)
            except ValueError:
                if isinstance(log_property.value, str):
                    split_log[:] = log_property.value
                elif isinstance(log_property.value, list):
                    split_log[:] = log_property.value[0]
                else:
                    raise ValueError('Cannot filter log "{}" of type "{}"'.format(log_name, log_dtype))

        return split_log

    def convert(self, use_mantid=False):
        """Main method to convert NeXus file to HidraProject File by

        1. split the workspace to sub runs
        2. for each split workspace, aka a sub run, get the total counts for each spectrum and save to a 1D array

        Parameters
        ----------
        use_mantid : bool
            Flag to use Mantid library to convert NeXus (True);
            Otherwise, use PyRS/Python algorithms to convert NeXus

        Returns
        -------
        pyrs.core.workspaces.HidraWorkspace
            HidraWorkspace for converted data

        """
        if use_mantid:
            raise RuntimeError('use_mantid=True is no longer supported')

        # set counts to each sub run
        sub_runs = self.split_events_sub_runs()

        # set mask
        if self.mask_array is not None:
            self._hidra_workspace.set_detector_mask(self.mask_array, is_default=True)

        self.split_sample_logs(sub_runs)

        # set the nominal wavelength from the nexus file
        runObj = self._event_wksp.run()
        if runObj.hasProperty('MonoSetting'):
            monosetting = MonoSetting.getFromIndex(runObj.getPropertyAsSingleValue('MonoSetting'))
        else:
            monosetting = MonoSetting.getFromRotation(runObj.getPropertyAsSingleValue('mrot'))
        self._hidra_workspace.set_wavelength(float(monosetting), calibrated=False)

        return self._hidra_workspace

    def save(self, projectfile):
        """
        Save workspace to Hidra project file
        """
        projectfile = os.path.abspath(projectfile)  # confirm absolute path to make logs more readable
        checkdatatypes.check_file_name(projectfile, check_exist=False, check_writable=True, is_dir=False,
                                       description='Converted Hidra project file')

        # remove file if it already exists
        if os.path.exists(projectfile):
            self._log.information('Projectfile "{}" exists, removing previous version'.format(projectfile))
            os.remove(projectfile)

        # save
        hydra_file = HidraProjectFile(projectfile, HidraProjectFileMode.OVERWRITE)

        # Set geometry
        hydra_file.write_instrument_geometry(HidraSetup(self._hidra_workspace.get_instrument_setup()))
        # save experimental data/detector counts
        self._hidra_workspace.save_experimental_data(hydra_file)
