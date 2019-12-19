"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from mantid.simpleapi import mtd, GenerateEventsFilter, LoadEventNexus, FilterEvents
from mantid.kernel import FloatTimeSeriesProperty, Int32TimeSeriesProperty, Int64TimeSeriesProperty, logger, Logger
import numpy
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import HidraSetup, AnglerCameraDetectorGeometry
from pyrs.utilities import checkdatatypes
from pyrs.utilities.rs_project_file import HidraConstants, HidraProjectFile, HidraProjectFileMode
import bisect


class NeXusConvertingApp(object):
    """
    Convert NeXus file to Hidra project file
    """
    def __init__(self, nexus_file_name):
        """ Initialization
        :param nexus_file_name:
        """
        # configure logging for this class
        self._log = Logger(__name__)

        checkdatatypes.check_file_name(nexus_file_name, True, False, False, 'NeXus file')

        self._nexus_name = nexus_file_name

        # workspaces
        self._event_ws_name = os.path.basename(nexus_file_name).split('.')[0]
        self._sub_run_workspace_dict = dict()
        self._sample_log_dict = dict()

        self._hydra_workspace = workspaces.HidraWorkspace(self._nexus_name)

        # project file
        self._project_file = None

        self._starttime = 0.  # start filtering at the beginning of the run

    def convert(self):
        """Main method to convert NeXus file to HidraProject File by

        1. split the workspace to sub runs
        2. for each split workspace, aka a sub run, get the total counts for each spectrum and save to a 1D array

        Parameters
        ----------
        start_time : float
            User defined run start time relative to DAS recorded run start time in unit of second

        Returns
        -------
        pyrs.core.workspaces.HidraWorkspace
            HidraWorkspace for converted data

        """
        # Load data file, split to sub runs and sample logs
        self._load_event_nexus()
        self._determine_start_time()
        self._sub_run_workspace_dict = self._split_sub_runs()

        self._set_counts()

        # Set sub runs to HidraWorkspace
        sub_runs = numpy.array(sorted(self._sub_run_workspace_dict.keys()))
        self._hydra_workspace.set_sub_runs(sub_runs)

        # Add the sample logs to the workspace
        sample_log_dict = self._create_sample_log_dict()

        for log_name in sample_log_dict:
            if log_name == HidraConstants.SUB_RUNS:
                continue  # skip 'SUB_RUNS'
            self._hydra_workspace.set_sample_log(log_name, sub_runs, sample_log_dict[log_name])

        return self._hydra_workspace

    def _set_counts(self):
        for sub_run, wkspname in self._sub_run_workspace_dict.items():
            self._hydra_workspace.set_raw_counts(sub_run, mtd[wkspname].extractY())

    def _create_sample_log_dict(self):
        # Get the sample log value
        sample_log_dict = dict()
        log_array_size = len(self._sub_run_workspace_dict.keys())

        # Construct the workspace
        sub_run_index = 0
        for sub_run in sorted(self._sub_run_workspace_dict.keys()):
            # this contains all of the sample logs
            runObj = mtd[str(self._sub_run_workspace_dict[sub_run])].run()
            # loop through all available logs
            for log_name in runObj.keys():
                log_value, log_dtype = self._get_log_value_and_type(runObj, log_name)

                # if the entry for this log is not created, create it!
                if log_name not in sample_log_dict:
                    sample_log_dict[log_name] = numpy.ndarray(shape=(log_array_size, ),
                                                              dtype=log_dtype)

                sample_log_dict[log_name][sub_run_index] = log_value
            # END-FOR

            sub_run_index += 1
        # END-FOR

        # create a fictional log for duration
        try:
            sample_log_dict[HidraConstants.SUB_RUN_DURATION] = self._calculate_sub_run_duration()
        except RuntimeError as run_err:
            self._log.error('Unable to calculate duration for sub runs: {}'.format(run_err))

        return sample_log_dict

    def _calculate_sub_run_duration(self):
        """Calculate the duration of each sub run

        The duration of each sub run is calculated from sample log 'splitter' with unit as second

        Exception: RuntimeError if there is no splitter

        Returns
        -------
        numpy.ndarray
            a vector of float as sub run's duration.  They are ordered by sub run number increasing monotonically

        """
        # Get sub runs and init returned value (array)
        sub_runs = sorted(self._sub_run_workspace_dict.keys())
        duration_vec = numpy.zeros(shape=(len(sub_runs),), dtype=float)

        sub_run_index = 0
        for sub_run in sorted(self._sub_run_workspace_dict.keys()):
            # get event workspace of sub run and then run object
            event_ws_i = mtd[str(self._sub_run_workspace_dict[sub_run])]

            # sample logs
            run_i = event_ws_i.run()

            # get splitter
            if not run_i.hasProperty('splitter'):
                # no splitter (which is not right), use NAN
                raise RuntimeError('sub run {} does not have splitter'.format(sub_run))
            else:
                # calculate duration
                splitter_times = run_i.getProperty('splitter').times.astype(float) * 1E-9
                splitter_value = run_i.getProperty('splitter').value

                if splitter_value[0] == 0:
                    splitter_times = splitter_times[1:]
                assert len(splitter_times) % 2 == 0, 'If splitter starts from 0, there will be odd number of ' \
                                                     'splitter times; otherwise, even number'

                sub_split_durations = splitter_times[1::2] - splitter_times[::2]

                duration_vec[sub_run_index] = numpy.sum(sub_split_durations)
            # END-FOR

            # Update
            sub_run_index += 1
        # END-FOR

        return duration_vec

    def save(self, projectfile, instrument):
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
        if instrument is None:
            # initialize instrument: hard code!
            instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
        hydra_file.write_instrument_geometry(HidraSetup(instrument))

        self._hydra_workspace.save_experimental_data(hydra_file)

    @staticmethod
    def _get_log_value_and_type(runObj, name):
        """
        Calculate the mean value of the sample log "within" the sub run time range
        :param name: Mantid run property's name
        :return:
        """
        log_property = runObj.getProperty(name)
        log_dtype = log_property.dtype()
        try:
            log_value = time_average_value(runObj, name)
            # return runObj.getPropertyAsSingleValue(name), log_dtype
            return log_value, log_dtype
        except ValueError:
            # if the value is a string, just return it
            if isinstance(log_property.value, str):
                return log_property.value, log_dtype
            elif isinstance(log_property.value, list):
                return log_property.value[0], log_dtype
            else:
                raise RuntimeError('Cannot convert "{}" to a single value'.format(name))

    def _load_event_nexus(self):
        '''Loads the event file using instance variables'''
        LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)
        # get the start time from the run object
        self._starttime = mtd[self._event_ws_name].run()['start_time'].value

    def _determine_start_time(self, abs_tolerance=0.05):
        '''This goes through a subset of logs and compares when they actually
        get to their specified setpoint, updating the start time for
        event filtering. When this is done ``self._starttime`` will have been updated.

        Parameters
        ----------
        abs_tolerance: float
            When then log is within this absolute tolerance of the setpoint, it is correct
        '''
        # static view of the run object
        runObj = mtd[self._event_ws_name].run()

        # loop through the 'special' logs
        for logname in ['sx', 'sy', 'sz', '2theta', 'omega', 'chi', 'phi']:
            if logname not in runObj:
                continue  # log doesn't exist - not a good one to look at
            if logname + 'Setpoint' not in runObj:
                continue  # log doesn't have a setpoint - not a good one to look at

            # get the observed values of the log
            observed = runObj[logname].value
            if len(observed) <= 1 or observed.std() <= .5 * abs_tolerance:
                continue  # don't bother if the log is constant within half of the tolerance

            # look for the setpoint and find when the log first got there
            setPoint = runObj[logname + 'Setpoint'].value[0]  # only look at first setpoint
            for i, value in enumerate(observed):
                if abs(value - setPoint) < abs_tolerance:
                    # pick the larger of what was found and the previous largest value
                    self._starttime = max(runObj[logname].times[i], self._starttime)
                    break

    def _split_sub_runs(self):
        """Performing event filtering according to sample log sub-runs

        DAS log may not be correct from the run start,

        Returns
        -------
        dict
            split workspaces: key = sub run number (integer), value = workspace name (string)

        """
        # Load data

        # Generate splitters by sample log 'scan_index'.  real sub run starts with scan_index == 1
        split_ws_name = 'Splitter_{}'.format(self._nexus_name)
        split_info_name = 'InfoTable_{}'.format(self._nexus_name)

        # StartTime can be relative in unit of second
        # https://docs.mantidproject.org/nightly/algorithms/GenerateEventsFilter-v1.html
        GenerateEventsFilter(InputWorkspace=self._event_ws_name,
                             OutputWorkspace=split_ws_name,
                             InformationWorkspace=split_info_name,
                             LogName='scan_index',
                             StartTime=str(self._starttime),
                             UnitOfTime='Seconds',
                             MinimumLogValue=0,
                             LogValueInterval=1)

        # Split
        base_out_name = self._event_ws_name + '_split'
        split_returns = FilterEvents(InputWorkspace=self._event_ws_name,
                                     SplitterWorkspace=split_ws_name,
                                     InformationWorkspace=split_info_name,
                                     OutputWorkspaceBaseName=base_out_name,
                                     DescriptiveOutputNames=False,  # requires split workspace ends with sub run
                                     OutputWorkspaceIndexedFrom1=False,  # as workspace 0 is kept for what left between
                                                                         # 2 sub runs
                                     GroupWorkspaces=True)

        # Fill in
        output_ws_names = split_returns.OutputWorkspaceNames
        sub_run_ws_dict = dict()   # [sub run number] = workspace name
        for ws_name in output_ws_names:
            try:
                sub_run_number = int(ws_name.split('_')[-1])
                if sub_run_number > 0 and len(output_ws_names) > 2:
                    sub_run_ws_dict[sub_run_number] = ws_name
                elif sub_run_number == 0:
                    sub_run_ws_dict[1] = ws_name
            except ValueError:
                # sub runs not ends with integer: unsplit
                pass
        # END-FOR

        return sub_run_ws_dict


def time_average_value(run_obj, log_name):
    """Get time averaged value for TimeSeriesProperty or single value

    Parameters
    ----------
    run_obj
    log_name

    Returns
    -------

    """
    # Get property
    log_property = run_obj.getProperty(log_name)

    has_splitter_log = run_obj.hasProperty('splitter')
    if has_splitter_log:
        splitter_times = run_obj.getProperty('splitter').times
        splitter_value = run_obj.getProperty('splitter').value
    else:
        splitter_times = splitter_value = None
    if has_splitter_log and isinstance(log_property,
                                       (Int32TimeSeriesProperty, Int64TimeSeriesProperty, FloatTimeSeriesProperty)):
        # Integer or float time series property and this is a split workspace
        try:
            log_value = calculate_log_time_average(log_property.times, log_property.value,
                                                   splitter_times, splitter_value)
        except RuntimeError as run_err:
            # Sample log may not meet requirement
            # TODO - log the error!
            logger.warning('Failed on sample log {}. Cause: {}'.format(log_name, run_err))
            # use current Mantid method instead
            log_value = run_obj.getPropertyAsSingleValue(log_name)

    else:
        # No a split workspace
        # If a split workspace: string, boolean and others won't have time average issue
        # Get single value
        log_value = run_obj.getPropertyAsSingleValue(log_name)

    return log_value


def calculate_log_time_average(log_times, log_value, splitter_times, splitter_value):
    """Calculate time average for sample log of split

    Parameters
    ----------
    log_times : ~numpy.ndarray
        sample log series time.  Allowed value are float (nanosecond) or numpy.datetime64
    log_value : numpy.ndarray
        sample log value series
    splitter_times : ndarray
        numpy array for splitter time
    splitter_value : ndarray
        numpy array for splitter value wit alternative 0 and 1.  1 stands for the period of events included

    Returns
    -------
    Float

    """
    # Determine T0 (starting of the time period)
    start_split_index = 0 if splitter_value[0] == 1 else 1

    # Convert time to float and in unit of second (this may not be necessary but good for test)
    splitter_times = splitter_times.astype(float) * 1E-9
    time_start = splitter_times[start_split_index]

    # convert splitter time to relative to time_start in unit of second
    splitter_times -= time_start

    # convert log time to relative to time_start in unit of second
    log_times = log_times.astype(float) * 1E-9
    log_times -= time_start

    # Calculate time average
    total_time = 0.
    weighted_sum = 0.
    num_periods = int(.5 * splitter_times.shape[0])

    for iperiod in range(num_periods):
        # determine the start and stop time
        start_time = splitter_times[2 * iperiod + start_split_index]
        stop_time = splitter_times[2 * iperiod + start_split_index + 1]

        logger.debug('Start/Stop Time: {} {}'.format(start_time, stop_time))

        # update the total time
        total_time += stop_time - start_time

        # get the (partial) interval from sample log
        # logger.debug('Log times: {}'.format(log_times))
        log_start_index = bisect.bisect(log_times, start_time)
        log_stop_index = bisect.bisect(log_times, stop_time)
        logger.debug('Start/Stop index: {} {}'.format(log_start_index, log_stop_index))

        if log_start_index == 0:
            logger.information('Sample log time start time:    {}'.format(log_times[0]))
            logger.information('Splitter star time:            {}'.format(start_time))
            raise RuntimeError('It is not expected that the run (splitter[0]) starts with no sample value recorded')

        # set the partial log time
        partial_time_size = log_stop_index - log_start_index + 2
        partial_log_times = numpy.ndarray((partial_time_size, ), dtype=float)
        partial_log_times[0] = start_time
        partial_log_times[-1] = stop_time
        # to 1 + time size - 2 = time size - 1
        partial_log_times[1:partial_time_size - 1] = log_times[log_start_index:log_stop_index]
        # logger.debug('Processed log times: {}'.format(partial_log_times))

        # set the partial value: v[0] is the value before log_start_index (to the right)
        # so it shall start from log_start_index - 1
        # the case with log_start_index == 0 is ruled out
        partial_log_value = log_value[log_start_index - 1:log_stop_index]
        # logger.debug('Partial log value: {}'.format(partial_log_value))

        # Now for time average
        weighted_sum += numpy.sum(partial_log_value * (partial_log_times[1:] - partial_log_times[:-1]))
    # END-FOR

    time_averaged = weighted_sum / total_time

    return time_averaged
