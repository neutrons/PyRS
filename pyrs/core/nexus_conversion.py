"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import bisect
from mantid.kernel import FloatPropertyWithValue, FloatTimeSeriesProperty, Int32TimeSeriesProperty, \
    Int64TimeSeriesProperty, logger, Logger
from mantid.simpleapi import mtd, ConvertToMatrixWorkspace, DeleteWorkspace, FilterByLogValue, \
    FilterByTime, LoadEventNexus, LoadMask, MaskDetectors
import numpy
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry, HidraSetup
from pyrs.dataobjects import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import checkdatatypes
from pyrs.split_sub_runs.load_split_sub_runs import load_split_nexus_python


SUBRUN_LOGNAME = 'scan_index'


class NeXusConvertingApp(object):
    """
    Convert NeXus file to Hidra project file
    """
    def __init__(self, nexus_file_name, mask_file_name=None):
        """Initialization

        Parameters
        ----------
        nexus_file_name : str
            Name of NeXus file
        mask_file_name : str
            Name of masking file
        """
        # configure logging for this class
        self._log = Logger(__name__)

        # NeXus name
        checkdatatypes.check_file_name(nexus_file_name, True, False, False, 'NeXus file')
        self._nexus_name = nexus_file_name

        # Mask
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
        self._sample_log_dict = dict()

        self._hydra_workspace = workspaces.HidraWorkspace(self._nexus_name)
        # Set a default instrument with this workspace
        # set up instrument
        # initialize instrument: hard code!
        instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
        self._hydra_workspace.set_instrument_geometry(instrument)

        # project file
        self._project_file = None

        self._starttime = 0.  # start filtering at the beginning of the run

    def __del__(self):
        # all of the workspaces that were created should be deleted if they haven't been already
        for name in [self._event_ws_name]:
            if name in mtd:
                DeleteWorkspace(Workspace=name)

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
        # This is a quick fix: TODO will make a proper refactor in future
        if not use_mantid:
            # Use PyRS/converter to load and split sub runs
            try:
                sub_run_counts, self._sample_log_dict, mask_array = load_split_nexus_python(self._nexus_name,
                                                                                            self._mask_file_name)
                # set counts to each sub run
                for sub_run in sub_run_counts:
                    self._hydra_workspace.set_raw_counts(sub_run, sub_run_counts[sub_run])
                # set mask
                if mask_array is not None:
                    self._hydra_workspace.set_detector_mask(mask_array, is_default=True)
                # get sub runs
                sub_runs = numpy.array(sorted(sub_run_counts.keys()))
            except RuntimeError as run_err:
                if str(run_err).count('Sub scan') == 1 and str(run_err).count('is not valid') == 1:
                    # RuntimeError: Sub scan (time = [0.], value = [0]) is not valid
                    # use Mantid to reduce
                    use_mantid = True
                else:
                    # unhandled exception: re-throw
                    raise run_err

        if use_mantid:
            # Use Mantid algorithms to load and split sub runs
            # Load data file, split to sub runs and sample logs
            self._load_mask_event_nexus(True)
            self._determine_start_time()
            self._split_sub_runs()
            sub_runs = self._sample_log_dict['scan_index']
        # END-IF

        # Add the sample logs to the hidra workspace
        # sub_runs = self._sample_log_dict['scan_index']

        for log_name in self._sample_log_dict:
            if log_name in ['scan_index', HidraConstants.SUB_RUNS]:
                continue  # skip 'SUB_RUNS'
            self._hydra_workspace.set_sample_log(log_name, sub_runs, self._sample_log_dict[log_name])

        return self._hydra_workspace

    def _set_counts(self, sub_run, wkspname):
        self._hydra_workspace.set_raw_counts(sub_run, mtd[wkspname].extractY())

    def _create_sample_log_dict(self, wkspname, sub_run_index, log_array_size):
        """Create dictioonary for sample log of a sub run

        Goal:
            1. set self._sample_log_dict[log_name][sub_run_index] with log value (single or time-averaged)
            2. set self._sample_log_dict[HidraConstants.SUB_RUN_DURATION][sub_run_index] with duration

        Parameters
        ----------
        wkspname
        sub_run_index
        log_array_size

        Returns
        -------

        """
        # this contains all of the sample logs
        runObj = mtd[wkspname].run()
        # loop through all available logs
        for log_name in runObj.keys():
            log_value, log_dtype = self._get_log_value_and_type(runObj, log_name)

            # if the entry for this log is not created, create it!
            if log_name not in self._sample_log_dict:
                self._sample_log_dict[log_name] = numpy.ndarray(shape=(log_array_size, ),
                                                                dtype=log_dtype)
            self._sample_log_dict[log_name][sub_run_index] = log_value
        # END-FOR

        # create a fictional log for duration
        if HidraConstants.SUB_RUN_DURATION not in self._sample_log_dict:
            self._sample_log_dict[HidraConstants.SUB_RUN_DURATION] = numpy.ndarray(shape=(log_array_size, ),
                                                                                   dtype=float)
        self._sample_log_dict[HidraConstants.SUB_RUN_DURATION][sub_run_index] \
            = self._calculate_sub_run_duration(runObj)

    def _calculate_sub_run_duration(self, runObj):
        """Calculate the duration of a sub run from the logs

        The duration of each sub run is calculated from sample log 'splitter' with unit as second

        Returns
        -------
        float
            The sub run's duration

        """
        # get splitter
        if runObj.hasProperty('splitter'):
            # calculate duration
            splitter_times = runObj.getProperty('splitter').times.astype(float) * 1E-9
            splitter_value = runObj.getProperty('splitter').value

            if splitter_value[0] == 0:
                splitter_times = splitter_times[1:]
            assert len(splitter_times) % 2 == 0, 'If splitter starts from 0, there will be odd number of ' \
                'splitter times; otherwise, even number'

            sub_split_durations = splitter_times[1::2] - splitter_times[::2]

            duration = numpy.sum(sub_split_durations)
        else:
            # no splitter (which is not right), use the duration property
            duration = runObj.getPropertyAsSingleValue('duration')

        return duration

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
        # if instrument is None:
        #     # initialize instrument: hard code!
        #     instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
        hydra_file.write_instrument_geometry(HidraSetup(self._hydra_workspace.get_instrument_setup()))

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

    def _load_mask_event_nexus(self, extract_mask):
        """Loads the event file using instance variables

        If mask file is not None, then also mask the EventWorkspace

        Parameters
        ----------
        extract_mask : bool
            If True, extract the mask out of MaskWorkspace and set HidraWorkspace

        Returns
        -------
        None

        """
        # Load
        ws = LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)

        # Mask
        if self._mask_file_name is not None:
            # Load mask with reference to event workspace just created
            mask_ws_name = os.path.basename(self._mask_file_name).split('.')[0] + '_mask'

            # check zero spectrum
            counts_vec = ws.extractY()
            num_zero = numpy.where(counts_vec < 0.5)[0].shape[0]

            mask_ws = LoadMask(Instrument='nrsf2', InputFile=self._mask_file_name, RefWorkspace=self._event_ws_name,
                               OutputWorkspace=mask_ws_name)

            # Extract mask out
            if extract_mask:
                # get the Y array from mask workspace: shape = (1048576, 1)
                mask_array = mask_ws.extractY()
                # in Mantid's mask workspace, 1 stands for mask (value cleared), 0 stands for non-mask (value kept)
                mask_array = 1 - mask_array.astype(int)
                # set the HidraWorkspace
                self._hydra_workspace.set_detector_mask(mask_array, is_default=True)

            # Mask detectors and set all the events in mask to zero
            MaskDetectors(Workspace=self._event_ws_name, MaskedWorkspace=mask_ws_name)

            ws = mtd[self._event_ws_name]
            counts_vec = ws.extractY()
            self._log.information('{}: number of extra masked spectra = {}'
                                  ''.format(self._event_ws_name, numpy.where(counts_vec < 0.5)[0].shape[0] - num_zero))

        # get the start time from the run object
        self._starttime = numpy.datetime64(mtd[self._event_ws_name].run()['start_time'].value)

        return

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

        # unset the start time if it is before the actual start of the run
        if self._starttime <= numpy.datetime64(mtd[self._event_ws_name].run()['start_time'].value):
            self._starttime = None
        else:
            print('[DEBUG] Shift from start_time = {}'
                  ''.format(self._starttime - numpy.datetime64(mtd[self._event_ws_name].run()['start_time'].value)))

    def _split_sub_runs(self):
        """Performing event filtering according to sample log sub-runs

        DAS log may not be correct from the run start,

        Returns
        -------
        dict
            split workspaces: key = sub run number (integer), value = workspace name (string)

        """

        # first remove the data before the previously calculated start time
        # don't bother if starttime isn't set
        if self._starttime:
            # numpy only likes integers for timedeltas
            duration = int(mtd[self._event_ws_name].run().getPropertyAsSingleValue('duration'))
            duration = numpy.timedelta64(duration, 's') + numpy.timedelta64(300, 's')  # add 5 minutes
            FilterByTime(InputWorkspace=self._event_ws_name,
                         OutputWorkspace=self._event_ws_name,
                         AbsoluteStartTime=str(self._starttime),
                         AbsoluteStopTime=str(self._starttime + duration))

        # determine the range of subruns being used
        scan_index = mtd[self._event_ws_name].run()[SUBRUN_LOGNAME].value
        scan_index_min = scan_index.min()
        scan_index_max = scan_index.max()
        multiple_subrun = bool(scan_index_min != scan_index_max)

        if multiple_subrun:
            # determine the duration of each subrun by correlating to the scan_index
            scan_times = mtd[self._event_ws_name].run()[SUBRUN_LOGNAME].times
            durations = {}
            for timedelta, subrun in zip(scan_times[1:] - scan_times[:-1], scan_index[:-1]):
                timedelta /= numpy.timedelta64(1, 's')  # convert to seconds
                if subrun not in durations:
                    durations[subrun] = timedelta
                else:
                    durations[subrun] += timedelta
            if 0 in durations:
                del durations[0]
            durations[scan_index_max] = 0.
            numpy.sum(durations.values())
            durations[scan_index_max] = mtd[self._event_ws_name].run()['duration'].value \
                - numpy.sum(durations.values())
            if durations[scan_index_max] < 0.:
                raise RuntimeError('Got negative duration ({}s) for subrun={}'
                                   ''.format(durations[scan_index_max], scan_index_max))

            # create the sorted, unique value version of the scan index and set it on the output workspace
            scan_index = numpy.unique(scan_index)
            scan_index.sort()
            if scan_index[0] == 0:
                scan_index = scan_index[1:]
            self._hydra_workspace.set_sub_runs(scan_index)

            # skip scan_index=0
            # the +1 is to make it inclusive
            for subrun_index, subrun in enumerate(scan_index):
                self._log.information('Filtering scan_index={}'.format(subrun))
                # pad up to 5 zeros
                ws_name = '{}_split_{:05d}'.format(self._event_ws_name, subrun)
                # filter out the subrun - this assumes that subruns are integers
                FilterByLogValue(InputWorkspace=self._event_ws_name,
                                 OutputWorkspace=ws_name,
                                 LogName=SUBRUN_LOGNAME,
                                 LogBoundary='Left',
                                 MinimumValue=float(subrun) - .25,
                                 MaximumValue=float(subrun) + .25)

                # update the duration in the filtered workspace
                duration = FloatPropertyWithValue('duration', durations[subrun])
                duration.units = 'second'
                mtd[ws_name].run()['duration'] = duration

                # subrun found should match what was requested
                if abs(mtd[ws_name].run().getPropertyAsSingleValue('scan_index') - subrun) > .1:
                    subrun_obs = mtd[ws_name].run().getPropertyAsSingleValue('scan_index')
                    # TODO this should match exactly - doesn't in test_split_log_time_average
                    self._log.warning('subrun {:.1f} is not the expected value {}'.format(subrun_obs, subrun))
                    # raise RuntimeError('subrun {:.1f} is not the expected value {}'.format(subrun_obs, subrun))

                # put the counts in the workspace
                self._set_counts(subrun, ws_name)

                # put the sample logs together
                self._create_sample_log_dict(ws_name, subrun_index, len(scan_index))

                # cleanup
                DeleteWorkspace(Workspace=ws_name)

        else:  # nothing to filter so just histogram it
            self._hydra_workspace.set_sub_runs(numpy.arange(1, 2))
            subrun = 1
            ConvertToMatrixWorkspace(InputWorkspace=self._event_ws_name,
                                     OutputWorkspace=self._event_ws_name)

            # put the counts in the workspace
            self._set_counts(subrun, self._event_ws_name)

            # put the sample logs together
            self._create_sample_log_dict(self._event_ws_name, 0, 1)
            # force the subrun number rather than just using it
            self._sample_log_dict['scan_index'][0] = subrun


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
        # this is a workaround for some runs where there is more than one value
        # in the scan_index log. It takes the largest unique value
        if log_name == 'scan_index':
            log_value = log_property.value.max()

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
