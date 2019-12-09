"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
from mantid.simpleapi import mtd, GenerateEventsFilter, LoadEventNexus, FilterEvents
from mantid.kernel import FloatTimeSeriesProperty, Int32TimeSeriesProperty, Int64TimeSeriesProperty
import numpy
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry, HidraSetup
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
        checkdatatypes.check_file_name(nexus_file_name, True, False, False, 'NeXus file')

        self._nexus_name = nexus_file_name

        # workspaces
        self._event_ws_name = os.path.basename(nexus_file_name).split('.')[0]
        self._sub_run_workspace_dict = dict()
        self._sample_log_dict = dict()

        self._hydra_workspace = workspaces.HidraWorkspace(self._nexus_name)

        # project file
        self._project_file = None

        return

    def convert(self, start_time):
        """Main method to convert NeXus file to HidraProject File by

        1. split the workspace to sub runs
        2. for each split workspace, aka a sub run, get the total counts for each spectrum and save to a 1D array

        Parameters
        ----------
        start_time : float
            User defined run start time relative to DAS recorded run start time in unit of second

        Returns
        -------

        """
        # Load data file, split to sub runs and sample logs
        self._sub_run_workspace_dict = self._split_sub_runs(start_time)

        # Get the sample log value
        sample_log_dict = dict()
        log_array_size = len(self._sub_run_workspace_dict.keys())

        # Construct the workspace
        sub_run_index = 0
        for sub_run in sorted(self._sub_run_workspace_dict.keys()):
            # counts
            event_ws_i = mtd[str(self._sub_run_workspace_dict[sub_run])]
            counts_i = event_ws_i.extractY()
            self._hydra_workspace.set_raw_counts(sub_run, counts_i)

            # sample logs
            runObj = event_ws_i.run()
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

        # Set sub runs to HidraWorkspace
        sub_runs = numpy.array(sorted(self._sub_run_workspace_dict.keys()))
        self._hydra_workspace.set_sub_runs(sub_runs)

        # Add the sample logs
        for log_name in sample_log_dict:
            if log_name == HidraConstants.SUB_RUNS:
                continue  # skip 'SUB_RUNS'
            self._hydra_workspace.set_sample_log(log_name, sub_runs, sample_log_dict[log_name])

    def save(self, projectfile, instrument=None):
        """
        Save workspace to Hidra project file
        """
        projectfile = os.path.abspath(projectfile)  # confirm absolute path to make logs more readable
        checkdatatypes.check_file_name(projectfile, check_exist=False, check_writable=True, is_dir=False,
                                       description='Converted Hidra project file')

        # remove file if it already exists
        if os.path.exists(projectfile):
            print('Projectfile "{}" exists, removing previous version'.format(projectfile))
            os.remove(projectfile)

        # save
        hydra_file = HidraProjectFile(projectfile, HidraProjectFileMode.OVERWRITE)

        # initialize instrument: hard code!
        if instrument is None:
            instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)

        # Set geometry
        hydra_file.write_instrument_geometry(HidraSetup(instrument))

        self._hydra_workspace.save_experimental_data(hydra_file)

    # @staticmethod
    def _get_log_value_and_type(self, runObj, name):
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

    def _split_sub_runs(self, relative_start_time=0):
        """Performing event filtering according to sample log sub-runs

        DAS log may not be correct from the run start,

        Parameters
        ----------
        relative_start_time : float or int
            Starting time from the run start time in unit of second

        Returns
        -------
        dict
            split workspaces: key = sub run number (integer), value = workspace name (string)

        """
        # Load data
        LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)

        # Generate splitters by sample log 'scan_index'.  real sub run starts with scan_index == 1
        split_ws_name = 'Splitter_{}'.format(self._nexus_name)
        split_info_name = 'InfoTable_{}'.format(self._nexus_name)

        # StartTime can be relative in unit of second
        # https://docs.mantidproject.org/nightly/algorithms/GenerateEventsFilter-v1.html
        GenerateEventsFilter(InputWorkspace=self._event_ws_name,
                             OutputWorkspace=split_ws_name,
                             InformationWorkspace=split_info_name,
                             LogName='scan_index',
                             StartTime='{}'.format(relative_start_time),
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
            print('Failed on sample log {}. Cause: {}'.format(log_name, run_err))
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
    num_periods = splitter_times.shape[0] / 2

    for iperiod in range(num_periods):
        # determine the start and stop time
        start_time = splitter_times[2 * iperiod + start_split_index]
        stop_time = splitter_times[2 * iperiod + start_split_index + 1]

        # print('Start/Stop Time: ', start_time, stop_time)

        # update the total time
        total_time += stop_time - start_time

        # get the (partial) interval from sample log
        # print('Log times: ', log_times)
        log_start_index = bisect.bisect(log_times, start_time)
        log_stop_index = bisect.bisect(log_times, stop_time)
        # print('Start/Stop index:', log_start_index, log_stop_index)

        if log_start_index == 0:
            print('Sample log time start time:    {}'.format(log_times[0]))
            print('Splitter star time:            {}'.format(start_time))
            raise RuntimeError('It is not expected that the run (splitter[0]) starts with no sample value recorded')

        # set the partial log time
        partial_time_size = log_stop_index - log_start_index + 2
        partial_log_times = numpy.ndarray((partial_time_size, ), dtype=float)
        partial_log_times[0] = start_time
        partial_log_times[-1] = stop_time
        # to 1 + time size - 2 = time size - 1
        partial_log_times[1:partial_time_size - 1] = log_times[log_start_index:log_stop_index]
        # print('Processed log times: ', partial_log_times)

        # set the partial value: v[0] is the value before log_start_index (to the right)
        # so it shall start from log_start_index - 1
        # the case with log_start_index == 0 is ruled out
        partial_log_value = log_value[log_start_index - 1:log_stop_index]
        # print('Partial log value:', partial_log_value)

        # Now for time average
        weighted_sum += numpy.sum(partial_log_value * (partial_log_times[1:] - partial_log_times[:-1]))
    # END-FOR

    time_averaged = weighted_sum / total_time

    return time_averaged
