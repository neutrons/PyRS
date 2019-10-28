"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""

from mantid.simpleapi import mtd, GenerateEventsFilter, LoadEventNexus, FilterEvents
import numpy
import os
from pyrs.core import workspaces
from pyrs.utilities import checkdatatypes
from pyrs.utilities import rs_project_file


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

    def convert(self):
        """
        Main method to convert NeXus file to HidraProject File by
        1. split the workspace to sub runs
        2. for each split workspace, aka a sub run, get the total counts for each spectrum and save to a 1D array
        :return:
        """
        # Load data file, split to sub runs and sample logs
        self._sub_run_workspace_dict = self._split_sub_runs()

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

        # Add the sample logs
        for log_name in sample_log_dict:
            self._hydra_workspace.set_sample_log(log_name, sample_log_dict[log_name])

    def save(self, projectfile):
        """
        Save workspace to Hidra project file
        """
        projectfile = os.path.abspath(projectfile)  # confirm absolute path to make logs more readable
        checkdatatypes.check_file_name(projectfile, check_exists=False, check_writable=True, is_dir=False,
                                       description='Converted Hidra project file')

        # remove file if it already exists
        if os.path.exists(projectfile):
            print('Projectfile "{}" exists, removing previous version'.format(projectfile))
            os.remove(projectfile)

        # save
        hydra_file = rs_project_file.HydraProjectFile(projectfile, rs_project_file.HydraProjectFileMode.OVERWRITE)
        self._hydra_workspace.save_experimental_data(hydra_file)

    @staticmethod
    def _get_log_value_and_type(runObj, name):
        """
        Calculate the mean value of the sample log "within" the sub run time range
        :param log_property: Mantid run property
        :return:
        """
        log_property = runObj.getProperty(name)
        log_dtype = log_property.dtype()
        try:
            # gets time average (if TimeSeriesProperty) or single value
            return runObj.getPropertyAsSingleValue(name), log_dtype
        except ValueError:
            # if the value is a string, just return it
            if isinstance(log_property.value, str):
                return log_property.value, log_dtype
            else:
                raise RuntimeError('Cannot convert "{}" to a single value'.format(name))

    def _split_sub_runs(self):
        """
        Performing event filtering according to sample log sub-runs
        :return: dictionary: key = sub run number (integer), value = workspace name (string)
        """
        # Load data
        LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)

        # Generate splitters by sample log 'scan_index'.  real sub run starts with scan_index == 1
        split_ws_name = 'Splitter_{}'.format(self._nexus_name)
        split_info_name = 'InfoTable_{}'.format(self._nexus_name)
        GenerateEventsFilter(InputWorkspace=self._event_ws_name,
                             OutputWorkspace=split_ws_name,
                             InformationWorkspace=split_info_name,
                             LogName='scan_index',
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
                if sub_run_number > 0:
                    sub_run_ws_dict[sub_run_number] = ws_name
            except ValueError:
                # sub runs not ends with integer: unsplit
                pass
        # END-FOR

        return sub_run_ws_dict
