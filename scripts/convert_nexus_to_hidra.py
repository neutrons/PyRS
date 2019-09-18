#!/usr/bin/python
import sys
import os
import argparse
import numpy
from pyrs.utilities import rs_project_file
from pyrs.core import workspaces
from pyrs.utilities import checkdatatypes
from mantid.simpleapi import GenerateEventsFilter, LoadEventNexus, FilterEvents

"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""


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
            event_ws_i = self._sub_run_workspace_dict[sub_run]
            counts_i = event_ws_i.extractY()
            self._hydra_workspace.set_raw_counts(sub_run, counts_i)

            # sample logs
            for log_property in event_ws_i.getProperties():
                name = log_property.name
                log_value = self._calculate_log_mean_value(log_property)

                # if the entry for this log is not created, create it!
                if name not in sample_log_dict:
                    sample_log_dict[name] = numpy.ndarray(shape=(log_array_size, ),
                                                          dtype=get_type(log_value))
                # END-OF

                sample_log_dict[name][sub_run_index] = log_value
            # END-FOR

            sub_run_index += 1
        # END-FOR

        # Add the sample logs
        for log_name in sample_log_dict:
            self._hydra_workspace.set_sample_log(log_name, sample_log_dict[log_name])

        return

    def save(self, output_dir):
        """
        Save workspace to Hidra project file
        :param output_dir:
        :return:
        """
        checkdatatypes.check_file_name(output_dir, True, True, True, 'Output directory for converted Hidra project'
                                                                     'file')

        # Save
        out_file_name = os.path.join(output_dir, os.path.basename(self._nexus_name).split('.')[0] +
                                     '.hdf')
        hydra_file = rs_project_file.HydraProjectFile(out_file_name, rs_project_file.HydraProjectFileMode.OVERWRITE)
        self._hydra_workspace.save_experimental_data(hydra_file)

        return

    @staticmethod
    def _calculate_log_mean_value(log_property):
        """
        Calculate the mean value of the sample log "within" the sub run time range
        :param log_property: Mantid run property
        :return:
        """
        # Single value

        # Time series property

        sample_log_times = log_property.times
        sample_log_value = log_property.value

        # TODO - FIXME - #84+ - Make this correct!
        log_mean_value = sample_log_value.mean()

        return log_mean_value

    def _split_sub_runs(self):
        """
        Performing event filtering according to sample log sub-runs
        :return: dictionary: key = sub run number (integer), value = workspace name (string)
        """
        # Load data
        event_ws = LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)

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


def parse_inputs(argv):
    """
    Parse inputs
    :param argv:
    :return: dictionary or None
    """
    # Define parser
    parser = argparse.ArgumentParser(description='Convert HB2B data to Hidra Project File')
    parser.add_argument('nexus', metavar='-n', type=str, help='name of nexus file')
    parser.add_argument('output', metavar='-o', type=str, help='Path to output directory')

    # Parse
    args = parser.parse_args()

    print('Args: {}'.format(args))

    return args


def get_type(value):
    """
    Get the numpy dtype for the input value if it is not a numpy
    :param value: any value
    :return: string for numpy data type
    """
    if type(value) == int:
        dtype = 'int'
    elif type(value) == str:
        dtype = 'object'
    else:
        dtype = 'float'

    return dtype


def main(argv):
    """
    Main
    :param argv:
    :return:
    """
    input_dict = parse_inputs(argv)
    if input_dict is None:
        sys.exit(-1)

    try:
        nexus_file = input_dict['nexus']
        output_dir = input_dict['output']

        converter = NeXusConvertingApp(nexus_file)
        converter.convert()
        converter.save(output_dir)

    except KeyError as key_err:
        print ('Unable to convert NeXus to Hidra project due to {}'.format(key_err))

    return


if __name__ == '__main__':
    main(sys.argv)
