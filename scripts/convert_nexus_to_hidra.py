#!/usr/bin/python
import sys
import os
from pyrs.core import reduction_manager
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util

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

        return

    def split_sub_runs(self):
        """
        Performing event filtering according to sample log sub-runs
        :return:
        """
        # Load data
        event_ws = LoadEventNexus(Filename=self._nexus_name, OutputWorkspace=self._event_ws_name)

        # Split
        GenerateEventFilters(InputWorkspace=self._event_ws_name, LogName='sub run',
                             OutputWorkspace='blabla', OutputInformationWorkspace='blablabla')

        split_returns = FilterEvents(InputWorkspace='')


        return


def parse_inputs(argv):
    """
    Parse inputs
    :param argv:
    :return: dictionary or None
    """

    return


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

    except KeyError as key_err:
        print ('Unable to convert NeXus to Hidra project due to {}'.format(key_err))


    return


if __name__ == '__main__':
    main(sys.argv)