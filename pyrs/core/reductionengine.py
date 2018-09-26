# Reduction engine including slicing
from pyrs.utilities import checkdatatypes
from pyrs.utilities import hb2b_utilities
from mantid.simpleapi import FilterEvents


class ReductionEngine(object):
    """ Main reduction engine that manage processing the raw neutron event NeXus
    """
    def __init__(self):
        """
        initialization
        """

        return

    def add_nexus_run(self, ipts_number, exp_number, run_number):
        """
        add a NeXus file to the project
        :param ipts_number:
        :param exp_number:
        :param run_number:
        :param file_name:
        :return:
        """
        nexus_file = hb2b_utilities.get_hb2b_raw_data(ipts_number, exp_number, run_number)

        self.add_nexus_file(ipts_number, exp_number, run_number, nexus_file)

        return

    def add_nexus_file(self, ipts_number, exp_number, run_number, nexus_file):
        """

        :param ipts_number:
        :param exp_number:
        :param run_number:
        :param nexus_file:
        :return:
        """
        if ipts_number is None or exp_number is None or run_number is None:
            # arbitrary single file
            self._single_file_manager.add_nexus(nexus_file)
        else:
            # well managed file
            self._archive_file_manager.add_nexus(ipts_number, exp_number, run_number, nexus_file)

        return