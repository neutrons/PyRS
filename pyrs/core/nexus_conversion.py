"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from mantid.kernel import Logger
from mantid.simpleapi import mtd, DeleteWorkspace, LoadEventNexus, LoadMask
import numpy
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry, HidraSetup
from pyrs.dataobjects import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import checkdatatypes
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor

SUBRUN_LOGNAME = 'scan_index'
NUM_PIXEL_1D = 1024
PIXEL_SIZE = 0.3 / NUM_PIXEL_1D
ARM_LENGTH = 0.985

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
        # initialize instrument with hard coded values
        instrument = AnglerCameraDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D, PIXEL_SIZE, PIXEL_SIZE, ARM_LENGTH, False)
        self._hydra_workspace.set_instrument_geometry(instrument)

        # project file
        self._project_file = None

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
        if use_mantid:
            raise RuntimeError('use_mantid=True is no longer supported')

        # Use PyRS/converter to load and split sub runs
        processor = NexusProcessor(self._nexus_name, self._mask_file_name)

        # set counts to each sub run
        sub_run_counts = processor.split_events_sub_runs()
        for sub_run in sub_run_counts:
            self._hydra_workspace.set_raw_counts(sub_run, sub_run_counts[sub_run])

        # set the sample logs# get sub runs
        sub_runs = numpy.array(sorted(sub_run_counts.keys()))

        self._sample_log_dict = processor.split_sample_logs()

        # set mask
        if processor.mask_array is not None:
            self._hydra_workspace.set_detector_mask(processor.mask_array, is_default=True)
        del processor

        for log_name in self._sample_log_dict:
            if log_name in ['scan_index', HidraConstants.SUB_RUNS]:
                continue  # skip 'SUB_RUNS'
            self._hydra_workspace.set_sample_log(log_name, sub_runs, self._sample_log_dict[log_name])

        return self._hydra_workspace

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
        hydra_file.write_instrument_geometry(HidraSetup(self._hydra_workspace.get_instrument_setup()))
        # save experimental data/detector counts
        self._hydra_workspace.save_experimental_data(hydra_file)
