"""
Convert HB2B NeXus file to Hidra project file for further reduction
"""
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from mantid.kernel import Logger
from mantid.simpleapi import mtd, DeleteWorkspace, LoadEventNexus, LoadMask, MaskDetectors
import numpy
import os
from pyrs.core import workspaces
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry, HidraSetup
from pyrs.dataobjects import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import checkdatatypes
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor

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
        pixel_size = 0.3 / 1024.0
        arm_length = 0.985
        instrument = AnglerCameraDetectorGeometry(1024, 1024, pixel_size, pixel_size, arm_length, False)
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
        # This is a quick fix: TODO will make a proper refactor in future
        if not use_mantid:
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

        if use_mantid:
            # Use Mantid algorithms to load and split sub runs
            # Load data file, split to sub runs and sample logs
            self._load_mask_event_nexus(extract_mask=True)
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

    def _create_sample_log_dict(self, wkspname):
        """Create dictioonary for sample log of a sub run

        Goal:
            1. set self._sample_log_dict[log_name] with log value (single or time-averaged)
            2. set self._sample_log_dict[HidraConstants.SUB_RUN_DURATION] with duration
        """
        SUBRUN_INDEX = 0
        NUM_SUBRUN = 1

        # this contains all of the sample logs
        runObj = mtd[wkspname].run()
        # loop through all available logs
        for log_name in runObj.keys():
            log_value, log_dtype = self._get_log_value_and_type(runObj, log_name)

            # if the entry for this log is not created, create it!
            if log_name not in self._sample_log_dict:
                self._sample_log_dict[log_name] = numpy.ndarray(shape=(NUM_SUBRUN, ),
                                                                dtype=log_dtype)
            self._sample_log_dict[log_name][SUBRUN_INDEX] = log_value
        # END-FOR

        # create a fictional log for duration
        if HidraConstants.SUB_RUN_DURATION not in self._sample_log_dict:
            self._sample_log_dict[HidraConstants.SUB_RUN_DURATION] = numpy.ndarray(shape=(NUM_SUBRUN, ),
                                                                                   dtype=float)
        self._sample_log_dict[HidraConstants.SUB_RUN_DURATION][SUBRUN_INDEX] \
            = runObj.getPropertyAsSingleValue('duration')

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

    def _split_sub_runs(self):
        """Performing event filtering according to sample log sub-runs

        DAS log may not be correct from the run start,

        Returns
        -------
        dict
            split workspaces: key = sub run number (integer), value = workspace name (string)

        """

        # determine the range of subruns being used
        scan_index = mtd[self._event_ws_name].run()[SUBRUN_LOGNAME].value

        if bool(scan_index.min() != scan_index.max()):
            raise RuntimeError('Multiple subruns is not supported')

        # nothing to filter so just use the histogram representation
        self._hydra_workspace.set_sub_runs(numpy.arange(1, 2))
        SUBRUN = 1  # there is only one

        # put the counts in the workspace
        self._set_counts(SUBRUN, self._event_ws_name)

        # put the sample logs together
        self._create_sample_log_dict(self._event_ws_name)
        DeleteWorkspace(Workspace=self._event_ws_name)
        # force the subrun number rather than just using it
        self._sample_log_dict['scan_index'][0] = SUBRUN


def time_average_value(run_obj, log_name):
    """Get time averaged value for TimeSeriesProperty or single value
    """
    # this is a workaround for some runs where there is more than one value
    # in the scan_index log. It takes the largest unique value
    if log_name == 'scan_index':
        log_property = run_obj.getProperty(log_name)
        log_value = log_property.value.max()
    else:
        # Get single value
        log_value = run_obj.getPropertyAsSingleValue(log_name)

    return log_value
