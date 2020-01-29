from __future__ import (absolute_import, division, print_function)  # python3 compatibility
from contextlib import contextmanager
import os
from mantid.simpleapi import Logger, GetIPTS
from mantid.api import FileFinder
from mantid.kernel import ConfigService
import numpy as np
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

DEFAULT_MASK_DIRECTORY = '/HFIR/HB2B/shared/CALIBRATION/'
DEFAULT_CALIBRATION_DIRECTORY = DEFAULT_MASK_DIRECTORY


@contextmanager
def archive_search():
    DEFAULT_FACILITY = 'default.facility'
    DEFAULT_INSTRUMENT = 'default.instrument'
    SEARCH_ARCHIVE = 'datasearch.searcharchive'
    HFIR = 'HFIR'
    HB2B = 'HB2B'

    # get the old values
    config = ConfigService.Instance()
    old_config = {}
    for property in [DEFAULT_FACILITY, DEFAULT_INSTRUMENT, SEARCH_ARCHIVE]:
        old_config[property] = config[property]

    # don't update things that are already set correctly
    if config[DEFAULT_FACILITY] == HFIR:
        del old_config[DEFAULT_FACILITY]
    else:
        config[DEFAULT_FACILITY] = HFIR

    if config[DEFAULT_INSTRUMENT] == HB2B:
        del old_config[DEFAULT_INSTRUMENT]
    else:
        config[DEFAULT_INSTRUMENT] = HB2B

    if HFIR in config[SEARCH_ARCHIVE]:
        del old_config[SEARCH_ARCHIVE]
    else:
        config[SEARCH_ARCHIVE] = HFIR

    try:
        # give back context
        yield

    finally:
        # set properties back to original values
        for property in old_config.keys():
            config[property] = old_config[property]


class ReductionController(object):
    """Control the data objects for manual reduction

    """
    def __init__(self):
        """Initialization of data structures

        """
        # current HidraWorkspace used in reduction
        self._curr_hidra_ws = None
        # record of previously and currently processed HidraWorksapce
        self._hidra_ws_dict = dict()
        # Working directory
        self._working_dir = '/HFIR/HB2B/'

    @property
    def working_dir(self):
        return self._working_dir

    @staticmethod
    def get_default_calibration_dir():
        """Default calibration directory on analysis cluster

        Returns
        -------
        str
            directory path

        """
        return DEFAULT_CALIBRATION_DIRECTORY

    @staticmethod
    def get_default_mask_dir():
        """Get default directory for masks on analysis cluster

        Returns
        -------
        str
            directory path for maskings

        """
        return DEFAULT_MASK_DIRECTORY

    @staticmethod
    def get_ipts_from_run(run_number):
        """Get IPTS number from run number

        Parameters
        ----------
        run_number : int
            run number

        Returns
        -------
        str or None
            IPTS path: example '/HFIR/HB2B/IPTS-22731/', None for not supported IPTS

        """
        # try with GetIPTS
        try:
            with archive_search():
                ipts = GetIPTS(RunNumber=run_number, Instrument='HB2B')
            return ipts
        except RuntimeError as e:
            print(e)
            return None  # indicate it wasn't found

    @staticmethod
    def get_nexus_file_by_run(run_number):
        """Get the Nexus path iin SNS data archive by run number

        Parameters
        ----------
        run_number : int or str
            run number

        Returns
        -------
        str or None
            file path to NeXus

        """
        # Find run: successful return is a size-one str array
        try:
            with archive_search():
                nexus_file = FileFinder.findRuns('HB2B{}'.format(run_number))[0]
            return nexus_file
        except RuntimeError as e:
            print(e)
            return None

    @staticmethod
    def get_default_output_dir(run_number):
        """Get default output directory for run number

        Exception: RuntimeError

        Parameters
        ----------
        run_number

        Returns
        -------

        """
        project_dir = None

        ipts_dir = ReductionController.get_ipts_from_run(run_number)
        if ipts_dir is not None:
            project_dir = os.path.join(ipts_dir, 'shared', 'manualreduce')

        return project_dir

    @staticmethod
    def get_nexus_dir(ipts_number):
        """Get NeXus directory

        Parameters
        ----------
        ipts_number : int
            IPTS number

        Returns
        -------
        str
            path to Nexus files

        """
        return '/HFIR/HB2B/IPTS-{}/nexus'.format(ipts_number)

    @staticmethod
    def get_hidra_project_dir(ipts_number, is_auto):
        """Get NeXus directory

        Parameters
        ----------
        ipts_number : int
            IPTS number
        is_auto : bool
            Flag for auto reduced data or manual reduced

        Returns
        -------
        str
            path to Nexus files

        """
        if is_auto:
            local_dir = 'autoreduce'
        else:
            local_dir = 'manualreduce'
        return '/HFIR/HB2B/IPTS-{}/shared/{}'.format(ipts_number, local_dir)

    def get_sub_runs(self):
        """Get sub runs of the current loaded HidraWorkspace

        Returns
        -------
        numpy.ndarray
            1D array for sorted sub runs

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        return self._curr_hidra_ws.get_sub_runs()

    def get_detector_counts(self, sub_run_number, output_matrix):
        """Get detector counts

        Exception: RuntimeError
            1. self._curr_hidra_ws does not exist
            2. sub run does not exist

        Parameters
        ----------
        sub_run_number : int
            sub run number
        output_matrix : bool
            True: output 2D, otherwise 1D

        Returns
        -------
        numpy.ndarray
            detector counts in 1D or 2D array

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        # Get detector counts from HidraWorkspace.  Possibly raise a RuntimeError from called method
        det_counts_array = self._curr_hidra_ws.get_detector_counts(sub_run_number)

        # Convert to 2D array for plotting as an option
        if output_matrix:
            # sanity check for array size
            counts_size = det_counts_array.shape[0]
            linear_size = int(np.sqrt(counts_size))
            assert linear_size == 1024

            # convert
            det_counts_array = det_counts_array.reshape((linear_size, linear_size))
        # END-IF

        return det_counts_array

    def get_powder_pattern(self, sub_run_number):
        """Retrieve powder pattern from current HidraWorkspace

        Exception: RuntimeError
            1. self._curr_hidra_ws does not exist
            2. sub run does not exist

        Parameters
        ----------
        sub_run_number : int
            sub run number

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            vector of 2theta and intensity

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        # Get powder pattern
        vec_2theta, vec_intensity = self._curr_hidra_ws.get_reduced_diffraction_data(sub_run=sub_run_number,
                                                                                     mask_id=None)

        return vec_2theta, vec_intensity

    def get_sample_log_value(self, log_name, sub_run_number):
        """Get sample log value

        Exception: RuntimeError
            1. self._curr_hidra_ws does not exist
            2. sub run does not exist

        Parameters
        ----------
        log_name : str
            log name
        sub_run_number : int
            sub run number

        Returns
        -------
        object
            sample log value

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        return self._curr_hidra_ws.get_sample_log_value(log_name, sub_run_number)

    def get_sample_logs_values(self, sample_log_names):
        """Get sample logs' values

        Note: as the heterogeneous type of sample logs, a dictionary as the return type is more
         convenient than numpy.ndarray

        Parameters
        ----------
        sample_log_names : ~list
            List of sample logs

        Returns
        -------
        ~dict
            sample log values in format of dictionary of numpy.ndarray

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        # Create a dictionary for sample logs
        logs_value_dict = dict()

        for log_name in sample_log_names:
            log_value_array = self._curr_hidra_ws.get_sample_log_values(log_name)
            logs_value_dict[log_name] = log_value_array
        # END-FOR

        return logs_value_dict

    def load_nexus_file(self, nexus_name):
        # TODO - ASAP - Need use case to decide functionality
        raise NotImplementedError('ASAP')

    def load_project_file(self, file_name, load_counts, load_pattern):
        from pyrs.core.pyrscore import PyRsCore
        core = PyRsCore()
        project_name = os.path.basename(file_name).split('.')[0]
        try:
            self._curr_hidra_ws = core.load_hidra_project(file_name, project_name, load_counts, load_pattern)
        except RuntimeError as run_err:
            raise RuntimeError('Failed to load project file {}: {}'.format(file_name, run_err))

        return

    def save_project(self):
        """Save HidraWorkspace to project file

        Exception: RuntimeError if
            (1) no curr_hidra_ws
            (2) curr_hidra_ws does not have file name associated

        Returns
        -------

        """
        if self._curr_hidra_ws is None:
            raise RuntimeError('No HidraWorkspace is created or loaded')

        project_file_name = self._curr_hidra_ws.hidra_project_file
        if project_file_name is None:
            raise RuntimeError('HiDRA workspace {} is not associated with any project file'
                               ''.format(self._curr_hidra_ws.name))

        # TODO - Need to find out the scenario!
        raise NotImplementedError('Need use cases!')

    def reduce_hidra_workflow(self, nexus, output_dir, progressbar, instrument=None, calibration=None, mask=None,
                              project_file_name=None):
        """Full workflow to reduce NeXus file

        Parameters
        ----------
        nexus
        output_dir
        progressbar
        instrument
        calibration
        mask
        project_file_name

        Returns
        -------

        """
        self._curr_hidra_ws = reduce_hidra_workflow(nexus, output_dir, progressbar, instrument,
                                                    calibration, mask, project_file_name)

        self._hidra_ws_dict[self._curr_hidra_ws.name] = self._curr_hidra_ws

        return self._curr_hidra_ws


def reduce_hidra_workflow(nexus, output_dir, progressbar, instrument=None, calibration=None, mask=None,
                          project_file_name=None):
    """Workflow of algorithms to reduce HB2B NeXus file to powder patterns

    Parameters
    ----------
    nexus
    output_dir
    progressbar
    instrument
    calibration : str
        calibration file name
    mask
    project_file_name : str or None
        if str, then the output file name won't use the default

    Returns
    -------
    pyrs.core.workspaces.HidraWorkspace
        HiDRA workspace

    """
    logger = Logger('reduce_HB2B')

    # Create project file (name) for default
    if project_file_name is None:
        project_file_name = os.path.basename(nexus).split('.')[0] + '.h5'
    project_file_name = os.path.join(output_dir, project_file_name)

    # Remove previous existing file
    if os.path.exists(project_file_name):
        logger.information('Will overwrite existing projectfile {}'.format(project_file_name))

    # Init logger
    logger = Logger('reduce_HB2B')

    # Set progress bar
    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(0.)

    # process the data
    converter = NeXusConvertingApp(nexus, mask)
    hidra_ws = converter.convert(use_mantid=False)

    # Update
    if progressbar:
        progressbar.setValue(50.)
    # add powder patterns

    # Calculate powder pattern
    logger.notice('Adding powder patterns to Hidra Workspace {}'.format(hidra_ws))

    # Initialize a reducer
    reducer = ReductionApp(False)
    # add workspace to reducer
    reducer.load_hidra_workspace(hidra_ws)
    # reduce
    reducer.reduce_data(instrument_file=instrument,
                        calibration_file=calibration,
                        mask=None,
                        sub_runs=list(hidra_ws.get_sub_runs()))

    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(95.)

    # Save
    reducer.save_diffraction_data(project_file_name)

    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(100.)

    return hidra_ws
