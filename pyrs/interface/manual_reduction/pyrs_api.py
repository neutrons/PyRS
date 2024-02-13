import os
from mantid.simpleapi import Logger
import numpy as np
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

DEFAULT_MASK_DIRECTORY = '/HFIR/HB2B/shared/CALIBRATION/'
DEFAULT_CALIBRATION_DIRECTORY = DEFAULT_MASK_DIRECTORY


class ReductionController:
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
    def get_default_nexus_dir(ipts_number):
        """Get the default NeXus files' directory

        Parameters
        ----------
        ipts_number : int, None
            IPTS number

        Returns
        -------
        str
            directory

        """
        nexus_path = '/HFIR/HB2B'

        if ipts_number is not None:
            nexus_path = os.path.join(nexus_path, 'IPTS-{}/nexus/'.format(ipts_number))

        return nexus_path

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
        vec_2theta, vec_intensity, vec_variance = self._curr_hidra_ws.get_reduced_diffraction_data(
            sub_run=sub_run_number, mask_id=None)

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
                              vanadium_file=None, project_file_name=None):
        """Full workflow to reduce NeXus file

        Parameters
        ----------
        nexus
        output_dir
        progressbar
        instrument
        calibration
        mask
        vanadium_file : str or None
            Vanadium file (reduced project file with vanadium counts at sub run 1)
        project_file_name

        Returns
        -------

        """
        self._curr_hidra_ws = reduce_hidra_workflow(nexus, output_dir, progressbar, instrument,
                                                    calibration, mask, vanadium_file, project_file_name)

        self._hidra_ws_dict[self._curr_hidra_ws.name] = self._curr_hidra_ws

        return self._curr_hidra_ws


def reduce_hidra_workflow(nexus, output_dir, progressbar, instrument=None, calibration=None, mask=None,
                          vanadium_file=None, project_file_name=None):
    """Workflow of algorithms to reduce HB2B NeXus file to powder patterns

    Parameters
    ----------
    nexus
    output_dir
    progressbar
    instrument
    calibration : str
        calibration file name
    mask : str or None
        Mask file (so far, only Mantid XML file)
    vanadium_file : str or None
        Vanadium file (reduced project file with vanadium counts at sub run 1)
    project_file_name : str or None
        if str, then the output file name won't use the default

    Returns
    -------
    pyrs.core.workspaces.HidraWorkspace
        HiDRA workspace

    """
    # Init logger
    logger = Logger('reduce_HB2B')

    # Create project file (name) for default
    if project_file_name is None:
        project_file_name = os.path.basename(nexus).split('.')[0] + '.h5'
    project_file_name = os.path.join(output_dir, project_file_name)

    # Remove previous existing file
    if os.path.exists(project_file_name):
        # overwrite existing file
        if os.access(project_file_name, os.W_OK):
            # log information
            logger.information('Will overwrite existing projectfile {}'.format(project_file_name))
        else:
            # no permission
            raise RuntimeError('User does not have permission to overwrite existing HiDRA project file {}'
                               ''.format(project_file_name))
    else:
        # file does not exist so far
        base_dir = os.path.dirname(project_file_name)
        if not (os.path.exists(base_dir) and os.access(base_dir, os.W_OK)):
            raise RuntimeError('User specified HiDRA project file path {} either does not exist or '
                               'user does not have write access.'.format(base_dir))
    # END-IF-ELSE

    # Set progress bar
    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(0)

    # process the data
    converter = NeXusConvertingApp(nexus_file_name=nexus, mask_file_name=mask)
    hidra_ws = converter.convert()

    # Update
    if progressbar:
        progressbar.setValue(50)
    # add powder patterns

    # Calculate powder pattern
    logger.notice('Adding powder patterns to Hidra Workspace {}'.format(hidra_ws))

    # Initialize a reducer
    reducer = ReductionApp()
    # add workspace to reducer
    reducer.load_hidra_workspace(hidra_ws)
    # reduce
    reducer.reduce_data(instrument_file=instrument,
                        calibration_file=calibration,
                        mask=None,
                        van_file=vanadium_file,
                        sub_runs=list(hidra_ws.get_sub_runs()))

    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(95)

    # Save
    reducer.save_diffraction_data(project_file_name)

    if progressbar:
        progressbar.setVisible(True)
        progressbar.setValue(100)

    return hidra_ws
