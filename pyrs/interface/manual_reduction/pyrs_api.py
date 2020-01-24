import os
from mantid.simpleapi import Logger, GetIPTS
from mantid.api import FileFinder
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from mantidqt.utils.asynchronous import BlockingAsyncTaskWithCallback
import numpy as np

from pyrs.core.pyrscore import PyRsCore
from pyrs.utilities import calibration_file_io
from pyrs.interface import gui_helper
from pyrs.utilities import checkdatatypes
from pyrs.dataobjects import HidraConstants

DEFAULT_MASK_DIRECTORY = '/HFIR/HB2B/shared/CALIBRATION/'
DEFAULT_CALIBRATION_DIRECTORY = DEFAULT_MASK_DIRECTORY


def _nexus_to_subscans(nexusfile, projectfile, mask_file_name, save_project_file, logger):
    """Split raw data from NeXus file to sub runs/scans
    Parameters
    ----------
    nexusfile : str
        HB2B event NeXus file's name
    projectfile : str
        Target HB2B HiDRA project file's name
    mask_file_name : str
        Mask file name; None for no mask
    save_project_file : str
        Project file to save to.  None for not being saved
    Returns
    -------
    pyrs.core.workspaces.HidraWorkspace
        Hidra workspace containing the raw counts and sample logs
    """

    if os.path.exists(projectfile):
        logger.information('Removing existing projectfile {}'.format(projectfile))
        os.remove(projectfile)

    logger.notice('Creating subscans from {} into project file {}'.format(nexusfile, projectfile))
    converter = NeXusConvertingApp(nexusfile, mask_file_name)
    hidra_ws = converter.convert()

    converter.save(projectfile)

    # save project file as an option
    if save_project_file:
        converter.save(projectfile)

    return hidra_ws


def _create_powder_patterns(hidra_workspace, instrument, calibration, mask, subruns, project_file_name, logger):
    logger.notice('Adding powder patterns to project file {}'.format(hidra_workspace))

    reducer = ReductionApp(False)
    reducer.load_hidra_workspace(hidra_workspace)

    # TODO - Need to add some debugging output for user to feel good
    reducer.reduce_data(instrument_file=instrument,
                        calibration_file=calibration,
                        mask=mask,
                        sub_runs=subruns)

    reducer.save_diffraction_data(project_file_name)


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
        ipts = GetIPTS(run_number)
        return ipts

    @staticmethod
    def get_nexus_file_by_run(run_number):
        nexus_file = FileFinder.findRuns('HB2B' + run_number)[0]
        return nexus_file

    @staticmethod
    def get_default_output_dir(run_number):
        try:
            ipts = ReductionController.get_ipts_from_run(run_number)
            project_dir = ipts + 'shared/manualreduce/'
        except RuntimeError:
            project_dir = None

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
        return self._curr_hidra_ws.get_sub_runs()

    def get_detector_counts(self, sub_run_number, output_matrix):
        """

        Parameters
        ----------
        sub_run_number
        output_matrix : bool
            True: output 2D, otherwise 1D

        Returns
        -------
        numpy.ndarray

        """
        # TODO - ASAP
        return np.ndarray((1024, 1024), dtype=int)

    def get_powder_pattern(self, sub_run_number):
        """

        Parameters
        ----------
        sub_run_number

        Returns
        -------

        """
        return np.arange(1000).astype(float) * 0.1,  np.arange(1000).astype(float) * 0.1

    def get_sample_log_value(self, log_name, sub_run_number):
        # TODO - ASAP
        return 69.5

    def get_sample_logs_values(self, sample_log_names):
        """Get sample logs' value

        Parameters
        ----------
        sample_log_names : ~list
            List of sample logs

        Returns
        -------
        numpy.ndarray

        """
        # TODO - ASAP
        return np.ndarray(shape=(2, 100))

    def load_nexus_file(self, nexus_name):
        # TODO - ASAP
        raise NotImplementedError('ASAP')

    def load_hidra_project(self, project_file_name, allow_no_counts):
        """Load Hidra project file to the core

        Parameters
        ----------
        project_file_name : str
            Hidra project file
        allow_no_counts : bool
            Flag ...

        Returns
        -------
        str
            project ID to refer

        """
        # Load data file
        project_name = os.path.basename(project_file_name).split('.')[0]
        try:
            self._curr_hidra_ws = self.parent._core.load_hidra_project(project_file_name,
                                                                       project_name=project_name,
                                                                       load_detector_counts=True,
                                                                       load_diffraction=True)
        except (KeyError, RuntimeError, IOError) as load_err:
            # Load
            try:
                self.parent._hydra_workspace = self.parent._core.load_hidra_project(project_file_name,
                                                                                    project_name=project_name,
                                                                                    load_detector_counts=False,
                                                                                    load_diffraction=True)
            except (KeyError, RuntimeError, IOError) as load_err:
                self.parent._hydra_workspace = None
                raise RuntimeError('Loading {} failed.\nNothing is loaded'.format(project_file_name))

            return

        # Set value for the loaded project
        self.parent._project_file_name = project_file_name
        self.parent._project_data_id = project_name

        # Fill sub runs to self.ui.comboBox_sub_runs
        self.parent._set_sub_runs()

        return self.parent._project_data_id

    def set_user_idf(self, idf_name):
        # set
        instrument = calibration_file_io.import_instrument_setup(idf_name)
        self._core.reduction_service.set_instrument(instrument)

    def save_project(self):
        output_project_name = os.path.join(self._output_dir, os.path.basename(self._project_file_name))
        if output_project_name != self._project_file_name:
            import shutil
            shutil.copyfile(self._project_file_name, output_project_name)

        self._core.reduction_service.save_project(self._project_data_id, output_project_name)


@staticmethod
def convert_to_project_file(nexus_filename):
    """
    Convert nexus_filename into a project file
    :param nexus_filename:
    """
    # TODO - Implement!

    return


@staticmethod
def load_project_file(parent, file_name):
    try:
        o_load = ReductionController(parent=parent)
        project_data_id = o_load.load_hidra_project(file_name)
    except RuntimeError as run_err:
        pop_message(parent,
                    'Failed to load project file {}: {}'.format(file_name, run_err),
                    None, 'error')
        project_data_id = None
    else:
        print('Loaded {} to {}'.format(file_name, project_data_id))

    return project_data_id


# TODO - Need to input a dictionary for HidraWorksapce generated
# TODO - Need to input the table to write the workspace result!
def reduce_hidra_workflow(nexus, outputdir, progressbar, subruns=list(), instrument=None, calibration=None, mask=None):

    project = os.path.basename(nexus).split('.')[0] + '.h5'
    project = os.path.join(outputdir, project)

    logger = Logger('reduce_HB2B')
    # process the data
    progressbar.setVisible(True)
    progressbar.setValue(0.)
    hidra_ws = _nexus_to_subscans(nexus, project, mask, False, logger)
    progressbar.setValue(50.)
    # add powder patterns
    _create_powder_patterns(hidra_ws, instrument, calibration,
                            None, subruns, project, logger)
    progressbar.setValue(100.)
    progressbar.setVisible(False)