import os
from mantid.simpleapi import Logger, GetIPTS
from mantid.api import FileFinder
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from mantidqt.utils.asynchronous import BlockingAsyncTaskWithCallback

from pyrs.core.pyrscore import PyRsCore
from pyrs.utilities import calibration_file_io
from pyrs.interface import gui_helper
from pyrs.utilities import checkdatatypes
from pyrs.dataobjects import HidraConstants
from pyrs.interface.manual_reduction.event_handler import EventHandler


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

    def load_hydra_file(self, project_file_name):
        """Load Hidra project file to the core

        Parameters
        ----------
        project_file_name

        Returns
        -------
        str
            project ID to refer

        """
        # Load data file
        project_name = os.path.basename(project_file_name).split('.')[0]
        try:
            self.parent._hydra_workspace = self.parent._core.load_hidra_project(project_file_name,
                                                                                project_name=project_name,
                                                                                load_detector_counts=True,
                                                                                load_diffraction=True)
        except (KeyError, RuntimeError, IOError) as load_err:
            self.parent._hydra_workspace = None
            pop_message(self.parent, 'Loading {} failed.\nTry to load diffraction only!'.format(project_file_name),
                        detailed_message='{}'.format(load_err),
                        message_type='error')

            # Load
            try:
                self.parent._hydra_workspace = self.parent._core.load_hidra_project(project_file_name,
                                                                                    project_name=project_name,
                                                                                    load_detector_counts=False,
                                                                                    load_diffraction=True)
            except (KeyError, RuntimeError, IOError) as load_err:
                self.parent._hydra_workspace = None
                pop_message(self.parent, 'Loading {} failed.\nNothing is loaded'.format(project_file_name),
                            detailed_message='{}'.format(load_err),
                            message_type='error')

            return

        # Set value for the loaded project
        self.parent._project_file_name = project_file_name
        self.parent._project_data_id = project_name

        # Fill sub runs to self.ui.comboBox_sub_runs
        self.parent._set_sub_runs()

        # Set to first sub run and plot
        self.parent.ui.comboBox_sub_runs.setCurrentIndex(0)

        # Fill in self.ui.frame_subRunInfoTable
        meta_data_array = self.parent._core.reduction_service.get_sample_logs_values(self.parent._project_data_id,
                                                                                     [HidraConstants.SUB_RUNS,
                                                                                      HidraConstants.TWO_THETA])
        self.parent.ui.rawDataTable.add_subruns_info(meta_data_array, clear_table=True)

        return self.parent._project_data_id

    def set_user_idf(self, idf_name):
        # set
        instrument = calibration_file_io.import_instrument_setup(idf_name)
        self._core.reduction_service.set_instrument(instrument)


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
        project_data_id = o_load.load_hydra_file(file_name)
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