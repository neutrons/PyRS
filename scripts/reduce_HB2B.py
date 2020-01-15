#!/usr/bin/python
from mantid.simpleapi import Logger
import os
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry

# DEFAULT VALUES FOR DATA PROCESSING
DEFAULT_CALIBRATION = None
DEFAULT_INSTRUMENT = None
DEFAULT_MASK = None


def _nexus_to_subscans(nexusfile, projectfile, mask_file_name, save_project_file):
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
    hydra_ws = converter.convert(use_mantid=False)

    # set up instrument
    # initialize instrument: hard code!
    instrument = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)

    # save project file as an option
    if save_project_file:
        converter.save(projectfile, instrument)
    else:
        hydra_ws.set_instrument_geometry(instrument)

    return hydra_ws


def _create_powder_patterns(hidra_workspace, instrument, calibration, mask, subruns, project_file_name):
    logger.notice('Adding powder patterns to Hidra Workspace{}'.format(hidra_workspace))

    reducer = ReductionApp(bool(options.engine == 'mantid'))
    # reducer.load_project_file(projectfile)
    # load HidraWorkspace
    reducer.load_hidra_workspace(hidra_workspace)

    reducer.reduce_data(instrument_file=instrument,
                        calibration_file=calibration,
                        mask=mask,
                        sub_runs=subruns)

    reducer.save_diffraction_data(project_file_name)


def _view_raw(hidra_workspace, mask, subruns, engine):
    reducer = ReductionApp(bool(engine == 'mantid'))
    # reducer.load_project_file(projectfile)
    reducer.load_hidra_workspace(hidra_workspace)

    # interpret None to be first subrun
    if not subruns:
        subruns = [0]

    # TODO pylint points out that this is a non-existant function
    # plot raw detector counts without reduction but possibly with masking
    reducer.plot_detector_counts(sub_run=subruns[0], mask=mask)


def reduce_hidra_workflow(user_options):

    # split into sub runs fro NeXus file
    hidra_ws = _nexus_to_subscans(user_options.nexus, user_options.project,
                                  mask_file_name=user_options.mask,
                                  save_project_file=False)

    if user_options.viewraw:  # plot data
        _view_raw(hidra_ws, None, user_options.subruns, user_options.engine)
    else:  # add powder patterns
        _create_powder_patterns(hidra_ws, user_options.instrument, user_options.calibration,
                                None, user_options.subruns, user_options.project)
        logger.notice('Successful reduced {}'.format(user_options.nexus))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Script for auto-reducing HB2B')
    parser.add_argument('nexus', help='name of nexus file')
    parser.add_argument('outputdir', nargs='?', help='Path to output directory')
    parser.add_argument('--project',
                        help='Project file with subscans and powder patterns (default is based on'
                        ' nexus filename in <outputdir>)')
    parser.add_argument('--instrument', nargs='?', default=DEFAULT_INSTRUMENT,
                        help='instrument configuration file overriding embedded (arm, pixel number'
                        ' and size) (default=%(default)s)')
    parser.add_argument('--calibration', nargs='?', default=DEFAULT_CALIBRATION,
                        help='instrument geometry calibration file overriding embedded (default=%(default)s)')
    parser.add_argument('--mask', nargs='?', default=DEFAULT_MASK,
                        help='masking file (PyRS hdf5 format or Mantid XML format) or '
                             'mask name (default=%(default)s)')
    parser.add_argument('--engine', choices=['mantid', 'pyrs'], default='pyrs',
                        help='reduction engine (default=%(default)s)')
    parser.add_argument('--viewraw', action='store_true',
                        help='viewing raw data with an option to mask (NO reduction)')
    parser.add_argument('--subruns', default=list(), nargs='*', type=int,
                        help='something about subruns (default is all runs)')  # TODO

    options = parser.parse_args()

    # generate project name if not already determined
    if not options.project:
        if not options.outputdir:
            parser.error('Need to specify either output directory or project filename')
        options.project = os.path.basename(options.nexus).split('.')[0] + '.h5'
        options.project = os.path.join(options.outputdir, options.project)

    logger = Logger('reduce_HB2B')

    # process data
    reduce_hidra_workflow(options)
