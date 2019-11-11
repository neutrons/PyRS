#!/usr/bin/python
from mantid.simpleapi import Logger
import os
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

# DEFAULT VALUES FOR DATA PROCESSING
DEFAULT_CALIBRATION = None
DEFAULT_INSTRUMENT = None
DEFAULT_MASK = None


def _nexus_to_subscans(nexusfile, projectfile):
    if os.path.exists(projectfile):
        logger.information('Removing existing projectfile {}'.format(projectfile))
        os.remove(projectfile)

    logger.notice('Creating subscans from {} into project file {}'.format(nexusfile, projectfile))
    converter = NeXusConvertingApp(nexusfile)
    converter.convert()
    converter.save(projectfile)


def _create_powder_patterns(projectfile, instrument, calibration, mask, subruns):
    logger.notice('Adding powder patterns to project file {}'.format(projectfile))

    reducer = ReductionApp(bool(options.engine == 'mantid'))
    reducer.load_project_file(projectfile)

    reducer.reduce_data(instrument_file=instrument,
                        calibration_file=calibration,
                        mask=mask,
                        sub_runs=subruns)

    reducer.save_diffraction_data(options.project)


def _view_raw(projectfile, mask, subruns, engine):
    reducer = ReductionApp(bool(engine == 'mantid'))
    reducer.load_project_file(projectfile)

    # interpret None to be first subrun
    if not subruns:
        subruns = [0]

    # TODO pylint points out that this is a non-existant function
    # plot raw detector counts without reduction but possibly with masking
    reducer.plot_detector_counts(sub_run=subruns[0], mask=mask)


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
                        help='masking file (PyRS hdf5 format) or mask name (default=%(default)s)')
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

    # process the data
    _nexus_to_subscans(options.nexus, options.project)

    if options.viewraw:  # plot data
        _view_raw(options.projectfile, options.mask, options.subruns, options.engine)
    else:  # add powder patterns
        _create_powder_patterns(options.project, options.instrument, options.calibration,
                                options.mask, options.subruns)
        logger.notice('Successful reduced {}'.format(options.nexus))
