#!/usr/bin/python
from pyrs.core.powder_pattern import ReductionApp


def _main(options):
    """
    main body
    """
    # call for reduction
    reducer = ReductionApp(bool(options.engine == 'mantid'))

    # Load Hidra project file
    reducer.load_project_file(options.project)

    # Process data
    if options.viewraw:
        # interpret None to be first subrun
        if not options.subruns:
            options.subruns = [0]
        # plot raw detector counts without reduction but possibly with masking
        reducer.plot_detector_counts(sub_run=options.subruns[0], mask=options.mask)
    else:
        # reduce
        reducer.reduce_data(instrument_file=options.instrument,
                            calibration_file=options.calibration,
                            mask=options.mask,
                            sub_runs=options.subruns)

        # save
        reducer.save_diffraction_data(options.project)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Script for auto-reducing HB2B')
    parser.add_argument('project', help='Project file, this will have the powder patterns appended to it')
    parser.add_argument('--instrument', nargs='?', default=None,
                        help='instrument configuration file overriding embedded (arm, pixel number and size)')
    parser.add_argument('--calibration', nargs='?', default=None,
                        help='instrument geometry calibration file overriding embedded')
    parser.add_argument('--mask', nargs='?', default=None,
                        help='masking file (PyRS hdf5 format) or mask name')
    parser.add_argument('--engine', choices=['mantid', 'pyrs'], default='pyrs',
                        help='reduction engine (default=%(default)s)')
    parser.add_argument('--viewraw', action='store_true',
                        help='viewing raw data with an option to mask (NO reduction)')
    parser.add_argument('--subruns', default=list(), nargs='*', type=int,
                        help='something about subruns (default is all runs)')  # TODO

    options = parser.parse_args()

    _main(options)
