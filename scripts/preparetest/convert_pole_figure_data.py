#!/usr/bin/python
""""
Convert the "old" HB2B data, now used to pole figure, to new HydraProject format

*How to run*
1. Add PyRS path to python path (refer to pyrsdev.sh)
1. Run this script

"""
from pyrs.utilities import rs_scan_io
from pyrs.utilities import rs_project_file


def main():
    """ Main to convert the data file to standard Hidra project format
    :return:
    """
    # Initial set up
    # Source data: 7 for files for 7 detector bank
    source_data_set = [(1, 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5'),
                       (2, 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5'),
                       (3, 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'),
                       (4, 'tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5'),
                       (5, 'tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5'),
                       (6, 'tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5'),
                       (7, 'tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5')]

    # TODO FIXME - FUTURE - HIDRA may have a different workflow to generate the reduced data sets across the detector
    # TODO ... ...        - Thus, how to form the raw data is still UNDER DISCUSSION


    return


if __name__ == '__main__':
    main()
