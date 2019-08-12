#!/usr/bin/python
import h5py
import sys
from pyrs.utilities import file_util
from pyrs.core import mantid_helper
from pyrs.utilities import rs_scan_io

# TODO - TONIGHT 0 - Make it work!
# TODO - TONIGHT 0 - Add it to script pysr_command
# This is a simple script to convert
#  (1) SPICE .bin
#  (2) Rotated TIFF from XRay
#  (3) Raw TIFF from Xray
# to a standard HDF5 format to simulate the NeXus file


def parse_xray_tiff(tiff_file_name, pixel_size=2048):
    """
    Parse XRay TIFF (image) data into memory
    :param tiff_file_name:
    :return: (1) 1D array (column major, from lower left qcorner) (2) (num_row, num_cols)
    """
    # is it rotated?
    if tiff_file_name.lower().count('rotate') == 0:
        # raw file
        counts_matrix = file_util.import_raw_tiff(tiff_file_name)
        # consider: load_data_from_tif(raw_tiff_name=tiff_file_name, pixel_size=2048, rotate=False)
    else:
        # rotated
        counts_matrix = file_util.load_gray_scale_tif(raw_tiff_name=tiff_file_name, pixel_size=pixel_size, rotate=True)

    counts_array = counts_matrix.flatten()

    return counts_array, counts_matrix.shape


def parse_spice_binary(bin_file_name):
    """
    Parse SPICE binary file
    :param bin_file_name:
    :return: (1) 1D array (column major, from lower left corner) (2) (num_row, num_cols)
    """
    bin_ws_name, pixel_sizes = file_util.import_spice_bin(bin_file_name, load_instrument=False)

    counts_array = mantid_helper.get_data_y(bin_ws_name, transpose=1)

    return counts_array, pixel_sizes


def save_to_h5(counts_array, detector_shape, two_theta, out_h5_name):
    """
    write the detector counts to an HDF5
    :param counts_array: 1d array as counts
    :param detector_shape: 2-tuple: num_rows, num_cols
    :param two_theta: 2-theta value
    :param out_h5_name:
    :return:
    """
    try:
        writer = rs_scan_io.DiffractionDataFile()
        writer.set_2theta(two_theta)
        writer.set_counts(counts_array, detector_shape)
        writer.save_rs_file(out_h5_name)
    except RuntimeError as run_err:
        print ('[ERROR] Failed to save to HDF5: {}'.format(run_err))
        return False

    return True


def main(argv):
    """
    main method
    :param argv:
    :return:
    """
    if len(argv) < 4:
        print ('Help\n{}  [Input File Name]   [Output File Name]   [2theta]'.format(argv[0]))
        sys.exit(1)

    raw_file_name = argv[1]
    output_file_name = argv[2]
    two_theta = float(argv[3])

    if raw_file_name.lower().endswith('tif') or raw_file_name.lower().endswith('tiff'):
        raw_data_set, dimension = parse_xray_tiff(raw_file_name)
    elif raw_file_name.lower.endswith('bin'):
        raw_data_set, dimension = parse_spice_binary(raw_file_name)
    else:
        print ('[ERROR] Input file {} of type {} is not supported (TIFF and .bin only)'
               ''.format(raw_file_name, raw_file_name.split('.')[-1]))
        sys.exit(-1)

    # export
    result = save_to_h5(raw_data_set, dimension, two_theta, output_file_name)
    if not result:
        sys.exit(-1)

    return


if __name__ == '__main__':
    main(sys.argv)
