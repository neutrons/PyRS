#!/usr/bin/python
# Test to verify that two reduction engines, Mantid and PyRS, will give out identical result

import os
import sys
from pyrs.core import reduce_hb2b_pyrs
from pyrs.utilities import calibration_file_io
from pyrs.core import reduction_manager
from pyrs.core import mask_util
import numpy
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication
import random
from matplotlib import pyplot as plt

# default testing directory is ..../PyRS/
print ('Current Working Directory: {}'.format(os.getcwd()))
test_data = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
if True:
    xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
    # xray_idf_name = 'tests/testdata/XRay_Definition_2K.xml'
elif False:  # IDF in test
    xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K_Mod.txt'
    # xray_idf_name = 'tests/testdata/XRay_Definition_2K_Mod.xml'
else:
    print ('Wrong configuration!')
    sys.exit(-1)

Mask_File = {0: 'tests/testdata/masks/Chi_0.hdf5',
             10: 'tests/testdata/masks/Chi_10.hdf5',
             20: 'tests/testdata/masks/Chi_20.hdf5',
             -10: 'tests/testdata/masks/Chi_Neg10.hdf5'}
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def create_instrument_load_data(calibrated, pixel_number, use_mantid):
    """ Create instruments: PyRS and Mantid and load data
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # instrument
    instrument = calibration_file_io.import_instrument_setup(xray_2k_instrument_file)

    # 2theta
    two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value
    arm_length_shift = 0.
    center_shift_x = 0.
    center_shift_y = 0.
    rot_x_flip = 0.
    rot_y_flip = 0.
    rot_z_spin = 0.

    test_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
    test_calibration.center_shift_x = center_shift_x
    test_calibration.center_shift_y = center_shift_y
    test_calibration.center_shift_z = arm_length_shift
    test_calibration.rotation_x = rot_x_flip  # + 0.001
    test_calibration.rotation_y = rot_y_flip  # - 0.00321
    test_calibration.rotation_z = rot_z_spin  # + 0.0022

    # reduction engine
    engine = reduction_manager.HB2BReductionManager()
    test_data_id, two_theta_tmp = engine.load_data(data_file_name=test_data, target_dimension=pixel_number,
                                                   load_to_workspace=use_mantid)

    # load instrument
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument, 1.239)
    pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                  rot_x_flip, rot_y_flip, rot_z_spin)

    return engine, pyrs_reducer


def reduce_data(mask_file, calibrated, pixel_number=2048):
    """
    Compare reduced data without mask
    :param mask_file: solid angle (integer)
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # create geometry/instrument
    engine, pyrs_reducer = create_instrument_load_data(False, pixel_number, use_mantid=False)

    # load mask: mask file
    if mask_file is not None:
        print ('Load masking file: {}'.format(mask_file))
        mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(mask_file)
        print ('Mask file {}: 2theta = {}'.format(mask_file, mask_2theta))
    else:
        mask_vec = None

    # reduce data
    min_2theta = 8.
    max_2theta = 64.
    num_bins = 1800

    # reduce PyRS (pure python)
    curr_id = engine.current_data_id
    pyrs_returns = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                           mask=mask_vec, x_range=(min_2theta, max_2theta),
                                                           num_bins=num_bins)
    pyrs_vec_x, pyrs_vec_y = pyrs_returns

    # plot
    if pyrs_vec_x.shape[0] != pyrs_vec_y.shape[0]:
        vec_2theta = pyrs_vec_x[:-1]
    else:
        vec_2theta = pyrs_vec_x

    # convert to dspacing
    vec_dspace = 0.5 * 1.239 / numpy.sin(0.5 * vec_2theta * numpy.pi / 180.)
    vec_dspace = vec_dspace[::-1]
    vec_y = pyrs_vec_y[::-1]

    print ('DSpacing: range = ({}, {})'.format(vec_dspace.min(), vec_dspace.max()))
    if mask_file:
        mask = os.path.basename(mask_file)
    else:
        mask = '(No Mask)'

    plt.plot(vec_dspace, vec_y, color='blue', label='from histogram-2theta')
    # plt.plot(vec_2theta, pyrs_vec_y, color='blue', label='PyRS: Mask {} Histogram by {}'
    #                                                      ''.format(mask, 'numpy.histogram'))
    # plt.legend()
    # plt.show()

    # Reduce to d-spacing
    min_d, max_d = 0.5, 5.0
    vec_d, vec_hist = pyrs_reducer.reduce_to_dspacing_histogram(counts_array=engine.get_counts(curr_id),
                                                                mask=mask_vec, x_range=(min_d, max_d),
                                                                num_bins=num_bins)

    plt.plot(vec_d, vec_hist, color='red', label='PyRS: Mask {} Histogram by {}'
                                                  ''.format(mask, 'numpy.histogram'))
    plt.legend()
    plt.show()

    # another comparison: bin size
    uneven_bins = vec_dspace[1:] - vec_dspace[:-1]
    plt.plot(vec_dspace[:-1], uneven_bins, color='blue')
    even_bins = vec_d[1:] - vec_d[:-1]
    plt.plot(vec_d[:-1], even_bins, color='red')
    plt.show()

    return


def main(argv):
    """
    main method
    :param argv:
    :return:
    """
    if len(argv) < 5:
        print ('{} [File Name] [2theta] [ROI=NONE/File] [ENGINE=M(antid)/P(yRS)]\n'.format(sys.argv[0]))
        sys.exit(0)

    # parse input
    file_name = argv[1]
    if not os.path.exists(file_name):
        print ('File {} does not exist.'.format(file_name))
        sys.exit(-1)

    two_theta = float(argv[2])
    roi_file = argv[3]
    if roi_file.lower() == 'none':
        roi_file = None
    engine = argv[4][0].lower()
    if engine != 'm' and engine != 'p':
        print ('ENGINE must be M or P')
        sys.exit(0)

    if engine == 'm':
        # mantid ...
        ignore

    else:
        # pyrs
        reduce_data(roi_file, False, pixel_number=2048)


if __name__ == '__main__':
    """ main
    """
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        # Test cases
        test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
        test_two_theta = '35.'
        engine = 'pyrs'

        if int(sys.argv[1]) == 5:
            roi_name = 'tests/testdata/masks/Chi_30.hdf5'
        elif int(sys.argv[1]) == 2:
            roi_name = 'tests/testdata/masks/Chi_0.hdf5'
        else:
            roi_name = 'none'

        main([sys.argv[0], test_file_name, test_two_theta, roi_name, engine])

    else:
        main(sys.argv)
