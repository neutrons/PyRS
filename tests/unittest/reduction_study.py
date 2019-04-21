#!/usr/bin/python
# Test to verify that two reduction engines, Mantid and PyRS, will give out identical result

import os
import sys
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import calibration_file_io
from pyrs.core import reductionengine
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


def create_instrument_load_data(calibrated, pixel_number):
    """ Create instruments: PyRS and Mantid and load data
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # instrument
    instrument = calibration_file_io.import_instrument_setup(xray_2k_instrument_file)

    # 2theta
    two_theta = 35.
    arm_length_shift = 0.
    center_shift_x = 0.
    center_shift_y = 0.
    rot_x_flip = 0.
    rot_y_flip = 0.
    rot_z_spin = 0.

    if calibrated:
        # Note: README/TODO-ING: ONLY shift Y
        center_shift_x = int(1000. * (random.random() - 0.5) * 2.0) / 1000.
        center_shift_y = int(1000. * (random.random() - 0.5) * 2.0) / 1000.
        arm_length_shift = int(1000. * (random.random() - 0.5) * 2.0) / 1000.  # 0.416 + (random.random() - 0.5) * 2.0
        # calibration  FIXME - Disable rotation calibration to find out the source of difference:  10-17 vs 10-7
        rot_x_flip = int(1000. * 2.0 * (random.random() - 0.5) * 5.0) / 1000.
        rot_y_flip = int(1000. * 2.0 * (random.random() - 0.5) * 5.0) / 1000.
        rot_z_spin = int(1000. * 2.0 * (random.random() - 0.5) * 5.0) / 1000.
        print ('[(Random) Calibration Setup]\n    Shift Linear (X, Y, Z) = {}, {}, {}\n    Shift Rotation '
               '(X, Y, Z) = {}, {}, {}'
               ''.format(center_shift_x, center_shift_y, arm_length_shift, rot_x_flip, rot_y_flip,
                         rot_z_spin))
    # END-IF: arbitrary calibration

    test_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
    test_calibration.center_shift_x = center_shift_x
    test_calibration.center_shift_y = center_shift_y
    test_calibration.center_shift_z = arm_length_shift
    test_calibration.rotation_x = rot_x_flip
    test_calibration.rotation_y = rot_y_flip
    test_calibration.rotation_z = rot_z_spin

    # reduction engine
    engine = reductionengine.HB2BReductionManager()
    test_data_id = engine.load_data(data_file_name=test_data, target_dimension=pixel_number, load_to_workspace=True)

    # load instrument
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument)
    pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                  rot_x_flip, rot_y_flip, rot_z_spin)

    return engine, pyrs_reducer


def convert_to_2theta(calibrated, pixel_number=2048):
    """
    study the 2 reduction engines' output to 2theta
    :param calibrated:
    :param pixel_number:
    :return:
    """
    engine, pyrs_reducer = create_instrument_load_data(calibrated, pixel_number)

    # compare 2theta
    pixel_matrix = pyrs_reducer.get_pixel_positions(is_matrix=False)
    print ('Pixel matrix shape = {}'.format(pixel_matrix.shape))
    pyrs_2theta_vec = pyrs_reducer.convert_to_2theta(pixel_matrix)
    print ('Shape: {}'.format(pyrs_2theta_vec.shape))
    # pyrs_2theta_vec = pyrs_2theta_vec.transpose().flatten()  # 1D vector

    return pyrs_reducer


def reduced_no_mask(calibrated, pixel_number=2048):
    """
    Compare reduced data without mask
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # load data and do 5 points geometry test
    engine, pyrs_reducer = create_instrument_load_data(calibrated, pixel_number)

    min_2theta = 8.
    max_2theta = 64.

    # reduce PyRS (pure python)
    curr_id = engine.current_data_id
    pyrs_returns = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                           mask=None, x_range=(min_2theta, max_2theta),
                                                           num_bins=2500)
    pyrs_vec_x, pyrs_vec_y = pyrs_returns
    print ('Debug Output: (pyrs) vec Y: sum = {}\n{}'.format(pyrs_vec_y.sum(), pyrs_vec_y))

    plt.plot(pyrs_vec_x[:-1], pyrs_vec_y, color='blue', label='PyRS')
    plt.legend()
    plt.show()

    return


def reduce_with_mask(mask_solid_angle, calibrated, pixel_number=2048):
    """
    Compare reduced data without mask
    :param mask_solid_angle: solid angle (integer)
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # create geometry/instrument
    engine, pyrs_reducer = create_instrument_load_data(calibrated, pixel_number)

    # load mask
    mask_file = Mask_File[mask_solid_angle]

    # load mask: mask file
    print ('Load masking file: {}'.format(mask_file))
    mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(mask_file)
    print ('Mask file {}: 2theta = {}'.format(mask_file, mask_2theta))

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
    plt.plot(pyrs_vec_x[:-1], pyrs_vec_y, color='blue', label='PyRS')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    """ main
    """
    if len(sys.argv) == 1:
        print ('{} [Test Options]\n'
               '\t\t1 = reducing to 2theta-intensity (basic)\n'
               '\t\t2 = reducing to 2theta-intensity (arbitrary calibration)\n'
               '\t\t3 = reducing to 2theta-intensity (ROI =   0 degree)\n'
               '\t\t4 = reducing to 2theta-intensity  (ROI =  10 degree)\n'
               '\t\t5 = reducing to 2theta-intensity  (ROI = -10 degree)\n'
               '\t\t6 = reducing to 2theta-intensity  (ROI =  20 degree)\n'
               '\t\t7 = reducing to 2theta-intensity  (ROI = -20 degree)\n'
               '\t\t8 = reducing to 2theta-intensity  (ROI =  30 degree)\n'
               '\t\t9 = reducing to 2theta-intensity  (ROI = -30 degree)\n'.format(sys.argv[0])
               )
    else:
        option = int(sys.argv[1])
        print ('Testing option: {}'.format(option))
        if option == 1:
            reduced_no_mask(False)
        elif option == 2:
            reduced_no_mask(True)
        elif option == 3:
            reduce_with_mask(mask_solid_angle=0, calibrated=False, pixel_number=2048)
        elif option == 4:
            reduce_with_mask(mask_solid_angle=10, calibrated=False, pixel_number=2048)
        elif option == 5:
            reduce_with_mask(mask_solid_angle=-10, calibrated=False, pixel_number=2048)
        elif option == 6:
            reduce_with_mask(mask_solid_angle=20, calibrated=False, pixel_number=2048)
        elif option == 7:
            reduce_with_mask(mask_solid_angle=-20, calibrated=False, pixel_number=2048)
        elif option == 8:
            reduce_with_mask(mask_solid_angle=30, calibrated=False, pixel_number=2048)
        elif option == 9:
            reduce_with_mask(mask_solid_angle=-30, calibrated=False, pixel_number=2048)
        else:
            raise NotImplementedError('ASAP')

