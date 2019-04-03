#!/usr/bin/python
# Test to verify that two reduction engines, Mantid and PyRS, will give out identical result

import os
import sys
from pyrs.core import reduce_hb2b_mtd
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
print (os.getcwd())
# therefore it is not too hard to locate testing data
# test_data = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'
test_data = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
xray_idf_name = 'tests/testdata/XRay_Definition_2K.xml'
test_mask = 'tests/testdata/masks/Chi_10.hdf5'
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
        center_shift_x = 1.0 * (random.random() - 0.5) * 2.0
        center_shift_y = 1.0 * (random.random() - 0.5) * 2.0
        arm_length_shift = (random.random() - 0.5) * 2.0  # 0.416 + (random.random() - 0.5) * 2.0
        # calibration
        rot_x_flip = 2.0 * (random.random() - 0.5) * 2.0
        rot_y_flip = 2.0 * (random.random() - 0.5) * 2.0
        rot_z_spin = 2.0 * (random.random() - 0.5) * 2.0
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

    mantid_reducer = reduce_hb2b_mtd.MantidHB2BReduction()
    data_ws_name = engine.get_raw_data(test_data_id, is_workspace=True)
    mantid_reducer.set_workspace(data_ws_name)
    mantid_reducer.load_instrument(two_theta, xray_idf_name, test_calibration)

    return engine, pyrs_reducer, mantid_reducer


# step 1: geometry must be correct!
def compare_geometry_test(calibrated, pixel_number=2048):
    """
    Compare the geometry
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = create_instrument_load_data(calibrated, pixel_number)

    # compare

    # construct the mantid pixel array
    workspace = mantid_reducer.get_workspace()
    num_pixels = workspace.getNumberHistograms()
    mantid_pixel_array = numpy.ndarray((num_pixels, 3))
    import time
    time0 = time.time()
    for iws in range(num_pixels):
        mantid_pixel_array[iws] = numpy.array(workspace.getDetector(iws).getPos())
    # END-FOR
    timef = time.time()
    print ('Construct pixel position array: time = {}'.format(timef - time0))

    pixel_array = pyrs_reducer.get_pixel_positions(is_matrix=False)

    # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
    pixel_locations = [(0, 0),
                       (0, pixel_number - 1),
                       (pixel_number - 1, 0),
                       (pixel_number - 1, pixel_number - 1),
                       (pixel_number / 2, pixel_number / 2)]

    is_same = True
    for index_i, index_j in pixel_locations:
        index1d = index_i + pixel_number * index_j
        pos_python = pixel_array[index1d]
        pos_mantid = workspace.getDetector(index1d).getPos()
        print ('({}, {}) / {}:   {:10s}   -   {:10s}    =   {:10s}'
               ''.format(index_i, index_j, index1d, 'PyRS', 'Mantid', 'Diff'))
        diff_sq = 0.
        for i in range(3):
            diff_sq += (float(pos_python[i] - pos_mantid[i])) ** 2
            print ('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
                   ''.format(i, float(pos_python[i]), float(pos_mantid[i]),
                             float(pos_python[i] - pos_mantid[i])))
        # END-FOR
        if diff_sq > 1.E-6:
            is_same = False
    # END-FOR

    # check the different of all the pixels
    diff_vec = numpy.sqrt(((pixel_array - mantid_pixel_array) ** 2).sum(1))
    print ('[DB...BAT] diff vec: shape = {}'.format(diff_vec.shape))
    print (diff_vec.min(), diff_vec.max())
    print (numpy.argmin(diff_vec), numpy.argmax(diff_vec))
    print (numpy.average(diff_vec))
    print ('...................................................................')

    assert is_same, 'Instrument geometries from 2 engines do not match!'

    # TODO - TONIGHT 1 - Keep the original 2D array for speed test against new 1D array

    return engine, pyrs_reducer, mantid_reducer


def compare_convert_2theta(calibrated, pixel_number=2048):
    """
    compare the 2 reduction engines' output to 2theta
    :param calibrated:
    :param pixel_number:
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number)

    # compare 2theta
    pixel_matrix = pyrs_reducer.get_pixel_positions(is_matrix=False)
    print ('Pixel matrix shape = {}'.format(pixel_matrix.shape))
    pyrs_2theta_vec = pyrs_reducer.convert_to_2theta(pixel_matrix)
    print ('Shape: {}'.format(pyrs_2theta_vec.shape))
    pyrs_2theta_vec = pyrs_2theta_vec.transpose().flatten()  # 1D vector

    raw_ws_name = mantid_reducer.get_workspace().name()
    mantid_2theta_vec = mantid_reducer.convert_from_raw_to_2theta(raw_ws_name, test_mode=True).readX(0)

    # compare shape
    assert pyrs_2theta_vec.shape == mantid_2theta_vec.shape, 'Shapes ({} vs {}) shall be same' \
                                                             ''.format(pyrs_2theta_vec.shape, mantid_2theta_vec.shape)

    diff = numpy.sqrt(numpy.sum(((pyrs_2theta_vec - mantid_2theta_vec) ** 2)))  # / pyrs_2theta_vec.shape[0]

    any_pixel = 30000
    for i in range(any_pixel, any_pixel + 30):
        print ('[DB...TEST...VISUAL... CHECK] {}   -   {}    =    {}'
               ''.format(pyrs_2theta_vec[i], mantid_2theta_vec[i], pyrs_2theta_vec[i] - mantid_2theta_vec[i]))
    print ('[TEST RESULT] For {} pixels, DIFF = {}'.format(pyrs_2theta_vec.shape, diff))

    assert diff < 1., 'Difference {} is too big!'.format(diff)

    return pyrs_reducer, mantid_reducer


def compare_reduced_no_mask(calibrated, pixel_number=2048):
    """
    Compare reduced data without mask
    :param calibrated:
    :param pixel_number:
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number)

    curr_id = engine.current_data_id
    pyrs_vec_x, pyrs_vec_y = pyrs_reducer.reduce_to_2theta_histogram(det_pos_matrix=None,
                                                                     counts_matrix=engine.get_counts(curr_id),
                                                                     mask=None,
                                                                     num_bins=2500)

    data_ws = mantid_reducer.get_workspace()
    resolution = (pyrs_vec_x[-1] - pyrs_vec_x[0]) / 2500
    reduced_data = mantid_reducer.reduce_to_2theta(data_ws.name(), two_theta_min=pyrs_vec_x[0],
                                                   two_theta_max=pyrs_vec_x[-1],
                                                   two_theta_resolution=resolution,
                                                   mask=None)
    mantid_vec_x = reduced_data[0]
    mantid_vec_y = reduced_data[1]

    diff_x = numpy.sqrt(numpy.sum((pyrs_vec_x - mantid_vec_x)**2))/mantid_vec_x.shape[0]
    diff_y = numpy.sqrt(numpy.sum((pyrs_vec_y - mantid_vec_y)**2))/mantid_vec_y.shape[0]

    print ('Diff[X]  =  {},  Diff[Y]  =  {}'.format(diff_x, diff_y))
    plt.plot(pyrs_vec_x, pyrs_vec_y, color='blue', label='PyRS')
    plt.plot(mantid_vec_x, mantid_vec_y, color='red', label='Mantid')
    plt.legend()
    plt.show()

    return


# TODO - TONIGHT 0 - Need to compare 7 masks
def compare_reduced_masked(calibrated, pixel_number=2048):
    """
    Compare reduced data without mask
    :param calibrated:
    :param pixel_number:
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number)

    # load mask: mask file
    print ('Load masking file: {}'.format(test_mask))
    mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(test_mask)

    # reduce data
    curr_id = engine.current_data_id
    pyrs_vec_x, pyrs_vec_y = pyrs_reducer.reduce_to_2theta_histogram(det_pos_matrix=None,
                                                                     counts_matrix=engine.get_counts(curr_id),
                                                                     mask=mask_vec,
                                                                     num_bins=2500)

    data_ws = mantid_reducer.get_workspace()
    resolution = (pyrs_vec_x[-1] - pyrs_vec_x[0]) / (2500 - 1)
    mantid_vec_x, mantid_vec_y = mantid_reducer.reduce_to_2theta(data_ws.name(), two_theta_min=pyrs_vec_x[0],
                                                                 two_theta_max=pyrs_vec_x[-1],
                                                                 two_theta_resolution=resolution,
                                                                 mask=mask_vec)

    diff_x = numpy.sqrt(numpy.sum((pyrs_vec_x - mantid_vec_x)**2))
    diff_y = numpy.sqrt(numpy.sum((pyrs_vec_y - mantid_vec_y)**2))

    print ('Diff[X]  =  {},  Diff[Y]  =  {}'.format(diff_x, diff_y))
    plt.plot(pyrs_vec_x, pyrs_vec_y, color='blue', xlabel='PyRS')
    plt.plot(mantid_vec_x, mantid_vec_y, color='red', xlable='Mantid')
    plt.show()

    return


if __name__ == '__main__':
    """ main
    """
    if len(sys.argv) == 1:
        print ('{} [Test Options]\n'
               '\t\t1 = geometry (basic)\n'
               '\t\t2 = geometry (arbitrary calibration)\n'
               '\t\t3 = converting to 2theta (basic)\n'
               '\t\t4 = converting to 2theta (arbitrary calibration)\n'
               '\t\t5 = reducing to 2theta-intensity (basic)\n'
               '\t\t6 = reducing to 2theta-intensity (arbitrary calibration)\n'
               '\t\t7 = reducing to 2theta-intensity with mask (basic)\n'
               '\t\t8 = reducing to 2theta-intensity with mask (arbitrary calibration)'.format(sys.argv[0]),
               '\t\t10 = counts on detector ID (raw)\n'
               '\t\t11 = counts on detector ID (ROI =   0 degree)\n',
               '\t\t12 = counts on detector ID (ROI =  10 degree)\n',
               '\t\t13 = counts on detector ID (ROI = -10 degree)\n',
               )
    else:
        option = int(sys.argv[1])
        print ('Testing option: {}'.format(option))
        if option == 1:
            compare_geometry_test(False)
        elif option == 2:
            compare_geometry_test(True)  # with calibration deviation
        elif option == 3:
            compare_convert_2theta(False)
        elif option == 4:
            compare_convert_2theta(True)
        elif option == 5:
            compare_reduced_no_mask(False)
        elif option == 6:
            compare_reduced_masked(True)
        elif option == 7:
            compare_reduced_masked(False)
        elif option == 8:
            compare_reduced_no_mask(True)
        else:
            raise NotImplementedError('ASAP')

