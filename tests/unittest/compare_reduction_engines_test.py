#!/usr/bin/python
# Test to verify that two reduction engines, Mantid and PyRS, will give out identical result
import os
import sys
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import calibration_file_io
from pyrs.core import reductionengine
from pyrs.core import mask_util
import time
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

# Section for XRay-Mock data
test_xray_data = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
xray_2theta = -35.
if True:
    xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
    xray_idf_name = 'tests/testdata/xray_data/XRAY_Definition_20190521_1342.xml'
if False:
    # Note: Tested Version
    xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
    xray_idf_name = 'tests/testdata/XRay_Definition_2K.xml'
XRay_Masks = {0: 'tests/testdata/masks/Chi_0.hdf5',
              10: 'tests/testdata/masks/Chi_10.hdf5',
              20: 'tests/testdata/masks/Chi_20.hdf5',
              -10: 'tests/testdata/masks/Chi_Neg10.hdf5',
              30: 'tests/testdata/masks/Chi_30.hdf5',
              -30: 'tests/testdata/masks/Chi_Neg30.hdf5'}

# Section for HBZ (mock) data
test_hbz_data = 'tests/testdata/HBZ_Project.h5'
hbz_2theta = whatever
hbz_scan_index = whatever
hbz_instrument_file = 'blabla'
hbz_idf = 'blabla'


def create_instrument_load_data(instrument, scan_index, calibrated, pixel_number):
    """ Create instruments: PyRS and Mantid and load data
    :param instrument: name of instrument
    :param scan_index: for multiple scan project file
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # instrument
    if instrument == 'XRay-XRayMock':
        hydra_idf = xray_2k_instrument_file
        mantid_idf = xray_idf_name
        two_theta = xray_2theta
    elif instrument == 'HBZ':
        hydra_idf = hbz_instrument_file
        mantid_idf = hbz_idf
        two_theta = hbz_2theta
    else:
        raise RuntimeError('Instrument {} does not have test data and IDF'.format(instrument))

    print ('----------- IDF in Test: {} vs {} ---------------'.format(hydra_idf, mantid_idf))

    instrument = calibration_file_io.import_instrument_setup(hydra_idf)

    # instrument geometry calibration
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
    else:
        arm_length_shift = center_shift_x = center_shift_y = 0.
        rot_x_flip = rot_y_flip = rot_z_spin = 0.
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
    test_data_id = engine.load_data(data_file_name=test_xray_data, target_dimension=pixel_number, load_to_workspace=True)

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
def compare_geometry_test(calibrated, instrument='XRayMock', pixel_number=2048, check_all_pixels=False):
    """
    Compare the geometry
    :param calibrated:
    :param instrument:
    :param pixel_number:
    :param check_all_pixels:
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = create_instrument_load_data(instrument, scan_index, calibrated, pixel_number)

    # compare
    # pyrs pixels
    print ('About to get pixel positions')
    pixel_array = pyrs_reducer.get_pixel_positions(is_matrix=False)

    # mantid workspace
    print ('About to get workspace')
    workspace = mantid_reducer.get_workspace()

    # check all pixels (optional)
    print ('Check All Pixels... Taking long time')
    if check_all_pixels:
        num_pixels = workspace.getNumberHistograms()
        mantid_pixel_array = numpy.ndarray((num_pixels, 3))
        time0 = time.time()
        for iws in range(num_pixels):
            mantid_pixel_array[iws] = numpy.array(workspace.getDetector(iws).getPos())
        # END-FOR
        timef = time.time()
        print ('Construct pixel position array: time = {}'.format(timef - time0))
        # check the different of all the pixels
        diff_vec = numpy.sqrt(((pixel_array - mantid_pixel_array) ** 2).sum(1))
        print ('Pixels Positions Difference:  Min = {}, Max = {}'.format(diff_vec.min(), diff_vec.max()))
        print ('Minimum difference @ {}-th Pixel; Maximum difference @ {}-th Pixel'.format(numpy.argmin(diff_vec), numpy.argmax(diff_vec)))

        average_difference = numpy.average(diff_vec)
        print ('Average pixel position difference among all pixels: {}'.format(numpy.average(diff_vec)))
        print ('...................................................................')
    else:
        average_difference = 1.E100

    if not check_all_pixels or average_difference > 1.E-10:
        # check corners
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

        assert is_same, 'Instrument geometries from 2 engines do not match!'
    # END-IF

    return engine, pyrs_reducer, mantid_reducer


def compare_convert_2theta(calibrated, pixel_number=2048):
    """
    compare the 2 reduction engines' output to 2theta
    :param calibrated:
    :param pixel_number:
    :return:
    """
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number, False)

    # compare 2theta
    pixel_matrix = pyrs_reducer.get_pixel_positions(is_matrix=False)
    print ('Pixel matrix shape = {}'.format(pixel_matrix.shape))
    pyrs_2theta_vec = pyrs_reducer.convert_to_2theta(pixel_matrix)
    print ('Shape: {}'.format(pyrs_2theta_vec.shape))
    # pyrs_2theta_vec = pyrs_2theta_vec.transpose().flatten()  # 1D vector

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
    # load data and do 5 points geometry test
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number, check_all_pixels=False)

    min_2theta = 15.0 # 8.
    max_2theta = 55.0 # 64.

    NUM_BINS = 1800  # 2500

    # reduce PyRS (pure python)
    curr_id = engine.current_data_id
    pyrs_returns = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                           mask=None, x_range=(min_2theta, max_2theta),
                                                           num_bins=NUM_BINS)
    # pyrs_vec_x, pyrs_vec_y, raw_vec_2theta, raw_vec_count = pyrs_returns
    pyrs_vec_x, pyrs_vec_y = pyrs_returns
    print ('Debug Output: (pyrs) vec Y: sum = {}\n{}'.format(pyrs_vec_y.sum(), pyrs_vec_y))

    # reduce Mantid
    data_ws = mantid_reducer.get_workspace()
    resolution = (pyrs_vec_x[-1] - pyrs_vec_x[0]) / NUM_BINS
    reduced_data = mantid_reducer.reduce_to_2theta(data_ws.name(), two_theta_min=min_2theta,
                                                   two_theta_max=max_2theta,
                                                   num_2theta_bins=NUM_BINS,
                                                   mask=None)
    mantid_vec_x = reduced_data[0]
    mantid_vec_y = reduced_data[1]

    diff_x = numpy.sqrt(numpy.sum((pyrs_vec_x - mantid_vec_x)**2))/mantid_vec_x.shape[0]
    diff_y = numpy.sqrt(numpy.sum((pyrs_vec_y - mantid_vec_y)**2))/mantid_vec_y.shape[0]

    print ('Diff[X]  =  {},  Diff[Y]  =  {}'.format(diff_x, diff_y))
    plt.plot(pyrs_vec_x[:-1], pyrs_vec_y, color='blue', label='PyRS')
    plt.plot(mantid_vec_x[:-1], mantid_vec_y, color='red', label='Mantid')
    plt.legend()
    plt.show()

    return


def compare_reduced_masked(angle, calibrated, pixel_number):
    """
    Compare reduced data without mask
    :param angle: solid angle (integer)
    :param calibrated:
    :param pixel_number:
    :return:
    """
    # create geometry/instrument
    engine, pyrs_reducer, mantid_reducer = compare_geometry_test(calibrated, pixel_number, False)

    # load mask
    test_mask = XRay_Masks[angle]

    # load mask: mask file
    print ('Load masking file: {}'.format(test_mask))
    mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(test_mask)
    print ('Mask file {}: 2theta = {}'.format(test_mask, mask_2theta))

    NUM_BINS = 1800

    # reduce data
    min_2theta = 15.
    max_2theta = 55.

    # reduce PyRS (pure python)
    curr_id = engine.current_data_id
    pyrs_returns = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                           mask=mask_vec, x_range=(min_2theta, max_2theta),
                                                           num_bins=NUM_BINS)
    pyrs_vec_x, pyrs_vec_y = pyrs_returns

    # reduce by Mantid
    data_ws = mantid_reducer.get_workspace()
    reduced_data = mantid_reducer.reduce_to_2theta(data_ws.name(),
                                                   two_theta_min=min_2theta,
                                                   two_theta_max=max_2theta,
                                                   num_2theta_bins=NUM_BINS,
                                                   mask=mask_vec)
    mantid_vec_x = reduced_data[0]
    mantid_vec_y = reduced_data[1]

    # possible convert
    if mantid_vec_x.shape[0] == pyrs_vec_x.shape[0] - 1:
        # partial compare result
        print ('Mantid ResampleX return: shape = {}, {}'.format(mantid_vec_x.shape,
                                                                mantid_vec_y.shape))

        diff_y = numpy.sqrt(numpy.sum((pyrs_vec_y - mantid_vec_y)**2))/mantid_vec_y.shape[0]

        print ('Diff[X]  =  Not Calculated,  Diff[Y]  =  {}'.format(diff_y))
        plt.plot(pyrs_vec_x[:-1], pyrs_vec_y, color='blue', label='PyRS')
        plt.plot(mantid_vec_x, mantid_vec_y, color='red', label='Mantid')
    else:
        # compare result
        diff_x = numpy.sqrt(numpy.sum((pyrs_vec_x - mantid_vec_x) ** 2)) / mantid_vec_x.shape[0]
        diff_y = numpy.sqrt(numpy.sum((pyrs_vec_y - mantid_vec_y)**2))/mantid_vec_y.shape[0]

        print ('Diff[X]  =  {},  Diff[Y]  =  {}'.format(diff_x, diff_y))
        plt.plot(pyrs_vec_x[:-1], pyrs_vec_y, color='blue', label='PyRS')
        plt.plot(mantid_vec_x[:-1], mantid_vec_y, color='red', label='Mantid')

    plt.legend()
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
               '\t\t21: geometry basic for HBZ\n',
               '\t\t22: geometry arbitrary calibration for HBZ\n',
               )
    else:
        option = int(sys.argv[1])
        print ('Testing option: {}'.format(option))
        if option == 1:
            compare_geometry_test(False, pixel_number=2048, check_all_pixels=False)
        elif option == 2:
            compare_geometry_test(True, pixel_number=2048, check_all_pixels=True)  # with calibration deviation
        elif option == 3:
            compare_convert_2theta(False)
        elif option == 4:
            compare_convert_2theta(True)
        elif option == 5:
            compare_reduced_no_mask(False)
        elif option == 6:
            compare_reduced_no_mask(True)
        elif option == 7:
            compare_reduced_masked(angle=0, calibrated=False, pixel_number=2048)
        elif option == 8:
            compare_reduced_masked(angle=0, calibrated=True, pixel_number=2048)
        elif option == 9:
            compare_reduced_masked(angle=10, calibrated=False, pixel_number=2048)
        elif option == 10:
            compare_reduced_masked(angle=10, calibrated=True, pixel_number=2048)
        elif option == 11:
            compare_reduced_masked(angle=20, calibrated=False, pixel_number=2048)
        elif option == 12:
            compare_reduced_masked(angle=20, calibrated=True, pixel_number=2048)
        elif option == 13:
            compare_reduced_masked(angle=30, calibrated=False, pixel_number=2048)
        elif option == 14:
            compare_reduced_masked(angle=30, calibrated=True, pixel_number=2048)
        elif option == 21:
            # HBZ
            compare_geometry_test(calibrated=False, instrument='HBZ', pixel_number=256, check_all_pixels=True)

        elif option == 22:
            # HBZ
            compare_geometry_test(calibrated=True, instrument='HBZ', pixel_number=256, check_all_pixels=True)
        else:
            raise NotImplementedError('ASAP')

