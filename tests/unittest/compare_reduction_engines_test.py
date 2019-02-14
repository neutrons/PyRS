#!/usr/bin/python
# Test to verify that two reduction engines, Mantid and PyRS, will give out identical result

import os
import sys
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import calibration_file_io
from pyrs.core import reductionengine

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

import random

# default testing directory is ..../PyRS/
print (os.getcwd())
# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'
xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
xray_idf_name = 'tests/testdata/XRay_Definition_2K.xml'
print ('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def compare_geometry_test(calibrated, pixel_number=2048):
    """
    Compare the geometry
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
    pixel_matrix = pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                                 rot_x_flip, rot_y_flip, rot_z_spin)

    mantid_reducer = reduce_hb2b_mtd.MantidHB2BReduction()
    data_ws_name = engine.get_raw_data(test_data_id, is_workspace=True)
    mantid_reducer.set_workspace(data_ws_name)
    mantid_reducer.load_instrument(two_theta, xray_idf_name, test_calibration)

    # compare
    workspace = mantid_reducer.get_workspace()

    # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
    pixel_locations = [(0, 0),
                       (0, pixel_number - 1),
                       (pixel_number - 1, 0),
                       (pixel_number - 1, pixel_number - 1),
                       (pixel_number / 2, pixel_number / 2)]

    is_same = True
    for index_i, index_j in pixel_locations:
        # print ('PyRS:   ', pixel_matrix[index_i, index_j])
        # print ('Mantid: ', workspace.getDetector(index_i + index_j * 1024).getPos())  # column major
        pos_python = pixel_matrix[index_i, index_j]
        index1d = index_i + pixel_number * index_j
        pos_mantid = workspace.getDetector(index1d).getPos()
        print ('({}, {} / {}):   {:10s}   -   {:10s}    =   {:10s}'
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

    assert is_same, 'Instrument geometry does not match!'

    return is_same


if __name__ == '__main__':
    """ main
    """
    if len(sys.argv) == 1:
        print ('{} [Test Options: 1 = geometry (basic), 2 = geometry (arbitrary calibration)')
    else:
        option = int(sys.argv[1])
        if option == 1:
            compare_geometry_test(False)
        elif option == 2:
            compare_geometry_test(True)
        else:
            raise NotImplementedError('ASAP')

