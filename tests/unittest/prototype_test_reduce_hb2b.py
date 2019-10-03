#!/usr/bin/python
# Test Reduce HB2B data with scripts built upon numpy arrays'
# Verify that Mantid and pure-python solution render the same result
from pyrs.core import pyrscore
from pyrs.core import reduce_hb2b_pyrs
import os
import matplotlib.pyplot as plt
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose, AddSampleLog, LoadInstrument, ConvertSpectrumAxis, ResampleX
from mantid.simpleapi import CreateWorkspace, ConvertToPointData
from mantid.api import AnalysisDataService as mtd
import time

XRAY_ARM_LENGTH = 0.416


def load_instrument(hb2b_builder, arm_length, two_theta=0., center_shift_x=0., center_shift_y=0.,
                    rot_x_flip=0., rot_y_flip=0., rot_z_spin=0., raw_data_ws_name=None, idf_name=None,
                    pixel_number=None):
    """ Load instrument to raw data file
    :param hb2b_builder:
    :param arm_length: full arm length
    :param two_theta: 2theta in sample log (instrument definition). It is opposite direction to Mantid coordinate
    :param center_shift_x:
    :param center_shift_y:
    :param rot_x_flip:
    :param rot_y_flip:
    :param rot_z_spin:
    :param raw_data_ws_name:
    :param idf_name:
    :param pixel_number: linear pixel size (row number and column number)
    :return: pixel matrix
    """
    pixel_matrix = hb2b_builder.build_instrument(arm_length_shift=arm_length, two_theta=-two_theta,
                                                 center_shift_x=center_shift_x, center_shift_y=center_shift_y,
                                                 rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip, rot_z_spin=rot_z_spin)

    if True:
        # using Mantid
        # check
        assert raw_data_ws_name is not None, 'data ws error'
        assert idf_name is not None, 'IDF cannot be None'
        assert pixel_number is not None, 'Pixel number to be given'

        # set up sample logs
        # cal::arm
        print ('Arm length correction  = {}'.format(arm_length))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::arm', LogText='{}'.format(arm_length),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::2theta
        print ('HB2B 2-theta = {}    (opposite to Mantid 2theta)'.format(two_theta))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='2theta', LogText='{}'.format(-two_theta),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')
        #
        # cal::deltax
        print ('Shift X = {}'.format(center_shift_x))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltax', LogText='{}'.format(center_shift_x),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::deltay
        print ('Shift Y = {}'.format(center_shift_y))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltay', LogText='{}'.format(center_shift_y),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::flip
        print ('Rotation X = {}'.format(rot_x_flip))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::flip', LogText='{}'.format(rot_x_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::roty
        print ('Rotation Y = {}'.format(rot_y_flip))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::roty', LogText='{}'.format(rot_y_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::spin
        print ('Rotation Z = {}'.format(rot_z_spin))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::spin', LogText='{}'.format(rot_z_spin),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        print ('Load instrument file : {}'.format(idf_name))
        LoadInstrument(Workspace=raw_data_ws_name,
                       Filename=idf_name,
                       InstrumentName='HB2B', RewriteSpectraMap='True')

        workspace = mtd[raw_data_ws_name]

        # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
        pixel_locations = [(0, 0),
                           (0, pixel_number - 1),
                           (pixel_number - 1, 0),
                           (pixel_number - 1, pixel_number - 1),
                           (pixel_number / 2, pixel_number / 2)]
        # compare position
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
                raise RuntimeError('Mantid PyRS mismatch!')
                # END-FOR
    # END-IF

    return pixel_matrix


def load_data_from_tif(raw_tiff_name, pixel_size=2048, rotate=True):
    """
    Load data from TIFF
    :param raw_tiff_name:
    :param pixel_size
    :param rotate:
    :return:
    """
    from skimage import io, exposure, img_as_uint, img_as_float
    from PIL import Image
    import numpy as np
    import pylab as plt

    ImageData = Image.open(raw_tiff_name)
    # im = img_as_uint(np.array(ImageData))
    io.use_plugin('freeimage')
    image_2d_data = np.array(ImageData, dtype=np.int32)
    print (image_2d_data.shape, type(image_2d_data), image_2d_data.min(), image_2d_data.max())
    # image_2d_data.astype(np.uint32)
    image_2d_data.astype(np.float64)
    if rotate:
        image_2d_data = image_2d_data.transpose()

    # Merge data if required
    if pixel_size == 1024:
        counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + image_2d_data[1::2, ::2] + image_2d_data[1::2, 1::2]
        pixel_type = '1K'
        # print (DataR.shape, type(DataR))
    else:
        # No merge
        counts_vec = image_2d_data
        pixel_type = '2K'

    counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
    print (counts_vec.min())

    data_ws_name = os.path.basename(raw_tiff_name).split('.')[0] + '_{}'.format(pixel_type)
    CreateWorkspace(DataX=np.zeros((pixel_size**2,)), DataY=counts_vec, DataE=np.sqrt(counts_vec), NSpec=pixel_size**2,
                    OutputWorkspace=data_ws_name, VerticalAxisUnit='SpectraNumber')

    return data_ws_name, counts_vec


def load_data_from_bin(bin_file_name):
    """
    """
    ws_name = os.path.basename(bin_file_name).split('.')[0]
    LoadSpiceXML2DDet(Filename=bin_file_name, OutputWorkspace=ws_name, LoadInstrument=False)

    # get vector of counts
    counts_ws = Transpose(InputWorkspace=ws_name, OutputWorkspace='temp')
    count_vec = counts_ws.readY(0)

    return ws_name, count_vec


def reduce_to_2theta(hb2b_builder, pixel_matrix, hb2b_data_ws_name, counts_vec, num_bins=2500):
    """
    reduce to 2theta unit
    :param hb2b_builder:
    :param pixel_matrix:
    :param hb2b_data_ws_name:
    :param counts_vec:
    :param num_bins:
    :return:
    """
    # pyrs_raw_ws = mtd[pyrs_raw_name]
    # counts_vec = pyrs_raw_ws.readY(0)

    # reduce by pure-Python
    time_pyrs_start = time.time()
    pyrs_reduced_name = '{}_pyrs_reduced'.format(hb2b_data_ws_name)
    bin_edgets, histogram = hb2b_builder.reduce_to_2theta_histogram(pixel_matrix, counts_vec, num_bins)

    for i in range(1000, 1020):
        print (bin_edgets[i], histogram[i])

    time_pyrs_mid = time.time()
    CreateWorkspace(DataX=bin_edgets, DataY=histogram, UnitX='degrees', OutputWorkspace=pyrs_reduced_name)
    ConvertToPointData(InputWorkspace=pyrs_reduced_name, OutputWorkspace=pyrs_reduced_name)
    time_pyrs_post = time.time()
    print ('PyRS Pure Python Reduction: Reduction Time = {}, Mantid post process time = {}'
           ''.format(time_pyrs_mid - time_pyrs_start, time_pyrs_post - time_pyrs_mid))

    # Mantid
    if True:
        # transfer to 2theta for data
        two_theta_ws_name = '{}_2theta'.format(hb2b_data_ws_name)
        mantid_reduced_name = '{}_mtd_reduced'.format(hb2b_data_ws_name)

        # For testing output
        # ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name,
        #                     OutputWorkspace=two_theta_ws_name, Target='Theta',
        #                     OrderAxis=True)
        # Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
        # two_theta_ws = mtd[two_theta_ws_name]
        # for i in range(10):
        #     print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))
        # for i in range(10010, 10020):
        #     print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))

        ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta')
        Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
        ResampleX(InputWorkspace=two_theta_ws_name, OutputWorkspace=mantid_reduced_name,
                  NumberBins=num_bins, PreserveEvents=False)
        mantid_ws = mtd[mantid_reduced_name]

        plt.plot(mantid_ws.readX(0), mantid_ws.readY(0), color='blue', label='Mantid')

        diff_y_vec = histogram - mantid_ws.readY(0)
        print ('Min/Max Diff = {} , {}'.format(min(diff_y_vec), max(diff_y_vec)))
    # END-IF

    pyrs_ws = mtd[pyrs_reduced_name]
    plt.plot(pyrs_ws.readX(0), histogram, color='red', label='Pure Python')
    plt.show()

    return


def test_main():
    """
    test main
    :return:
    """
    raise NotImplementedError('Not used!  Replaced by test_with_mantid')
    rs_core = pyrscore.PyRsCore()

    # Determine PyRS root
    pyrs_root_dir = os.path.abspath('.')
    test_data_dir = os.path.join(pyrs_root_dir, 'tests/testdata/')
    assert os.path.exists(test_data_dir), 'Test data directory {} does not exist'.format(test_data_dir)

    # Instrument factor
    row_col_size = 1024
    wavelength_kv = 1.E5  # kev
    wavelength = 1.296  # A

    Beam_Center_X = 0.000805
    Beam_Center_Y = -0.006026

    # set up data file and IDF
    if True:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.bin')
        two_theta = 35.

    if False:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_0deg-00000_Rotated.bin')
        two_theta = 0.

    hb2b_file_name = os.path.join(pyrs_root_dir, test_file_name)
    # instrument geometry
    idf_name = os.path.join(test_data_dir, 'XRay_Definition_1K.xml')

    # Load data
    raw_data_ws_name = os.path.basename(hb2b_file_name).split('.')[0]
    LoadSpiceXML2DDet(Filename=hb2b_file_name, OutputWorkspace=raw_data_ws_name, DetectorGeometry='0,0',
                      LoadInstrument=False)
    Transpose(InputWorkspace=raw_data_ws_name, OutputWorkspace=raw_data_ws_name)
    raw_data_ws = mtd[raw_data_ws_name]

    # if False:
    #     # old Mantid based type
    #     # Set up instrument geometry parameter
    #     load_instrument(raw_data_ws_name, idf_name, two_theta, cal_shift_x, cal_shift_y)
    #     reduced_ws_name = convert_to_2theta(raw_data_ws_name)

    xray_1k = True
    xray_2k = False
    hb2b_1k = False
    num_bins = 1000

    if xray_1k:
        # numpy array based script
        # build instrument
        num_rows = 1024
        num_columns = 1024
        pixel_size_x = 0.00029296875
        pixel_size_y = 0.00029296875
        arm_length = 0.416

        center_shift_x = 0.000805
        center_shift_y = -0.006026
        rot_x_flip = 0.
        rot_y_flip = 0.
        rot_z_spin = 0.

        hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)
        pixel_matrix = hb2b_builder.build_instrument(arm_length=arm_length, two_theta=-two_theta,
                                                     center_shift_x=center_shift_x, center_shift_y=center_shift_y,
                                                     rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip, rot_z_spin=0.)
        bin_edgets, histograms = hb2b_builder.reduce_to_2theta_histogram(pixel_matrix, raw_data_ws.readY(0), num_bins)

    elif xray_2k:
        raise NotImplementedError('No there yet!')

    else:
        raise RuntimeError('Nothing to test with!')
    # END-IF

    # plot
    vec_x = bin_edgets[:-1]
    vec_y = histograms
    plt.plot(vec_x, vec_y)

    # reduced_ws = mtd[reduced_ws_name]
    # vec_x = reduced_ws.readX(0)
    # vec_y = reduced_ws.readY(0)
    # plt.plot(vec_x, vec_y)
    #
    # non_norm_ws = mtd[raw_data_ws_name]
    # vec_x_nn = non_norm_ws.readX(0)
    # vec_y_nn = non_norm_ws.readY(0) / 1000.
    # plt.plot(vec_x_nn, vec_y_nn)

    plt.show()

    return


def main(argv):
    """
    test PyRS reduction with Mantid
    :return:
    """
    import random

    # Init setup for instrument geometry
    pixel_length = 1024
    pixel_length = 2048

    # Determine PyRS root and load data
    pyrs_root_dir = os.path.abspath('.')
    test_data_dir = os.path.join(pyrs_root_dir, 'tests/testdata/')
    assert os.path.exists(test_data_dir), 'Test data directory {} does not exist'.format(test_data_dir)

    # set up data file and IDF
    if pixel_length == 1024:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.bin')
        two_theta = 35.
        hb2b_ws_name, hb2b_count_vec = load_data_from_bin(test_file_name)

        num_rows = 2048/2
        num_columns = 2048/2
        pixel_size_x = 0.00020*2
        pixel_size_y = 0.00020*2
        idf_name = 'XRay_Definition_1K.xml'

    elif pixel_length == 2048:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.tif')
        two_theta = 35.
        hb2b_ws_name, hb2b_count_vec = load_data_from_tif(test_file_name, pixel_length)

        num_rows = 2048
        num_columns = 2048
        pixel_size_x = 0.00020
        pixel_size_y = 0.00020
        idf_name = 'XRay_Definition_2K.xml'

    else:
        raise NotImplementedError('No test file given!')

    # create HB2B-builder (python)
    # hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)
    from pyqr.utilities import calibration_file_io
    xray_instrument = calibration_file_io.InstrumentSetup()
    xray_instrument.detector_rows = num_rows
    xray_instrument.detector_columns = num_columns
    xray_instrument.pixel_size_x = pixel_size_x
    xray_instrument.pixel_size_y = pixel_size_y
    xray_instrument.arm_length = 0.416
    # num_rows, num_columns, pixel_size_x, pixel_size_y
    hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(xray_instrument)
    # set up IDF (Mantid)
    idf_name = os.path.join('tests/testdata/', idf_name)

    # load instrument
    print ('2THETA = {}'.format(two_theta))
    for iter in range(1):

        if False:
            arm_length = 0.416 + (random.random() - 0.5) * 2.0
            # two_theta = -80 + (random.random() - 0.5) * 20.

            # calibration
            rot_x_flip = 2.0 * (random.random() - 0.5) * 2.0
            rot_y_flip = 2.0 * (random.random() - 0.5) * 2.0
            rot_z_spin = 2.0 * (random.random() - 0.5) * 2.0

            center_shift_x = 1.0 * (random.random() - 0.5) * 2.0
            center_shift_y = 1.0 * (random.random() - 0.5) * 2.0
        else:
            arm_length = 0.  #0.416 + (random.random() - 0.5) * 2.0
            # two_theta = -80 + (random.random() - 0.5) * 20.

            # calibration
            rot_x_flip = 0.  # 2.0 * (random.random() - 0.5) * 2.0
            rot_y_flip = 0.  # 2.0 * (random.random() - 0.5) * 2.0
            rot_z_spin = 0.  # 2.0 * (random.random() - 0.5) * 2.0

            center_shift_x = 0.  # 1.0 * (random.random() - 0.5) * 2.0
            center_shift_y = 0.  # 1.0 * (random.random() - 0.5) * 2.0
        # END

        hb2b_pixel_matrix = load_instrument(hb2b_builder, arm_length, two_theta,
                                            center_shift_x, center_shift_y,
                                            rot_x_flip, rot_y_flip, rot_z_spin,
                                            hb2b_ws_name, idf_name, pixel_length)

        # reduce data
        reduce_to_2theta(hb2b_builder, hb2b_pixel_matrix, hb2b_ws_name, hb2b_count_vec)
    # END-FOR

    # # Load data
    # if True:
    #     # load data from SPICE binary
    #     raw_data_ws_name = os.path.basename(test_file_name).split('.')[0]
    #     LoadSpiceXML2DDet(Filename=test_file_name, OutputWorkspace=raw_data_ws_name, DetectorGeometry='0,0',
    #                       LoadInstrument=False)
    #     # get data Y (counts) for PyRS reduction
    #     pyrs_raw_name = '{}_pyrs'.format(raw_data_ws_name)
    #     Transpose(InputWorkspace=raw_data_ws_name, OutputWorkspace=pyrs_raw_name)
    # else:
    #     raise NotImplementedError('Next: Load data from image file')
    #
    # # Instrument factor
    # wavelength_kv = 1.E5  # kev
    # wavelength = 1.296  # A
    #
    # Beam_Center_X = 0.000805
    # Beam_Center_Y = -0.006026
    #
    # xray_1k = True
    # xray_2k = False
    # hb2b_1k = False
    #
    # num_bins = 1000
    #
    # if xray_1k:
    #     # numpy array based script
    #     # build instrument
    #     num_rows = 1024
    #     num_columns = 1024
    #     pixel_size_x = 0.00040
    #     pixel_size_y = 0.00040
    #     arm_length = 0.416
    #
    #     rot_x_flip = 0.
    #     rot_y_flip = 0.
    #     rot_z_spin = 0.
    #
    #     idf_name = os.path.join(test_data_dir, 'Xray_HB2B_1K.xml')
    #
    #     hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)
    #
    # else:
    #     raise NotImplementedError('No instrument configuration given!')
    #
    # # Load instrument and convert to 2Theta
    # hb2b_pixel = build_instrument(hb2b_builder, arm_length, two_theta, Beam_Center_X, Beam_Center_Y,
    #                               rot_x_flip, rot_y_flip, rot_z_spin, raw_data_ws_name, idf_name=idf_name)
    #
    # # reduce data
    # reduce_to_2theta(hb2b_builder, hb2b_pixel, raw_data_ws_name, pyrs_raw_name)

    return


if __name__ == '__main__':
    # test_main()
    main('whatever')
