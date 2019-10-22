#!/usr/bin/python
# Compare Mantid instrument builder and PyRS numpy-builder
# This script to read in a TIFF file or SPICE-compatible binary file
from pyrs.core import reduce_hb2b_pyrs
import os
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose, AddSampleLog, LoadInstrument, CreateWorkspace
from mantid.api import AnalysisDataService as mtd

XRAY_ARM_LENGTH = 0.416


def load_instrument(hb2b_builder, arm_length_shift, two_theta=0., center_shift_x=0., center_shift_y=0.,
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
    # pixel_matrix = hb2b_builder.build_instrument(arm_length=arm_length, two_theta=-two_theta,
    #                                              center_shift_x=center_shift_x, center_shift_y=center_shift_y,
    #                                              rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip, rot_z_spin=rot_z_spin)
    pixel_matrix = hb2b_builder.build_instrument(two_theta=-two_theta,
                                                 arm_length_shift=arm_length_shift,
                                                 center_shift_x=center_shift_x,
                                                 center_shift_y=center_shift_y,
                                                 rot_x_flip=rot_x_flip,
                                                 rot_y_flip=rot_y_flip,
                                                 rot_z_spin=rot_z_spin)

    if True:
        # using Mantid
        # check
        assert raw_data_ws_name is not None, 'data ws error'
        assert idf_name is not None, 'IDF cannot be None'
        assert pixel_number is not None, 'Pixel number to be given'

        # set up sample logs
        # cal::arm
        print('Arm length correction  = {}'.format(arm_length_shift))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::arm', LogText='{}'.format(arm_length_shift),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::2theta
        print('HB2B 2-theta = {}    (opposite to Mantid 2theta)'.format(two_theta))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='2theta', LogText='{}'.format(-two_theta),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')
        #
        # cal::deltax
        print('Shift X = {}'.format(center_shift_x))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltax', LogText='{}'.format(center_shift_x),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::deltay
        print('Shift Y = {}'.format(center_shift_y))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltay', LogText='{}'.format(center_shift_y),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::flip
        print('Rotation X = {}'.format(rot_x_flip))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::flip', LogText='{}'.format(rot_x_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::roty
        print('Rotation Y = {}'.format(rot_y_flip))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::roty', LogText='{}'.format(rot_y_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::spin
        print('Rotation Z = {}'.format(rot_z_spin))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::spin', LogText='{}'.format(rot_z_spin),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        print('Load instrument file : {}'.format(idf_name))
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
            print('({}, {} / {}):   {:10s}   -   {:10s}    =   {:10s}'
                  ''.format(index_i, index_j, index1d, 'PyRS', 'Mantid', 'Diff'))
            diff_sq = 0.
            for i in range(3):
                diff_sq += (float(pos_python[i] - pos_mantid[i]))**2
                print('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
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
    from skimage import io
    from PIL import Image
    import numpy as np

    ImageData = Image.open(raw_tiff_name)
    # im = img_as_uint(np.array(ImageData))
    io.use_plugin('freeimage')
    image_2d_data = np.array(ImageData, dtype=np.int32)
    print(image_2d_data.shape, type(image_2d_data), image_2d_data.min(), image_2d_data.max())
    # image_2d_data.astype(np.uint32)
    image_2d_data.astype(np.float64)
    if rotate:
        image_2d_data = image_2d_data.transpose()

    # Merge data if required
    if pixel_size == 1024:
        counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + \
            image_2d_data[1::2, ::2] + image_2d_data[1::2, 1::2]
        pixel_type = '1K'
        # print (DataR.shape, type(DataR))
    else:
        # No merge
        counts_vec = image_2d_data
        pixel_type = '2K'

    counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
    print(counts_vec.min())

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


def main(argv):
    """
    main args
    :param argv:
    :return:
    """
    import random

    # Init setup for instrument geometry
    pixel_length = 1024
    pixel_length = 2048

    # Data
    if pixel_length == 2048:
        image_file = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'
        hb2b_ws_name, hb2b_count_vec = load_data_from_tif(image_file, pixel_length)
    else:
        bin_file = 'tests/testdata/LaB6_10kev_0deg-00000_Rotated.bin'
        hb2b_ws_name, hb2b_count_vec = load_data_from_bin(bin_file)

    # create instrument
    if pixel_length == 2048:
        num_rows = 2048
        num_columns = 2048
        pixel_size_x = 0.00020
        pixel_size_y = 0.00020
        idf_name = 'XRay_Definition_2K.xml'
    elif pixel_length == 1024:
        num_rows = 2048/2
        num_columns = 2048/2
        pixel_size_x = 0.00020*2
        pixel_size_y = 0.00020*2
        idf_name = 'XRay_Definition_1K.xml'
    else:
        raise RuntimeError('Wrong setup')

    from pyqr.utilities import calibration_file_io
    xray_instrument = calibration_file_io.InstrumentSetup()
    xray_instrument.detector_rows = num_rows
    xray_instrument.detector_columns = num_columns
    xray_instrument.pixel_size_x = pixel_size_x
    xray_instrument.pixel_size_y = pixel_size_y
    xray_instrument.arm_length = 0.416
    # num_rows, num_columns, pixel_size_x, pixel_size_y
    hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(xray_instrument)
    idf_name = os.path.join('tests/testdata/', idf_name)

    # load instrument
    for iter in range(1):

        if False:
            arm_length_shift = (random.random() - 0.5) * 2.0  # 0.416 + (random.random() - 0.5) * 2.0
            two_theta = 35. + (random.random() - 0.5) * 20.

            # calibration
            rot_x_flip = 2.0 * (random.random() - 0.5) * 2.0
            rot_y_flip = 2.0 * (random.random() - 0.5) * 2.0
            rot_z_spin = 2.0 * (random.random() - 0.5) * 2.0

            center_shift_x = 1.0 * (random.random() - 0.5) * 2.0
            center_shift_y = 1.0 * (random.random() - 0.5) * 2.0
        else:
            arm_length_shift = 0.  # arm length shift
            two_theta = 35.

            # calibration
            rot_x_flip = 0.  # 2.0 * (random.random() - 0.5) * 2.0
            rot_y_flip = 0.  # 2.0 * (random.random() - 0.5) * 2.0
            rot_z_spin = 0.  # 2.0 * (random.random() - 0.5) * 2.0

            center_shift_x = 0.  # 1.0 * (random.random() - 0.5) * 2.0
            center_shift_y = 0.  # 1.0 * (random.random() - 0.5) * 2.0
        # END

        hb2b_pixel_matrix = load_instrument(hb2b_builder, arm_length_shift, two_theta,
                                            center_shift_x, center_shift_y,
                                            rot_x_flip, rot_y_flip, rot_z_spin,
                                            hb2b_ws_name, idf_name, pixel_length)
    # END-FOR


if __name__ == '__main__':
    main(['do it'])
