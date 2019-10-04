#!/usr/bin/python
# Test Reduce HB2B data with scripts built upon numpy arrays'
# Verify that Mantid and pure-python solution render the same result
from pyrs.core import pyrscore
from pyrs.core import reduce_hb2b_pyrs
import os
import matplotlib.pyplot as plt
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose, AddSampleLog, LoadInstrument, ConvertSpectrumAxis, ResampleX
from mantid.simpleapi import CreateWorkspace, ConvertToPointData, SaveNexusProcessed, Multiply
from mantid.api import AnalysisDataService as mtd
from pyrs.utilities import file_utilities
import time
from pyrs.core import rs_scan_io


# TODO - NIGHT - Continue to verify the cases for all masks!
# TODO         - Wait for hdf5/numpy binary mask files


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
    pixel_matrix = hb2b_builder.build_instrument(arm_length=arm_length, two_theta=-two_theta,
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
        print('Arm length correction  = {}'.format(arm_length - XRAY_ARM_LENGTH))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::arm', LogText='{}'.format(arm_length - XRAY_ARM_LENGTH),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::2theta
        print('HB2B 2-theta = {}    (opposite to Mantid 2theta)'.format(two_theta))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::2theta', LogText='{}'.format(-two_theta),
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
                diff_sq += (float(pos_python[i] - pos_mantid[i])) ** 2
                print('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
                      ''.format(i, float(pos_python[i]), float(pos_mantid[i]),
                                float(pos_python[i] - pos_mantid[i])))
            # END-FOR
            if diff_sq > 1.E-6:
                raise RuntimeError('Mantid PyRS mismatch!')
                # END-FOR
    # END-IF

    return pixel_matrix


# def load_data_from_tif(raw_tiff_name, pixel_size=2048, rotate=True):
#     """
#     Load data from TIFF
#     :param raw_tiff_name:
#     :param pixel_size
#     :param rotate:
#     :return:
#     """
#     from skimage import io, exposure, img_as_uint, img_as_float
#     from PIL import Image
#     import numpy as np
#     import pylab as plt
#
#     ImageData = Image.open(raw_tiff_name)
#     # im = img_as_uint(np.array(ImageData))
#     io.use_plugin('freeimage')
#     image_2d_data = np.array(ImageData, dtype=np.int32)
#     print (image_2d_data.shape, type(image_2d_data), image_2d_data.min(), image_2d_data.max())
#     # image_2d_data.astype(np.uint32)
#     image_2d_data.astype(np.float64)
#     if rotate:
#         image_2d_data = image_2d_data.transpose()
#
#     # Merge data if required
#     if pixel_size == 1024:
#         counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + image_2d_data[1::2, ::2] + image_2d_data[1::2, 1::2]
#         pixel_type = '1K'
#         # print (DataR.shape, type(DataR))
#     else:
#         # No merge
#         counts_vec = image_2d_data
#         pixel_type = '2K'
#
#     counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
#     print (counts_vec.min())
#
#     data_ws_name = os.path.basename(raw_tiff_name).split('.')[0] + '_{}'.format(pixel_type)
#     CreateWorkspace(DataX=np.zeros((pixel_size**2,)), DataY=counts_vec, DataE=np.sqrt(counts_vec), NSpec=pixel_size**2,
#                     OutputWorkspace=data_ws_name, VerticalAxisUnit='SpectraNumber')
#
#     return data_ws_name, counts_vec
#
#
# def load_data_from_tif_ver2():
#     """ Use matplotlib import TIFF
#     :return:
#     """
#     # TODO - NIGHT - Implement version 2 as the main TIFF reader for test on analysis cluster
#     data = matplotlib.image.imread(tiff_name)
#     data.astype(np.int32)
#
#     return


# def load_data_from_bin(bin_file_name):
#     """
#     """
#     ws_name = os.path.basename(bin_file_name).split('.')[0]
#     LoadSpiceXML2DDet(Filename=bin_file_name, OutputWorkspace=ws_name, LoadInstrument=False)
#
#     # get vector of counts
#     counts_ws = Transpose(InputWorkspace=ws_name, OutputWorkspace='temp')
#     count_vec = counts_ws.readY(0)
#
#     return ws_name, count_vec


def reduce_to_2theta(hb2b_builder, pixel_matrix, hb2b_data_ws_name, counts_vec, mask_vec, mask_ws_name,
                     num_bins=1000):
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
    # mask the input detector counts
    if mask_vec is not None:
        counts_vec.astype('float64')
        mask_vec.astype('float64')  # mask vector shall be float64 already!
        masked_count_vec = mask_vec * counts_vec
        counts_vec = masked_count_vec
    # END-IF

    # reduce by pure-Python
    time_pyrs_start = time.time()
    pyrs_reduced_name = '{}_pyrs_reduced'.format(hb2b_data_ws_name)
    bin_edgets, histogram = hb2b_builder.reduce_to_2theta_histogram(pixel_matrix, counts_vec, num_bins)

    time_pyrs_mid = time.time()

    CreateWorkspace(DataX=bin_edgets, DataY=histogram, UnitX='degrees', OutputWorkspace=pyrs_reduced_name)
    ConvertToPointData(InputWorkspace=pyrs_reduced_name, OutputWorkspace=pyrs_reduced_name)

    time_pyrs_post = time.time()
    print('PyRS Pure Python Reduction: Reduction Time = {}, Mantid post process time = {}'
          ''.format(time_pyrs_mid - time_pyrs_start, time_pyrs_post - time_pyrs_mid))

    SaveNexusProcessed(InputWorkspace=pyrs_reduced_name, Filename='{}.nxs'.format(pyrs_reduced_name),
                       Title='PyRS reduced: {}'.format(hb2b_data_ws_name))

    # Mantid
    if True:
        # transfer to 2theta for data
        two_theta_ws_name = '{}_2theta'.format(hb2b_data_ws_name)
        mantid_reduced_name = '{}_mtd_reduced'.format(hb2b_data_ws_name)

        # Mask
        if mask_ws_name:
            # Multiply by masking workspace
            masked_ws_name = '{}_masked'.format(hb2b_data_ws_name)
            Multiply(LHSWorkspace=hb2b_data_ws_name, RHSWorkspace=mask_ws_name,
                     OutputWorkspace=masked_ws_name, ClearRHSWorkspace=False)
            hb2b_data_ws_name = masked_ws_name
            SaveNexusProcessed(InputWorkspace=hb2b_data_ws_name, Filename='{}_raw.nxs'.format(hb2b_data_ws_name))
        # END-IF

        ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta')
        Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
        ResampleX(InputWorkspace=two_theta_ws_name, OutputWorkspace=mantid_reduced_name,
                  NumberBins=num_bins, PreserveEvents=False)
        mantid_ws = mtd[mantid_reduced_name]

        SaveNexusProcessed(InputWorkspace=mantid_reduced_name, Filename='{}.nxs'.format(mantid_reduced_name),
                           Title='Mantid reduced: {}'.format(hb2b_data_ws_name))

        plt.plot(mantid_ws.readX(0), mantid_ws.readY(0), color='blue', label='Mantid')

        diff_y_vec = histogram - mantid_ws.readY(0)
        print('Min/Max Diff = {} , {}'.format(min(diff_y_vec), max(diff_y_vec)))
    # END-IF

    plt.plot(bin_edgets[:-1], histogram, color='red', label='Pure Python')

    plt.show()

    return


def create_mask(mantid_mask_xml, pixel_number, is_mask):
    """
    create Mask vector and workspace
    :param mantid_mask_xml:
    :param pixel_number: total pixel number
    :return:
    """
    masking_array = file_utilities.load_mantid_mask(pixel_number, mantid_mask_xml, is_mask)

    mask_ws_name = os.path.basename(mantid_mask_xml).split('.')[0]

    CreateWorkspace(DataX=[0], DataY=masking_array, NSpec=pixel_number, OutputWorkspace=mask_ws_name)

    return masking_array, mask_ws_name


def main(argv):
    """
    test PyRS reduction with Mantid
    :return:
    """
    import random

    # load masks
    temp_list = ['Chi_0_Mask.xml', 'Chi_10_Mask.xml',
                 'Chi_20_Mask.xml', 'Chi_30_Mask.xml', 'NegZ_Mask.xml']
    mask_xml_list = [os.path.join('tests/testdata/masks', xml_name) for xml_name in temp_list]

    # Init setup for instrument geometry
    pixel_length = 2048

    # Determine PyRS root and load data
    pyrs_root_dir = os.path.abspath('.')
    test_data_dir = os.path.join(pyrs_root_dir, 'tests/testdata/')
    assert os.path.exists(test_data_dir), 'Test data directory {} does not exist'.format(test_data_dir)

    # Load data file and set up IDF
    if pixel_length == 2048:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.tif')
        two_theta = 35.
        hb2b_ws_name, hb2b_count_vec = load_data_from_tif(test_file_name, pixel_length)

        num_rows = 2048
        num_columns = 2048
        pixel_size_x = 0.00020
        pixel_size_y = 0.00020
        idf_name = 'XRay_Definition_2K.xml'

    else:
        raise NotImplementedError('Masking only support 2048 x 2048 case (not {})'.format(pixel_length))

    # Masking
    masking_list = list()   # tuple: mask array and mask workspace
    if pixel_length == 2048:
        for mask_xml in mask_xml_list:
            if 'Chi_0' in mask_xml:
                is_mask = True
                print('mask {} with is_mask = True'.format(mask_xml))
            else:
                is_mask = False
            mask_array, mask_ws_name = create_mask(mask_xml, pixel_length ** 2, is_mask)
            masking_list.append((mask_array, mask_ws_name))
    else:
        raise NotImplementedError('Masking only support 2048 x 2048 case (not {})'.format(pixel_length))

    # create HB2B-builder (python)
    hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)
    # set up IDF (Mantid)
    idf_name = os.path.join('tests/testdata/', idf_name)

    # Build and load instrument
    random.seed(1)
    arm_length = 0.416 + (random.random() - 0.5) * 2.0

    # calibration
    rot_x_flip = 2.0 * (random.random() - 0.5) * 2.0
    rot_y_flip = 2.0 * (random.random() - 0.5) * 2.0
    rot_z_spin = 2.0 * (random.random() - 0.5) * 2.0

    center_shift_x = 1.E-3 * (random.random() - 0.5) * 2.0
    center_shift_y = 1.E-3 * (random.random() - 0.5) * 2.0

    hb2b_pixel_matrix = load_instrument(hb2b_builder, arm_length, two_theta,
                                        center_shift_x, center_shift_y,
                                        rot_x_flip, rot_y_flip, rot_z_spin,
                                        hb2b_ws_name, idf_name, pixel_length)

    for mask_index in [0]:
        mask_array, mask_ws_name = masking_list[mask_index]
        reduce_to_2theta(hb2b_builder, hb2b_pixel_matrix, hb2b_ws_name, hb2b_count_vec,
                         mask_array, mask_ws_name, num_bins=2500)

        # reduce data
        reduce_to_2theta(hb2b_builder, hb2b_pixel_matrix, hb2b_ws_name, hb2b_count_vec,
                         mask_array, mask_ws_name, num_bins=2500)
    # END-FOR

    return


if __name__ == '__main__':
    # test_main()
    main('whatever')
