#!/usr/bin/python
# a script to read in a TIFF file, mask it optionally and reduce to 2theta
# output shall be in ProcessedNeXus file for mantid to import
from pyrs.core import reduce_hb2b_pyrs
from pyrs.utilities import file_utilities
import os
import matplotlib.pyplot as plt
from mantid.simpleapi import Transpose, AddSampleLog, LoadInstrument, ConvertSpectrumAxis, ResampleX
from mantid.simpleapi import CreateWorkspace, Multiply, SaveNexusProcessed
from mantid.api import AnalysisDataService as mtd
from PIL import Image
import numpy as np


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
                                                 rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip, rot_z_spin=0.)

    if True:
        # using Mantid
        # check
        assert raw_data_ws_name is not None, 'data ws error'
        assert idf_name is not None, 'IDF cannot be None'
        assert pixel_number is not None, 'Pixel number to be given'

        # set up sample logs
        # cal::arm
        # FIXME - No arm length calibration
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::arm', LogText='{}'.format(0.),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::2theta
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::2theta', LogText='{}'.format(-two_theta),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')
        #
        # cal::deltax
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltax', LogText='{}'.format(center_shift_x),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::deltay
        print('Shift Y = {}'.format(center_shift_y))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltay', LogText='{}'.format(center_shift_y),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::roty
        print('Rotation Y = {}'.format(rot_y_flip))
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::roty', LogText='{}'.format(-two_theta - rot_y_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::flip
        AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::flip', LogText='{}'.format(rot_x_flip),
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
            for i in range(3):
                print('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
                      ''.format(i, float(pos_python[i]), float(pos_mantid[i]),
                                float(pos_python[i] - pos_mantid[i])))
            # END-FOR
        # END-FOR
    # END-IF

    return pixel_matrix


def reduce_to_2theta(hb2b_builder, pixel_matrix, hb2b_data_ws_name, counts_array, mask_vec, mask_ws_name,
                     num_bins=1000):
    """
    Reduce to 2theta with Masks
    :param hb2b_builder:
    :param pixel_matrix:
    :param hb2b_data_ws_name:
    :param counts_array:
    :param mask_vec:
    :param num_bins:
    :return:
    """
    # reduce by PyRS
    if False:
        pyrs_raw_ws = mtd[pyrs_raw_name]
        vec_counts = pyrs_raw_ws.readY(0)
    else:
        vec_counts = counts_array.astype('float64')

    # mask
    if mask_vec is not None:
        print(mask_vec.dtype)
        vec_counts.astype('float64')
        mask_vec.astype('float64')
        vec_counts *= mask_vec
    # reduce
    bin_edgets, histogram = hb2b_builder.reduce_to_2theta_histogram(pixel_matrix, vec_counts, num_bins)

    # create workspace
    pyrs_reduced_name = '{}_pyrs_reduced'.format(hb2b_data_ws_name)
    CreateWorkspace(DataX=bin_edgets, DataY=histogram, NSpec=1, OutputWorkspace=pyrs_reduced_name)
    SaveNexusProcessed(InputWorkspace=pyrs_reduced_name, Filename='{}.nxs'.format(pyrs_reduced_name),
                       Title='PyRS reduced: {}'.format(hb2b_data_ws_name))

    if True:
        # Mantid
        # transfer to 2theta for data
        two_theta_ws_name = '{}_2theta'.format(hb2b_data_ws_name)

        # Mask
        if mask_ws_name:
            # Multiply by masking workspace
            masked_ws_name = '{}_masked'.format(hb2b_data_ws_name)
            Multiply(LHSWorkspace=hb2b_data_ws_name, RHSWorkspace=mask_ws_name,
                     OutputWorkspace=masked_ws_name, ClearRHSWorkspace=False)
            hb2b_data_ws_name = masked_ws_name
            SaveNexusProcessed(InputWorkspace=hb2b_data_ws_name, Filename='{}_raw.nxs'.format(hb2b_data_ws_name))
        # END-IF

        # # this is for test only!
        # ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta',
        #                     OrderAxis=False)
        # Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
        # two_theta_ws = mtd[two_theta_ws_name]
        # for i in range(10):
        #     print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))
        # for i in range(10010, 10020):
        #     print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))

        ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta')
        Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
        # final:
        mantid_reduced_name = '{}_mtd_reduced'.format(hb2b_data_ws_name)
        ResampleX(InputWorkspace=two_theta_ws_name, OutputWorkspace=mantid_reduced_name,
                  NumberBins=num_bins, PreserveEvents=False)
        mantid_ws = mtd[mantid_reduced_name]

        SaveNexusProcessed(InputWorkspace=mantid_reduced_name, Filename='{}.nxs'.format(mantid_reduced_name),
                           Title='Mantid reduced: {}'.format(hb2b_data_ws_name))

        plt.plot(mantid_ws.readX(0), mantid_ws.readY(0), color='blue', mark='o')

    # END-IF

    plt.plot(bin_edgets[:-1], histogram, color='red')

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


def load_data_from_tif(raw_tiff_name, pixel_size=2048, rotate=True):
    """
    Load data from TIFF
    :param raw_tiff_name:
    :param pixel_size
    :param rotate:
    :return:
    """
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


def main(argv):
    """
    main args
    :param argv:
    :return:
    """
    # init setup
    image_file = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'
    two_theta = 0.  # it is 35 degree... using 0 for debugging

    # load masks
    temp_list = ['Chi_0_Mask.xml', 'Chi_10_Mask.xml',
                 'Chi_20_Mask.xml', 'Chi_30_Mask.xml', 'NegZ_Mask.xml']
    mask_xml_list = [os.path.join('tests/testdata/masks', xml_name) for xml_name in temp_list]

    # Now it is the setup for real reduction
    pixel_length = 1024

    # Load raw data
    hb2b_ws_name, hb2b_count_vec = load_data_from_tif(image_file, pixel_length)

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

    # create instrument
    if pixel_length == 2048:
        num_rows = 2048
        num_columns = 2048
        pixel_size_x = 0.00020
        pixel_size_y = 0.00020
        idf_name = 'Xray_HB2B_2K.xml'
    elif pixel_length == 1024:
        num_rows = 2048/2
        num_columns = 2048/2
        pixel_size_x = 0.00020*2
        pixel_size_y = 0.00020*2
        idf_name = 'Xray_HB2B_1K.xml'
        idf_name = 'XRay_Definition_1K.xml'
    else:
        raise RuntimeError('Wrong setup')

    hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)
    idf_name = os.path.join('tests/testdata/', idf_name)

    # load instrument
    arm_length = 0.416
    # calibration
    rot_x_flip = 0.  # 0.01
    rot_y_flip = 30.  # with trouble -0.142
    rot_z_spin = 0.  # 0.98  # still no good

    center_shift_x = 0.  # 0.001
    center_shift_y = 0.  # -0.02

    hb2b_pixel_matrix = load_instrument(hb2b_builder, arm_length, two_theta,
                                        center_shift_x, center_shift_y,
                                        rot_x_flip, rot_y_flip, rot_z_spin,
                                        hb2b_ws_name, idf_name, pixel_length)

    for mask_index in [0]:
        mask_array, mask_ws_name = masking_list[mask_index]
        reduce_to_2theta(hb2b_builder, hb2b_pixel_matrix, hb2b_ws_name, hb2b_count_vec,
                         mask_array, mask_ws_name, num_bins=2500)


if __name__ == '__main__':
    main(['do it'])
