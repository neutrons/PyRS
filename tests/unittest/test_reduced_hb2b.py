#!/usr/bin/python
# Test Reduce HB2B data with scripts built upon numpy arrays
from pyrs.core import pyrscore
from pyrs.core import reduce_hb2b_pyrs
import os
import matplotlib.pyplot as plt
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose, AddSampleLog, LoadInstrument, ConvertSpectrumAxis, ResampleX
from mantid.api import AnalysisDataService as mtd


def build_instrument(hb2b_builder, arm_length=0.95, two_theta=0., center_shift_x=0., center_shift_y=0.,
                     rot_x_flip=0., rot_y_flip=0., rot_z_spin=0., raw_data_ws_name=None, idf_name=None):
    """
    :param arm_length: full arm length: engineered = 0.95 m
    :param two_theta: 2theta in sample log (instrument definition). It is opposite direction to Mantid coordinate
    :return:
    """
    pixel_matrix = hb2b_builder.build_instrument(arm_length=arm_length, two_theta=-two_theta,
                                                 center_shift_x=center_shift_x, center_shift_y=center_shift_y,
                                                 rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip, rot_z_spin=0.)

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
    AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltay', LogText='{}'.format(center_shift_y),
                 LogType='Number Series', LogUnit='meter',
                 NumberType='Double')

    # cal::roty
    AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::roty', LogText='{}'.format(-two_theta - rot_y_flip),
                 LogType='Number Series', LogUnit='degree',
                 NumberType='Double')

    # cal::flip
    AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::flip', LogText='{}'.format(rot_x_flip),
                 LogType='Number Series', LogUnit='degree',
                 NumberType='Double')

    # cal::spin
    AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::spin', LogText='{}'.format(rot_z_spin),
                 LogType='Number Series', LogUnit='degree',
                 NumberType='Double')

    LoadInstrument(Workspace=raw_data_ws_name,
                   Filename=idf_name,
                   InstrumentName='HB2B', RewriteSpectraMap='True')

    workspace = mtd[raw_data_ws_name]

    # test 5 spots: (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
    for index_i, index_j in [(0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)]:
        print ('PyRS:   ', pixel_matrix[index_i, index_j])
        print ('Mantid: ', workspace.getDetector(index_i + index_j * 1024).getPos())  # column major
        pos_python = pixel_matrix[index_i, index_j]
        pos_mantid = workspace.getDetector(index_i + 1024 * index_j).getPos()
        for i in range(3):
            print ('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
                   ''.format(i, float(pos_python[i]), float(pos_mantid[i]),
                             float(pos_python[i] - pos_mantid[i])))
        # END-FOR
    # END-FOR

    return pixel_matrix


def test_with_mantid():
    """
    test PyRS reduction with Mantid
    :return:
    """
    # Determine PyRS root and load data
    pyrs_root_dir = os.path.abspath('.')
    test_data_dir = os.path.join(pyrs_root_dir, 'tests/testdata/')
    assert os.path.exists(test_data_dir), 'Test data directory {} does not exist'.format(test_data_dir)

    # set up data file and IDF
    if True:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.bin')
        two_theta = 35.

    elif False:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_0deg-00000_Rotated.bin')
        two_theta = 0.

    else:
        raise NotImplementedError('No test file given!')

    # Load data
    if True:
        # load data from SPICE binary
        raw_data_ws_name = os.path.basename(test_file_name).split('.')[0]
        LoadSpiceXML2DDet(Filename=test_file_name, OutputWorkspace=raw_data_ws_name, DetectorGeometry='0,0',
                          LoadInstrument=False)
        # get data Y (counts) for PyRS reduction
        pyrs_raw_name = '{}_pyrs'.format(raw_data_ws_name)
        Transpose(InputWorkspace=raw_data_ws_name, OutputWorkspace=pyrs_raw_name)
    else:
        raise NotImplementedError('Next: Load data from image file')

    # Instrument factor
    wavelength_kv = 1.E5  # kev
    wavelength = 1.296  # A

    Beam_Center_X = 0.000805
    Beam_Center_Y = -0.006026

    xray_1k = True
    xray_2k = False
    hb2b_1k = False

    num_bins = 1000

    if xray_1k:
        # numpy array based script
        # build instrument
        num_rows = 1024
        num_columns = 1024
        pixel_size_x = 0.00040
        pixel_size_y = 0.00040
        arm_length = 0.416

        rot_x_flip = 0.
        rot_y_flip = 0.
        rot_z_spin = 0.

        idf_name = os.path.join(test_data_dir, 'Xray_HB2B_1K.xml')

        hb2b_builder = reduce_hb2b_pyrs.PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)

    else:
        raise NotImplementedError('No instrument configuration given!')

    # Load instrument and convert to 2Theta
    hb2b_pixel = build_instrument(hb2b_builder, arm_length, two_theta, Beam_Center_X, Beam_Center_Y,
                                  rot_x_flip, rot_y_flip, rot_z_spin, raw_data_ws_name, idf_name=idf_name)

    # reduce data
    reduce_to_2theta(hb2b_builder, hb2b_pixel, raw_data_ws_name, pyrs_raw_name)

    return


def reduce_to_2theta(hb2b_builder, pixel_matrix, hb2b_data_ws_name, pyrs_raw_name, num_bins=1000):

    pyrs_raw_ws = mtd[pyrs_raw_name]
    bin_edgets, histogram = hb2b_builder.reduce_to_2theta_histogram(pixel_matrix, pyrs_raw_ws.readY(0), num_bins)
    plt.plot(bin_edgets[:-1], histogram, color='red')

    # Mantid
    # transfer to 2theta for data
    two_theta_ws_name = '{}_2theta'.format(hb2b_data_ws_name)

    # this is for test only!
    ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta',
                        OrderAxis=False)
    Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
    two_theta_ws = mtd[two_theta_ws_name]
    for i in range(10):
        print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))
    for i in range(10010, 10020):
        print ('{}: x = {}, y = {}'.format(i, two_theta_ws.readX(0)[i], two_theta_ws.readY(0)[i]))

    ConvertSpectrumAxis(InputWorkspace=hb2b_data_ws_name, OutputWorkspace=two_theta_ws_name, Target='Theta')
    Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name)
    # final:
    mantid_reduced_name = '{}_mtd_reduced'
    ResampleX(InputWorkspace=two_theta_ws_name, OutputWorkspace=mantid_reduced_name,
              NumberBins=num_bins, PreserveEvents=False)
    mantid_ws = mtd[mantid_reduced_name]
    plt.plot(mantid_ws.readX(0), mantid_ws.readY(0), color='blue')

    plt.show()

    print ('compare bin edge:')
    print (bin_edgets[:10])
    print (mantid_ws.readX(0)[:10])

    return


def test_main():
    """
    test main
    :return:
    """
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


if __name__ == '__main__':
    # test_main()
    test_with_mantid()
