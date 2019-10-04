# NOTE : script is for prototyping with MantidPlot
import os
import time
from mantid.simpleapi import LoadSpiceXML2DDet, CloneWorkspace, AddSampleLog, LoadInstrument, ConvertSpectrumAxis, Transpose, ResampleX, Divide
from mantid.api import AnalysisDataService as mtd
from matplotlib import pyplot as plt


def test_main():
    """
    :return:
    """
    # Determine PyRS root
    user_root_dir = os.path.expanduser('~')
    pyrs_root = os.path.join(user_root_dir, 'Projects/PyRS')
    test_data_dir = os.path.join(pyrs_root, 'tests/testdata/')

    # Instrument factor
    row_col_size = 1024
    wavelength_kv = 1.E5  # kev
    wavelength = 1.296  # A
    Beam_Center_X = 0.000805
    Beam_Center_Y = -0.006026

    # set up data file and IDF
    if True:
        # Test 1024
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_35deg-00004_Rotated.bin')
        two_theta = 35.

    if False:
        test_file_name = os.path.join(test_data_dir, 'LaB6_10kev_0deg-00000_Rotated.bin')
        two_theta = 0.

    hb2b_file_name = os.path.join(pyrs_root, test_file_name)
    # instrument geometry
    idf_name = os.path.join(test_data_dir, 'XRay_Definition_1K.xml')

    # instrument calibration setup
    cal_shift_x = Beam_Center_X
    cal_shift_y = Beam_Center_Y

    # Load data
    raw_data_ws_name = os.path.basename(hb2b_file_name).split('.')[0]
    LoadSpiceXML2DDet(Filename=hb2b_file_name, OutputWorkspace=raw_data_ws_name, DetectorGeometry='0,0',
                      LoadInstrument=False)

    # Set up instrument geometry parameter
    load_instrument(raw_data_ws_name, idf_name, two_theta, cal_shift_x, cal_shift_y)

    reduced_ws_name = convert_to_2theta(raw_data_ws_name)

    # plot
    reduced_ws = mtd[reduced_ws_name]
    vec_x = reduced_ws.readX(0)
    vec_y = reduced_ws.readY(0)
    plt.plot(vec_x, vec_y)

    non_norm_ws = mtd[raw_data_ws_name]
    vec_x_nn = non_norm_ws.readX(0)
    vec_y_nn = non_norm_ws.readY(0) / 1000.
    plt.plot(vec_x_nn, vec_y_nn)

    plt.show()

    return
# END


def load_instrument(raw_data_ws_name, idf_name, two_theta, cal_shift_x, cal_shift_y):
    """ Set up parameters to workspace and load instrument
    """
    # instrument position
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::2theta', LogText='{}'.format(-two_theta), LogType='Number Series',
                 LogUnit='degree', NumberType='Double')

    # calibration information
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::deltax', LogText='{}'.format(cal_shift_x), LogType='Number Series', LogUnit='meter',
                 NumberType='Double')
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::deltay', LogText='{}'.format(-cal_shift_y), LogType='Number Series', LogUnit='meter',
                 NumberType='Double')
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::flip', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::roty', LogText='{}'.format(-two_theta),
                 LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=raw_data_ws_name,
                 LogName='cal::spin', LogText='90.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=raw_data_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[raw_data_ws_name])

    return mtd[raw_data_ws_name]


def print_position(workspace, row_col_size=1024):
    """
    base on the 1024 x 1024 size
    :param workspace:
    :return:
    """
    det_id_list = [0, 1023, 1023*1024, 1024*1024-1, 511*1024+511]
    # Lower left, upper left, lower right, upper right, lower-left-middle
    for det_id in det_id_list:
        row_index = det_id / row_col_size
        col_index = det_id % row_col_size
        print 'Det {}, {} @ {}'.format(row_index, col_index, workspace.getDetector(det_id).getPos())

    return


def convert_to_2theta(ws_name, num_bins=1000):
    """
    """
    # duplicate for vanadium
    vanadium = CloneWorkspace(InputWorkspace=ws_name, OutputWorkspace='vanadium')

    # transfer to 2theta for data
    ConvertSpectrumAxis(InputWorkspace=ws_name, OutputWorkspace=ws_name, Target='Theta')
    Transpose(InputWorkspace=ws_name, OutputWorkspace=ws_name)
    ResampleX(InputWorkspace=ws_name, OutputWorkspace=ws_name, NumberBins=num_bins, PreserveEvents=False)

    # vanadium: set to 1 for now
    time_van_start = time.time()
    for iws in range(vanadium.getNumberHistograms()):
        vanadium.dataY(iws)[0] = 1.
    time_van_stop = time.time()
    ConvertSpectrumAxis(InputWorkspace='vanadium', OutputWorkspace='vanadium', Target='Theta')
    Transpose(InputWorkspace='vanadium', OutputWorkspace='vanadium')
    ResampleX(InputWorkspace='vanadium', OutputWorkspace='vanadium', NumberBins=num_bins, PreserveEvents=False)

    norm_ws_name = ws_name+'_normalized'
    Divide(LHSWorkspace=ws_name, RHSWorkspace='vanadium', OutputWorkspace=norm_ws_name)

    print('Create vanadium workspace : {} seconds'.format(time_van_stop - time_van_start))

    return norm_ws_name


# Test script
test_main()
