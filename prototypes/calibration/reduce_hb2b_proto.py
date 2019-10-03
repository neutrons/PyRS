# TODO - ASAP - A script to reduce HB2B data
# NOTE : script is for prototyping with MantidPlot
import os

def print_position(workspace):
    det_id_list = [0, 1023, 1023*1024, 1024*1024-1, 511*1024+511]
    # Lower left, upper left, lower right, upper right, lower-left-middle
    for det_id in det_id_list:
        print workspace.getDetector(det_id).getPos()

    return

def test_direct_beam(idf_name):
    """
    """
    output_ws_name = 'directbeam'
    # at z = r + 0.04
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.04', LogType='Number Series', LogUnit='meter', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return


def test_rotate_2theta(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='-35.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='-35.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return


def test_rotate_2theta_45(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='-45.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='-45.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return


def test_no_2theta_rotation_flip_5(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='5.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return


def test_rotate_2theta_90_flip_5(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='-90.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='5.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='-90.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return

def test_rotate_2theta_45_flip_5(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='-45.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='5.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='-45.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return


def convert_to_2theta(ws_name):
    """
    """
    num_bins = 100

    ConvertSpectrumAxis(InputWorkspace=ws_name, OutputWorkspace=ws_name, Target='Theta')
    Transpose(InputWorkspace=ws_name, OutputWorkspace=ws_name)

    ResampleX(InputWorkspace=ws_name, OutputWorkspace=ws_name, NumberBins=num_bins, PreserveEvents=False)

    # vanadium
    vanadium = CloneWorkspace(InputWorkspace='2theta_position', OutputWorkspace='vanadium')
    for iws in range(vanadium.getNumberHistograms()):
        vanadium.dataY(iws)[0] = 1.
    ConvertSpectrumAxis(InputWorkspace='vanadium', OutputWorkspace='vanadium', Target='Theta')
    Transpose(InputWorkspace='vanadium', OutputWorkspace='vanadium')
    ResampleX(InputWorkspace='vanadium', OutputWorkspace='vanadium', NumberBins=num_bins, PreserveEvents=False)

    Divide(LHSWorkspace='data2theta2', RHSWorkspace='vanadium', OutputWorkspace='reduced')


# Set up
# data file
pyrs_root = '/home/wzz/Projects/PyRS/'
test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.bin'
hb2b_file_name = os.path.join(pyrs_root, test_file_name)

# instrument geometry
idf_name = 'HB2B_Definition_Prototype.xml'
idf_name_v3 = 'HB2B_Definition_v3.xml'
idf_name_v4 = 'HB2B_Definition_v4.xml'


idf_name_v0 = 'HB2B_Definition_Prototype.xml'

# Load data
if True:
    LoadSpiceXML2DDet(Filename=hb2b_file_name, OutputWorkspace='hb2b', DetectorGeometry='0,0', LoadInstrument=False)

# test_direct_beam(idf_name)
test_rotate_2theta(idf_name_v4)
# test_rotate_2theta_45(idf_name)
#test_no_2theta_rotation_flip_5(idf_name_v4)
#print ('\n')
#test_rotate_2theta_90_flip_5(idf_name_v4)
#print ('\n')
#test_rotate_2theta_45_flip_5(idf_name_v4)
# Similar to WANDPowderReduction
