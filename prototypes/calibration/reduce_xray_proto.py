# TODO - ASAP - A script to reduce HB2B data 
# NOTE : script is for prototyping with MantidPlot
import os

def print_position(workspace):
    det_id_list = [0, 1023, 1023*1024, 1024*1024-1, 511*1024+511]
    # Lower left, upper left, lower right, upper right, lower-left-middle
    for det_id in det_id_list:
        print workspace.getDetector(det_id).getPos()

    return


def convert_to_2theta(ws_name):
    """
    """
    num_bins = 1000
    
    # duplicate for vanadium
    vanadium = CloneWorkspace(InputWorkspace=ws_name, OutputWorkspace='vanadium')
   
    # transfer to 2theta for data 
    ConvertSpectrumAxis(InputWorkspace=ws_name, OutputWorkspace=ws_name, Target='Theta')
    Transpose(InputWorkspace=ws_name, OutputWorkspace=ws_name)
    ResampleX(InputWorkspace=ws_name, OutputWorkspace=ws_name, NumberBins=num_bins, PreserveEvents=False)

    # vanadium: set to 1 for now
    for iws in range(vanadium.getNumberHistograms()):
        vanadium.dataY(iws)[0] = 1.
    ConvertSpectrumAxis(InputWorkspace='vanadium', OutputWorkspace='vanadium', Target='Theta')
    Transpose(InputWorkspace='vanadium', OutputWorkspace='vanadium')
    ResampleX(InputWorkspace='vanadium', OutputWorkspace='vanadium', NumberBins=num_bins, PreserveEvents=False)
    
    Divide(LHSWorkspace=ws_name, RHSWorkspace='vanadium', OutputWorkspace='reduced')
  
    return

#---------------------------------------------------------------------------    
wavelength = 1.E5 # kev
wavelength = 1.296  # A
Beam_Center_X = 0.000805
Beam_Center_Y = -0.006026


def test_rotate_2theta(idf_name):
    """
    """
    output_ws_name = 'rotate_2theta'
    CloneWorkspace(InputWorkspace='hb2b', OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='0.0', LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='-30.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='{}'.format(Beam_Center_X), LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='{}'.format(-Beam_Center_Y), LogType='Number Series', LogUnit='meter', NumberType='Double')
    
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='0.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='-30.0', LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='90.0', LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, RewriteSpectraMap=True)
    # print output
    print_position(mtd[output_ws_name])

    return output_ws_name
    


# Set up
# data file
pyrs_root = '/home/wzz/Projects/PyRS/'
test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.bin'
hb2b_file_name = os.path.join(pyrs_root, test_file_name)

# instrument geometry
idf_name = 'XRay_Definition_1K.xml'

# Load data
if True:
    LoadSpiceXML2DDet(Filename=hb2b_file_name, OutputWorkspace='hb2b', DetectorGeometry='0,0', LoadInstrument=False)

ws_name = test_rotate_2theta(idf_name)
convert_to_2theta(ws_name)
# Similar to WANDPowderReduction
