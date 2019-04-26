image_file = '/home/wzz/Projects/PyRS/tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'

from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image
import numpy as np
import pylab as plt

ImageData = Image.open(image_file)
#im = img_as_uint(np.array(ImageData))
io.use_plugin('freeimage')
Data = np.array(ImageData, dtype=np.int32)
print (Data.shape, type(Data), Data.min(), Data.max())
Data.astype(np.uint32)

# merge
DataR = Data[::2, ::2] + Data[::2, 1::2] + Data[1::2, ::2] + Data[1::2, 1::2]
print (DataR.shape, type(DataR))

DataR = DataR.reshape((1024*1024, ))
print (DataR.min())

CreateWorkspace(DataX=np.zeros((1024**2, )), DataY=DataR, DataE=np.sqrt(DataR), NSpec=1024**2, OutputWorkspace='from_tif', VerticalAxisUnit='SpectraNumber')
# Transpose(InputWorkspace='from_tif', OutputWorkspace='from_tif')

# TODO - ASAP - A script to reduce HB2B data 
# NOTE : script is for prototyping with MantidPlot
import os

def print_position(workspace):
    det_id_list = [0, 1023, 1023*1024, 1024*1024-1, 511*1024+511]
    # Lower left, upper left, lower right, upper right, lower-left-middle
    for det_id in det_id_list:
        print workspace.getDetector(det_id).getPos()

    return


def convert_to_2theta(ws_name, reduced_ws_name):
    """
    """
    num_bins = 1000
    
    # duplicate for vanadium
    vanadium = CloneWorkspace(InputWorkspace=ws_name, OutputWorkspace='vanadium')
   
    # transfer to 2theta for data 
    ws_name_theta1 = '{}_theta'.format(ws_name)
    ws_name_theta2 = '{}_theta_transpose'.format(ws_name)
    ws_name_resample = '{}_resample'.format(ws_name)
    ConvertSpectrumAxis(InputWorkspace=ws_name, OutputWorkspace=ws_name_theta1, Target='Theta')
    Transpose(InputWorkspace=ws_name_theta1, OutputWorkspace=ws_name_theta2)
    ResampleX(InputWorkspace=ws_name_theta2, OutputWorkspace=ws_name_resample, NumberBins=num_bins, PreserveEvents=False)

    # vanadium: set to 1 for now
    for iws in range(vanadium.getNumberHistograms()):
        vanadium.dataY(iws)[0] = 1.
    ConvertSpectrumAxis(InputWorkspace='vanadium', OutputWorkspace='vanadium', Target='Theta')
    Transpose(InputWorkspace='vanadium', OutputWorkspace='vanadium')
    ResampleX(InputWorkspace='vanadium', OutputWorkspace='vanadium', NumberBins=num_bins, PreserveEvents=False)
    
    Divide(LHSWorkspace=ws_name_resample, RHSWorkspace='vanadium', OutputWorkspace=reduced_ws_name)
  
    return

#---------------------------------------------------------------------------    
wavelength = 1.E5 # kev
wavelength = 1.296  # A
Beam_Center_X = 0.000805
Beam_Center_Y = -0.006026


def test_rotate_2theta(ws_name, idf_name):
    """
    """
    output_ws_name = ws_name
    CloneWorkspace(InputWorkspace=ws_name, OutputWorkspace=output_ws_name)
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

# ws_name = test_rotate_2theta(idf_name)
test_rotate_2theta(ws_name='from_tif', idf_name=idf_name)
convert_to_2theta(ws_name='from_tif', reduced_ws_name='tif_test')
# Similar to WANDPowderReduction
