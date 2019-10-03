# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
import os
import sys
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyqr.utilities import calibration_file_io
from pyrs.core import reduction_manager
from pyrs.core import mask_util
import numpy
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping


# TODO - NIGHT - Use pure-python reduction to replace Mantid reduction


def print_position(workspace):
    det_id_list = [0, 1023, 1023*1024, 1024*1024-1, 511*1024+511]
    # Lower left, upper left, lower right, upper right, lower-left-middle
    for det_id in det_id_list:
        print workspace.getDetector(det_id).getPos()

    return

def convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=None):
    
    if Mask is not None:
        CloneWorkspace(InputWorkspace=vanadium, OutputWorkspace='V_Clone')
        MaskDetectors('V_Clone', MaskedWorkspace=Mask)
        ConvertSpectrumAxis(InputWorkspace='V_Clone', OutputWorkspace=vanadium+'_Reduce', Target='Theta')
    else:
        ConvertSpectrumAxis(InputWorkspace=vanadium, OutputWorkspace=vanadium+'_Reduce', Target='Theta')
    
    Transpose(InputWorkspace=vanadium+'_Reduce', OutputWorkspace=vanadium+'_Reduce')
    ResampleX(InputWorkspace=vanadium+'_Reduce', OutputWorkspace=vanadium+'_Reduce', XMin=15, XMax=55, NumberBins=num_bins, PreserveEvents=False)
    
    return vanadium+'_Reduce'

def convert_to_2theta(ws_name, vanadium):
    """
    """
    num_bins = 1900
       
    # transfer to 2theta for data 
    ConvertSpectrumAxis(InputWorkspace=ws_name, OutputWorkspace=ws_name+'_reduced', Target='Theta')
    Transpose(InputWorkspace=ws_name+'_reduced', OutputWorkspace=ws_name+'_reduced')
    ResampleX(InputWorkspace=ws_name+'_reduced', OutputWorkspace=ws_name+'_reduced', XMin=15, XMax=55, NumberBins=num_bins, PreserveEvents=False)
  
    Divide(LHSWorkspace=ws_name+'_reduced', RHSWorkspace=vanadium, OutputWorkspace=ws_name+'_normalized')
  
    return

#----------------s-----------------------------------------------------------    
wavelength = 1.E5 # kev
wavelength = 1.296  # A

def test_rotate_2theta(idf_name, InputWorkspace, output_ws_name, DetDistance = 0.0, DetTTH = 35.0, DetTTH_Shift=0, Beam_Center_X = -0.008, Beam_Center_Y = 0.0, DetFlit = 0, DetSpin = 0):
    """
    """
    CloneWorkspace(InputWorkspace=InputWorkspace, OutputWorkspace=output_ws_name)
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::arm', LogText='{}'.format(DetDistance), LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::2theta', LogText='{}'.format(DetTTH), LogType='Number Series', LogUnit='degree', NumberType='Double')

    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltax', LogText='{}'.format(Beam_Center_X), LogType='Number Series', LogUnit='meter', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::deltay', LogText='{}'.format(-Beam_Center_Y), LogType='Number Series', LogUnit='meter', NumberType='Double')
    
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::flip', LogText='{}'.format(DetFlit), LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::roty', LogText='{}'.format(-1*DetTTH+DetTTH_Shift), LogType='Number Series', LogUnit='degree', NumberType='Double')
    AddSampleLog(Workspace=output_ws_name,
                 LogName='cal::spin', LogText='{}'.format(DetSpin), LogType='Number Series', LogUnit='degree', NumberType='Double')

    # Load instrument
    LoadInstrument(Workspace=output_ws_name, Filename=idf_name, MonitorList='0,1', RewriteSpectraMap=True)

    return output_ws_name


# ------------------   PRYS  -----------------

def create_instrument(test_data_file, calibrated, pixel_number):
    """
    Create instruments: PyRS and Mantid
    :param calibrated:
    :param pixel_number:
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

    if False:
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
    engine = reduction_manager.HB2BReductionManager()
    test_data_id, two_the_tmp = engine.load_data(data_file_name=test_data_file,
                                                 target_dimension=pixel_number, load_to_workspace=True)

    # load instrument
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument)
    pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                  rot_x_flip, rot_y_flip, rot_z_spin)

    mantid_reducer = None
    # mantid_reducer = reduce_hb2b_mtd.MantidHB2BReduction()
    # data_ws_name = engine.get_raw_data(test_data_id, is_workspace=True)
    # mantid_reducer.set_workspace(data_ws_name)
    # mantid_reducer.load_instrument(two_theta, xray_idf_name, test_calibration)

    return engine, pyrs_reducer, mantid_reducer

# ------------------   Main -----------------

# Set up
# data file
pyrs_root = './tests/testdata/'
# test_file_name = 'LaB6_10kev_35deg-00004.tif'  # None rotated image file  FIXME - So far doesn't work Issue #51
test_file_name = 'LaB6_10kev_35deg-00004_Rotated.tif'
xray_2k_instrument_file = 'tests/testdata/xray_data/XRay_Definition_2K.txt'

# Temporarily disabled: # FIXME Vanadium = 'Vanadium.bin'
#test_file_name = 'tests/testdata/BNT_7BT_2KNN_6kV_mm-03425-001.xml'

hb2b_file_name = os.path.join(pyrs_root, test_file_name)
# Temporarily disabled: # FIXME hb2b_Vfile_name = os.path.join(pyrs_root, Vanadium)
assert os.path.exists(hb2b_file_name), hb2b_file_name
# Temporarily disabled: # FIXME assert os.path.exists(hb2b_Vfile_name), hb2b_Vfile_name

# instrument geometry
idf_name = os.path.join(pyrs_root, 'XRay_Definition_2K.xml')
assert os.path.exists(idf_name)
#idf_name = 'CG2_Definition.xml'

import os 
os.system('cp ' + idf_name + ' ~/.mantid/instrument/HB2B_Definition.xml')
# Load data
print ('Loading {} ....'.format(hb2b_file_name))
engine, pyrs_reducer, mantid_reducer = create_instrument(hb2b_file_name, False, 2048)
print ('.... Done')
if False:
    LoadSpice2D(Filename=hb2b_file_name, OutputWorkspace='hb2b')
    LoadSpice2D(Filename=hb2b_Vfile_name, OutputWorkspace='hb2b_V')

# Reduce: convert to 2theta

if False:
    ws_name = test_rotate_2theta(idf_name, 'hb2b', 'hb2b_rotate')
    vanadium = test_rotate_2theta(idf_name, 'hb2b_V', 'hb2b_V_rotate')
    vanadium_line = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=None)
    convert_to_2theta(ws_name, vanadium_line)


NegMask = 'NegMask'
PosMask = 'PosMask'
ZeMask  = 'ZeroMask'
p10Mask = 'p10Mask'
p20Mask = 'p20Mask'
p30Mask = 'p30Mask'
n30Mask = 'n30Mask'

if True:
    for solid_angle in [30, -30]:  # shall be extended to 0, +/- 10, +/- 20, +/-30
        mask_vec, mask_2theta, note = mask_util.load_pyrs_mask('tests/testdata/masks/Chi_30.hdf5')

else:
    LoadMask(Instrument='HB2B', InputFile='/SNS/users/hcf/HFIR_TESTING/NegZ_Mask.xml', OutputWorkspace=NegMask)
    LoadMask(Instrument='HB2B', InputFile='/SNS/users/hcf/HFIR_TESTING/Chi_30_Mask.xml', OutputWorkspace=p30Mask)

    InvertMask(InputWorkspace=NegMask, OutputWorkspace=PosMask)
    InvertMask(InputWorkspace=p30Mask, OutputWorkspace=p30Mask)

    CloneWorkspace(InputWorkspace=p30Mask, OutputWorkspace=n30Mask)

    MaskDetectors(Workspace=p30Mask, MaskedWorkspace=PosMask)
    MaskDetectors(Workspace=n30Mask, MaskedWorkspace=NegMask)

    WS_p30deg = test_rotate_2theta(idf_name, 'hb2b','hb2b_rotate_p30deg')
    WS_n30deg = test_rotate_2theta(idf_name, 'hb2b','hb2b_rotate_n30deg')

    MaskDetectors(WS_p30deg, MaskedWorkspace=p30Mask)
    MaskDetectors(WS_n30deg, MaskedWorkspace=n30Mask)

vanadium_P30 = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=p30Mask)
vanadium_N30 = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=n30Mask)

def MinDifference(x):

    
    WS_p30deg_Rot = test_rotate_2theta(idf_name, WS_p30deg,'hb2b_rotate_p30deg_Rot', DetDistance = 0.0, DetTTH = 35.0, DetTTH_Shift = 0.0, Beam_Center_X = -0.002, Beam_Center_Y = -0.007, DetFlit = x[0], DetSpin = 0.0)
    WS_n30deg_Rot = test_rotate_2theta(idf_name, WS_n30deg,'hb2b_rotate_n30deg_Rot', DetDistance = 0.0, DetTTH = 35.0, DetTTH_Shift = 0.0, Beam_Center_X = -0.002, Beam_Center_Y = -0.007, DetFlit = x[0], DetSpin = 0.0)

    convert_to_2theta(WS_p30deg_Rot, vanadium_P30)
    convert_to_2theta(WS_n30deg_Rot, vanadium_N30)
    
    N30_Fit = 'Fit_N30'
    P30_Fit = 'Fit_P30'

    FitPeaks(InputWorkspace='hb2b_rotate_n30deg_Rot_reduced', OutputWorkspace=N30_Fit, StartWorkspaceIndex=0, StopWorkspaceIndex=0,
             PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55', FittedPeaksWorkspace='hb2b_rotate_n30deg_reduced_Output', OutputPeakParametersWorkspace='hb2b_rotate_n30deg_reduced_FITS', OutputParameterFitErrorsWorkspace='hb2b_rotate_n30deg_reduced_Errors')
    
    FitPeaks(InputWorkspace='hb2b_rotate_p30deg_Rot_reduced', OutputWorkspace=P30_Fit, StartWorkspaceIndex=0, StopWorkspaceIndex=0,
             PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5', FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_p30deg_reduced_Output', OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS', OutputParameterFitErrorsWorkspace='hb2b_rotate_p23deg_reduced_Errors')

    Error3 = (mtd[N30_Fit].extractY()[0] - mtd[P30_Fit].extractY()[0])
    
    print x
    print  Error3*Error3
    return (Error3*Error3) * 1e8


x0 = [0,0,-0.002,-0.007,-0.922,0]

x0 = [-1.]
DE_Res = leastsq(MinDifference, x0, xtol = 1e-15, maxfev=3000, epsfcn=1e-2)

DD           = 0.0
D_Shift    = 0
Center_X = -0.002
Center_Y = -0.007
Flip          = -1
Spin        = 0.0

DE_Res = leastsq(MinDifference, [-1], xtol = 1e-15, maxfev=3000)


WS_p10deg_Rot = test_rotate_2theta(idf_name, WS_p10deg,'hb2b_rotate_p10deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
WS_p20deg_Rot = test_rotate_2theta(idf_name, WS_p20deg,'hb2b_rotate_p20deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
WS_p30deg_Rot = test_rotate_2theta(idf_name, WS_p30deg,'hb2b_rotate_p30deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
WS_n10deg_Rot = test_rotate_2theta(idf_name, WS_n10deg,'hb2b_rotate_n10deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
WS_n20deg_Rot = test_rotate_2theta(idf_name, WS_n20deg,'hb2b_rotate_n20deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
WS_n30deg_Rot = test_rotate_2theta(idf_name, WS_n30deg,'hb2b_rotate_n30deg_Rot', DetDistance = DD, DetTTH = 35.0, DetTTH_Shift = D_Shift, Beam_Center_X = Center_X, Beam_Center_Y = Center_Y, DetFlit = Flip, DetSpin = Spin)
    
convert_to_2theta(WS_p10deg_Rot, vanadium_P10)
convert_to_2theta(WS_p20deg_Rot, vanadium_P20)
convert_to_2theta(WS_p30deg_Rot, vanadium_P30)
convert_to_2theta(WS_n10deg_Rot, vanadium_N10)
convert_to_2theta(WS_n20deg_Rot, vanadium_N20)
convert_to_2theta(WS_n30deg_Rot, vanadium_N30)
    
