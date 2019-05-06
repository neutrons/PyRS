# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py

# TODO - TODAY 0 - Use pure-python reduction to replace Mantid reduction

import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping


#----------------s-----------------------------------------------------------    
wavelength = 1.E5 # kev
wavelength = 1.296  # A


def MinDifference_Chris(x):
    WS_p30deg_Rot = test_rotate_2theta(idf_name, WS_p30deg, 'hb2b_rotate_p30deg_Rot', DetDistance=0.0, DetTTH=35.0,
                                       DetTTH_Shift=0.0, Beam_Center_X=-0.002, Beam_Center_Y=-0.007, DetFlit=x[0],
                                       DetSpin=0.0)
    WS_n30deg_Rot = test_rotate_2theta(idf_name, WS_n30deg, 'hb2b_rotate_n30deg_Rot', DetDistance=0.0, DetTTH=35.0,
                                       DetTTH_Shift=0.0, Beam_Center_X=-0.002, Beam_Center_Y=-0.007, DetFlit=x[0],
                                       DetSpin=0.0)

    convert_to_2theta(WS_p30deg_Rot, vanadium_P30)
    convert_to_2theta(WS_n30deg_Rot, vanadium_N30)

    N30_Fit = 'Fit_N30'
    P30_Fit = 'Fit_P30'

    FitPeaks(InputWorkspace='hb2b_rotate_n30deg_Rot_reduced', OutputWorkspace=N30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_n30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_n30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_n30deg_reduced_Errors')

    FitPeaks(InputWorkspace='hb2b_rotate_p30deg_Rot_reduced', OutputWorkspace=P30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_p30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_p23deg_reduced_Errors')

    Error3 = (mtd[N30_Fit].extractY()[0] - mtd[P30_Fit].extractY()[0])

    print x
    print  Error3 * Error3
    return (Error3 * Error3) * 1e8
# This is main!!!


def MinDifference(x):
    """ Cost function to align the peaks!
    :param x:
    :return:
    """
    if False:
        WS_p30deg_Rot = test_rotate_2theta(idf_name, WS_p30deg, 'hb2b_rotate_p30deg_Rot', DetDistance=0.0, DetTTH=35.0,
                                           DetTTH_Shift=0.0, Beam_Center_X=-0.002, Beam_Center_Y=-0.007, DetFlit=x[0],
                                           DetSpin=0.0)
        WS_n30deg_Rot = test_rotate_2theta(idf_name, WS_n30deg, 'hb2b_rotate_n30deg_Rot', DetDistance=0.0, DetTTH=35.0,
                                           DetTTH_Shift=0.0, Beam_Center_X=-0.002, Beam_Center_Y=-0.007, DetFlit=x[0],
                                           DetSpin=0.0)

        convert_to_2theta(WS_p30deg_Rot, vanadium_P30)
        convert_to_2theta(WS_n30deg_Rot, vanadium_N30)
    else:
        # TODO - TONIGHT 0 - From here!
        ws_positive_roi = convert_to_

    N30_Fit = 'Fit_N30'
    P30_Fit = 'Fit_P30'

    FitPeaks(InputWorkspace='hb2b_rotate_n30deg_Rot_reduced', OutputWorkspace=N30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_n30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_n30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_n30deg_reduced_Errors')

    FitPeaks(InputWorkspace='hb2b_rotate_p30deg_Rot_reduced', OutputWorkspace=P30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_p30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_p23deg_reduced_Errors')

    Error3 = (mtd[N30_Fit].extractY()[0] - mtd[P30_Fit].extractY()[0])

    print x
    print  Error3 * Error3
    return (Error3 * Error3) * 1e8
# This is main!!!


def main():
    import os

    # Set up
    # data file
    if False:
        # pyrs_root = '/SNS/users/hcf/HFIR_TESTING/'
        # test_file_name = 'LaB6_10kev_35deg-00004.xml'
        # Vanadium = 'Vanadium.xml'
        test_file_name = 'tests/testdata/BNT_7BT_2KNN_6kV_mm-03425-001.xml'
        pass
    else:
        test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'

    # instrument geometry
    if False:
        idf_name = os.path.join(pyrs_root, 'XRay_Definition_1K.xml')
    else:
        idf_name = 'tests/testdata/xray_data/XRay_Definition_2K.txt'

    # Load, mask and reduce data
    if False:
        # old way
        os.system('cp ' + idf_name + ' ~/.mantid/instrument/HB2B_Definition.xml')
        # Load data
        LoadSpice2D(Filename=hb2b_file_name, OutputWorkspace='hb2b')
        LoadSpice2D(Filename=hb2b_Vfile_name, OutputWorkspace='hb2b_V')

        ws_name = test_rotate_2theta(idf_name, 'hb2b', 'hb2b_rotate')
        vanadium = test_rotate_2theta(idf_name, 'hb2b_V', 'hb2b_V_rotate')
        vanadium_line = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=None)
        convert_to_2theta(ws_name, vanadium_line)

        NegMask = 'NegMask'
        PosMask = 'PosMask'
        ZeMask = 'ZeroMask'
        p10Mask = 'p10Mask'
        p20Mask = 'p20Mask'
        p30Mask = 'p30Mask'
        n30Mask = 'n30Mask'

        LoadMask(Instrument='HB2B', InputFile='/SNS/users/hcf/HFIR_TESTING/NegZ_Mask.xml', OutputWorkspace=NegMask)
        LoadMask(Instrument='HB2B', InputFile='/SNS/users/hcf/HFIR_TESTING/Chi_30_Mask.xml', OutputWorkspace=p30Mask)

        InvertMask(InputWorkspace=NegMask, OutputWorkspace=PosMask)
        InvertMask(InputWorkspace=p30Mask, OutputWorkspace=p30Mask)

        CloneWorkspace(InputWorkspace=p30Mask, OutputWorkspace=n30Mask)

        MaskDetectors(Workspace=p30Mask, MaskedWorkspace=PosMask)
        MaskDetectors(Workspace=n30Mask, MaskedWorkspace=NegMask)

        WS_p30deg = test_rotate_2theta(idf_name, 'hb2b', 'hb2b_rotate_p30deg')
        WS_n30deg = test_rotate_2theta(idf_name, 'hb2b', 'hb2b_rotate_n30deg')

        MaskDetectors(WS_p30deg, MaskedWorkspace=p30Mask)
        MaskDetectors(WS_n30deg, MaskedWorkspace=n30Mask)

        vanadium_P30 = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=p30Mask)
        vanadium_N30 = convert_to_2thetaVanadium(vanadium, num_bins=1900, Mask=n30Mask)

    else:
        from pyrs.core import reduce_hb2b_pyrs
        from pyrs.core import calibration_file_io
        from pyrs.core import reductionengine
        from pyrs.core import mask_util

        # build instrument
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        # 2theta
        two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value
        arm_length_shift = 0.
        center_shift_x = 0.
        center_shift_y = 0.
        rot_x_flip = 0.
        rot_y_flip = 0.
        rot_z_spin = 0.

        test_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
        test_calibration.center_shift_x = center_shift_x
        test_calibration.center_shift_y = center_shift_y
        test_calibration.center_shift_z = arm_length_shift
        test_calibration.rotation_x = rot_x_flip  # + 0.001
        test_calibration.rotation_y = rot_y_flip  # - 0.00321
        test_calibration.rotation_z = rot_z_spin  # + 0.0022

        # reduction engine
        engine = reductionengine.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name,
                                        target_dimension=2048,
                                        load_to_workspace=False)

        # load instrument
        pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument, 1.239)
        pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                      rot_x_flip, rot_y_flip, rot_z_spin)


    x0 = [0, 0, -0.002, -0.007, -0.922, 0]

    x0 = [-1.]
    DE_Res = leastsq(MinDifference, x0, xtol=1e-15, maxfev=3000, epsfcn=1e-2)

    DD = 0.0
    D_Shift = 0
    Center_X = -0.002
    Center_Y = -0.007
    Flip = -1
    Spin = 0.0

    DE_Res = leastsq(MinDifference, [-1], xtol=1e-15, maxfev=3000)



