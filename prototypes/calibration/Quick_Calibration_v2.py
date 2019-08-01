# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
print ('Prototype Calibration: Quick_Calibration_v2')
# Original can be found at .../tests/Quick_Calibration_v2.py
# TODO - TODAY 0 - Use pure-python reduction to replace Mantid reduction
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import calibration_file_io
from pyrs.core import reduction_manager
from pyrs.core import mask_util
from mantid.simpleapi import CreateWorkspace, FitPeaks
from mantid.api import AnalysisDataService as mtd
from matplotlib import pyplot as plt

#----------------s-----------------------------------------------------------    
wavelength = 1.E5 # kev
wavelength = 1.296  # A
two_theta = 30.


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


def MinDifference(x, engine, hb2b_setup, positive_roi_vec, negative_roi_vec):
    """ Cost function to align the peaks!
    :param x:
    :return:
    """
    def convert_to_2theta(two_theta, instrument_setup, roi_vec, geometry_shift, ws_name):
        # load instrument: as it changes
        pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument_setup, 1.239)
        pyrs_reducer.build_instrument(two_theta, geometry_shift.center_shift_z,
                                      geometry_shift.center_shift_x, geometry_shift.center_shift_y,
                                      geometry_shift.rotation_x, geometry_shift.rotation_y,
                                      geometry_shift.rotation_z)

        # reduce data
        min_2theta = 8.
        max_2theta = 64.
        num_bins = 1800

        # reduce PyRS (pure python)
        curr_id = engine.current_data_id
        vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                                       mask=roi_vec, x_range=(min_2theta, max_2theta),
                                                                       num_bins=num_bins,
                                                                       is_point_data=True,
                                                                       use_mantid_histogram=False)

        CreateWorkspace(DataX=vec_2theta, DataY=vec_hist, DataE=np.sqrt(vec_hist), NSpec=1,
                        OutputWorkspace=ws_name)

        return vec_2theta, vec_hist

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

        geom_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
        geom_calibration.center_shift_x = x[0]
        geom_calibration.center_shift_y = x[1]
        geom_calibration.center_shift_z = x[2]
        geom_calibration.rotation_x = x[3]
        geom_calibration.rotation_y = x[4]
        geom_calibration.rotation_z = x[5]

        positive_roi = convert_to_2theta(two_theta, hb2b_setup, positive_roi_vec, geom_calibration,
                                            'positive_roi_ws')
        negative_roi = convert_to_2theta(two_theta, hb2b_setup, negative_roi_vec, geom_calibration,
                                            'negative_roi_ws')

        # plt.plot(positive_roi[0], positive_roi[1], color='red')
        # plt.plot(negative_roi[0], negative_roi[1], color='green')
        # plt.show()

    N30_Fit = 'Fit_N30'
    P30_Fit = 'Fit_P30'



    FitPeaks(InputWorkspace='positive_roi_ws', OutputWorkspace=N30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_n30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_n30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_n30deg_reduced_Errors')

    FitPeaks(InputWorkspace='negative_roi_ws', OutputWorkspace=P30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace='hb2b_rotate_p30deg_reduced_Output',
             OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_p23deg_reduced_Errors')

    Error3 = (mtd[N30_Fit].extractY()[0] - mtd[P30_Fit].extractY()[0])

    print ('Parameters:     {}'.format(x))
    print ('Fitted Peaks +: {}'.format(mtd[P30_Fit].readY(0))) 
    print ('Fitted Peaks -: {}'.format(mtd[N30_Fit].readY(0))) 
    print ('Diff**2       = {}'.format(Error3 * Error3))
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
        pos_mask_h5 = None
        neg_mask_h5 = None
    else:
        test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
        pos_mask_h5 = 'tests/testdata/masks/Chi_30.hdf5'
        neg_mask_h5 = 'tests/testdata/masks/Chi_Neg30.hdf5'

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

    elif False:
        # build instrument: for FUN
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        # 2theta
        two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value

        calibration = [-2.28691912e-04, 3.42766839e-06, -1.99762398e-03, -5.59805308e-02, -8.32593462e-01, 7.66556036e-04]

        arm_length_shift = calibration[2]
        center_shift_x = calibration[0]
        center_shift_y = calibration[1]
        rot_x_flip = calibration[3]
        rot_y_flip = calibration[4]
        rot_z_spin = calibration[5]


        # reduction engine
        engine = reduction_manager.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name,
                                        target_dimension=2048,
                                        load_to_workspace=False)

        # load instrument
        pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(instrument, 1.239)
        pyrs_reducer.build_instrument(two_theta, arm_length_shift, center_shift_x, center_shift_y,
                                      rot_x_flip, rot_y_flip, rot_z_spin)

        # reduce data
        min_2theta = 8.
        max_2theta = 64.
        num_bins = 1800

        # reduce PyRS (pure python)
        curr_id = engine.current_data_id
        # mask
        roi_vec_pos, mask_2theta, note = mask_util.load_pyrs_mask(pos_mask_h5)
        roi_vec_neg, mask_2thetA, notE = mask_util.load_pyrs_mask(neg_mask_h5)
        pos_2theta, pos_hist = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                                       mask=roi_vec_pos, x_range=(min_2theta, max_2theta),
                                                                       num_bins=num_bins,
                                                                       is_point_data=True,
                                                                       use_mantid_histogram=False)
        neg_2theta, neg_hist = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
                                                                       mask=roi_vec_neg, x_range=(min_2theta, max_2theta),
                                                                       num_bins=num_bins,
                                                                       is_point_data=True,
                                                                       use_mantid_histogram=False)
        plt.plot(pos_2theta, pos_hist, color='red')
        plt.plot(neg_2theta, neg_hist, color='blue')
        plt.show()
        
        print ('RESULT EXAMINATION IS OVER')

    else:
        import time
        t_start = time.time()

        # reduction engine
        engine = reduction_manager.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)
        # instrument
        instrument = calibration_file_io.import_instrument_setup(idf_name)
        # mask
        roi_vec_pos, mask_2theta, note = mask_util.load_pyrs_mask(pos_mask_h5)
        roi_vec_neg, mask_2thetA, notE = mask_util.load_pyrs_mask(neg_mask_h5)

        x0 = [0, 0, -0.002, -0.007, -0.922, 0]
        # x0 = [-1.]
        # engine, hb2b_setup, positive_roi_vec, negative_roi_vec
        DE_Res = leastsq(MinDifference, np.array(x0), args=(engine, instrument, roi_vec_pos, roi_vec_neg),
                         xtol=1e-15, maxfev=3000, epsfcn=1e-2)

        t_stop = time.time()
        print ('Total Time: {}'.format(t_stop - t_start))
        print (DE_Res[0])
        print (DE_Res[1])

        DD = 0.0
        D_Shift = 0
        Center_X = -0.002
        Center_Y = -0.007
        Flip = -1
        Spin = 0.0

        # DE_Res = leastsq(MinDifference, [-1], xtol=1e-15, maxfev=3000)
    # END-IF-ELSE

    return

main()
