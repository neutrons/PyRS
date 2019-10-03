# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
print ('Prototype Calibration: Quick_Calibration_v4')
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.optimize import least_squares
# from scipy.optimize import minimize
# from scipy.optimize import basinhopping
import itertools
import math
import lmfit

from pyrs.core import pyrscore
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import instrument_geometry
from pyrs.utilities import calibration_file_io
from pyrs.core import reductionengine
from pyrs.core import mask_util
from mantid.simpleapi import CreateWorkspace, FitPeaks
from mantid.api import AnalysisDataService as mtd
from matplotlib import pyplot as plt

colors = ['black', 'red', 'blue', 'green', 'yellow']

dSpace = np.array( [4.156826, 2.93931985, 2.39994461, 2.078413, 1.8589891, 1.69701711, 1.46965993, 1.38560867, 1.3145038, 1.2533302, 1.19997231, 1.1528961, 1.11095848, 1.0392065, 1.00817839, 0.97977328, 0.95364129, 0.92949455, 0.9070938, 0.88623828, 0.84850855, 0.8313652, 0.81522065, 0.79998154, 0.77190321, 0.75892912, 0.73482996] )

class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return

def check_alignment_inputs(roi_vec_set, two_theta):
    """ Check inputs for alignment routine for required formating
    :param roi_vec_set: list/array of ROI/mask vector
    :param two_theta:
    :return: num_reduced_set, two_theta, num_two_theta
    """
    # check
    assert isinstance(roi_vec_set, list), 'must be list'
    if len(roi_vec_set) < 2:
        raise RuntimeError('User must specify more than 1 ROI/MASK vector')
    else:
        num_reduced_set = len(roi_vec_set)

    if not isinstance(two_theta, list): two_theta = [ two_theta ]

    return (num_reduced_set, two_theta)

def get_alignment_residual(x, engine, hb2b_setup, two_theta, roi_vec_set ):
    """ Cost function for peaks alignment to determine wavelength
    :param x: list/array of detector shift/rotation and neutron wavelength values
    :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
    :param engine:
    :param hb2b_setup: HB2B class containing instrument definitions
    :param two_theta: list/array of detector positions
    :param roi_vec_set: list/array of ROI/mask vector
    :return:
    """

    GlobalParameter.global_curr_sequence += 1

    residual = np.array( [] )
    resNone  = 0.

    TTH_Calib   = np.arcsin( x[6] / 2. / dSpace ) * 360. / np.pi
    TTH_Calib   = TTH_Calib[ ~np.isnan( TTH_Calib ) ]

    for i_tth in range( len( two_theta ) ):
        #reduced_data_set[i_tth] = [None] * num_reduced_set
        # load instrument: as it changes
        pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(hb2b_setup, x[6] )
        pyrs_reducer.build_instrument_prototype(two_theta[i_tth], x[0], x[1], x[2], x[3], x[4], x[5] )

        DetectorAngle   = np.abs( two_theta[i_tth] )

        Eta_val = pyrs_reducer.get_eta_Values()
        TTH_val = pyrs_reducer.generate_2theta_histogram_vector( None, 0.01, None)
        maxtth  = np.max( TTH_val )
        mintth  = np.min( TTH_val )

        peak_centers    = ''
        fit_windows     = ''

        CalibPeaks      = TTH_Calib[ np.where( (TTH_Calib > DetectorAngle-8) == (TTH_Calib < DetectorAngle+8) )[0] ]
        for ipeak in CalibPeaks:
            if (ipeak > mintth) and (ipeak < maxtth):
                peak_centers+='%.4f,'%ipeak
                fit_windows+='%.4f,%.4f,'%(ipeak-1, ipeak+1)

        peak_centers = peak_centers[:-1]
        fit_windows = fit_windows[:-1]

        if peak_centers == '':residual = np.concatenate([residual, np.array( [20000] ) ])

        else:
            # reduce data

            num_rows = 1 + len( roi_vec_set )  / 2 + len( roi_vec_set )  % 2

            ax1 = plt.subplot(num_rows, 1, num_rows)

            ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit

            for i_roi in range( len( roi_vec_set ) ):
                ws_name_i = 'reduced_data_{:02}'.format(i_roi)
                out_peak_pos_ws = 'peaks_positions_{:02}'.format(i_roi)
                fitted_ws = 'fitted_peaks_{:02}'.format(i_roi)

                # Define Mask
                Mask = np.zeros_like( Eta_val )
                if abs(roi_vec_set[i_roi]) == roi_vec_set[i_roi]:
                    index = np.where( (Eta_val < (roi_vec_set[i_roi]+5) ) == ( Eta_val > (roi_vec_set[i_roi]-5 ) ))[0]
                else:
                    index = np.where( (Eta_val > (roi_vec_set[i_roi]-5) ) == ( Eta_val < (roi_vec_set[i_roi]+5 ) ))[0]

                Mask[index] = 1.

                # reduce
                reduced_i = convert_to_2theta(engine, pyrs_reducer, Mask, ws_name_i, 'SimulatedData_%d'%np.abs(DetectorAngle), min_2theta=mintth, max_2theta=maxtth, num_bins=750 )

                # fit peaks
                FitPeaks(InputWorkspace=ws_name_i, OutputWorkspace=out_peak_pos_ws,
                         StartWorkspaceIndex=0, StopWorkspaceIndex=0,
                         PeakCenters=peak_centers,
                         FitWindowBoundaryList=fit_windows,
                         FittedPeaksWorkspace=fitted_ws,
                         OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',  # FIXME - need to give a good name too
                         OutputParameterFitErrorsWorkspace='hb2b_rotate_p30deg_reduced_Errors')

    #            print ( "\n\n\n\n\n\n" )
    #            print ( mtd[ out_peak_pos_ws ].readY(0) )
    #            print ( CalibPeaks )
    #            print ( CalibPeaks - mtd[ out_peak_pos_ws ].readY(0)  )
    #            print ( "\n\n\n\n\n\n" )

                for p_index in range( len(mtd[ out_peak_pos_ws ].readY(0)) ):
                    fitted_pos_i = mtd[ out_peak_pos_ws ].readY(0)[p_index]

                    if fitted_pos_i < 0.:
                        residual_sq = 1000
                    else:
                        residual_sq = ( 100.0 * np.abs(fitted_pos_i - CalibPeaks[p_index]) )
                        resNone += fitted_pos_i - CalibPeaks[p_index]

                    residual = np.concatenate([residual, np.array( [residual_sq] ) ])

            # plot

                ax1.plot(reduced_i[0], reduced_i[1], color=colors[i_roi % 5])
                index_i = i_roi + 1
                ax2 = plt.subplot(num_rows, 2, index_i)
                ax2.plot(reduced_i[0], reduced_i[1], color='black')
                ax2.plot(mtd[fitted_ws].readX(0), mtd[fitted_ws].readY(0), color='red')


            plt.savefig('Round{:010}_{:02}.png'.format(GlobalParameter.global_curr_sequence, i_tth))
            plt.clf()


    c_n_2 = math.factorial(len( roi_vec_set )*len(two_theta)) / (math.factorial(2) * math.factorial(len( roi_vec_set )*len(two_theta) - 2))
    norm_cost = residual.sum() / c_n_2

    print ( "\n\n\n" )
    print ('Residual      = {}'.format(norm_cost))
    print ('Residual      = {}'.format(resNone))
    print ( "\n\n\n" )

    return (residual)

def convert_to_2theta(engine, pyrs_reducer, roi_vec, ws_name, curr_id, min_2theta=16., max_2theta=61., num_bins=1800):
    # reduce data
    TTHStep     = (max_2theta - min_2theta) / num_bins
    pyrs_reducer.set_mask( roi_vec )
    pyrs_reducer._detector_counts = engine.get_counts(curr_id)

    vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram((min_2theta, max_2theta), TTHStep, True,
                                   is_point_data=True, normalize_pixel_bin=True, use_mantid_histogram=False)

    CreateWorkspace(DataX=vec_2theta, DataY=vec_hist, DataE=np.sqrt(vec_hist), NSpec=1,
                    OutputWorkspace=ws_name)

    return vec_2theta, vec_hist


def peak_alignment_wavelength( x, engine, hb2b_setup, two_theta, roi_vec_set, detectorCalib, ScalarReturn=False ):
    """ Cost function for peaks alignment to determine wavelength
    :param x:
    :param engine:
    :param hb2b_setup:
    :param two_theta:
    :param roi_vec_set: list/array of ROI/mask vector
    :param detectorCalib:
    :param ScalarReturn:
    :return:
    """

    num_reduced_set, two_theta = check_alignment_inputs(roi_vec_set, two_theta)

    paramVec        = np.zeros(7)
    paramVec[0:6]   = detectorCalib
    paramVec[6]     = x[0]

    residual = get_alignment_residual(paramVec, engine, hb2b_setup, two_theta, roi_vec_set )

    if ScalarReturn:
        return np.sum( residual )
    else:
        return residual

def peak_alignment_shift( x, engine, hb2b_setup, two_theta, roi_vec_set, detectorRotation=[0,0,0], Wavelength=1.452, ScalarReturn=False ):
    """ Cost function for peaks alignment to determine detector shift
    :param x:
    :param engine:
    :param hb2b_setup:
    :param two_theta:
    :param roi_vec_set: list/array of ROI/mask vector
    :param detectorCalib:
    :param ScalarReturn:
    :return:
    """

    num_reduced_set, two_theta = check_alignment_inputs(roi_vec_set, two_theta)

    paramVec        = np.zeros(7)
    paramVec[0:3]   = x[:]
    paramVec[3:6]   = detectorRotation
    paramVec[6]     = Wavelength

    print ('\n\n\n\n\n')
    print (paramVec)
    print ('\n\n\n\n\n')

    residual = get_alignment_residual(paramVec, engine, hb2b_setup, two_theta, roi_vec_set )

    if ScalarReturn:
        return np.sum( residual )
    else:
        return residual

def peak_alignment_rotation( x, engine, hb2b_setup, two_theta, roi_vec_set, detectorShift=[0,0,0], Wavelength=1.452, ScalarReturn=False ):
    """ Cost function for peaks alignment to determine detector rotation
    :param x:
    :param engine:
    :param hb2b_setup:
    :param two_theta:
    :param roi_vec_set: list/array of ROI/mask vector
    :param detectorCalib:
    :param ScalarReturn:
    :return:
    """
    num_reduced_set, two_theta = check_alignment_inputs(roi_vec_set, two_theta)

    paramVec        = np.zeros(7)
    paramVec[0:3]   = detectorShift
    paramVec[3:6]   = x[:]
    paramVec[6]     = Wavelength

    residual = get_alignment_residual(paramVec, engine, hb2b_setup, two_theta, roi_vec_set )

    if ScalarReturn:
        return np.sum( residual )
    else:
        return residual

def peaks_alignment_all(x, engine, hb2b_setup, two_theta, roi_vec_set, plot=False, ScalarReturn=False):
    """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
    :param x:
    :param engine:
    :param hb2b_setup:
    :param two_theta:
    :param roi_vec_set: list/array of ROI/mask vector
    :param plot:
    :return:
    """

    num_reduced_set, two_theta = check_alignment_inputs(roi_vec_set, two_theta)

    residual = get_alignment_residual(x, engine, hb2b_setup, two_theta, roi_vec_set )

    if ScalarReturn:
        return np.sum( residual )
    else:
        return residual

# This is main!!!


def main():

    # # ----------------s-----------------------------------------------------------
    # wavelength = 1.296  # A
    # two_theta = 30.

    # Set up
    # data, mask and etc
    test_file_name1 = 'tests/testdata/SimulatedData_60.tiff'
    test_file_name2 = 'tests/testdata/SimulatedData_61.tiff'
    test_file_name3 = 'tests/testdata/SimulatedData_62.tiff'
    test_file_name4 = 'tests/testdata/SimulatedData_63.tiff'
    test_file_name5 = 'tests/testdata/SimulatedData_64.tiff'
    test_file_name6 = 'tests/testdata/SimulatedData_65.tiff'
#    test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'


    # reduction engine
    DataSets    = []
    two_theta   = []
    TTHS = np.arange(60, 65.1, 1)
    engine      = reductionengine.HB2BReductionManager()
    TTHS = np.array( [56., 60.0, 65., 70., 81.] )

    for TTH in TTHS:
        engine.load_data(data_file_name='tests/testdata/SimulatedData_%.0f.tiff'%TTH, target_dimension=1024,
                                    load_to_workspace=False)
        DataSets.append( 'SimulatedData_%.0f'%TTH )
        two_theta.append( -1* TTH )

    # instrument geometry
    idf_name = 'tests/testdata/xray_data/XRay_Definition_2K.txt'
    idf_name = 'tests/testdata/xray_data/XRay_Definition_1K.txt'

    t_start = time.time()

    # instrument
    instrument  = calibration_file_io.import_instrument_setup(idf_name)

    HB2B        = instrument_geometry.AnglerCameraDetectorGeometry( instrument.detector_rows, instrument.detector_columns, instrument.pixel_size_x, instrument.pixel_size_y, instrument.arm_length, False )

#        roi_vec_list = [30, -30, 20, -20, 10, -10]
    roi_vec_list = [5, 2, 0, -2, -5]
    roi_vec_list = [5, 0, -5]

    start_calibration = np.array( [0.0] * 7, dtype = np.float)
    start_calibration[6] =  1.452

    GlobalParameter.global_curr_sequence = 0

    #out = minimize(peaks_alignment_all, start_calibration, args=(engine, HB2B, two_theta, roi_vec_list, False, True, 1.452), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-05, 'gtol': 1e-03, 'eps': 1e-03, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})
    t_stop1 = time.time()

    #out2 = minimize(peaks_alignment_all, out.x, args=(engine, HB2B, two_theta, roi_vec_list, False, True, 1.452), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-07, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})
    t_stop2 = time.time()

    #start_calibration = np.array( [0.0] * 3, dtype = np.float)

    #out = minimize(peak_alignment_shift, start_calibration, args=(engine, HB2B, two_theta, roi_vec_list, [0, 0, 0], 1.452, True), method='L-BFGS-B', jac=None, bounds=([-.1, .1], [-.1, .1], [ -.1, .1 ] ), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-05, 'gtol': 1e-03, 'eps': 1e-03, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})

#    out1 = least_squares(peak_alignment_shift, out.x, jac='2-point', bounds=([-.1, -.1, -.1 ], [ .1, .1, .1 ]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, HB2B, two_theta, roi_vec_list, [0, 0, 0], 1.452, False), kwargs={})

#    start_calibration = np.array( [0.0] * 3, dtype = np.float)
#    out2 = least_squares(peak_alignment_rotation, start_calibration, jac='2-point', bounds=([ -np.pi, -np.pi, -np.pi], [  np.pi, np.pi, np.pi, ]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, HB2B, two_theta, roi_vec_list, out1.x, 1.452, False ), kwargs={})

    start_calibration = np.array( [0.0] * 7, dtype = np.float)
#    start_calibration[0:3]  = out1.x
#    start_calibration[3:6]  = out2.x
    start_calibration[6]    =  1.452

    out3 = least_squares(peaks_alignment_all, start_calibration, jac='2-point', bounds=([-.1, -.1, -.1, -np.pi, -np.pi, -np.pi, 1.4], [ .1, .1, .1, np.pi, np.pi, np.pi, 1.5]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, HB2B, two_theta, roi_vec_list, False, False), kwargs={})

    #out3 = least_squares(peaks_alignment_all, out2.x, jac='2-point', bounds=([-.1, -.1, -.1, -np.pi, -np.pi, -np.pi ], [.1, .1, .1, np.pi, np.pi, np.pi]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, HB2B, two_theta, roi_vec_list, False, False, 1.452), kwargs={})



    t_stop = time.time()
    print ('Global Refine: {}'.format(t_stop1 - t_start))
    print ('Local Refine: {}'.format(t_stop2 - t_start))
    print ('Total Time: {}'.format(t_stop - t_start))
    print (out3.x)

    with open('SimulatedData_Calibration.txt', 'w') as fOUT:
        fOUT.write( '# Detector Calibration\n')
        fOUT.write( 'Shift_x: %.8f\n'%out3.x[0] )
        fOUT.write( 'Shift_y: %.8f\n'%out3.x[1] )
        fOUT.write( 'Shift_z: %.8f\n'%out3.x[2] )
        fOUT.write( 'Rot_x: %.8f\n'%out3.x[3] )
        fOUT.write( 'Rot_y: %.8f\n'%out3.x[4] )
        fOUT.write( 'Rot_z: %.8f\n'%out3.x[5] )
        if start_calibration.shape[0] == 7:
            fOUT.write( 'Lambda: %.8f\n'%out3.x[6] )

    if start_calibration.shape[0] == 7:CalibData = dict( zip( ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda'], out3.x ) )
    else: CalibData = dict( zip( ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z'], out3.x ) )

    import json
    with open('Calibration.txt', 'w') as outfile:
        json.dump(CalibData, outfile)
#        peaks_alignment_score(out3.x, engine, instrument, two_theta, roi_vec_list, True, True)

    # DE_Res = leastsq(MinDifference, [-1], xtol=1e-15, maxfev=3000)
# END-IF-ELSE

    return

main()
