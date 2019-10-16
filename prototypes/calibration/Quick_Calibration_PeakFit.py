# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
from matplotlib import pyplot as plt
from mantid.api import AnalysisDataService as mtd
from mantid.simpleapi import CreateWorkspace, FitPeaks
from pyrs.core import mask_util
from pyrs.core import reductionengine
from pyqr.utilities import calibration_file_io
from pyrs.core import instrument_geometry
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import pyrscore
import lmfit
import math
import itertools
from scipy.optimize import least_squares
from scipy.optimize import minimize
import os
import time
import numpy as np
print('Prototype Calibration: Quick_Calibration_v4')
# from scipy.optimize import minimize
# from scipy.optimize import basinhopping


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return


def convert_to_2theta(engine, pyrs_reducer, roi_vec, ws_name):
    # reduce data
    min_2theta = 16.
    max_2theta = 61.
    num_bins = 1800
    TTHStep = (max_2theta - min_2theta) / num_bins
    # reduce PyRS (pure python)
    curr_id = engine.current_data_id

    pyrs_reducer.set_mask(roi_vec)
    pyrs_reducer._detector_counts = engine.get_counts(curr_id)
#    vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram(counts_array=engine.get_counts(curr_id),
#                                                                   mask=roi_vec, x_range=(min_2theta, max_2theta),
#                                                                   num_bins=num_bins,
#                                                                   is_point_data=True,
#                                                                   use_mantid_histogram=False)

    vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram((min_2theta, max_2theta), TTHStep, True,
                                                                   is_point_data=True, normalize_pixel_bin=True, use_mantid_histogram=False)

    CreateWorkspace(DataX=vec_2theta, DataY=vec_hist, DataE=np.sqrt(vec_hist), NSpec=1,
                    OutputWorkspace=ws_name)

    return vec_2theta, vec_hist


def peaks_alignment_score(x, engine, hb2b_setup, two_theta, roi_vec_set, plot=False, ScalarReturn=False):
    """ Cost function for peaks alignment
    :param x:
    :param engine:
    :param hb2b_setup:
    :param two_theta:
    :param roi_vec_set: list/array of ROI/mask vector
    :param plot:
    :return:
    """

    peak_centers = '17.5,24.5,30.25,35.2,39.4,43.2,50.0,53.5,56.4,59.45'
    fit_windows = '16,19,23,26,29,32,33,37,38,41,42,44.5,49.1,51.5,51.5,55,55,58,58,61'

    peak_centers = '24.5,30.25,50.0,53.5,56.4,59.45'
    fit_windows = '23,26,29,32,49.1,51.5,51.5,55,55,58,58,61'

    # check
    assert isinstance(roi_vec_set, list), 'must be list'
    if len(roi_vec_set) < 2:
        raise RuntimeError('User must specify more than 1 ROI/MASK vector')
    else:
        num_reduced_set = len(roi_vec_set)

    # convert the input X array (to be refined) to geometry calibration values
#    geom_calibration = instrument_geometry.AnglerCameraDetectorShift()
#    geom_calibration.center_shift_x = x[0]
#    geom_calibration.center_shift_y = x[1]
#    geom_calibration.center_shift_z = x[2]
#    geom_calibration.rotation_x = x[3]
#    geom_calibration.rotation_y = x[4]
#    geom_calibration.rotation_z = x[5]

    #geom_shift = instrument_geometry.AnglerCameraDetectorShift( x[0], x[1], x[2], x[3], x[4], x[5] )

#AnglerCameraDetectorGeometry.apply_shift( instrument_geometry.AnglerCameraDetectorShift( x[0], x[1], x[2], x[3], x[4], x[5] ) )

    # load instrument: as it changes
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(hb2b_setup, 1.239)
    pyrs_reducer.build_instrument_prototype(two_theta, x[0], x[1], x[2], x[3], x[4], x[5])

#    pyrs_reducer.build_instrument_prototype()

    Eta_val = pyrs_reducer.get_eta_Values()

    # reduce data
    reduced_data_set = [None] * num_reduced_set

    for i_roi in range(num_reduced_set):
        ws_name_i = 'reduced_data_{:02}'.format(i_roi)
        out_peak_pos_ws = 'peaks_positions_{:02}'.format(i_roi)
        fitted_ws = 'fitted_peaks_{:02}'.format(i_roi)

        # Define Mask
        Mask = np.zeros_like(Eta_val)
        if abs(roi_vec_set[i_roi]) == roi_vec_set[i_roi]:
            index = np.where((Eta_val < (roi_vec_set[i_roi]+5)) == (Eta_val > (roi_vec_set[i_roi]-5)))[0]
        else:
            index = np.where((Eta_val > (roi_vec_set[i_roi]-5)) == (Eta_val < (roi_vec_set[i_roi]+5)))[0]

        Mask[index] = 1.

        # reduce
        reduced_i = convert_to_2theta(engine, pyrs_reducer, Mask, ws_name_i)

        # fit peaks
        FitPeaks(InputWorkspace=ws_name_i, OutputWorkspace=out_peak_pos_ws,
                 StartWorkspaceIndex=0, StopWorkspaceIndex=0,
                 PeakCenters=peak_centers,
                 FitWindowBoundaryList=fit_windows,
                 FittedPeaksWorkspace=fitted_ws,
                 OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',  # FIXME - need to give a good name too
                 OutputParameterFitErrorsWorkspace='hb2b_rotate_p30deg_reduced_Errors')

        reduced_data_set[i_roi] = reduced_i, ws_name_i, out_peak_pos_ws, fitted_ws
    # END-FOR

    # calculate the quality of peak alignment for each pair of ROI
    residual = None
    for roi_i, roi_j in list(itertools.combinations(range(num_reduced_set), 2)):
        # get data
        r_t_i = reduced_data_set[roi_i]
        r_t_j = reduced_data_set[roi_j]

        # calculate residual/cost
        num_peaks = len(mtd[r_t_i[2]].readY(0))
        residual_sq = np.zeros(shape=(num_peaks,), dtype='float')
        for p_index in range(num_peaks):
            pos_pos_i = mtd[r_t_i[2]].readY(0)[p_index]
            neg_pos_i = mtd[r_t_j[2]].readY(0)[p_index]
            if pos_pos_i < 0. and neg_pos_i < 0.:
                # both failed to fit
                residual_sq[p_index] = 200 ** 2
            elif pos_pos_i * neg_pos_i < 0.:
                # 1 failed to fit
                residual_sq[p_index] = 100 ** 2
            else:
                residual_sq[p_index] = ((pos_pos_i - neg_pos_i) ** 2)
        # END-FOR

        if residual is None:
            residual = residual_sq
        else:
            residual = np.concatenate([residual, residual_sq])
    # END-IF-ELSE
    c_n_2 = math.factorial(num_reduced_set) / (math.factorial(2) * math.factorial(num_reduced_set - 2))
    norm_cost = residual.sum() / c_n_2

    # plot
    num_rows = 1 + num_reduced_set / 2 + num_reduced_set % 2

    ax1 = plt.subplot(num_rows, 1, num_rows)
    ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit
    colors = ['black', 'red', 'blue', 'green', 'yellow']
    for roi_i in range(num_reduced_set):
        r_t_i = reduced_data_set[roi_i]
        ax1.plot(r_t_i[0][0], r_t_i[0][1], color=colors[roi_i % 5])
    ax1.set_title('Normalized Cost = {}'.format(norm_cost))

    for roi_i in range(num_reduced_set):
        index_i = roi_i + 1
        print('subplot: {}, {}, {}'.format(num_rows, 2, index_i))
        ax2 = plt.subplot(num_rows, 2, index_i)
        ax2.plot(reduced_data_set[roi_i][0][0], reduced_data_set[roi_i][0][1], color='black')
        ax2.plot(mtd[reduced_data_set[roi_i][3]].readX(0), mtd[reduced_data_set[roi_i][3]].readY(0), color='red')
        ax2.set_title('{}'.format(roi_i))

    if plot:
        plt.show()
    else:
        plt.savefig('Round{:010}.png'.format(GlobalParameter.global_curr_sequence))
        GlobalParameter.global_curr_sequence += 1
    plt.clf()

    # print ('Parameters:     {}'.format(x))
    # print ('Fitted Peaks +: {}'.format(mtd[P30_Fit].readY(0)))
    # print ('Fitted Peaks -: {}'.format(mtd[N30_Fit].readY(0)))
    print('Residual      = {}'.format(norm_cost))

    if ScalarReturn:
        return np.sum(residual)
    else:
        return residual

# This is main!!!


def main():

    # # ----------------s-----------------------------------------------------------
    # wavelength = 1.296  # A
    # two_theta = 30.

    # Set up
    # data, mask and etc
    test_file_name = 'tests/testdata/Simulated_LaB6_rotate.tiff'
#    test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'

    # instrument geometry
    idf_name = 'tests/testdata/xray_data/XRay_Definition_2K.txt'

    # 2theta
    two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value

    # Load, mask and reduce data
    if False:
        # check
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        calibration = [-7.86738387e-05, 9.18988206e-05, -5.55805387e-05, -1.44470481e-01,
                       -6.45203851e-01, 1.33199903e+00]

        calibration = [6.03052044e-05, -5.21351130e-04,  2.45533336e-04,
                       7.71328703e-01, -5.45174980e-01,  1.24916005e+00]
        # reduction engine
        engine = reductionengine.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)

        roi_vec_list = [30, -30, 10, -10]
        peaks_alignment_score(calibration, engine, instrument, two_theta, roi_vec_list, plot=True)

        print('RESULT EXAMINATION IS OVER')

    else:
        t_start = time.time()

        # reduction engine
        engine = reductionengine.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)
        # instrument
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        HB2B = instrument_geometry.AnglerCameraDetectorGeometry(
            instrument.detector_rows, instrument.detector_columns, instrument.pixel_size_x, instrument.pixel_size_y, instrument.arm_length, False)

        start_calibration = [-3.90985615e-05, -2.72036598e-04, 3.91642084e-04, 5.99667751e-03,
                             -8.15624721e-01, 1.42673120e+00]

        roi_vec_list = [30, -30, 20, -20, 10, -10]

        start_calibration = np.array([0.00026239, 0.0055485, 0.00048748, 0.00130224, -0.00025911, -0.0006577])
        start_calibration = np.array([0] * 6)

        out = minimize(peaks_alignment_score, start_calibration, args=(engine, HB2B, two_theta, roi_vec_list, False, True), method='L-BFGS-B', jac=None, bounds=None, tol=None,
                       callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-05, 'gtol': 1e-03, 'eps': 1e-03, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})
        t_stop1 = time.time()

        out2 = minimize(peaks_alignment_score, out.x, args=(engine, HB2B, two_theta, roi_vec_list, False, True), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None,
                        options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-07, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})
        t_stop2 = time.time()

        out3 = least_squares(peaks_alignment_score, out2.x, jac='2-point', bounds=([-.1, -.1, -.1, -np.pi, -np.pi, -np.pi], [.1, .1, .1, np.pi, np.pi, np.pi]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
                             x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, HB2B, two_theta, roi_vec_list, False, False), kwargs={})

#        # optimize
#        GlobalParameter.global_curr_sequence = 0  # reset output
#        DE_Res = leastsq(peaks_alignment_score, np.array(start_calibration),
#                         args=(engine, instrument, two_theta, roi_vec_list, False),
#                         xtol=1e-15, ftol=1e-12, gtol=1e-12, maxfev=30000, epsfcn=1e-2)

        t_stop = time.time()
        print('Global Refine: {}'.format(t_stop1 - t_start))
        print('Local Refine: {}'.format(t_stop2 - t_start))
        print('Total Time: {}'.format(t_stop - t_start))
        print(out3.x)

        peaks_alignment_score(out3.x, engine, instrument, two_theta, roi_vec_list, True, True)

        # DE_Res = leastsq(MinDifference, [-1], xtol=1e-15, maxfev=3000)
    # END-IF-ELSE

    return


main()
