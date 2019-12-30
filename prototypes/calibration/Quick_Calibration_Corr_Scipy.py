# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
from matplotlib import pyplot as plt
from mantid.api import AnalysisDataService as mtd
from mantid.simpleapi import CreateWorkspace, FitPeaks
from pyrs.core import mask_util
from pyrs.core import reductionengine
from pyrs.core.instrument_geometry import AnglerCameraDetectorShift
from pyqr.utilities import calibration_file_io
from pyrs.core import reduce_hb2b_pyrs
import lmfit
import math
import itertools
from scipy.optimize import minimize
try:
    from scipy.optimize import least_squares
except ImportError:
    from scipy.optimize import leastsq as least_squares  # for older scipy
import os
import time
import numpy as np
print('Prototype Calibration: Quick_Calibration_v4')
# from scipy.optimize import basinhopping


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return


def convert_to_2theta(hb2b, pyrs_reducer, roi_vec, min_2theta, max_2theta, num_bins):
    # reduce data
    # reduce PyRS (pure python)
    curr_id = hb2b.current_data_id

    vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram(counts_array=hb2b.get_counts(curr_id),
                                                                   mask=roi_vec, x_range=(min_2theta, max_2theta),
                                                                   num_bins=num_bins,
                                                                   is_point_data=True,
                                                                   use_mantid_histogram=False)

    vec_2theta_V, vec_hist_V = pyrs_reducer.reduce_to_2theta_histogram(counts_array=np.ones_like(hb2b.get_counts(curr_id)),
                                                                       mask=roi_vec, x_range=(min_2theta, max_2theta),
                                                                       num_bins=num_bins,
                                                                       is_point_data=True,
                                                                       use_mantid_histogram=False)

    return vec_2theta, vec_hist/vec_hist_V


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
    peak_centers = '17.5,24.5,30.25,35.2,39.4,43.2,53.5'
    fit_windows = '16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55'

    peak_pos = [17.5, 24.5, 30.25, 35.2, 39.4, 43.2, 53.5]

    peak_pos = [25.5, 30.25, 35.2, 39.4, 43.2, 50.7, 54., 57., 59]

    # check
    #assert isinstance(roi_vec_set, list), 'must be list'
    if len(roi_vec_set) < 2:
        raise RuntimeError('User must specify more than 1 ROI/MASK vector')
    else:
        num_reduced_set = len(roi_vec_set)
        num_peaks = len(peak_pos)

    # convert the input X array (to be refined) to geometry calibration values
    geom_calibration = AnglerCameraDetectorShift()
    geom_calibration.center_shift_x = x[0]
    geom_calibration.center_shift_y = x[1]
    geom_calibration.center_shift_z = x[2]
    geom_calibration.rotation_x = x[3]
    geom_calibration.rotation_y = x[4]
    geom_calibration.rotation_z = x[5]

    # load instrument: as it changes
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(hb2b_setup, 1.239)
    pyrs_reducer.build_instrument(two_theta, geom_calibration.center_shift_z,
                                  geom_calibration.center_shift_x, geom_calibration.center_shift_y,
                                  geom_calibration.rotation_x, geom_calibration.rotation_y,
                                  geom_calibration.rotation_z)

    Eta_val = pyrs_reducer.get_eta_value()

    # reduce data
    reduced_data_set = [None] * num_reduced_set

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for i_roi in range(num_reduced_set):
            # Define Mask
            Mask = np.zeros_like(Eta_val)
            if abs(roi_vec_set[i_roi]) == roi_vec_set[i_roi]:
                index = np.where((Eta_val < (roi_vec_set[i_roi]+5)) == (Eta_val > (roi_vec_set[i_roi]-5)))[0]
            else:
                index = np.where((Eta_val > (roi_vec_set[i_roi]-5)) == (Eta_val < (roi_vec_set[i_roi]+5)))[0]

            Mask[index] = 1.

            #x_vec, y_vec = convert_to_2theta(engine, pyrs_reducer, Mask, 18, 63, 1800 )
            vec_x, vec_y = convert_to_2theta(engine, pyrs_reducer, Mask, 18, 63, 1800)
            ax.plot(vec_x.T, vec_y.T, label=roi_vec_set[i_roi])

        ax.set_ylabel('Int (cts.)')
        ax.set_ylabel('2theta (deg.)')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.show()
        return

    for i_roi in range(num_reduced_set):
        Peaks = [None] * num_peaks

        # Define Mask
        Mask = np.zeros_like(Eta_val)
        if abs(roi_vec_set[i_roi]) == roi_vec_set[i_roi]:
            index = np.where((Eta_val < (roi_vec_set[i_roi]+5)) == (Eta_val > (roi_vec_set[i_roi]-5)))[0]
        else:
            index = np.where((Eta_val > (roi_vec_set[i_roi]-5)) == (Eta_val < (roi_vec_set[i_roi]+5)))[0]

        Mask[index] = 1.
        for i_peak in range(num_peaks):
            # reduce
            Peaks[i_peak] = convert_to_2theta(engine, pyrs_reducer, Mask,
                                              peak_pos[i_peak]-1, peak_pos[i_peak]+1, 60)[1]

        reduced_data_set[i_roi] = Peaks

    # END-FOR

    # calculate the quality of peak alignment for each pair of ROI
    residual = None

    for i_roi in range(num_reduced_set):
        for peak_i, peak_j in list(itertools.combinations(range(num_peaks), 2)):
            # get data

            #residual_sq = 1. / np.min( np.corrcoef( reduced_data_set[i_roi][peak_i], reduced_data_set[i_roi][peak_j] ) )

            temp_p1 = reduced_data_set[i_roi][peak_i]
            temp_p2 = reduced_data_set[i_roi][peak_j]
            residual_sq = (1. / np.correlate(temp_p1 / np.linalg.norm(temp_p1),
                                             temp_p2 / np.linalg.norm(temp_p2))) - 1.
#            residual_sq = np.correlate( reduced_data_set[i_roi][peak_i], reduced_data_set[i_roi][peak_j] )
            if not np.isfinite(residual_sq):
                residual_sq = np.array([1000.])
            if residual is None:
                residual = residual_sq
            else:
                residual = np.concatenate([residual, residual_sq])

    # END-IF-ELSE
    c_n_2 = math.factorial(num_reduced_set) / (math.factorial(2) * math.factorial(num_reduced_set - 2))
    norm_cost = residual.sum() / c_n_2

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
    Van_file_name = 'tests/testdata/Simulated_V_rotate.tiff'

    # instrument geometry
    idf_name = 'tests/testdata/xray_data/XRay_Definition_2K.txt'

    # 2theta
    two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value

    # instrument
    instrument = calibration_file_io.import_instrument_setup(idf_name)

    # reduction engine
    engine = reductionengine.HB2BReductionManager()
    test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                    load_to_workspace=False)

    # Load, mask and reduce data
    if False:

        calibration = [-7.86738387e-05, 9.18988206e-05, -5.55805387e-05, -1.44470481e-01,
                       -6.45203851e-01, 1.33199903e+00]

        calibration = [6.03052044e-05, -5.21351130e-04,  2.45533336e-04,
                       7.71328703e-01, -5.45174980e-01,  1.24916005e+00]

        roi_vec_list = [30, -30, 10, -10]
        peaks_alignment_score(calibration, engine, instrument, two_theta, roi_vec_list, plot=True)

        print('RESULT EXAMINATION IS OVER')

    else:
        t_start = time.time()

        start_calibration = [-3.90985615e-05, -2.72036598e-04, 3.91642084e-04, 5.99667751e-03,
                             -8.15624721e-01, 1.42673120e+00]

        start_calibration = np.array([0] * 6)

        roi_vec_list = np.arange(-30, 30.1, 15)

        out = minimize(peaks_alignment_score, start_calibration, args=(engine, instrument, two_theta, roi_vec_list, False, True), method='L-BFGS-B', jac=None, bounds=None, tol=None,
                       callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-05, 'gtol': 1e-03, 'eps': 1e-03, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})
        t_stop1 = time.time()

        out2 = minimize(peaks_alignment_score, out.x, args=(engine, instrument, two_theta, roi_vec_list, False, True), method='L-BFGS-B', jac=None, bounds=None, tol=None,
                        callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-07, 'maxfun': 2, 'maxiter': 2, 'iprint': -1, 'maxls': 20})
        t_stop2 = time.time()

        #out3 = least_squares(peaks_alignment_score, start_calibration, jac='2-point', bounds=([-.1, -.1, -.1, -np.pi, -np.pi, -np.pi], [.1, .1, .1, np.pi, np.pi, np.pi]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, instrument, two_theta, roi_vec_list, False, False), kwargs={})

        out3 = least_squares(peaks_alignment_score, out2.x, jac='2-point', bounds=([-.1, -.1, -.1, -np.pi, -np.pi, -np.pi], [.1, .1, .1, np.pi, np.pi, np.pi]), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
                             loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(engine, instrument, two_theta, roi_vec_list, False, False), kwargs={})

        t_stop = time.time()
        print('Global Refine: {}'.format(t_stop1 - t_start))
        print('Local Refine: {}'.format(t_stop2 - t_start))
        print('Total Time: {}'.format(t_stop - t_start))
        print(out3.x)

        peaks_alignment_score(out3.x, engine, instrument, two_theta, roi_vec_list, True, True)
    # END-IF-ELSE

    return


main()
