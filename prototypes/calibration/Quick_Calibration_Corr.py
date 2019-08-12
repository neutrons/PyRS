# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
print ('Prototype Calibration: Quick_Calibration_v4')
import numpy as np
import time
import os
from scipy.optimize import leastsq
# from scipy.optimize import minimize
# from scipy.optimize import basinhopping
import itertools
import math
import lmfit

from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import calibration_file_io
from pyrs.core import reductionengine
from pyrs.core import mask_util
from mantid.simpleapi import CreateWorkspace, FitPeaks
from mantid.api import AnalysisDataService as mtd
from matplotlib import pyplot as plt


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return


def convert_to_2theta(engine, pyrs_reducer, roi_vec, ws_name):
    # reduce data
    min_2theta = 16.
    max_2theta = 61.
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


def peaks_alignment_score(x, engine, hb2b_setup, two_theta, roi_vec_set, plot=False):
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

    # check
    assert isinstance(roi_vec_set, list), 'must be list'
    if len(roi_vec_set) < 2:
        raise RuntimeError('User must specify more than 1 ROI/MASK vector')
    else:
        num_reduced_set = len(roi_vec_set)

    # convert the input X array (to be refined) to geometry calibration values
    geom_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
    geom_calibration.center_shift_x = x['center_shift_x'].value
    geom_calibration.center_shift_y = x['center_shift_y'].value
    geom_calibration.center_shift_z = x['center_shift_z'].value
    geom_calibration.rotation_x = x['rotation_x'].value
    geom_calibration.rotation_y = x['rotation_y'].value
    geom_calibration.rotation_z = x['rotation_z'].value

    # load instrument: as it changes
    pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(hb2b_setup, 1.239)
    pyrs_reducer.build_instrument(two_theta, geom_calibration.center_shift_z,
                                  geom_calibration.center_shift_x, geom_calibration.center_shift_y,
                                  geom_calibration.rotation_x, geom_calibration.rotation_y,
                                  geom_calibration.rotation_z)

    Eta_val = pyrs_reducer.get_eta_Values()

    # reduce data
    reduced_data_set = [None] * num_reduced_set

    for i_roi in range(num_reduced_set):
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
                residual_sq[p_index] = 20 ** 2
            elif pos_pos_i * neg_pos_i < 0.:
                # 1 failed to fit
                residual_sq[p_index] = 10 ** 2
            else:
                residual_sq[p_index] = (pos_pos_i - neg_pos_i) ** 2
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
        print ('subplot: {}, {}, {}'.format(num_rows, 2, index_i))
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
    print ('Residual      = {}'.format(norm_cost))

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

        calibration = [ 6.03052044e-05, -5.21351130e-04,  2.45533336e-04,  7.71328703e-01, -5.45174980e-01,  1.24916005e+00]
        # reduction engine
        engine = reductionengine.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)

        roi_vec_list = [30, -30, 10, -10]
        peaks_alignment_score(calibration, engine, instrument, two_theta, roi_vec_list, plot=True)

        print ('RESULT EXAMINATION IS OVER')

    else:
        t_start = time.time()

        # reduction engine
        engine = reductionengine.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)
        # instrument
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        start_calibration = [-3.90985615e-05, -2.72036598e-04, 3.91642084e-04, 5.99667751e-03,
                                 -8.15624721e-01, 1.42673120e+00]

        roi_vec_list = [30, -30, 10, -10]

        params = lmfit.Parameters()
        params.add('center_shift_x', value=start_calibration[0], min=-.1, max=.1)
        params.add('center_shift_y', value=start_calibration[1], min=-.1, max=.1)
        params.add('center_shift_z', value=start_calibration[2], min=-.1, max=.1)
        params.add('rotation_x', value=start_calibration[3], min=-np.pi, max=np.pi)
        params.add('rotation_y', value=start_calibration[4], min=-np.pi, max=np.pi)
        params.add('rotation_z', value=start_calibration[5], min=-np.pi, max=np.pi)


        out = lmfit.minimize( peaks_alignment_score, params, args=(engine, instrument, two_theta, roi_vec_list, False) )


#        # optimize
#        GlobalParameter.global_curr_sequence = 0  # reset output
#        DE_Res = leastsq(peaks_alignment_score, np.array(start_calibration),
#                         args=(engine, instrument, two_theta, roi_vec_list, False),
#                         xtol=1e-15, ftol=1e-12, gtol=1e-12, maxfev=30000, epsfcn=1e-2)

        t_stop = time.time()
        print ('Total Time: {}'.format(t_stop - t_start))
        print (out.params)


        # DE_Res = leastsq(MinDifference, [-1], xtol=1e-15, maxfev=3000)
    # END-IF-ELSE

    return

main()
