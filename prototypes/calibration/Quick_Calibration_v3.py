# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at .../tests/Quick_Calibration_v2.py and ./Quick_Calibration_v2.py
print ('Prototype Calibration: Quick_Calibration_v3')
import numpy as np
import time
import os
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from pyrs.core import reduce_hb2b_pyrs
from pyqr.utilities import calibration_file_io
from pyrs.core import reduction_manager
from pyrs.core import mask_util
from mantid.simpleapi import CreateWorkspace, FitPeaks
from mantid.api import AnalysisDataService as mtd
from matplotlib import pyplot as plt


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return


def convert_to_2theta(engine, two_theta, instrument_setup, roi_vec, geometry_shift, ws_name):
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


def CostFunction(x, engine, hb2b_setup, two_theta, positive_roi_vec, negative_roi_vec, plot=False):
    """ Cost function to align the peaks!
    :param x:
    :return:
    """
    # Reduce
    geom_calibration = calibration_file_io.ResidualStressInstrumentCalibration()
    geom_calibration.center_shift_x = x[0]
    geom_calibration.center_shift_y = x[1]
    geom_calibration.center_shift_z = x[2]
    geom_calibration.rotation_x = x[3]
    geom_calibration.rotation_y = x[4]
    geom_calibration.rotation_z = x[5]

    positive_roi = convert_to_2theta(engine, two_theta, hb2b_setup, positive_roi_vec, geom_calibration,
                                     'positive_roi_ws')
    negative_roi = convert_to_2theta(engine, two_theta, hb2b_setup, negative_roi_vec, geom_calibration,
                                     'negative_roi_ws')

    # Fit peaks
    N30_Fit = 'Fit_N30'
    P30_Fit = 'Fit_P30'
    p30_fitted = 'hb2b_rotate_p30deg_reduced_Output'
    n30_fitted = 'hb2b_rotate_n30deg_reduced_Output'

    FitPeaks(InputWorkspace='positive_roi_ws', OutputWorkspace=P30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace=p30_fitted,
             OutputPeakParametersWorkspace='hb2b_rotate_p30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_p30deg_reduced_Errors')

    FitPeaks(InputWorkspace='negative_roi_ws', OutputWorkspace=N30_Fit, StartWorkspaceIndex=0,
             StopWorkspaceIndex=0, PeakCenters='17.5,24.5,30.25,35.2,39.4,43.2,53.5',
             FitWindowBoundaryList='16,19,23,26,29,32,33,37,38,41,42,44.5,51.5,55',
             FittedPeaksWorkspace=n30_fitted,
             OutputPeakParametersWorkspace='hb2b_rotate_n30deg_reduced_FITS',
             OutputParameterFitErrorsWorkspace='hb2b_rotate_n30deg_reduced_Errors')

    assert len(mtd[P30_Fit].readY(0)) == len(mtd[N30_Fit].readY(0)), 'Number of peaks fitted must be equal'

    # calculate residual/cost
    num_peaks = len(mtd[N30_Fit].readY(0))
    residual_sq = np.zeros(shape=(num_peaks, ), dtype='float')
    for p_index in range(len(mtd[P30_Fit].readY(0))):
        pos_pos_i = mtd[P30_Fit].readY(0)[p_index]
        neg_pos_i = mtd[N30_Fit].readY(0)[p_index]
        if pos_pos_i < 0. and neg_pos_i < 0.:
            # both failed to fit
            residual_sq[p_index] = 20**2
        elif pos_pos_i * neg_pos_i < 0.:
            # 1 failed to fit
            residual_sq[p_index] = 10**2
        else:
            residual_sq[p_index] = (pos_pos_i - neg_pos_i) ** 2
    # END-FOR
    residual = np.sqrt(residual_sq)

    # plot
    ax1 = plt.subplot(212)
    ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit
    ax1.plot(r[0], positive_roi[1], color='red')
    ax1.plot(negative_roi[0], negative_roi[1], color='green')
    ax1.set_title('Cost = {}'.format(residual))

    ax2 = plt.subplot(221)
    # ax2.margins(2, 2)  # Values >0.0 zoom out
    ax2.plot(positive_roi[0], positive_roi[1], color='black')
    ax2.plot(mtd[p30_fitted].readX(0), mtd[p30_fitted].readY(0), color='red')
    ax2.set_title('Positive')

    ax3 = plt.subplot(222)
    # ax3.margins(x=0, y=-0.25)  # Values in (-0.5, 0.0) zooms in to center
    ax3.plot(negative_roi[0], negative_roi[1], color='black')
    ax3.plot(mtd[n30_fitted].readX(0), mtd[n30_fitted].readY(0), color='red')
    ax3.set_title('Negative')

    if plot:
        plt.show()
    else:
        plt.savefig('Round{:010}.png'.format(GlobalParameter.global_curr_sequence))
        GlobalParameter.global_curr_sequence += 1

    print ('Parameters:     {}'.format(x))
    print ('Fitted Peaks +: {}'.format(mtd[P30_Fit].readY(0))) 
    print ('Fitted Peaks -: {}'.format(mtd[N30_Fit].readY(0))) 
    print ('Residual      = {}'.format(residual.sum()))

    return residual
# This is main!!!


def main():

    # # ----------------s-----------------------------------------------------------
    # wavelength = 1.296  # A
    # two_theta = 30.

    # Set up
    # data, mask and etc
    test_file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
    pos_mask_h5 = 'tests/testdata/masks/Chi_30.hdf5'
    neg_mask_h5 = 'tests/testdata/masks/Chi_Neg30.hdf5'

    # instrument geometry
    idf_name = 'tests/testdata/xray_data/XRay_Definition_2K.txt'

    # 2theta
    two_theta = -35.  # TODO - TONIGHT 1 - Make this user specified value

    # Load, mask and reduce data
    if True:
        # check
        instrument = calibration_file_io.import_instrument_setup(idf_name)

        # calibration = [-2.28691912e-04, 3.42766839e-06, -1.99762398e-03, -5.59805308e-02, -8.32593462e-01,
        #                7.66556036e-04]
        calibration = [0.] * 6
        calibration = [2.33235126e-05,  -1.85337378e-04, -1.87855142e-03,  -2.20924269e-02,
                       -1.64058209e+00, 1.41293750e+00]

        roi_vec_pos, mask_2theta, note = mask_util.load_pyrs_mask(pos_mask_h5)
        roi_vec_neg, mask_2thetA, notE = mask_util.load_pyrs_mask(neg_mask_h5)

        # reduction engine
        engine = reduction_manager.HB2BReductionManager()
        test_data_id = engine.load_data(data_file_name=test_file_name, target_dimension=2048,
                                        load_to_workspace=False)

        CostFunction(calibration, engine, instrument, two_theta, roi_vec_pos, roi_vec_neg, plot=True)

        print ('RESULT EXAMINATION IS OVER')

    else:
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

        start_calibration = [0, 0, -0.002, -0.007, -0.922, 0]
        GlobalParameter.global_curr_sequence = 0  # reset output
        DE_Res = leastsq(CostFunction, np.array(start_calibration),
                         args=(engine, instrument, two_theta, roi_vec_pos, roi_vec_neg, False),
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
