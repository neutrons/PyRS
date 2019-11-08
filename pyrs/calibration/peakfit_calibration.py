import numpy as np
import time
import glob
import dateutil.parser
import json
import os
from pyrs.core import reduce_hb2b_pyrs
from matplotlib import pyplot as plt

from pyrs.utilities.calibration_file_io import write_calibration_to_json
try:
    from scipy.optimize import least_squares
    UseLSQ = False
except ImportError:
    UseLSQ = True
    from scipy.optimize import leastsq  # for older scipy


colors = ['black', 'red', 'blue', 'green', 'yellow']

dSpace = np.array([4.156826, 2.93931985, 2.39994461, 2.078413, 1.8589891, 1.69701711, 1.46965993,
                   1.38560867, 1.3145038, 1.2533302, 1.19997231, 1.1528961, 1.11095848, 1.0392065,
                   1.00817839, 0.97977328, 0.95364129, 0.92949455, 0.9070938, 0.88623828, 0.84850855,
                   0.8313652, 0.81522065, 0.79998154, 0.77190321, 0.75892912, 0.73482996])


def runCalib():
    return


def quadratic_background(x, p0, p1, p2):
    """Quadratic background

    Y = p2 * x**2 + p1 * x + p0

    Parameters
    ----------
    x: float or ndarray
        x value
    p0: float
    p1: float
    p2: float

    Returns
    -------
    float or numpy array
        background
    """
    return p2*x*x + p1*x + p0


def GaussianModel(x, mu, sigma, Amp):
    """Gaussian Model

    Y = Amp/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x: float or ndarray
        x value
    mu: float
    sigma: float
    Amp: float

    Returns
    -------
    float or numpy array
        Peak
    """
    return Amp/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


class GlobalParameter(object):
    global_curr_sequence = 0

    def __init__(self):
        return


class PeakFitCalibration(object):
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """
    def __init__(self, hb2b_instrument, hidra_data):
        """
        Initialization
        """

        self._instrument = hb2b_instrument
        self._engine = hidra_data
        # calibration: numpy array. size as 7 for ... [6] for wave length
        self._calib = np.array(7 * [0], dtype=np.float)
        # calibration error: numpy array. size as 7 for ...
        self._caliberr = np.array(7 * [-1], dtype=np.float)

        # Set wave length
        self._calib[6] = \
            np.array([1.452, 1.452, 1.540, 1.731, 1.886, 2.275, 2.667])[self._engine.get_log_value('MonoSetting')[0]]
        self._calibstatus = -1

        GlobalParameter.global_curr_sequence = 0

        return
    
    @staticmethod
    def check_alignment_inputs(roi_vec_set):
        """ Check inputs for alignment routine for required formating
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        # check
        if roi_vec_set is not None:
            assert isinstance(roi_vec_set, list), 'must be list'
            if len(roi_vec_set) < 2:
                raise RuntimeError('User must specify more than 1 ROI/MASK vector')

        return

    @staticmethod
    def convert_to_2theta(pyrs_reducer, roi_vec, min_2theta=16., max_2theta=61., num_bins=1800):
        """Convert data, with ROI, to 2theta

        Parameters
        ----------
        pyrs_reducer
        roi_vec
        min_2theta
        max_2theta
        num_bins

        Returns
        -------
        ndarray, ndarray
            2theta, intensity
        """
        # reduce data
        two_theta_step = (max_2theta - min_2theta) / num_bins
        pyrs_reducer.set_mask(roi_vec)
        vec_2theta, vec_hist = pyrs_reducer.reduce_to_2theta_histogram((min_2theta, max_2theta), two_theta_step, True,
                                                                       is_point_data=True, normalize_pixel_bin=True,
                                                                       use_mantid_histogram=False)

        return vec_2theta, vec_hist

    @staticmethod
    def FitPeaks(x, y, Params, Peak_Num):

        def CalcPatt(x, y, PAR, Peak_Num):
            Model = np.zeros_like(x)
            Model += quadratic_background(x, PAR['p0'], PAR['p1'], PAR['p2'])
            for ipeak in Peak_Num:
                Model += GaussianModel(x, PAR['g%d_center'%ipeak], PAR['g%d_sigma'%ipeak], PAR['g%d_amplitude'%ipeak])
            return Model 

        def residual(x0, x, y, ParamNames, Peak_Num):
            PAR = dict(zip(ParamNames, x0))
            Model = CalcPatt(x, y, PAR, Peak_Num)    
            return (y-Model) / np.sqrt(y)
        
        x0 = list()
        ParamNames = list()
        LL, UL = [], []
        
        Params['p0'] = [y[0], -np.inf, np.inf]
        for pkey in list(Params.keys()):
            x0.append(Params[pkey][0])
            LL.append(Params[pkey][1])
            UL.append(Params[pkey][2])   
            
            ParamNames.append(pkey)
            
        if UseLSQ:            
            out = leastsq(residual, x0, args=(x, y, ParamNames, Peak_Num), Dfun=None, ftol=1e-8, xtol=1e-8,
                          gtol=1e-8, maxfev=0, factor=1.0)
            returnSetup = [dict(zip(out[0], ParamNames)), CalcPatt(x, y, dict(zip(out[0], ParamNames)), Peak_Num)]
        else:
            out = least_squares(residual, x0, bounds=[LL, UL], method='dogbox', ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                f_scale=1.0, max_nfev=None, args=(x, y, ParamNames, Peak_Num))
            returnSetup = [dict(zip(ParamNames, out.x)), CalcPatt(x, y, dict(zip(ParamNames, out.x)), Peak_Num)]

        return returnSetup
        
    def FitDetector(self, fun, x0, jac='2-point', bounds=[], method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
                    x_scale=1.0, loss='linear', \
                    f_scale=1.0, diff_step=None, tr_solver=None, max_nfev=None, verbose=0, args=(),
                    kwargs={}, full_output=0, col_deriv=0, maxfev=0):
        
        if UseLSQ:
            if type(max_nfev) == type(None):max_nfev=0
            out = leastsq(self.peak_alignment_rotation, x0, args=args, Dfun=None, ftol=ftol, xtol=xtol, gtol=gtol,
                          maxfev=max_nfev, factor=f_scale)
            
        else:
            if len(bounds[0]) != len(bounds[1]):
                raise RuntimeError('User must specify bounds of equal length')
            
            out = least_squares(self.peak_alignment_rotation, x0, jac=jac, bounds=bounds, method='dogbox',
                                ftol=ftol, xtol=xtol, gtol=gtol, \
                                x_scale=x_scale, loss=loss, f_scale=f_scale, diff_step=diff_step,
                                tr_solver=tr_solver, max_nfev=max_nfev, args=args)
        
        return out
    
    def get_alignment_residual(self, x, roi_vec_set=None):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        GlobalParameter.global_curr_sequence += 1

        residual = np.array([])
        # resNone = 0.

        two_theta_calib = np.arcsin(x[6] / 2. / dSpace) * 360. / np.pi
        two_theta_calib = two_theta_calib[~np.isnan(two_theta_calib)]

        # background = LinearModel()

        for i_tth in range(self._engine.get_log_value('2Theta').shape[0]):
            # reduced_data_set[i_tth] = [None] * num_reduced_set
            # load instrument: as it changes
            pyrs_reducer = reduce_hb2b_pyrs.PyHB2BReduction(self._instrument, x[6])
            pyrs_reducer.build_instrument_prototype(-1. * self._engine.get_log_value('2Theta')[i_tth],
                                                    self._instrument._arm_length,
                                                    x[0], x[1], x[2], x[3], x[4], x[5])
            pyrs_reducer._detector_counts = self._engine.get_raw_counts(i_tth)

            DetectorAngle = self._engine.get_log_value('2Theta')[i_tth]
            mintth = DetectorAngle-8.0
            maxtth = DetectorAngle+8.7

            Eta_val = pyrs_reducer.get_eta_Values()
            maxEta = np.max(Eta_val) - 2
            minEta = np.min(Eta_val) + 2

#            FitModel = Model(quadratic_background)
#            pars1 = FitModel.make_params(p0=100, p1=1, p2=0.01)

            CalibPeaks = two_theta_calib[np.where((two_theta_calib > mintth) == (two_theta_calib < maxtth))[0]]
            for ipeak in range(len(CalibPeaks)):
                Peaks = []
                pars1 = {}
                pars1['p1'] = [0, -np.inf, np.inf] 
                pars1['p2'] = [0, -np.inf, np.inf] 
                if (CalibPeaks[ipeak] > mintth) and (CalibPeaks[ipeak] < maxtth):

                    Peaks.append(ipeak)
#                    PeakModel = GaussianModel(prefix='g%d_' % ipeak)
#                    FitModel += PeakModel

#                    pars1.update(PeakModel.make_params())
#                    pars1['g%d_center' % ipeak].set(value=CalibPeaks[ipeak], min=CalibPeaks[ipeak] - 2,
#                                                    max=CalibPeaks[ipeak] + 2)
#                    pars1['g%d_sigma' % ipeak].set(value=0.5, min=1e-1, max=1.5)
#                    pars1['g%d_amplitude' % ipeak].set(value=50., min=10, max=1e6)
                    pars1['g%d_center' % ipeak] = [CalibPeaks[ipeak], CalibPeaks[ipeak] - 2,
                                                    CalibPeaks[ipeak] + 2]
                    pars1['g%d_sigma' % ipeak] = [0.5, 1e-1, 1.5]
                    pars1['g%d_amplitude' % ipeak] = [50., 10, 1e6]

            if roi_vec_set is None:
                eta_roi_vec = np.arange(minEta, maxEta + 0.2, 2)
            else:
                eta_roi_vec = np.array(roi_vec_set)

            num_rows = 1 + len(Peaks) / 2 + len(Peaks) % 2
#            ax1 = plt.subplot(num_rows, 1, num_rows)
#            ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit

            for i_roi in range(eta_roi_vec.shape[0]):
                # ws_name_i = 'reduced_data_{:02}'.format(i_roi)
                # out_peak_pos_ws = 'peaks_positions_{:02}'.format(i_roi)
                # fitted_ws = 'fitted_peaks_{:02}'.format(i_roi)

                # Define Mask
                Mask = np.zeros_like(Eta_val)
                if abs(eta_roi_vec[i_roi]) == eta_roi_vec[i_roi]:
                    index = np.where((Eta_val < (eta_roi_vec[i_roi]+1)) == (Eta_val > (eta_roi_vec[i_roi] - 1)))[0]
                else:
                    index = np.where((Eta_val > (eta_roi_vec[i_roi]-1)) == (Eta_val < (eta_roi_vec[i_roi] + 1)))[0]

                Mask[index] = 1.

                # reduce
                reduced_i = self.convert_to_2theta(pyrs_reducer, Mask, min_2theta=mintth, max_2theta=maxtth,
                                                   num_bins=720)

                # fit peaks
#                Fitresult = FitModel.fit(reduced_i[1], pars1, x=reduced_i[0])
                Fitresult = self.FitPeaks(reduced_i[0], reduced_i[1], pars1, Peaks)

                for p_index in Peaks:
                    residual = np.concatenate([residual,
                                               np.array([(100.0 * (Fitresult[0]['g%d_center' % p_index] -
                                                                   CalibPeaks[p_index]))])])

                # plot results
                backgroundShift = np.average(quadratic_background(reduced_i[0],
                                                                  Fitresult[0]['p0'],
                                                                  Fitresult[0]['p1'],
                                                                  Fitresult[0]['p2']))
                
#                ax1.plot(reduced_i[0], reduced_i[1], color=colors[i_roi % 5])

#                for index_i in range(len(Peaks)):
#                    ax2 = plt.subplot(num_rows, 2, index_i+1)
#                    ax2.plot(reduced_i[0], reduced_i[1], 'x', color='black')
#                    ax2.plot(reduced_i[0], Fitresult[1], color='red')
#                    ax2.plot([CalibPeaks[index_i], CalibPeaks[index_i]],
#                             [backgroundShift, backgroundShift + Fitresult.params['g0_amplitude'].value],
#                             'k', linewidth=2)
#                    ax2.set_xlim([CalibPeaks[index_i]-1.5, CalibPeaks[index_i] + 1.5])
                # END-FOR

            # Optionally save the least square figure of this round for further reference
#            if os.path.exists('FitFigures'):
#                plt.savefig('./FitFigures/Round{:010}_{:02}.png'.format(GlobalParameter.global_curr_sequence, i_tth))
#                plt.clf()
        # END-FOR(tth)

        print ("\n")
        print ('Iteration      {}'.format(GlobalParameter.global_curr_sequence))
        print ('RMSE         = {}'
               ''.format(np.sqrt(residual.sum()**2 / (len(eta_roi_vec) *
                                                      self._engine.get_log_value('2Theta').shape[0]))))
        print ('Residual Sum = {}'.format(np.sum(residual) / 100.))
        print ("\n")

        return residual

    def peak_alignment_wavelength(self, x):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """
        # self.check_alignment_inputs(roi_vec_set)
        roi_vec_set = None
        paramVec = np.copy(self._calib)
        paramVec[6] = x[0]

        return self.get_alignment_residual(paramVec, roi_vec_set)

    def peak_alignment_shift(self, x, roi_vec_set=None, return_scalar=False):
        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        self.check_alignment_inputs(roi_vec_set)

        paramVec = np.copy(self._calib)
        paramVec[0:3] = x[:]

        print ('\n')
        print (paramVec)
        print ('\n')

        residual = self.get_alignment_residual(paramVec, roi_vec_set)

        if return_scalar:
            return np.sum(residual)

        return residual

    def peak_alignment_rotation(self, x, roi_vec_set=None, return_scalar=False):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        self.check_alignment_inputs(roi_vec_set)

        paramVec = np.copy(self._calib)
        paramVec[3:6] = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set)

        if return_scalar:
            return np.sum(residual)

        return residual

    def peaks_alignment_all(self, x, roi_vec_set=None, return_scalar=False):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        self.check_alignment_inputs(roi_vec_set)

        residual = self.get_alignment_residual(x, roi_vec_set)

        if return_scalar:
            return np.sum(residual)

        return residual

    def singleEval(self, x=None, roi_vec_set=None, return_scalar=False):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        GlobalParameter.global_curr_sequence = -10

        if x is None:
            residual = self.get_alignment_residual(self._calib, roi_vec_set)
        else:
            residual = self.get_alignment_residual(x, roi_vec_set)

        if return_scalar:
            residual = np.sum(residual)

        return residual

    def calibrate_wave_length(self, initial_guess=None):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            initial_guess = self.get_wavelength()

        out = least_squares(self.peak_alignment_wavelength, initial_guess, jac='2-point',
                            bounds=([self._calib[6]-.05], [self._calib[6]+.05]), method='dogbox',
                            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,
                            diff_step=None, tr_solver='exact',
                            tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})

        self.set_wavelength(out)

        return

    def CalibrateShift(self, initalGuess=None):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_shift()

        out = least_squares(self.peak_alignment_shift, initalGuess, jac='2-point',
                            bounds=([-.05, -.05, -.05], [.05, .05, .05]), method='dogbox',
                            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                            f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={},
                            jac_sparsity=None, max_nfev=None, verbose=0, args=(None, False), kwargs={})

        self.set_shift(out)

        return

    def CalibrateRotation(self, initalGuess=None):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_rotation()

        out = least_squares(self.peak_alignment_rotation, initalGuess, jac='3-point',
                            bounds=([-np.pi/20, -np.pi/20, -np.pi/20], [np.pi/20, np.pi/20, np.pi/20]),
                            method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                            f_scale=1.0, diff_step=None,
                            tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                            args=(None, False), kwargs={})

        self.set_rotation(out)

        return

    def FullCalibration(self, initalGuess=None):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_calib()

        out = least_squares(self.peaks_alignment_all, initalGuess, jac='3-point',
                            bounds=([-.05, -.05, -.05, -np.pi/20, -np.pi/20, -np.pi/20, 1.4],
                                    [.05, .05, .05, np.pi/20, np.pi/20, np.pi/20, 1.5]),
                            method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
                            loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={},
                            jac_sparsity=None, max_nfev=None, verbose=0,
                            args=(None, False), kwargs={})

        self.set_calibration(out)

        return

    def get_calib(self):
        return np.array(self._calib)

    def get_shift(self):
        return np.array([self._calib[0], self._calib[1], self._calib[2]])

    def get_rotation(self):
        return np.array([self._calib[3], self._calib[4], self._calib[5]])

    def get_wavelength(self):
        return np.array([self._calib[6]])

    def set_shift(self, out):
        self._calib[0:3] = out.x
        self._calibstatus = out.status

        J = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[0:3] = var

        return

    def set_rotation(self, out):
        self._calib[3:6] = out.x
        self._calibstatus = out.status

        J = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[3:6] = var

        return

    def set_wavelength(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """
        self._calib[6] = out.x[0]
        self._calibstatus = out.status

        J = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr[6] = var

        return

    def set_calibration(self, out):
        """Set calibration to calibration data structure

        Parameters
        ----------
        out

        Returns
        -------

        """
        self._calib = out.x
        self._calibstatus = out.status

        J = out.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        self._caliberr = var

        return

    def get_archived_calibration(self):
        """Get calibration from archived JSON file

        Output: result is written to self._calib[i]

        Returns
        -------
        None
        """
        # Monochromator setting
        mono_setting_index = self._engine.get_log_value('MonoSetting')[0]
        mono_setting = ['Si333', 'Si511', 'Si422', 'Si331', 'Si400', 'Si311', 'Si220'][mono_setting_index]

        for files in glob.glob('/HFIR/HB2B/shared/CAL/{}/*.json'.format(mono_setting)):
            # get date
            datetime = files.split('.json')[0].split('_CAL_')[1]
            if dateutil.parser.parse(datetime) < dateutil.parser.parse(self._engine.get_log_value('MonoSetting')[0]):
                CalibData = json.read(files)
                keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda']
                for i in range(len(keys)):
                    self._calib[i] = CalibData[keys[i]]

        return

    # TODO - #86 - Clean up!
    def write_calibration(self, file_name=None):
        """Write the calibration to a Json file

        Parameters
        ----------
        file_name: str or None
            output Json file name.  If None, write to /HFIR/HB2B/shared/CAL/

        Returns
        -------
        None
        """
        from pyrs.core.instrument_geometry import AnglerCameraDetectorShift
        #
        # CalibData = dict(zip(['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda'],
        #                      self._calib))
        # CalibData.update(dict(zip(['error_Shift_x', 'error_Shift_y', 'error_Shift_z', 'error_Rot_x', 'error_Rot_y',
        #                            'error_Rot_z', 'error_Lambda'], self._caliberr)))
        # CalibData.update({'Status': self._calibstatus})

        # Year, Month, Day, Hour, Min = time.localtime()[0:5]
        mono_setting_index = self._engine.get_log_value('MonoSetting')[0]
        Mono = ['Si333', 'Si511', 'Si422', 'Si331', 'Si400', 'Si311', 'Si220'][mono_setting_index]

        # Form AnglerCameraDetectorShift objects
        cal_shift = AnglerCameraDetectorShift(self._calib[0], self._calib[1], self._calib[2], self._calib[3],
                                              self._calib[4], self._calib[5])
        cal_shift_error = AnglerCameraDetectorShift(self._caliberr[0], self._caliberr[1], self._caliberr[2],
                                                    self._caliberr[3], self._caliberr[4], self._caliberr[5])
        wl = self._calib[6]
        wl_error = self._caliberr[6]

        # Determine output file name
        if file_name is None:
            # default case: write to archive
            if os.access('/HFIR/HB2B/shared', os.W_OK):
                file_name = '/HFIR/HB2B/shared/CAL/%s/HB2B_CAL_%s.json' % (Mono,
                                                                           time.strftime('%Y-%m-%dT%H:%M',
                                                                                         time.localtime()))
            else:
                raise IOError('User does not privilege to write to {}'.format('/HFIR/HB2B/shared'))
        # END-IF

        write_calibration_to_json(cal_shift, cal_shift_error, wl, wl_error, self._calibstatus, file_name)

        # with open(file_name, 'w') as outfile:
        #     json.dump(CalibData, outfile)
        # print('[INFO] Calibration file is written to {}'.format(file_name))

        return
