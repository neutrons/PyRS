# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
import numpy as np
import time
import json
import os

# Import pyrs modules
from pyrs.core import MonoSetting  # type: ignore
from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from pyrs.core.reduction_manager import HB2BReductionManager
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

# Import instrument constants
from pyrs.core.nexus_conversion import NUM_PIXEL_1D, PIXEL_SIZE, ARM_LENGTH

# Import scipy libraries for minimization
from scipy.optimize import least_squares
from scipy.optimize import brute
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping


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
        background    calibrator.eta_slices = 10

    """
    return p2*x*x + p1*x + p0


def linear_background(x, p0, p1, p2):
    """linear background

    Y = p1 * x + p0

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


class GlobalParameter:
    global_curr_sequence = 0
    current_plot = 0

    def __init__(self):
        return


def get_ref_flags(powder_engine, pin_engine):
    def get_mono_setting(_engine):
        try:
            monosetting = MonoSetting.getFromIndex(_engine.get_sample_log_value('MonoSetting', 1))
        except ValueError:
            monosetting = MonoSetting.getFromRotation(_engine.get_sample_log_value('mrot', 1))

        return monosetting

    def get_tth_ref(_engine):
        try:
            _engine.get_sample_log_value('2theta', 1)
            tth_ref = '2theta'
        except ValueError:
            tth_ref = '2Theta'
        return tth_ref

    if (powder_engine is not None) and (pin_engine is not None):
        if get_mono_setting(powder_engine) == get_mono_setting(pin_engine):
            monosetting = get_mono_setting(powder_engine)
        else:
            raise RuntimeError('Powder and Pin data measured using different mono settings')

        if get_tth_ref(powder_engine) == get_tth_ref(pin_engine):
            tth_ref = get_tth_ref(powder_engine)
        else:
            raise RuntimeError('Powder and Pin data have different 2theta reference keys\n')
    elif powder_engine is not None:
        monosetting = get_mono_setting(powder_engine)
        tth_ref = get_tth_ref(powder_engine)
    elif pin_engine is not None:
        monosetting = get_mono_setting(pin_engine)
        tth_ref = get_tth_ref(pin_engine)
    else:
        raise RuntimeError('No data were provided as an input\n')

    return monosetting, tth_ref


class PeakFitCalibration:
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """

    def __init__(self, hb2b_inst=None, powder_engine=None, pin_engine=None, powder_lines=None,
                 single_material=True, wavelength=None):
        """
        Initialization

        Parameters
        ----------
        hb2b_inst : DENEXDetectorGeometry
            Overide default instrument configuration
        powder_engine : HiDraWorksapce
            HiDraWorksapce with powder raw counts and log data
        pin_engine : HiDraWorksapce
            HiDraWorksapce with pin raw counts and log data
        powder_lines : list
            list of dspace for reflections in the field of view during the experiment
        single_material : bool
            Flag if powder data are from a single material

        """

        self.engines = []

        # define instrument setup
        if hb2b_inst is None:
            self._instrument = DENEXDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D,
                                                     PIXEL_SIZE, PIXEL_SIZE,
                                                     ARM_LENGTH, False)
        else:
            self._instrument = hb2b_inst

        if pin_engine is not None:
            dSpace = 3.59188696 * np.array([1./np.sqrt(11), 1./np.sqrt(12)])
            self.engines.append([pin_engine, dSpace, True])

        if powder_engine is not None:
            if powder_lines is None:
                raise RuntimeError('User must define a list of dspace')

            self.engines.append([powder_engine, powder_lines, single_material])

        # calibration: numpy array. size as 7 for ... [6] for wave length
        self._calib = np.array(8 * [0], dtype=np.float)
        # calibration error: numpy array. size as 7 for ...
        self._caliberr = np.array(8 * [-1], dtype=np.float)

        # calibration starting point: numpy array. size as 7 for ...
        self._calib_start = np.array(8 * [0], dtype=np.float)

        if wavelength is None:
            self.monosetting, self.tth_ref = get_ref_flags(powder_engine, pin_engine)

            # Set wave length
            self._calib[6] = float(self.monosetting)
            self._calib_start[6] = float(self.monosetting)
        else:
            self.monosetting = 1
            self._calib[6] = wavelength
            self._calib_start[6] = wavelength


        # Initalize calibration status to -1
        self._calibstatus = -1

        self.ReductionResults = {}
        self._residualpoints = None
        self.singlepeak = False

        self.refinement_summary = ''

        self.plot_res = False

        # define inital peak fitting data
        self.min_tth = None
        self.max_tth = None
        self.min_eta = None
        self.max_eta = None
        self.min_subrun = 0
        self.max_subrun = 0
        self.eta_slices = 3
        self.bins = 512
        self.inital_width = 0.5
        self.inital_AMP = 0.5

        GlobalParameter.global_curr_sequence = 0

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
    class RandomDisplacementBounds(object):
        """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            Modified! (dropped acceptance-rejection sampling for a more specialized approach)
        """
        def __init__(self, xmin, xmax, stepsize=0.5):
            self.xmin = xmin
            self.xmax = xmax
            self.stepsize = stepsize

        def __call__(self, x):
            """take a random step but ensure the new position is within the bounds """
            min_step = np.maximum(self.xmin - x, -self.stepsize)
            max_step = np.minimum(self.xmax - x, self.stepsize)

            random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
            xnew = x + random_step

            return xnew


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
        # Default minimum and maximum 2theta are related with
        pixel_2theta_array = pyrs_reducer.instrument.get_pixels_2theta(1)

        bin_boundaries_2theta = HB2BReductionManager.generate_2theta_histogram_vector(min_2theta, num_bins,
                                                                                      max_2theta,
                                                                                      pixel_2theta_array,
                                                                                      roi_vec)

        # Histogram
        data_set = pyrs_reducer.reduce_to_2theta_histogram(bin_boundaries_2theta,
                                                           mask_array=roi_vec,
                                                           is_point_data=True,
                                                           vanadium_counts_array=None)

        vec_2theta, vec_hist = data_set[:2]

        return vec_2theta, vec_hist

    def plot_data(self, x, y, fit):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y, 'kx')
        ax.plot(x, fit, 'b')
        ax.plot(x, y-fit, 'r')
        fig.savefig('fit_{}.png'.format(GlobalParameter.current_plot))
        GlobalParameter.current_plot += 1

        return

    def FitPeaks(self, x, y, Params, Peak_Num):

        def CalcPatt(x, y, PAR, Peak_Num):
            Model = np.zeros_like(x)
            Model += quadratic_background(x, PAR['p0'], PAR['p1'], PAR['p2'])
            for ipeak in Peak_Num:
                Model += GaussianModel(x, PAR['g%d_center' % ipeak], PAR['g%d_sigma' %
                                                                         ipeak], PAR['g%d_amplitude' % ipeak])
            return Model

        def residual(x0, x, y, ParamNames, Peak_Num):
            PAR = dict(zip(ParamNames, x0))
            Model = CalcPatt(x, y, PAR, Peak_Num)
            return (y-Model)

        x0 = list()
        ParamNames = list()
        LL, UL = [], []

        Params['p0'] = [y[0], -np.inf, np.inf]

        for pkey in list(Params.keys()):
            x0.append(Params[pkey][0])
            LL.append(Params[pkey][1])
            UL.append(Params[pkey][2])

            ParamNames.append(pkey)

        # fit the functions
        out = least_squares(residual, x0, bounds=[LL, UL], method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8,
                            f_scale=1.0, max_nfev=None, args=(x, y, ParamNames, Peak_Num))

        if self.plot_res:
            self.plot_data(x, y, CalcPatt(x, y, dict(zip(ParamNames, x0)), Peak_Num))
            self.plot_data(x, y, CalcPatt(x, y, dict(zip(ParamNames, out.x)), Peak_Num))

        returnSetup = [dict(zip(ParamNames, out.x)), CalcPatt(x, y, dict(zip(ParamNames, out.x)), Peak_Num),
                       out.status]

        return returnSetup

    def FitDetector(self, fun, x0, jac='3-point', bounds=[], method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
                    x_scale=1.0, loss='linear', tr_options={}, jac_sparsity=None, f_scale=1.0, diff_step=None,
                    tr_solver=None, max_nfev=None, verbose=0, ROI=None, ConPos=False, kwargs='', epsfcn=1e-6,
                    factor=100.0, i_index=2, Brute=False, fDiff=1e-9):

        self.check_alignment_inputs(ROI)

        BOUNDS = []
        lL = bounds[1]
        uL = bounds[0]
        for i_b in range(len(uL)):
            BOUNDS.append([lL[i_b], uL[i_b]])

        if Brute is True:
            out1 = brute(fun, ranges=BOUNDS, args=(ROI, ConPos, True, i_index), Ns=11)
            return [out1, np.array([0]), 1]
        elif Brute == 2:
            out1 = differential_evolution(fun, bounds=BOUNDS, args=(ROI, ConPos, True, i_index))
            return [out1.x, np.array([0]), 1]
        elif Brute == 3:
            bounded_step = self.RandomDisplacementBounds(np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS]))
            minimizer_kwargs = {"method":"L-BFGS-B", "args":(ROI, ConPos, True, i_index), "bounds": BOUNDS}
#            out1 = basinhopping(fun, x0, minimizer_kwargs=minimizer_kwargs, take_step=bounded_step)
            out1 = basinhopping(fun, x0, minimizer_kwargs=minimizer_kwargs)
            return [out1.x, np.array([0]) * out1.x.shape[0], 1]
        else:
            if len(bounds[0]) != len(bounds[1]):
                raise RuntimeError('User must specify bounds of equal length')

            if len(x0) != len(bounds[1]):
                raise RuntimeError('User must specify bounds of equal length')

            self._residualpoints = self.singleEval(ConstrainPosition=ConPos).shape[0]
            GlobalParameter.global_curr_sequence = 0

            out = least_squares(fun, x0, jac=jac, bounds=bounds, method=method,
                                ftol=ftol, xtol=xtol, gtol=gtol, tr_options=tr_options,
                                jac_sparsity=jac_sparsity, x_scale=x_scale, loss=loss, f_scale=f_scale,
                                diff_step=diff_step, tr_solver=tr_solver, max_nfev=max_nfev,
                                args=(ROI, ConPos, False, i_index))

            J = out.jac

            if np.sum(J.T.dot(J)) < 1e-8:
                var = -2 * np.zeros_like(J.T.dot(J))
            else:
                try:
                    cov = np.linalg.inv(J.T.dot(J))
                    var = np.sqrt(np.diagonal(cov))
                except np.linalg.LinAlgError:
                    var = -2 * np.zeros_like(out.x)

            return [out.x, var, out.status]

    def calculate_1d_pattern(self):

        x = self._calib

        pyrs_reducer = PyHB2BReduction(self._instrument, x[6])

        reduction = []
        for engine_setup in self.engines:
            datasets, dSpace, single_material = engine_setup

            self._engine = datasets

            two_theta_calib = np.arcsin(x[6] / 2. / dSpace) * 360. / np.pi
            two_theta_calib = two_theta_calib[~np.isnan(two_theta_calib)]

            if datasets is None:
                sub_runs = []
            else:
                sub_runs = self._engine.get_sub_runs()

            reduced_data = np.zeros((sub_runs.shape[0], 2, self.bins))

            for i_tth in sub_runs:

                pyrs_reducer.build_instrument_prototype(-1. * self._engine.get_sample_log_value('2theta', i_tth),
                                                        self._instrument._arm_length,
                                                        x[0], x[1], x[2], x[3], x[4], x[5], x[7])

                # Load raw counts
                pyrs_reducer._detector_counts = self._engine.get_detector_counts(i_tth)

                tths = pyrs_reducer._instrument.get_pixels_2theta(1)

                if self.min_tth is None:
                    self.min_tth = tths.min() + .5
                    self.max_tth = tths.max() - .5

                reduced_data[i_tth - 1, :, :] = self.convert_to_2theta(pyrs_reducer,
                                                                       np.ones_like(tths),
                                                                       min_2theta=tths.min(),
                                                                       max_2theta=tths.max(),
                                                                       num_bins=self.bins)

                pyrs_reducer._detector_counts = np.ones_like(tths)

                reduced_data[i_tth - 1, 1, :] /= self.convert_to_2theta(pyrs_reducer,
                                                                        np.ones_like(tths),
                                                                        min_2theta=tths.min(),
                                                                        max_2theta=tths.max(),
                                                                        num_bins=self.bins)[1]

            reduction.append(reduced_data)

        return reduction

    def get_sub_runs(self, dataset):
        '''
        get subruns from datset

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.

        Returns
        -------
        sub_runs : np.array
            array of subrun names.

        '''
        if dataset is None:
            sub_runs = []
        elif self.max_subrun == 0:
            sub_runs = dataset.get_sub_runs()
        else:
            sub_runs = dataset.get_sub_runs()[self.min_subrun:self.max_subrun]

        return sub_runs

    def get_alignment_residual(self, x, roi_vec_set=None, ConPeaks=False, ReturnFit=False):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        pyrs_reducer = PyHB2BReduction(self._instrument, x[6])

        GlobalParameter.global_curr_sequence += 1
        residual = np.array([])

        if ConPeaks:
            for engine_setup in self.engines:
                datasets, dSpace, single_material = engine_setup

                self._engine = datasets

                two_theta_calib = np.arcsin(x[6] / 2. / dSpace) * 360. / np.pi
                two_theta_calib = two_theta_calib[~np.isnan(two_theta_calib)]

                sub_runs = self.get_sub_runs(datasets)

                for i_tth in sub_runs:
                    if ReturnFit:
                        self.ReductionResults[i_tth] = {}

                    pyrs_reducer.build_instrument_prototype(-1. * self._engine.get_sample_log_value('2theta', i_tth),
                                                            self._instrument._arm_length,
                                                            x[0], x[1], x[2], x[3], x[4], x[5], x[7])

                    # Load raw counts
                    pyrs_reducer._detector_counts = self._engine.get_detector_counts(i_tth)

                    tths = pyrs_reducer._instrument.get_pixels_2theta(1)
                    Eta_val = pyrs_reducer.get_eta_value()

                    if self.min_tth is None:
                        self.min_tth = tths.min() + .5
                        self.max_tth = tths.max() - .5

                    if self.min_eta is None:
                        self.max_eta = Eta_val.max() - 2
                        self.min_eta = Eta_val.min() + 2

                    if roi_vec_set is None:
                        eta_delta = (self.max_eta - self.min_eta) / self.eta_slices

                        eta_roi_vec = np.arange(self.min_eta + eta_delta / 2,
                                                self.max_eta - eta_delta / 2,
                                                self.eta_slices)
                    else:
                        eta_roi_vec = np.array(roi_vec_set)

                    resq = []

                    if single_material:
                        CalibPeaks = two_theta_calib[np.where((two_theta_calib > self.min_tth) ==
                                                              (two_theta_calib < self.max_tth))[0]]
                    else:
                        CalibPeaks = np.array([two_theta_calib[i_tth - 1]])

                    if CalibPeaks.shape[0] > 0 and (not ConPeaks or self.singlepeak):
                        CalibPeaks = np.array([CalibPeaks[0]])

                    if CalibPeaks.shape[0] >= 1:
                        Peaks = []
                        pars1 = {}
                        pars1['p1'] = [0, -np.inf, np.inf]
                        pars1['p2'] = [0, -np.inf, np.inf]

                        for ipeak in range(len(CalibPeaks)):
                            if (CalibPeaks[ipeak] > self.min_tth) and (CalibPeaks[ipeak] < self.max_tth):
                                resq.append([])
                                Peaks.append(ipeak)

                                pars1['g%d_center' % ipeak] = [CalibPeaks[ipeak], CalibPeaks[ipeak] - 0.5,
                                                               CalibPeaks[ipeak] + 0.5]
                                pars1['g%d_sigma' % ipeak] = [self.inital_width, 1e-3, 1.5]
                                pars1['g%d_amplitude' % ipeak] = [self.inital_AMP, 0.001, 1e9]

                        for i_roi in range(eta_roi_vec.shape[0]):
                            # Define Mask
                            Mask = np.zeros_like(Eta_val)
                            if abs(eta_roi_vec[i_roi]) == eta_roi_vec[i_roi]:
                                index = np.where((Eta_val < (eta_roi_vec[i_roi] + 1)) ==
                                                 (Eta_val > (eta_roi_vec[i_roi] - 1)))[0]
                            else:
                                index = np.where((Eta_val > (eta_roi_vec[i_roi] - 1)) ==
                                                 (Eta_val < (eta_roi_vec[i_roi] + 1)))[0]

                            Mask[index] = 1.

                            # reduce
                            reduced_i = self.convert_to_2theta(pyrs_reducer, Mask, min_2theta=self.min_tth,
                                                               max_2theta=self.max_tth, num_bins=self.bins)

                            # fit peaks
                            Fitresult = self.FitPeaks(reduced_i[0], reduced_i[1], pars1, Peaks)

                            if ReturnFit:
                                self.ReductionResults[i_tth][(i_roi, GlobalParameter.global_curr_sequence)] = \
                                                        [reduced_i[0], reduced_i[1], Fitresult[1]]

                            if Fitresult[2] == 5 or Fitresult[2] < 2:
                                pass

                            elif ConPeaks:
                                for p_index in Peaks:
                                    if Fitresult[0]['g%d_center' % p_index] == CalibPeaks[p_index]:
                                        residual = np.concatenate([residual, np.array([1000.])])
                                    else:
                                        residual = np.concatenate([residual,
                                                                   np.array([(Fitresult[0]['g%d_center' % p_index] /
                                                                              CalibPeaks[p_index]) - 1])])
                            else:
                                for p_index in Peaks:
                                    residual = np.concatenate([residual,
                                                               np.array([((Fitresult[0]['g%d_center' % p_index]))])])
        else:
            reduced_data = self.get_2d_reduced_data(x)

            for i in range(reduced_data.shape[0]):
                for j in range(i+1, reduced_data.shape[0]):
                    residual = np.concatenate([residual,
                                               np.array([1. / np.corrcoef(reduced_data[i], reduced_data[j])[0, 1]])])
            residual = np.abs(np.array(residual)) - 1

        # if not ConPeaks:
        #     residual -= np.average(residual)

        if self._residualpoints is not None:
            if residual.shape[0] < self._residualpoints:
                residual = np.concatenate([residual, np.array([1000.0] * (self._residualpoints-residual.shape[0]))])
            elif residual.shape[0] > self._residualpoints:
                residual = residual[:self._residualpoints]
        
        print("")
        print(x)
        print('Iteration      {}'.format(GlobalParameter.global_curr_sequence))
        print('RMSE         = {}'.format(np.sqrt((residual**2).sum() / residual.shape[0])))
        print('Residual Sum = {}'.format(np.sum(residual)))
        print('Residual min = {}'.format(residual.min()))
        print('Residual max = {}'.format(residual.max()))
        print('Residual avg delta = {}'.format(np.abs(residual).sum() / residual.shape[0]))

        if np.all(residual == 0.):
            residual += 1000

        return residual

    def singleEval(self, x=None, roi_vec_set=None, ConstrainPosition=True, ReturnFit=True, ReturnScalar=False):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        GlobalParameter.global_curr_sequence = -10

        if x is None:
            residual = self.get_alignment_residual(self._calib, roi_vec_set, ConstrainPosition,
                                                   ReturnFit)
        else:
            residual = self.get_alignment_residual(x, roi_vec_set, ConstrainPosition, ReturnFit)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def get_2d_reduced_data(self, x):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        roi_vec_set = None
        ConPeaks = False
        ReturnFit = False

        pyrs_reducer = PyHB2BReduction(self._instrument, x[6])

        for engine_setup in self.engines:

            datasets, dSpace, single_material = engine_setup

            self._engine = datasets

            two_theta_calib = np.arcsin(x[6] / 2. / dSpace) * 360. / np.pi
            two_theta_calib = two_theta_calib[~np.isnan(two_theta_calib)]

            sub_runs = self.get_sub_runs(datasets)

            for i_tth in sub_runs:
                if ReturnFit:
                    self.ReductionResults[i_tth] = {}

                # print(self._instrument._instrument_geom_params.detector_size)
                pyrs_reducer.build_instrument_prototype(-1. * self._engine.get_sample_log_value('2theta', i_tth),
                                                        self._instrument._arm_length,
                                                        x[0], x[1], x[2], x[3], x[4], x[5], x[7])

                # Load raw counts
                pyrs_reducer._detector_counts = self._engine.get_detector_counts(i_tth)

                tths = pyrs_reducer._instrument.get_pixels_2theta(1)
                Eta_val = pyrs_reducer.get_eta_value()

                if self.min_tth is None:
                    self.min_tth = tths.min() + .5
                    self.max_tth = tths.max() - .5

                if self.min_eta is None:
                    self.max_eta = Eta_val.max() - 2
                    self.min_eta = Eta_val.min() + 2

                if roi_vec_set is None:
                    eta_roi_vec = np.arange(self.min_eta, self.max_eta, self.eta_slices)
                else:
                    eta_roi_vec = np.array(roi_vec_set)

                if single_material:
                    CalibPeaks = two_theta_calib[np.where((two_theta_calib > self.min_tth) ==
                                                          (two_theta_calib < self.max_tth))[0]]
                else:
                    CalibPeaks = np.array([two_theta_calib[i_tth - 1]])

                if CalibPeaks.shape[0] > 0 and (not ConPeaks or self.singlepeak):
                    CalibPeaks = np.array([CalibPeaks[0]])

                reduced_data = np.zeros((eta_roi_vec.shape[0], self.bins))
                if CalibPeaks.shape[0] >= 1:
                    Peaks = []
                    pars1 = {}
                    pars1['p1'] = [0, -np.inf, np.inf]
                    pars1['p2'] = [0, -np.inf, np.inf]

                    for ipeak in range(len(CalibPeaks)):
                        if (CalibPeaks[ipeak] > self.min_tth) and (CalibPeaks[ipeak] < self.max_tth):
                            Peaks.append(ipeak)
                            pars1['g%d_center' % ipeak] = [CalibPeaks[ipeak], CalibPeaks[ipeak] - 0.5,
                                                           CalibPeaks[ipeak] + 0.5]
                            pars1['g%d_sigma' % ipeak] = [self.inital_width, 1e-3, 1.5]
                            pars1['g%d_amplitude' % ipeak] = [self.inital_AMP, 0.001, 1e9]

                    for i_roi in range(eta_roi_vec.shape[0]):
                        # Define Mask
                        # refactor to own function
                        Mask = np.zeros_like(Eta_val)
                        if abs(eta_roi_vec[i_roi]) == eta_roi_vec[i_roi]:
                            index = np.where((Eta_val < (eta_roi_vec[i_roi] + 1)) ==
                                             (Eta_val > (eta_roi_vec[i_roi] - 1)))[0]
                        else:
                            index = np.where((Eta_val > (eta_roi_vec[i_roi] - 1)) ==
                                             (Eta_val < (eta_roi_vec[i_roi] + 1)))[0]

                        Mask[index] = 1.

                        # reduce
                        reduced_data[i_roi, :] = self.convert_to_2theta(pyrs_reducer, Mask, min_2theta=self.min_tth,
                                                                        max_2theta=self.max_tth, num_bins=self.bins)[1]

        return reduced_data

    def peak_alignment_wavelength(self, x, roi_vec_set=None, ConstrainPosition=True, ReturnScalar=False,
                                  i_index=2):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """
        paramVec = np.copy(self._calib)
        # paramVec[0] = x[0]
        paramVec[6] = x

        residual = self.get_alignment_residual(paramVec, roi_vec_set, True)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_single(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                              i_index=2):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[i_index] = x
        print(paramVec)
        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_shift(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                             i_index=2):
        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[0:3] = x[:]
        print(paramVec)
        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_xy_shift(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                                i_index=2):
        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[0:2] = x[:]

        print(paramVec)
        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_rotation(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                                i_index=2):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[3:6] = x[:]

        print(paramVec)

        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_geometry(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                                i_index=2):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[:6] = x[:]

        print(paramVec)
        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peaks_alignment_all(self, x, roi_vec_set=None, ConstrainPosition=False,
                            ReturnScalar=False, i_index=2):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[:7] = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set, True, False)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def calibrate_single(self, initial_guess=None, ConstrainPosition=True, LL=[], UL=[],
                         i_index=0, Brute=True, diff_step=None):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        out = self.FitDetector(self.peak_alignment_single, initial_guess, jac='3-point', bounds=(LL, UL),
                               method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                               f_scale=1.0, diff_step=diff_step, tr_solver='exact', factor=100., epsfcn=1e-8,
                               tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, i_index=i_index, Brute=Brute)

        return out

    def calibrate_wave_length(self, initial_guess=None, ConstrainPosition=True):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = self.get_wavelength()
            # initial_guess = np.concatenate((self.get_shiftx(), self.get_wavelength()))

        out = self.FitDetector(self.peak_alignment_wavelength, initial_guess, jac='3-point',
                               bounds=([self._calib[6]-.01], [self._calib[6]+.01]), method='dogbox',
                               ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,
                               diff_step=None, tr_solver='exact', factor=100., epsfcn=1e-8,
                               tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, Brute=False, fDiff=1e-4)

        self.set_wavelength(out)

        return

    def calibrate_shiftx(self, initial_guess=None, ConstrainPosition=False, Brute=False,
                         bounds=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[0]])

        if bounds is None:
            bounds = [[-0.05, -0.05], [0.05, 0.05]]

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=bounds[0], UL=bounds[1], i_index=0,
                                    Brute=Brute)

        self.set_shiftx(out)

    def calibrate_xy_shift(self, initial_guess=None, ConstrainPosition=False, bounds=None,
                           Brute=False, diff_step=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            initial_guess = self.get_calib()[0:2]

        if bounds is None:
            bounds = [[-0.05, -0.05], [0.05, 0.05]]

        out = self.FitDetector(self.peak_alignment_xy_shift, initial_guess, jac='3-point',
                               bounds=(bounds[0], bounds[1]), method='dogbox',
                               ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,
                               diff_step=diff_step, tr_solver='exact', factor=100., epsfcn=1e-8,
                               tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, Brute=Brute, fDiff=1e-4)

        self.set_xy_shift(out)

        return

    def calibrate_shifty(self, initial_guess=None, ConstrainPosition=False, Brute=False,
                         bounds=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[1]])

        if bounds is None:
            bounds = [[-0.05], [0.05]]

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=bounds[0], UL=bounds[1], i_index=1,
                                    Brute=Brute)

        self.set_shifty(out)

        return

    def calibrate_distance(self, initial_guess=None, ConstrainPosition=False, Brute=False,
                           bounds=None, diff_step=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[2]])

        if bounds is None:
            bounds = [[-0.05], [0.05]]

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=bounds[0], UL=bounds[1], i_index=2,
                                    Brute=Brute, diff_step=diff_step)

        self.set_distance(out)

        return

    def CalibrateShift(self, initalGuess=None, ConstrainPosition=True, Brute=False,
                       bounds=None):

        if initalGuess is None:
            initalGuess = self.get_shift()
        if bounds is None:
            bounds = [[-.05, -.05, -.15], [.05, .05, .15]]

        out = self.FitDetector(self.peak_alignment_shift, initalGuess, jac='3-point',
                               bounds=(bounds[0], bounds[1]), method='dogbox',
                               ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                               f_scale=1.0, diff_step=1e-3, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0, Brute=Brute,
                               ROI=None, ConPos=ConstrainPosition)

        self.set_shift(out)

        return

    def CalibrateRotation(self, initalGuess=None, ConstrainPosition=False, Brute=False):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_rotation()

        out = self.FitDetector(self.peak_alignment_rotation, initalGuess, jac='3-point',
                               bounds=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                               method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                               f_scale=1.0, diff_step=None,
                               tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, Brute=Brute)

        self.set_rotation(out)

        return

    def CalibrateGeometry(self, initalGuess=None, ConstrainPosition=False, diff_step=None):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_calib()[:6]

        out = self.FitDetector(self.peak_alignment_geometry, initalGuess, jac='3-point',
                               bounds=([-.05, -.05, -.05, -5.0, -5.0, -5.0],
                                       [.05, .05, .05, 5.0, 5.0, 5.0]),
                               method='dogbox', ftol=1e-12, xtol=1e-12, gtol=1e-12, x_scale=1.0,
                               loss='linear', f_scale=1.0, diff_step=diff_step, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0, factor=100., epsfcn=1e-2,
                               ROI=None, ConPos=ConstrainPosition)

        self.set_geo(out)

        return

    def FullCalibration(self, initalGuess=None, ConstrainPosition=False, Brute=False):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_calib()[:-1]
            # initalGuess = self.get_calib()

        out = self.FitDetector(self.peaks_alignment_all, initalGuess, jac='3-point',
                               bounds=([-.05, -.05, -.15, -5.0, -5.0, -5.0, self._calib[6]-.01],
                                       [.05, .05, .15, 5.0, 5.0, 5.0, self._calib[6]+.01]),
                               # bounds=([-.05, -.05, -.15, -5.0, -5.0, -5.0, self._calib[6]-.01, -0.05],
                               #         [.05, .05, .15, 5.0, 5.0, 5.0, self._calib[6]+.01, 0.05]),
                               method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
                               loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, Brute=Brute)

        self.set_calibration(out)

        return

    def get_calib(self):
        return np.array(self._calib)

    def get_shift(self):
        return np.array([self._calib[0], self._calib[1], self._calib[2]])

    def get_shiftx(self):
        return np.array([self._calib[0]])

    def get_rotation(self):
        return np.array([self._calib[3], self._calib[4], self._calib[5]])

    def get_wavelength(self):
        return np.array([self._calib[6]])

    def set_shift(self, out):
        self._calib[0:3] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0:3] = out[1]

        return

    def set_xy_shift(self, out):
        self._calib[0:2] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0:2] = out[1]

        return

    def set_distance(self, out):
        self._calib[2] = out[0]
        self._calibstatus = out[2]
        self._caliberr[2] = out[1]

        return

    def set_rotation(self, out):
        '''


        Parameters
        ----------
        out : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        self._calib[3:6] = out[0]
        self._calibstatus = out[2]

        if len(out[1].shape) == 2:
            self._caliberr[3:6] = -2. * np.ones_like(out[0])
        else:
            self._caliberr[3:6] = out[1]

        return

    def set_geo(self, out):
        self._calib[0:6] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0:6] = out[1]

        return

    def set_wavelength(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """
        if len(out[0]) > 1:
            self._calib[0] = out[0][0]
            self._calib[6] = out[0][1]
            self._calibstatus = out[2]
            self._caliberr[0] = out[1][0]
            self._caliberr[6] = out[1][1]
        else:
            self._calib[6] = out[0]
            self._calibstatus = out[2]
            self._caliberr[6] = out[1]

        return

    def set_shiftx(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[0] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0] = out[1]

        return

    def set_shifty(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[1] = out[0]
        self._calibstatus = out[2]
        self._caliberr[1] = out[1]

        return

    def set_calibration(self, out):
        """Set calibration to calibration data structure

        Parameters
        ----------
        out

        Returns
        -------

        """
        self._calib = out[0]
        self._calibstatus = out[2]
        self._caliberr = out[1]

        return

    def get_archived_calibration(self, file_name):
        """Get calibration from archived JSON file

        Output: result is written to self._calib[i]

        Returns
        -------
        None
        """
        with open(file_name) as fIN:
            CalibData = json.load(fIN)
            keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda', 'two_theta_0']
            for i in range(len(keys)):
                self._calib[i] = CalibData[keys[i]]

        self._calib_start = np.copy(self._calib)

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
        from pyrs.core.instrument_geometry import DENEXDetectorShift
        # Form DENEXDetectorShift objects
        cal_shift = DENEXDetectorShift(self._calib[0], self._calib[1], self._calib[2], self._calib[3],
                                       self._calib[4], self._calib[5], self._calib[7])

        cal_shift_error = DENEXDetectorShift(self._caliberr[0], self._caliberr[1], self._caliberr[2],
                                             self._caliberr[3], self._caliberr[4], self._caliberr[5],
                                             self._caliberr[7])

        wl = self._calib[6]
        wl_error = self._caliberr[6]

        # Determine output file name
        if file_name is None:
            # default case: write to archive
            if os.access('/HFIR/HB2B/shared', os.W_OK):
                file_name = '/HFIR/HB2B/shared/CAL/%s/HB2B_CAL_%s.json' % (float(self.monosetting),
                                                                           time.strftime('%Y-%m-%dT%H:%M',
                                                                                         time.localtime()))
            else:
                raise IOError('User does not privilege to write to {}'.format('/HFIR/HB2B/shared'))
        # END-IF

        write_calibration_to_json(cal_shift, cal_shift_error, wl, wl_error, self._calibstatus, file_name)

        return

    def print_calibration(self, print_to_screen=True, refine_step=None):
        """Print the calibration results to screen

        Parameters
        ----------

        Returns
        -------
        None
        """

        res = self.singleEval()

        keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda', 'two_theta_0']
        print_string = '\n###########################################'
        print_string += '\n########### Calibration Summary ###########'
        print_string += '\n###########################################\n'
        if refine_step is not None:
            print_string += '\nrefined using {}\n'.format(refine_step)

        print_string += 'Iterations     {}\n'.format(GlobalParameter.global_curr_sequence)
        print_string += 'RMSE         = {}\n'.format(np.sqrt((res**2).sum() / res.shape[0]))
        print_string += 'Residual Sum = {}\n'.format(np.sum(res))

        print_string += "Parameter:  inital guess  refined value\n"
        for i in range(len(keys)):
            print_string += '{:10s}{:^15.5f}{:^14.5f}\n'.format(keys[i], self._calib_start[i], self._calib[i])

        self.refinement_summary += print_string

        if print_to_screen:
            print(print_string)

        return
