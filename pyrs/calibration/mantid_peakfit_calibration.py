import numpy as np
import time
import json
import os

# Import pyrs modules
from pyrs.core import MonoSetting  # type: ignore
from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

# Import instrument constants
from pyrs.core.nexus_conversion import NUM_PIXEL_1D, PIXEL_SIZE, ARM_LENGTH

# Import scipy libraries for minimization
from scipy.optimize import least_squares
from scipy.optimize import brute


class GlobalParameter:
    global_curr_sequence = 0
    current_plot = 0

    def __init__(self):
        return


class FitCalibration:
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """

    def __init__(self, _inst=None, powder_engine=None, nexus_file=None,
                 mask_file=None, eta_slice=3, bins=512):
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
        self.eta_slices = eta_slice
        self.bins = bins

        # define instrument setup
        if _inst is None:
            self._instrument = DENEXDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D,
                                                     PIXEL_SIZE, PIXEL_SIZE,
                                                     ARM_LENGTH, False)
        else:
            self._instrument = _inst
        # # 'shift_x', 'shift_y', 'shift_z', 'rotation_x', 'rotation_y', and 'rotation_z'
        # self._detectorshift = DENEXDetectorShift()

        if powder_engine is not None:
            self._hidra_ws = powder_engine
        elif nexus_file is not None:
            self._reduce_diffraction_data(nexus_file)
        else:
            exit()

        # calibration: numpy array. size as 7 for ... [6] for wave length
        self._calib = np.array(8 * [0], dtype=np.float64)
        # calibration error: numpy array. size as 7 for ...
        self._caliberr = np.array(8 * [-1], dtype=np.float64)

        # calibration starting point: numpy array. size as 7 for ...
        self._calib_start = np.array(8 * [0], dtype=np.float64)

        # Set wave length
        self.monosetting = MonoSetting.getFromRotation(self._hidra_ws.get_sample_log_value('mrot', 1))
        self._calib[6] = float(self.monosetting)
        self._calib_start[6] = float(self.monosetting)

        # Initalize calibration status to -1
        self._calibstatus = -1

        self.ReductionResults = {}
        self._residualpoints = None
        self.singlepeak = False

        self.refinement_summary = ''

        self.plot_res = False

        self._ref_powders = np.array(['Ni', 'Fe', 'Mo'])
        self._ref_powders_sy = np.array([62, 12, -13])
        self._ref_structure = np.array(['FCC', 'BCC', 'BCC'])
        self._ref_lattice = [3.523799438, 2.8663982, 3.14719963]



        self.get_powder_lines()
        # self._fitting_ws = HidraWorkspace()
        # self._fit_engine = PeakFitEngineFactory.getInstance(self._fitting_ws, 'PseudoVoigt', 'Linear',
        #                                                     wavelength=self._calib[6], out_of_plane_angle=None)

        GlobalParameter.global_curr_sequence = 0

    @property
    def sy(self):
        return self._hidra_ws.get_sample_log_values('sy')

    @property
    def sub_runs(self):
        return self._hidra_ws.get_sub_runs()
    
    @property
    def powders(self):
        return self._powders

    def set_tthbins(self, tth_bins):
        self.bins = tth_bins

    def set_etabins(self, eta_bins):
        self.eta_slices = eta_bins

    def get_diff_peaks(self, sub_run):
        '''
        Determine peak position for reference lattice

        Parameters
        ----------
        sub_run : int
            sub-run index.

        Returns
        -------
        np.array
            array of two theta peaks for the reference powder.

        '''

        dspace = self._powders_lattice[sub_run] / self._diff_peaks[self._powders_struc[sub_run]]
        return np.rad2deg(np.arcsin(self._calib[6] / 2 / dspace) * 2)

    def get_powder_lines(self):
        self._powders = [''] * self.sy.size
        self._powders_struc = [''] * self.sy.size
        self._powders_lattice = np.zeros_like(self.sy)

        for i_pos in range(self.sy.size):
            try:
                index = np.where(np.abs(self._ref_powders_sy - self.sy[i_pos]) < 2)[0]
                self._powders[i_pos] = self._ref_powders[index[0]]
                self._powders_struc[i_pos] = self._ref_structure[index[0]]
                self._powders_lattice[i_pos] = self._ref_lattice[index[0]]
            except IndexError:
                pass

        self._diff_peaks = {}
        self._diff_peaks['BCC'] = np.sqrt(np.array([2, 4, 6, 8, 12, 10, 14, 18]))
        self._diff_peaks['FCC'] = np.sqrt(np.array([3, 4, 8, 12, 11, 19]))

    def _reduce_diffraction_data(self, nexus_file, mask_file=None, calibration=None):
        converter = NeXusConvertingApp(nexus_file, mask_file)
        self._hidra_ws = converter.convert()
        if calibration is not None:
            self._hidra_ws.set_instrument_geometry(calibration)

        self.reducer = ReductionApp()
        self.reducer.load_hidra_workspace(self._hidra_ws)
        self.reducer.reduce_data(sub_runs=None, instrument_file=None, calibration_file=None,
                                 mask=None, num_bins=self.bins, eta_step=self.eta_slices)

    def plot_data(self, x, y, fit):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y, 'kx')
        ax.plot(x, fit, 'b')
        ax.plot(x, y-fit, 'r')
        fig.savefig('fit_{}.png'.format(GlobalParameter.current_plot))
        GlobalParameter.current_plot += 1

        return

    def fit_peaks(self):

        self._fitting_ws = HidraWorkspace()
        self._fitting_ws.set_sub_runs([1])

        for sub_run in self.sub_runs:
            peaks = self.get_diff_peaks(sub_run)
            for mask in self._hidra_ws.reduction_masks:
                tth, int_vec, error_vec = self.reducer.get_diffraction_data(sub_run, mask_id=mask)
                self._fitting_ws.set_reduced_diffraction_data(1, None, tth, int_vec, error_vec)
                _fit_engine = PeakFitEngineFactory.getInstance(self._fitting_ws, 'PseudoVoigt', 'Linear',
                                                               wavelength=self._calib[6], out_of_plane_angle=None)

                # print((tth.min + 1) > peaks)))

            print(tth.min(), tth.max)
            print((tth.min() + 1) > peaks)
            print(((tth.max() - 1) < peaks))
            print(((tth.min() + 1) > peaks) * ((tth.max() - 1) < peaks))
            peaks = peaks[((tth.min() + 1) > peaks) * ((tth.max() - 1) < peaks)]

            print(sub_run, peaks)

        # return returnSetup

    def FitDetector(self, fun, x0, jac='3-point', bounds=[], method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
                    x_scale=1.0, loss='linear', tr_options={}, jac_sparsity=None, f_scale=1.0, diff_step=None,
                    tr_solver=None, max_nfev=None, verbose=0, ROI=None, ConPos=False, kwargs='', epsfcn=1e-6,
                    factor=100.0, i_index=2, Brute=False, fDiff=1e-9, start=0, stop=0):

        self.check_alignment_inputs(ROI)

        BOUNDS = []
        lL = bounds[1]
        uL = bounds[0]
        for i_b in range(len(uL)):
            BOUNDS.append([lL[i_b], uL[i_b]])

        if Brute:
            out1 = brute(fun, ranges=BOUNDS, args=(ROI, ConPos, True, i_index), Ns=11)
            return [out1, np.array([0]), 1]

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
                                args=(ROI, ConPos, False, i_index, start, stop))

            J = out.jac

            if np.sum(J.T.dot(J)) < 1e-8:
                var = -2 * np.zeros_like(J.T.dot(J))
            else:
                cov = np.linalg.inv(J.T.dot(J))
                var = np.sqrt(np.diagonal(cov))

            return [out.x, var, out.status]

    def get_alignment_residual(self, x, roi_vec_set=None, ConPeaks=False, ReturnFit=False, start=0, stop=0):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        pyrs_reducer = PyHB2BReduction(self._instrument, x[6])

        GlobalParameter.global_curr_sequence += 1

        residual = np.array([])

        for engine_setup in self.engines:

            datasets, dSpace, single_material = engine_setup

            self._engine = datasets

            two_theta_calib = np.arcsin(x[6] / 2. / dSpace) * 360. / np.pi
            two_theta_calib = two_theta_calib[~np.isnan(two_theta_calib)]

            if datasets is None:
                stop = start
                sub_runs = []
            elif stop == 0:
                sub_runs = self._engine.get_sub_runs()

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

                if self.min_tth is None:
                    self.min_tth = tths.min() + .5
                    self.max_tth = tths.max() - .5

                Eta_val = pyrs_reducer.get_eta_value()
                maxEta = Eta_val.max() - 2
                minEta = Eta_val.min() + 2

                if roi_vec_set is None:
                    eta_roi_vec = np.arange(minEta, maxEta, self.eta_slices)
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
                            print(CalibPeaks[ipeak])
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
                                                               np.array([((Fitresult[0]['g%d_center' % p_index] -
                                                                           CalibPeaks[p_index]))])])
                        else:
                            for p_index in Peaks:
                                residual = np.concatenate([residual,
                                                           np.array([((Fitresult[0]['g%d_center' % p_index]))])])

                # END-FOR

            # END-FOR(tth)

        if not ConPeaks:
            residual -= np.average(residual)

        if self._residualpoints is not None:
            if residual.shape[0] < self._residualpoints:
                residual = np.concatenate([residual, np.array([1000.0] * (self._residualpoints-residual.shape[0]))])
            elif residual.shape[0] > self._residualpoints:
                residual = residual[:self._residualpoints]

        print("")
        print('Iteration      {}'.format(GlobalParameter.global_curr_sequence))
        print('RMSE         = {}'.format(np.sqrt((residual**2).sum() / residual.shape[0])))
        print('Residual Sum = {}'.format(np.sum(residual)))

        if np.all(residual == 0.):
            residual += 1000

        return residual

    def singleEval(self, x=None, roi_vec_set=None, ConstrainPosition=True, ReturnFit=True, ReturnScalar=False,
                   start=0, stop=0):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """
        GlobalParameter.global_curr_sequence = -10

        if x is None:
            residual = self.get_alignment_residual(self._calib, roi_vec_set, ConstrainPosition,
                                                   ReturnFit, start=start, stop=stop)
        else:
            residual = self.get_alignment_residual(x, roi_vec_set, ConstrainPosition, ReturnFit,
                                                   start=start, stop=stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_wavelength(self, x, roi_vec_set=None, ConstrainPosition=True, ReturnScalar=False,
                                  i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[0] = x[0]
        paramVec[6] = x[1]

        residual = self.get_alignment_residual(paramVec, roi_vec_set, True)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_single(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                              i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[i_index] = x

        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False, start, stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_shift(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                             i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[0:3] = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False, start, stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_rotation(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                                i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[3:6] = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False, start, stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_geometry(self, x, roi_vec_set=None, ConstrainPosition=False, ReturnScalar=False,
                                i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[:6] = x[:]

        residual = self.get_alignment_residual(paramVec, roi_vec_set, ConstrainPosition, False, start, stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peaks_alignment_all(self, x, roi_vec_set=None, ConstrainPosition=False,
                            ReturnScalar=False, i_index=2, start=0, stop=0):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        residual = self.get_alignment_residual(x, roi_vec_set, True, False, start, stop)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def calibrate_single(self, initial_guess=None, ConstrainPosition=True, LL=[], UL=[],
                         i_index=0, Brute=True, start=0, stop=0):
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
                               f_scale=1.0, diff_step=None, tr_solver='exact', factor=100., epsfcn=1e-8,
                               tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, start=start, stop=stop,
                               ROI=None, ConPos=ConstrainPosition, i_index=i_index, Brute=Brute)

        return out

    def calibrate_wave_length(self, initial_guess=None, ConstrainPosition=True, start=0, stop=0):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

#        GlobalParameter.global_curr_sequence = 0

        if initial_guess is None:
            # initial_guess = self.get_wavelength()
            initial_guess = np.concatenate((self.get_shiftx(), self.get_wavelength()))

        out = self.FitDetector(self.peak_alignment_wavelength, initial_guess, jac='3-point',
                               bounds=([-0.05, self._calib[6]-.01], [0.05, self._calib[6]+.01]), method='dogbox',
                               ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0,
                               diff_step=None, tr_solver='exact', factor=100., epsfcn=1e-8,
                               tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, start=start, stop=stop,
                               ROI=None, ConPos=ConstrainPosition, Brute=False, fDiff=1e-4)

        self.set_wavelength(out)

        return

    def calibrate_shiftx(self, initial_guess=None, ConstrainPosition=False, start=0, stop=0, Brute=False):
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

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=[-0.05], UL=[0.05], i_index=0,
                                    start=start, stop=stop, Brute=Brute)

        self.set_shiftx(out)

    def calibrate_shifty(self, initial_guess=None, ConstrainPosition=False, start=0, stop=0, Brute=False):
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

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=[-0.05], UL=[0.05], i_index=1,
                                    start=start, stop=stop, Brute=Brute)

        self.set_shifty(out)

        return

    def calibrate_distance(self, initial_guess=None, ConstrainPosition=False, start=0, stop=0, Brute=False):
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

        out = self.calibrate_single(initial_guess=initial_guess, ConstrainPosition=ConstrainPosition,
                                    LL=[-0.1], UL=[0.1], i_index=2,
                                    start=start, stop=stop, Brute=Brute)

        self.set_distance(out)

        return

    def CalibrateShift(self, initalGuess=None, ConstrainPosition=True, start=0, stop=0, bounds=None):

        if initalGuess is None:
            initalGuess = self.get_shift()
        if bounds is None:
            bounds = [[-.05, -.05, -.15], [.05, .05, .15]]

        out = self.FitDetector(self.peak_alignment_shift, initalGuess, jac='3-point',
                               bounds=(bounds[0], bounds[1]), method='dogbox',
                               ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                               f_scale=1.0, diff_step=1e-3, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition, start=start, stop=stop)

        self.set_shift(out)

        return

    def CalibrateRotation(self, initalGuess=None, ConstrainPosition=False, start=0, stop=0):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_rotation()

        out = self.FitDetector(self.peak_alignment_rotation, initalGuess, jac='3-point',
                               bounds=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                               method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
                               f_scale=1.0, diff_step=None, start=start, stop=stop,
                               tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                               ROI=None, ConPos=ConstrainPosition)

        self.set_rotation(out)

        return

    def CalibrateGeometry(self, initalGuess=None, ConstrainPosition=False, start=0, stop=0):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_calib()[:6]

        out = self.FitDetector(self.peak_alignment_geometry, initalGuess, jac='3-point',
                               bounds=([-.05, -.05, -.05, -5.0, -5.0, -5.0],
                                       [.05, .05, .05, 5.0, 5.0, 5.0]),
                               method='dogbox', ftol=1e-12, xtol=1e-12, gtol=1e-12, x_scale=1.0,
                               loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0, factor=100., epsfcn=1e-2,
                               ROI=None, ConPos=ConstrainPosition, start=start, stop=stop)

        self.set_geo(out)

        return

    def FullCalibration(self, initalGuess=None, ConstrainPosition=False, start=0, stop=0):

        GlobalParameter.global_curr_sequence = 0

        if initalGuess is None:
            initalGuess = self.get_calib()

        out = self.FitDetector(self.peaks_alignment_all, initalGuess, jac='3-point',
                               bounds=([-.05, -.05, -.15, -5.0, -5.0, -5.0, self._calib[6]-.05, -3.0],
                                       [.05, .05, .15, 5.0, 5.0, 5.0, self._calib[6]+.05, 3.0]),
                               method='dogbox', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0,
                               loss='linear', f_scale=1.0, diff_step=None, tr_solver='exact', tr_options={},
                               jac_sparsity=None, max_nfev=None, verbose=0, start=start, stop=stop,
                               ROI=None, ConPos=ConstrainPosition)

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

    def set_distance(self, out):
        self._calib[2] = out[0]
        self._calibstatus = out[2]
        self._caliberr[2] = out[1]

        return

    def set_rotation(self, out):
        self._calib[3:6] = out[0]
        self._calibstatus = out[2]
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
            keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'Lambda']
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
