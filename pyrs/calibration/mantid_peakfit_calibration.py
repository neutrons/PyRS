import numpy as np
import time
import json
import os

# Import pyrs modules
from pyrs.core import MonoSetting  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

# Import scipy libraries for minimization
from scipy.optimize import least_squares
from scipy.optimize import brute


class FitCalibration:
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """

    def __init__(self, _inst=None, nexus_file=None, mask_file=None, vanadium=None,
                 eta_slice=3, bins=512, reduction_engine=None, pow_lines=None,
                 max_nfev=None, method='trf'):

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

        self.eta_slices = eta_slice
        self.bins = bins
        self.vanadium = vanadium
        self.mask_file = mask_file

        self._keep_subrun_list = None

        if nexus_file is not None:
            self._reduce_diffraction_data(nexus_file)
        else:
            self._hidra_ws = reduction_engine

        self.initalize_calib_arrays()

        # Initalize calibration status to -1
        self._calibstatus = -1

        self.ReductionResults = {}
        self._residualpoints = None
        self.singlepeak = False

        self.refinement_summary = ''

        self.fitted_ws = None
        self._max_nfev = max_nfev
        self._ref_method = method

        self._ref_powders = np.array(['Ni', 'Fe', 'Mo'])
        self._ref_powders_sy = np.array([62, 12, -13])
        self._ref_structure = np.array(['FCC', 'BCC', 'BCC'])
        # self._ref_lattice = [3.523799438, 2.8663982, 3.14719963]
        self._ref_lattice = [3.526314, 2.865579, 3.147664]

        if pow_lines is None:
            self.get_powder_lines()

    @property
    def sy(self):
        return self._hidra_ws.get_sample_log_values('sy')

    @property
    def sub_runs(self):
        return self._hidra_ws.get_sub_runs()

    @property
    def powders(self):
        return self._powders

    @property
    def calibration_array(self):
        return self._calib

    @property
    def calibration_error_array(self):
        return self._caliberr

    @property
    def residual_sum(self):
        return self._residual_sum[-1]

    @property
    def residual_rmse(self):
        return self._residual_rmse[-1]

    def set_calibration_array(self, params):
        self._calib = np.array(params)

    def set_refinement_method(self, method):
        self._ref_method = method

    def set_max_nfev(self, nfev):
        self._max_nfev = nfev

    def set_inst_shifts_wl(self, params):
        self._hidra_ws.set_detector_shift(DENEXDetectorShift(params[0], params[1], params[2],
                                                             params[3], params[4], params[5], params[6]))

        self._hidra_ws.set_wavelength(params[7], False)

        return

    def set_keep_subrun_list(self, keep_list):
        self._keep_subrun_list = keep_list

    def set_tthbins(self, tth_bins):
        self.bins = tth_bins

    def set_etabins(self, eta_bins):
        self.eta_slices = eta_bins

    def _reduce_diffraction_data(self, nexus_file, mask_file=None, calibration=None):
        converter = NeXusConvertingApp(nexus_file, mask_file)
        self._hidra_ws = converter.convert()
        if calibration is not None:
            self._hidra_ws.set_instrument_geometry(calibration)

        # self._hidra_ws.set_detector_shift(DENEXDetectorShift(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))

        self.reducer = ReductionApp()
        self.reducer.load_hidra_workspace(self._hidra_ws)
        self.reducer.reduce_data(sub_runs=None,
                                 instrument_file=None, calibration_file=None,
                                 mask=self.mask_file, num_bins=self.bins, eta_step=self.eta_slices,
                                 van_file=self.vanadium)

    def reduce_data(self):

        self._hidra_ws._mask_dict = dict()
        self.reducer = ReductionApp()
        self.reducer.load_hidra_workspace(self._hidra_ws)

        self.reducer.reduce_data(sub_runs=self.sub_runs[np.array(self._keep_subrun_list)],
                                 instrument_file=None, calibration_file=None,
                                 mask=self.mask_file, num_bins=self.bins, eta_step=self.eta_slices,
                                 van_file=self.vanadium)

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
        return np.rad2deg(np.arcsin(self._hidra_ws.get_wavelength(False, False) / 2 / dspace) * 2)

    def get_powder_lines(self):
        self._powders = [''] * self.sy.size
        self._powders_struc = [''] * self.sy.size
        self._powders_lattice = np.zeros_like(self.sy)
        self._keep_subrun_list = np.array([True] * self.sy.size)

        for i_pos in range(self.sy.size):
            try:
                index = np.where(np.abs(self._ref_powders_sy - self.sy[i_pos]) < 2)[0]
                self._powders[i_pos] = self._ref_powders[index[0]]
                self._powders_struc[i_pos] = self._ref_structure[index[0]]
                self._powders_lattice[i_pos] = self._ref_lattice[index[0]]
            except IndexError:
                pass

        self._diff_peaks = {}
        self._diff_peaks['BCC'] = np.sqrt(np.array([2, 4, 6, 8, 10, 12]))
        self._diff_peaks['FCC'] = np.sqrt(np.array([3, 4, 8, 11, 12, 19]))

    def initalize_calib_arrays(self):
        # calibration: numpy array. size as 7 for ... [6] for wave length
        self._calib = np.array(8 * [0], dtype=np.float64)
        # calibration error: numpy array. size as 7 for ...
        self._caliberr = np.array(8 * [-1], dtype=np.float64)
        # calibration starting point: numpy array. size as 7 for ...
        self._calib_start = np.array(8 * [0], dtype=np.float64)

        # Set wave length
        self.monosetting = MonoSetting.getFromRotation(self._hidra_ws.get_sample_log_value('mrot', 1))
        self._calib[7] = float(self.monosetting)
        self._calib_start[7] = float(self.monosetting)

        self._calibration = self._calib.reshape(-1, 1)
        self._residual_sum = []
        self._residual_rmse = []

    def fit_peaks(self):
        self.reduce_data()
        self._fitting_ws = HidraWorkspace()
        self._fitting_ws.set_sub_runs(range(1, len(self._hidra_ws.reduction_masks) + 1))

        center_errors = np.array([])
        self.fitted_ws = [None] * self.sub_runs.size

        for i_run, sub_run in enumerate(self.sub_runs):
            if self._keep_subrun_list[i_run]:
                peaks = self.get_diff_peaks(sub_run - 1)
                for i_mask, mask in enumerate(self._hidra_ws.reduction_masks):
                    tth, int_vec, error_vec = self.reducer.get_diffraction_data(sub_run, mask_id=mask)
                    self._fitting_ws.set_reduced_diffraction_data(i_mask + 1, None, tth, int_vec, error_vec)

                _fit_engine = PeakFitEngineFactory.getInstance(self._fitting_ws, 'PseudoVoigt', 'Linear',
                                                               wavelength=self._calib[7], out_of_plane_angle=None)

                peaks = peaks[(peaks > (tth.min() + 1)) * (peaks < (tth.max() - 1))]

                if peaks.size > 1:
                    peaks = peaks[:1]

                _ws = []
                for peak in peaks:
                    # print(sub_run, peak, peaks)
                    fits = _fit_engine.fit_multiple_peaks(["peak_tags"], [peak - 3], [peak + 3])

                    _ws.append(fits.fitted)
                    center_errors = np.concatenate((center_errors,
                                                    fits.peakcollections[0].get_effective_params()[0]['Center'] -
                                                    peak))

                self.fitted_ws[i_run] = _ws

        return center_errors

    def get_alignment_residual(self, x):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: tth_0,
        x[7]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        self._calibration = np.concatenate((self._calibration, x.reshape(-1, 1)), axis=1)

        self.set_inst_shifts_wl(x)
        # self._hidra_ws.set_wavelength(x[7], False)
        # self._hidra_ws.set_detector_shift(DENEXDetectorShift(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))

        residual = self.fit_peaks()

        print("")
        print('Iteration      {}'.format(self._calibration.shape[1]))
        print('RMSE         = {}'.format(np.sqrt((residual**2).sum() / residual.shape[0])))
        print('Residual Sum = {}'.format(np.sum(residual)))

        self._residual_sum.append(residual.sum())
        self._residual_rmse.append(np.sqrt((residual**2).sum() / residual.shape[0]))

        return residual

    def FitDetector(self, fun, x0, jac='3-point', bounds=[],
                    i_index=2, Brute=False):

        if Brute:
            BOUNDS = []
            lL = bounds[1]
            uL = bounds[0]
            for i_b in range(len(uL)):
                BOUNDS.append([lL[i_b], uL[i_b]])
            out1 = brute(fun, ranges=BOUNDS, args=(Brute, i_index), Ns=11)
            return [out1, np.array([0]), 1]

        else:
            if len(bounds[0]) != len(bounds[1]):
                raise RuntimeError('User must specify bounds of equal length')

            if len(x0) != len(bounds[1]):
                raise RuntimeError('User must specify bounds of equal length')

            if self._ref_method == 'lm':
                out = least_squares(fun, x0, jac='3-point', method=self._ref_method,
                                    max_nfev=self._max_nfev, args=(Brute, i_index))
            else:
                out = least_squares(fun, x0, jac='3-point', bounds=bounds, method=self._ref_method,
                                    max_nfev=self._max_nfev, args=(Brute, i_index))

            J = out.jac

            if np.sum(J.T.dot(J)) < 1e-8:
                var = np.diagonal(-2 * np.zeros_like(J.T.dot(J)))
            else:
                try:
                    cov = np.linalg.inv(J.T.dot(J))
                    var = np.sqrt(np.diagonal(cov))
                except np.linalg.LinAlgError:
                    var = np.diagonal(-2 * np.zeros_like(J.T.dot(J)))

            return [out.x, var, out.status]

    def peak_alignment_wave_shift(self, x, ReturnScalar=False, i_index=2):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[6] = x[0]
        paramVec[7] = x[1]

        residual = self.get_alignment_residual(paramVec)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_single(self, x, ReturnScalar=False, i_index=2):
        """ Cost function for peaks alignment to determine wavelength
        :param x:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[i_index] = x[0]

        residual = self.get_alignment_residual(paramVec)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_shift(self, x, ReturnScalar=False, i_index=2):

        """ Cost function for peaks alignment to determine detector shift
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """
        paramVec = np.copy(self._calib)
        paramVec[0:3] = x[:]

        residual = self.get_alignment_residual(paramVec)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_rotation(self, x, ReturnScalar=False,
                                i_index=2):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[3:6] = x[:]

        residual = self.get_alignment_residual(paramVec)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peak_alignment_geometry(self, x, ReturnScalar=False,
                                i_index=2):
        """ Cost function for peaks alignment to determine detector rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :param return_scalar:
        :return:
        """

        paramVec = np.copy(self._calib)
        paramVec[:7] = x[:]

        residual = self.get_alignment_residual(paramVec)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def peaks_alignment_all(self, x, ReturnScalar=False, i_index=2):
        """ Cost function for peaks alignment to determine wavelength and detector shift and rotation
        :param x:
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        residual = self.get_alignment_residual(x)

        if ReturnScalar:
            residual = np.sqrt(np.mean(residual**2))

        return residual

    def calibrate_single(self, initial_guess=None, ConstrainPosition=True, LL=[], UL=[],
                         i_index=0):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        out = self.FitDetector(self.peak_alignment_single, initial_guess, jac='3-point', bounds=(LL, UL),
                               i_index=i_index)

        return out

    def calibrate_wave_length(self, initial_guess=None):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = self.get_wavelength()

        out = self.calibrate_single(initial_guess=initial_guess,
                                    LL=[self._calib[7] - .025], UL=[self._calib[7] + .025], i_index=7)

        self.set_wavelength(out)

        return

    def calibrate_wave_shift(self, initial_guess=None):
        """Calibrate wave length

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = np.concatenate((self.get_tth0(), self.get_wavelength()))

        out = self.FitDetector(self.peak_alignment_wave_shift, initial_guess,
                               bounds=([-5.0, self._calib[7]-.025], [5.0, self._calib[7]+.025]))

        self.set_wave_shift(out)

        return

    def calibrate_shiftx(self, initial_guess=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[0]])

        out = self.calibrate_single(initial_guess=initial_guess,
                                    LL=[-0.05], UL=[0.05], i_index=0)

        self.set_shiftx(out)

    def calibrate_shifty(self, initial_guess=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[1]])

        out = self.calibrate_single(initial_guess=initial_guess,
                                    LL=[-0.05], UL=[0.05], i_index=1)

        self.set_shifty(out)

        return

    def calibrate_distance(self, initial_guess=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[2]])

        out = self.calibrate_single(initial_guess=initial_guess,
                                    LL=[-0.1], UL=[0.1], i_index=2)

        self.set_distance(out)

        return

    def calibrate_tth0(self, initial_guess=None):
        """Calibrate Detector Distance

        Parameters
        ----------
        initial_guess

        Returns
        -------

        """

        if initial_guess is None:
            initial_guess = np.array([self.get_calib()[6]])

        out = self.calibrate_single(initial_guess=initial_guess,
                                    LL=[-5.0], UL=[5.0], i_index=6)

        self.set_tth0(out)

        return

    def CalibrateShift(self, initalGuess=None, bounds=None):

        if initalGuess is None:
            initalGuess = self.get_shift()
        if bounds is None:
            bounds = [[-.05, -.05, -.15], [.05, .05, .15]]

        out = self.FitDetector(self.peak_alignment_shift, initalGuess,
                               bounds=(bounds[0], bounds[1]))

        self.set_shift(out)

        return

    def CalibrateRotation(self, initalGuess=None):

        if initalGuess is None:
            initalGuess = self.get_rotation()

        out = self.FitDetector(self.peak_alignment_rotation, initalGuess,
                               bounds=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]))

        self.set_rotation(out)

        return

    def CalibrateGeometry(self, initalGuess=None):

        if initalGuess is None:
            initalGuess = self.get_calib()[:7]

        out = self.FitDetector(self.peak_alignment_geometry, initalGuess,
                               bounds=([-.05, -.05, -.05, -5.0, -5.0, -5.0, -5.0],
                                       [.05, .05, .05, 5.0, 5.0, 5.0, 5.0]))

        self.set_geo(out)

        return

    def FullCalibration(self, initalGuess=None):

        if initalGuess is None:
            initalGuess = self.get_calib()

        out = self.FitDetector(self.peaks_alignment_all, initalGuess,
                               bounds=([-.05, -.05, -.15, -5.0, -5.0, -5.0, -5.0, self._calib[7]-.05],
                                       [.05, .05, .15, 5.0, 5.0, 5.0, 5.0, self._calib[7]+.05]))

        self.set_calibration(out)

        return

    def get_calib(self):
        """
        :return np.array: array of calibration poarams
        """
        return np.array(self._calib)

    def get_shift(self):
        return np.array([self._calib[0], self._calib[1], self._calib[2]])

    def get_shiftx(self):
        return np.array([self._calib[0]])

    def get_rotation(self):
        return np.array([self._calib[3], self._calib[4], self._calib[5]])

    def get_wavelength(self):
        return np.array([self._calib[7]])

    def get_tth0(self):
        return np.array([self._calib[6]])

    def set_shift(self, out):
        self._calib[0:3] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0:3] = out[1]

        return

    def set_distance(self, out):
        self._calib[2] = out[0][0]
        self._calibstatus = out[2]
        self._caliberr[2] = out[1][0]

        return

    def set_tth0(self, out):
        self._calib[6] = out[0][0]
        self._calibstatus = out[2]
        self._caliberr[6] = out[1][0]

        return

    def set_rotation(self, out):
        self._calib[3:6] = out[0]
        self._calibstatus = out[2]
        self._caliberr[3:6] = out[1]

        return

    def set_geo(self, out):
        self._calib[0:7] = out[0]
        self._calibstatus = out[2]
        self._caliberr[0:7] = out[1]

        return

    def set_wave_shift(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[6] = out[0][0]
        self._calib[7] = out[0][1]
        self._calibstatus = out[2]
        self._caliberr[6] = out[1][0]
        self._caliberr[7] = out[1][1]

        return

    def set_wavelength(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[7] = out[0][0]
        self._calibstatus = out[2]
        self._caliberr[7] = out[1][0]

        return

    def set_shiftx(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[0] = out[0][0]
        self._calibstatus = out[2]
        self._caliberr[0] = out[1][0]

        return

    def set_shifty(self, out):
        """See wave length to calibration data

        Parameters
        ----------
        out

        Returns
        -------

        """

        self._calib[1] = out[0][0]
        self._calibstatus = out[2]
        self._caliberr[1] = out[1][0]

        return

    def set_calibration(self, out):
        """Set calibration to calibration data structure

        Parameters
        ----------
        out

        Returns
        -------

        """
        self._calib[:] = out[0][:]
        self._calibstatus = out[2]
        self._caliberr[:] = out[1][:]

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
            keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'TTH_0', 'Lambda']
            for i in range(len(keys)):
                self._calib[i] = CalibData[keys[i]]

        self._calib_start = np.copy(self._calib)

        return

    def write_calibration(self, file_name=None, write_latest=False):
        """Write the calibration to a Json file

        Parameters
        ----------
        file_name: str or None
            output Json file name.  If None, write to /HFIR/HB2B/shared/CAL/
        write_latest: bool
            bool saying that the calibration should write HB2B_Cal_Latest
        Returns
        -------
        None
        """

        # Form DENEXDetectorShift objects
        cal_shift = DENEXDetectorShift(self._calib[0], self._calib[1], self._calib[2], self._calib[3],
                                       self._calib[4], self._calib[5], self._calib[6])

        cal_shift_error = DENEXDetectorShift(self._caliberr[0], self._caliberr[1], self._caliberr[2],
                                             self._caliberr[3], self._caliberr[4], self._caliberr[5],
                                             self._caliberr[6])

        wl = self._calib[7]
        wl_error = self._caliberr[7]

        # Determine output file name
        if file_name is None:
            # default case: write to archive
            if os.access('/HFIR/HB2B/shared', os.W_OK):
                file_name = '/HFIR/HB2B/shared/CALIBRATION/{}/HB2B_CAL_{}.json'.format(self.monosetting.name,
                                                                                       time.strftime('%Y-%m-%dT%H:%M',
                                                                                                     time.localtime()))
                write_calibration_to_json(cal_shift, cal_shift_error, wl, wl_error, self._calibstatus, file_name)

            else:
                print('User does not privilege to write to {}'.format('/HFIR/HB2B/shared'))

        else:
            write_calibration_to_json(cal_shift, cal_shift_error, wl, wl_error, self._calibstatus, file_name)

        if write_latest:
            if os.access('/HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json', os.W_OK):
                write_calibration_to_json(cal_shift, cal_shift_error, wl, wl_error, self._calibstatus,
                                          '/HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json')
            else:
                print('User does not privilege to write /HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json')

        return

    def print_calibration(self, print_to_screen=True, refine_step=None):
        """Print the calibration results to screen

        Parameters
        ----------

        Returns
        -------
        None
        """

        keys = ['Shift_x', 'Shift_y', 'Shift_z', 'Rot_x', 'Rot_y', 'Rot_z', 'TTH_0', 'Lambda']
        print_string = '\n###########################################'
        print_string += '\n########### Calibration Summary ###########'
        print_string += '\n###########################################\n'
        if refine_step is not None:
            print_string += '\nrefined using {}\n'.format(refine_step)

        print_string += 'Iterations     {}\n'.format(self._calibration.shape[1])
        print_string += 'RMSE         = {}\n'.format(self._residual_rmse[-1])
        print_string += 'Residual Sum = {}\n'.format(self._residual_sum[-1])

        print_string += "Parameter:  inital guess  refined value\n"
        for i in range(len(keys)):
            print_string += '{:10s}{:^15.5f}{:^14.5f}\n'.format(keys[i], self._calib_start[i], self._calib[i])

        self.refinement_summary += print_string

        if print_to_screen:
            print(print_string)

        return
