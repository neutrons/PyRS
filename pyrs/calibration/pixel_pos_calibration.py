import numpy as np
import time
import json
import os
import sys
import copy

sys.path.append("/home/hcf/ResearchSoftware/PyRS/build/lib")

import matplotlib.pyplot as plt

# Import pyrs modules
from pyrs.core import MonoSetting  # type: ignore
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

# Import scipy libraries for minimization
from scipy.optimize import least_squares
from scipy.optimize import brute

NUM_PIXEL_1D = 1024
DET_SIZE = 0.3
PIXEL_RATIO = 1.0
PIXEL_SIZE_X = DET_SIZE / NUM_PIXEL_1D
PIXEL_SIZE_Y = PIXEL_SIZE_X * PIXEL_RATIO
ARM_LENGTH = 0.985


def quadratic(x, p0, p1, p2):
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

def interpolate_ring(pixel_x_positions, pixel_y_positions, pixel_y_eval=None):
    '''
    fit the x and y  position of the ring using a quadradic function and return a super sampled fit

    Parameters
    ----------
    pixel_x_positions : np.array
        array with x positions for the ring
    pixel_y_positions : np.array
        array with y position for the ring
    pixel_y_eval :  np.array, optional
        .

    Returns
    -------
    ring_eval : np.array
        DESCRIPTION.

    '''

    if pixel_y_eval is None:
        pixel_y_eval = np.arange(40, 1024 - 40)

    def residual(x0, x, y):
        p0, p1, p2 = x0
        return x - quadratic(y, p0, p1, p2)

    out = least_squares(residual, np.array([pixel_x_positions.mean(), 0, 0]), args=(pixel_x_positions,
                                                                                    pixel_y_positions))

    return [quadratic(pixel_y_eval, out.x[0], out.x[1], out.x[2]), pixel_y_eval]


class PixelCalibration:
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """

    def __init__(self, _inst=None, nexus_file=None, mask_file=None, vanadium=None,
                 eta_slice=3, bins=512, reduction_engine=None, pow_lines=None, chunks=15,
                 max_nfev=None, method='trf', interpolate=False):

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

        self._plot_peak_fits = True
        self._plot_peak_fits_savepath = os.getcwd()
        self._offset = 40
        self.Chunks = chunks
        self.vertical_slice = int((1024 - self._offset - 40) / self.Chunks)
        self.eta_slices = eta_slice
        self.bins = bins
        self.vanadium = vanadium
        self.mask_file = mask_file
        self._interpolate_ring = interpolate

        self._keep_subrun_list = None

        if nexus_file is not None:
            self._reduce_diffraction_data(nexus_file)
        else:
            self._hidra_ws = reduction_engine

        self.initalize_calib_arrays()

        self.singlepeak = False

        self.refinement_summary = ''

        self.fitted_ws = {}
        self._max_nfev = max_nfev
        self._ref_method = method

        self._ref_powders = np.array(['Ni', 'Fe', 'Mo'])
        self._ref_powders_sy = np.array([62, 12, -13])
        self._ref_structure = np.array(['FCC', 'BCC', 'BCC'])
        self._ref_lattice = [3.523799438, 2.8663982, 3.14719963]
        # self._ref_lattice = [3.526314, 2.865579, 3.147664]

        if pow_lines is None:
            self.get_powder_lines()

        self._get_fitted_pixels()

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
        self.hb2b_inst = reduce_hb2b_pyrs.PyHB2BReduction(self._hidra_ws.get_instrument_setup())
        if calibration is not None:
            self._hidra_ws.set_instrument_geometry(calibration)

        self.reducer = ReductionApp()
        self.reducer.load_hidra_workspace(self._hidra_ws)
        self.reducer.reduce_data(sub_runs=None,
                                 instrument_file=None, calibration_file=None,
                                 mask=self.mask_file, num_bins=self.bins, eta_step=self.eta_slices,
                                 van_file=self.vanadium)

    def fit_calibrated_detector_pixels(self):
        self._get_fitted_pixels(self._calib[0], self._calib[1], self._calib[2],
                                self._calib[3], self._calib[4], self._calib[5],
                                self._calib[6])

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
        self.fit_cycle = 1
    
        # Initalize calibration status to -1
        self._calibstatus = -1

        self.ReductionResults = {}
        self._residualpoints = None

    @staticmethod
    def get_pixel_pos_from_fit(fitted_twotheta, two_theta_vector):
        '''
        get the pixel position of the ring from the fitted two theta position
    
        Parameters
        ----------
        fitted_twotheta : TYPE
            DESCRIPTION.
        two_theta_vector : TYPE
            DESCRIPTION.
    
        Returns
        -------
        pix_pos : TYPE
            DESCRIPTION.
    
        '''
    
        pixel_center = two_theta_vector.shape[1] / 2
        index = int(pixel_center)
        fraction = 1 - pixel_center % 1

        lateral_pixels = fraction * two_theta_vector[:, index] + pixel_center % 1 * two_theta_vector[:, index + 1]

        try:
            index_1 = np.where(fitted_twotheta < lateral_pixels)[0].min()
            pix_pos = (fitted_twotheta - lateral_pixels[index_1]) / (lateral_pixels[index_1 + 1] - lateral_pixels[index_1])

            return pix_pos + index_1
        except:
            return -1

    def get_pixel_tth(self, pixel_vector, two_theta, shift_x, shift_y, shift_z, rot_x, rot_y, rot_z, two_theta_0):
        '''
        convert array of x and y pixel coordinates into corresponding x, y, and z position
    
        Parameters
        ----------
        pixel_vector : TYPE
            DESCRIPTION.
        two_theta : TYPE
            DESCRIPTION.
        shift_x : TYPE
            DESCRIPTION.
        shift_y : TYPE
            DESCRIPTION.
        shift_z : TYPE
            DESCRIPTION.
        rot_x : TYPE
            DESCRIPTION.
        rot_y : TYPE
            DESCRIPTION.
        rot_z : TYPE
            DESCRIPTION.
    
        Returns
        -------
        pixel_matrix : TYPE
            DESCRIPTION.
    
        '''
    
        pixel_matrix = np.ndarray(shape=(1, pixel_vector[0].shape[0], 3), dtype=np.float64)
    
        # set Y as different from each row
        start_y_pos = -(NUM_PIXEL_1D * 0.5 - 0.5) * PIXEL_SIZE_Y
        start_x_pos = (NUM_PIXEL_1D * 0.5 - 0.5) * PIXEL_SIZE_X
    
        for i_row, row_index in enumerate(pixel_vector[1]):
            pixel_matrix[0, i_row, 1] = start_y_pos + float(row_index) * PIXEL_SIZE_Y
        # set X as different from each column
        for i_col, col_index in enumerate(pixel_vector[0]):
            pixel_matrix[:, i_col, 0] = start_x_pos - float(col_index) * PIXEL_SIZE_X
        # # set Z: zero at origin
        pixel_matrix[:, :, 2] = 0.
    
        # shift center
        pixel_matrix[:, :, 0] += shift_x
        pixel_matrix[:, :, 1] += shift_y
        pixel_matrix[:, :, 2] += shift_z
        pixel_matrix[:, :, 2] += ARM_LENGTH
    
        # rotation around instrument center
        # get rotation matrix at origin (for flip, spin and vertical): all data from calibration value
        rot_x_flip = rot_x * np.pi / 180.
        rot_y_flip = rot_y * np.pi / 180.
        rot_z_spin = rot_z * np.pi / 180.

        instrument = self.reducer._reduction_manager._last_reduction_engine.instrument
        calib_matrix = instrument.generate_rotation_matrix(rot_x_flip,
                                                           rot_y_flip,
                                                           rot_z_spin)

        # print ('[DB...BAT] Calibration rotation matrix:\n{}'.format(calib_matrix))
        # and rotate at origin
        pixel_matrix = instrument._rotate_detector(pixel_matrix,
                                                   calib_matrix)

        # rotate detector (2theta) if it is not zero
        two_theta_rotation = (two_theta + two_theta_0) * np.pi / 180.
        pixel_matrix = instrument._rotate_detector(pixel_matrix,
                                                   instrument._cal_rotation_matrix_y(two_theta_rotation))

        det_pos_norm_matrix = np.sqrt(pixel_matrix[:][:, :, 0] ** 2 +
                                      pixel_matrix[:][:, :, 1] ** 2 +
                                      pixel_matrix[:][:, :, 2] ** 2)
    
        twotheta_matrix = np.arccos(pixel_matrix[:, :, 2] / det_pos_norm_matrix) * 180 / np.pi

        return np.squeeze(twotheta_matrix)

    @staticmethod
    def _plot_peaks(ax, ny, i_peak, i_chunk, twotheta, intensity, fit_ws, center=None):
        xvec = fit_ws[0].readX(0)[1:]
        yvec = fit_ws[0].readY(0)[1:]
        i_col = i_chunk % ny
        i_row = int(i_chunk / ny)
        ax[i_row, i_col].set_title(f'Peak {i_peak} section {i_chunk}')
        ax[i_row, i_col].plot(xvec, yvec, 'r')
        ax[i_row, i_col].plot(twotheta, intensity, 'kx')
        # ax[i_row, i_col].plot(twotheta, intensity - fit_intensity, 'b')
    
        if center is not None:
            ax[i_row, i_col].axvline(center, c='k', ls='--')
        
    def _get_fitted_pixels(self, shift_x=0.0, shift_y=0.0, shift_z=-0.0, rot_x=0.0,
                           rot_y=0.0, rot_z=0.0, two_theta_0=0.0, interpolate=False):
    
        sub_runs = self.sub_runs[np.array(self._keep_subrun_list)]

        self.fitted_ws = {}

        self.fitted_pix = {}
        for tth_i in sub_runs:
            powder = self.reducer.get_raw_counts(tth_i).reshape((1024, 1024))
            reduction_engine = self.reducer._reduction_manager.setup_reduction_engine(self._hidra_ws,
                                                                                      tth_i,
                                                                                      DENEXDetectorShift(shift_x, shift_y, shift_z,
                                                                                                         rot_x, rot_y, rot_z,
                                                                                                         two_theta_0))
        
            tth_vec = reduction_engine._instrument.get_pixels_2theta(1).reshape((1024, 1024))

            # center = 2. * np.rad2deg(np.arcsin(wavelength / 2 / dspace[tth_i]))
    
            ring_y = np.ndarray(shape=self.Chunks)
            ring_x = np.ndarray(shape=self.Chunks)
            # delta_peak_pos = []
            if self._plot_peak_fits:
                nx = np.int32(np.ceil(np.sqrt(self.Chunks)))
                ny = np.int32(np.ceil(self.Chunks / nx))
                peak_fig, peaks_ax = plt.subplots(nx, ny, sharex=True, sharey=True, 
                                                  figsize=(3.5 * nx * 1,
                                                           3.5 * ny * 1))
            else:
                peaks_ax = None

            for i_chunk in range(self.Chunks):
                pix_0 = i_chunk * self.vertical_slice + self._offset
                pix_1 = (i_chunk + 1) * self.vertical_slice + self._offset
        
                slicked_tth = tth_vec[:, pix_0:pix_1]
    
                _tth, _int, _var = reduction_engine.histogram_by_numpy(slicked_tth.reshape(-1),
                                                                       powder[:, pix_0:pix_1].reshape(-1),
                                                                       self.bins, True, None)

                peak_center, fit_ws, _peak = self.fit_peaks(_tth, _int, tth_i, i_chunk)
                pixcel_center = self.get_pixel_pos_from_fit(peak_center, slicked_tth)

                if peaks_ax is not None:
                    self._plot_peaks(peaks_ax, ny, tth_i, i_chunk, _tth, _int, fit_ws, center=_peak)

                if pixcel_center > 0:
                    ring_x[i_chunk] = self.get_pixel_pos_from_fit(peak_center, slicked_tth)
                    ring_y[i_chunk] = (pix_0 + pix_1) / 2

                # ring_x.append(get_pixel_pos_from_fit(dat[0]['g0_center'], slicked_tth))
                # ring_y.append((pix_0 + pix_1) / 2)
                # if plot_peaks:
                #     _plot_peaks(peaks_ax, ny, tth_i, i_chunk, _tth[tth_index], _int[tth_index], dat[1],
                #                 center=center)
    
            # _plot_gradient(_ax, ny_s, tth_i, delta_peak_pos)
    
            if self._interpolate_ring:
                self.fitted_pix[tth_i] = interpolate_ring(ring_x, ring_y)
            else:
                self.fitted_pix[tth_i] = [np.array(ring_x), np.array(ring_y)]

            peak_fig.savefig(f'{self._plot_peak_fits_savepath}/FitSummary_peak_{tth_i + 1}_fits_it_{self.fit_cycle}.png')
            plt.close(peak_fig)

        self.fit_cycle += 1

        return

    def fit_peaks(self, _two_theta, _intensity, tth_i, ring_chunk):
        self._fitting_ws = HidraWorkspace()
        self._fitting_ws.set_sub_runs(range(1, 1 + 1))

        center_errors = np.array([])

        peaks = self.get_diff_peaks(tth_i - 1)
        peaks = peaks[(peaks > (_two_theta.min() + 1)) * (peaks < (_two_theta.max() - 1))]

        self._fitting_ws.set_reduced_diffraction_data(1, None, two_theta_array=_two_theta, intensity_array=_intensity)
        
        _fit_engine = PeakFitEngineFactory.getInstance(self._fitting_ws, 'PseudoVoigt', 'Linear',
                                                       wavelength=self._calib[7], out_of_plane_angle=None)

        if peaks.size > 1:
            peaks = peaks[:1]

        _ws = []
        for peak in peaks:
            fits = _fit_engine.fit_multiple_peaks(["peak_tags"], [peak - 2], [peak + 2])

            _ws.append(fits.fitted)
            center_errors = np.concatenate((center_errors,
                                            fits.peakcollections[0].get_effective_params()[0]['Center']))

        self.fitted_ws[tth_i, ring_chunk] = copy.copy(_ws)

        return center_errors, _ws, peak

    def get_ring_d(self, sub_run, wavelength, two_theta):
        peaks = self.get_diff_peaks(sub_run - 1)
        peaks = peaks[(peaks > (two_theta.min() - 2)) * (peaks < (two_theta.max() + 2))]

        return wavelength / 2 / np.sin(np.deg2rad(peaks[0]) / 2)

    def get_alignment_residual(self, x):
        """ Cost function for peaks alignment to determine wavelength
        :param x: list/array of detector shift/rotation and neutron wavelength values
        :x[0]: shift_x, x[1]: shift_y, x[2]: shift_z, x[3]: rot_x, x[4]: rot_y, x[5]: rot_z, x[6]: tth_0,
        x[7]: wavelength
        :param roi_vec_set: list/array of ROI/mask vector
        :return:
        """

        self._calibration = np.concatenate((self._calibration, x.reshape(-1, 1)), axis=1)

        shift_x, shift_y, shift_z, rot_x, rot_y, rot_z, two_theta_0, wavelength = x

        residual = np.array([])
        for ring_i in self.fitted_pix.keys():
            pix_x, pix_y = self.fitted_pix[ring_i]

            two_theta = self.get_pixel_tth([pix_x, pix_y],
                                      -1. * self._hidra_ws.get_detector_2theta(ring_i),
                                      shift_x=shift_x,
                                      shift_y=shift_y,
                                      shift_z=shift_z,
                                      rot_x=rot_x,
                                      rot_y=rot_y,
                                      rot_z=rot_z,
                                      two_theta_0=two_theta_0)

            if two_theta.std() > 2.:
                two_theta = two_theta[two_theta > (two_theta.mean() - 2 * two_theta.std())]

            expected_d = self.get_ring_d(ring_i, wavelength, two_theta)

            residual = np.concatenate((residual,
                                       expected_d - wavelength / 2 / np.sin(np.deg2rad(two_theta) / 2)))

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
                out = least_squares(fun, x0, jac='2-point', method=self._ref_method,
                                    max_nfev=self._max_nfev, args=(Brute, i_index))
            else:
                out = least_squares(fun, x0, jac='3-point', bounds=bounds, method=self._ref_method,
                                    max_nfev=self._max_nfev, args=(Brute, i_index))

            residual = fun(out.x)

            try:
                var = np.sqrt(np.diag(np.linalg.inv(2 * np.dot(out.jac.T, out.jac)) * \
                                      np.abs(residual).sum() / (residual.size - len(x0))))
            except np.linalg.LinAlgError:
                var = [-2] * out.x.size

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
        self._calib[0:3] = out[0][:]
        self._calibstatus = out[2]
        self._caliberr[0:3] = out[1][:]

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
        self._calib[3:6] = out[0][:]
        self._calibstatus = out[2]
        self._caliberr[3:6] = out[1][:]

        return

    def set_geo(self, out):
        self._calib[0:7] = out[0][:]
        self._calibstatus = out[2]
        self._caliberr[0:7] = out[1][:]

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
                print('User does not have privilege to write /HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json')

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

        print_string += "Parameter:  inital guess  refined value, refined error\n"
        for i in range(len(keys)):
            print_string += '{:10s}{:^15.5f}{:^14.5f}{:^14.5f}\n'.format(keys[i], self._calib_start[i], self._calib[i], self._caliberr[i])

        self.refinement_summary += print_string

        if print_to_screen:
            print(print_string)

        return


calibration = PixelCalibration(nexus_file='/home/hcf/ResearchSoftware/PyRS/Calib_Testing/HB2B_4971.nxs.h5',
                               chunks=10, interpolate=True)

calibration.CalibrateShift()
calibration.print_calibration(print_to_screen=False)
# calibration.fit_calibrated_detector_pixels()

calibration.CalibrateRotation()
calibration.print_calibration(print_to_screen=False)
# calibration.fit_calibrated_detector_pixels()

calibration.calibrate_wave_shift()
calibration.print_calibration(print_to_screen=False)
print(calibration.refinement_summary)
calibration.fit_calibrated_detector_pixels()

# calibration.FullCalibration()
# calibration.CalibrateRotation()
# calibration.fit_calibrated_detector_pixels()
# calibration.calibrate_wave_length()
# calibration.print_calibration()
# calibration.CalibrateRotation()
# calibration.print_calibration()
