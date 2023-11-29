import numpy as np
from qtpy.QtCore import Signal, QObject  # type:ignore
from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

# Import instrument constants
from pyrs.core.nexus_conversion import NUM_PIXEL_1D


class DetectorCalibrationModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        super().__init__()
        self._peak_fit = peak_fit_core
        self._hidra_ws = None
        self.peak_fit_engine = None
        self._run_number = None
        self._calibration_obj = None
        self._nexus_file = None

        self.detector_params = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.sub_runs = np.array([1])

        self._calibration_obj = None

    @property
    def nexus_file(self):
        return self._nexus_file

    @property
    def sub_runs(self):
        return self._calibration_obj._hidra_ws.get_sub_runs()

    @property
    def powders(self):
        return self._calibration_obj.powders

    @property
    def get_wavelength(self):
        return self._calibration_obj._hidra_ws.get_wavelength(False, False)

    @property
    def reduction_masks(self):
        return self._calibration_obj._hidra_ws.reduction_masks

    @property
    def rmse(self):
        if self._calibration_obj is not None:
            return self._calibration_obj.residual_rmse
        else:
            return None

    @property
    def r_sum(self):
        if self._calibration_obj is not None:
            return self._calibration_obj.residual_sum
        else:
            return None

    def export_calibration(self, filename=None, write_latest=False):
        if self._calibration_obj is not None:
            self._calibration_obj.write_calibration(file_name=filename, write_latest=write_latest)

    def set_refinement_params(self, method, max_nfev):
        self._calibration_obj.set_refinement_method(method)
        self._calibration_obj.set_max_nfev(int(max_nfev))

    def set_calibration_params(self, params_list):
        self._calibration_obj.set_calibration_array(params_list)
        self._calibration_obj.set_inst_shifts_wl(params_list)

    def set_reduction_param(self, tth_bins, eta_bins):
        if self._calibration_obj is not None:
            self._calibration_obj = FitCalibration(nexus_file=self._nexus_file, eta_slice=eta_bins, bins=tth_bins)

    def _init_calibration(self, nexus_file, tth=512, eta=3.):
        self._calibration_obj = FitCalibration(nexus_file=nexus_file, eta_slice=eta, bins=tth)
        self._nexus_file = nexus_file

    def fit_diffraction_peaks(self):
        if self._calibration_obj is not None:
            residual = self._calibration_obj.fit_peaks()
            return np.sqrt((residual**2).sum() / residual.shape[0]), residual.sum()

    def get_calibration_values(self, x_item, y_item):
        if x_item == 0:
            _x = np.arange(self._calibration_obj._calibration.shape[1] - 1)
        else:
            _x = self._calibration_obj._calibration[x_item - 1, 1:]

        return _x, self._calibration_obj._calibration[y_item, 1:]

    def calibrate_detector(self, fit_recipe):

        if self._calibration_obj is not None:
            # self._calibration_obj.initalize_calib_results()
            calibration = []
            calibration_error = []
            for recipe in fit_recipe:
                if recipe == "wavelength":
                    self._calibration_obj.calibrate_wave_length()
                elif recipe == "rotations":
                    self._calibration_obj.CalibrateRotation()
                elif recipe == "geometry":
                    self._calibration_obj.CalibrateGeometry()
                elif recipe == "shifts":
                    self._calibration_obj.CalibrateShift()
                elif recipe == "shift x":
                    self._calibration_obj.calibrate_shiftx()
                elif recipe == "shift y":
                    self._calibration_obj.calibrate_shifty()
                elif recipe == "distance":
                    self._calibration_obj.calibrate_distance()
                elif recipe == "full":
                    self._calibration_obj.FullCalibration()
                elif recipe == 'wavelength_tth0':
                    self._calibration_obj.calibrate_wave_shift()
                elif recipe == 'tth0':
                    self._calibration_obj.calibrate_tth0()

                if recipe != '':
                    calibration.append(np.copy(self._calibration_obj.calibration_array))
                    calibration_error.append(np.copy(self._calibration_obj.calibration_error_array))

            try:
                return calibration, calibration_error, self._calibration_obj.residual_sum, \
                    self._calibration_obj.residual_rmse
            except IndexError:
                return calibration, -1, -1, -1

    def set_keep_subrun_list(self, keep_list):
        if self._calibration_obj is not None:
            self._calibration_obj.set_keep_subrun_list(keep_list)

    def get_reduced_diffraction_data(self, sub_run, mask):
        return self._calibration_obj.reducer.get_diffraction_data(sub_run, mask_id=mask)

    def get_fitted_diffraction_data(self, sub_run):
        if self._calibration_obj.fitted_ws is not None:
            return self._calibration_obj.fitted_ws[sub_run - 1]

    def get_2D_diffraction_counts(self, sub_run):
        if self._nexus_file is not None:
            return self._calibration_obj._hidra_ws.get_detector_counts(sub_run).reshape(NUM_PIXEL_1D, NUM_PIXEL_1D)
