import json
import traceback
import os
import numpy as np

from qtpy.QtWidgets import QTableWidgetItem  # type:ignore
from qtpy.QtCore import Signal, QObject, Slot  # type:ignore

from pyrs.calibration.mantid_peakfit_calibration import FitCalibration

# Import instrument constants
from pyrs.core.nexus_conversion import NUM_PIXEL_1D


class WorkerObject(QObject):

    signalStatus = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._parent = parent

        self._calib_routine = ''

    def set_calib(self, recipe):
        self._calib_routine = recipe

    @Slot()
    def startWork(self):
        for recipe in self._calib_routine:
            if recipe == "wavelength":
                self._parent._calibration_obj.calibrate_wave_length()
            elif recipe == "rotations":
                self._parent._calibration_obj.CalibrateRotation()
            elif recipe == "geometry":
                self._parent._calibration_obj.CalibrateGeometry()
            elif recipe == "shifts":
                self._parent._calibration_obj.CalibrateShift()
            elif recipe == "shift x":
                self._parent._calibration_obj.calibrate_shiftx()
            elif recipe == "shift y":
                self._parent._calibration_obj.calibrate_shifty()
            elif recipe == "distance":
                self._parent._calibration_obj.calibrate_distance()
            elif recipe == "full":
                self._parent._calibration_obj.FullCalibration()
            elif recipe == 'wavelength_tth0':
                self._parent._calibration_obj.calibrate_wave_shift()
            elif recipe == 'tth0':
                self._parent._calibration_obj.calibrate_tth0()

        self.signalStatus.emit('Idle.')


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

        self._worker_object = WorkerObject(parent=self)

    @property
    def runnumber(self):
        return self._run_number

    @property
    def sub_runs(self):
        return self._calibration_obj._hidra_ws.get_sub_runs()

    @property
    def powders(self):
        return self._calibration_obj.powders

    @property
    def sy(self):
        return self._calibration_obj.sy

    @property
    def get_wavelength(self):
        return self._calibration_obj._hidra_ws.get_wavelength(False, False)

    @property
    def reduction_masks(self):
        return self._calibration_obj._hidra_ws.reduction_masks

    def export_calibration(self, filename=None):
        if self._calibration_obj is not None:
            self._calibration_obj.write_calibration(file_name=filename)

    def set_calibration_params(self, params_list):
        self._calibration_obj.set_calibration_array(params_list)
        self._calibration_obj.set_inst_shifts_wl(params_list)

    def set_reduction_param(self, tth_bins, eta_bins):
        if self._calibration_obj is not None:
            self._calibration_obj = FitCalibration(nexus_file=self._nexus_file, eta_slice=eta_bins, bins=tth_bins)

    def _init_calibration(self, nexus_file, tth=512, eta=3.):
        self._nexus_file = nexus_file
        self._calibration_obj = FitCalibration(nexus_file=nexus_file, eta_slice=eta, bins=tth)

    def fit_diffraction_peaks(self):
        if self._calibration_obj is not None:
            residual =  self._calibration_obj.fit_peaks()
            return np.sqrt((residual**2).sum() / residual.shape[0]), residual.sum()

    def get_calibration_values(self, x_item, y_item):
        if x_item == 0:
            _x = np.arange(self._calibration_obj._calibration.shape[1])
        else:
            _x = self._calibration_obj._calibration[x_item - 1, :]

        return _x, self._calibration_obj._calibration[y_item, :]

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
                    calibration.append(self._calibration_obj.calibration_array)
                    calibration_error.append(self._calibration_obj.calibration_error_array)

            return calibration, calibration_error, self._calibration_obj.residual_sum,\
                self._calibration_obj.residual_rmse

    def set_keep_subrun_list(self, keep_list):
        if self._calibration_obj is not None:
            self._calibration_obj.set_keep_subrun_list(keep_list)

    def get_reduced_diffraction_data(self, sub_run, mask):
        return self._calibration_obj.reducer.get_diffraction_data(sub_run, mask_id=mask)

    def get_fitted_diffraction_data(self, sub_run):
        if self._calibration_obj.fitted_ws is not None:
            return self._calibration_obj.fitted_ws[sub_run - 1]

    def get_2D_diffraction_counts(self, sub_run):
        return self._calibration_obj._hidra_ws.get_detector_counts(sub_run).reshape(NUM_PIXEL_1D, NUM_PIXEL_1D)

    def to_json(self, filename, fit_range_table):

        fileParts = os.path.splitext(filename)

        if fileParts[1] != '.json':
            filename = '{}.json'.format(fileParts[0])

        try:
            json_output = dict()

            for peak_row in range(fit_range_table.rowCount()):
                if (fit_range_table.item(peak_row, 0) is not None and
                        fit_range_table.item(peak_row, 1) is not None):

                    if fit_range_table.item(peak_row, 2) is None:
                        peak_tag = 'peak_{}'.format(peak_row + 1)
                    else:
                        peak_tag = fit_range_table.item(peak_row, 2).text()

                    if fit_range_table.item(peak_row, 3) is None:
                        d0 = 1.0
                    else:
                        d0 = float(fit_range_table.item(peak_row, 3).text())

                json_output[str(peak_row)] = {"peak_range": [float(fit_range_table.item(peak_row, 0).text()),
                                                             float(fit_range_table.item(peak_row, 1).text())],
                                              "peak_label": peak_tag,
                                              "d0": d0}

            with open(filename, 'w') as f:
                json.dump(json_output, f)

        except Exception as e:
            self.failureMsg.emit(f"Failed save json file to {filename}",
                                 str(e),
                                 traceback.format_exc())

    def from_json(self, filename, fit_range_table):
        with open(filename) as f:
            data = json.load(f)

        for peak_entry in data.keys():

            try:
                float(data[peak_entry]["d0"])
            except TypeError:
                data[peak_entry]["d0"] = 1.0

            fit_range_table.insertRow(fit_range_table.rowCount())
            fit_range_table.setItem(fit_range_table.rowCount() - 1, 0,
                                    QTableWidgetItem(str(data[peak_entry]["peak_range"][0])))
            fit_range_table.setItem(fit_range_table.rowCount() - 1, 1,
                                    QTableWidgetItem(str(data[peak_entry]["peak_range"][1])))
            fit_range_table.setItem(fit_range_table.rowCount() - 1, 2,
                                    QTableWidgetItem(str(data[peak_entry]["peak_label"])))
            fit_range_table.setItem(fit_range_table.rowCount() - 1, 3,
                                    QTableWidgetItem(str(data[peak_entry]["d0"])))

        return
