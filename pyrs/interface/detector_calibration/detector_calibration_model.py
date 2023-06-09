import json
import traceback
import os
import numpy as np

from qtpy.QtWidgets import QTableWidgetItem  # type:ignore
from qtpy.QtCore import Signal, QObject  # type:ignore
from pyrs.core.instrument_geometry import DENEXDetectorGeometry

from pyrs.core.powder_pattern import ReductionApp
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core import MonoSetting  # type: ignore

# Import instrument constants
from pyrs.core.nexus_conversion import NUM_PIXEL_1D, PIXEL_SIZE, ARM_LENGTH


def get_ref_flags(powder_engine, pin_engine):
    def get_mono_setting(_engine):
        try:
            monosetting = MonoSetting.getFromIndex(_engine.get_sample_log_value('MonoSetting', 1))
        except ValueError:
            monosetting = MonoSetting.getFromRotation(_engine.get_sample_log_value('mrot', 1))

        return monosetting


class DetectorCalibrationModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        super().__init__()
        self._peak_fit = peak_fit_core
        self.ws = None
        self.peak_fit_engine = None
        self._run_number = None

        self._instrument = DENEXDetectorGeometry(NUM_PIXEL_1D, NUM_PIXEL_1D,
                                                 PIXEL_SIZE, PIXEL_SIZE,
                                                 ARM_LENGTH, False)

        self.detector_params = [0, 0, 0, 0, 0, 0, 0, 0]
        self.sub_runs = np.array([1])


    @property
    def runnumber(self):
        return self._run_number

    @property
    def sy(self):
        return self._hydra_ws.get_sample_log_values('sy')

    def set_subruns(self):
        self.sub_runs = self._hydra_ws.get_sub_runs()

    def _load_nexus_data(self, nexus_file, mask_file=None):
        converter = NeXusConvertingApp(nexus_file, mask_file)
        self._hydra_ws = converter.convert()
        self.detector_params[6] = MonoSetting.getFromRotation(self._hydra_ws.get_sample_log_value('mrot', 1))
        self.set_subruns()
        self._reduce_diffraction_data()

        return

    def _reduce_diffraction_data(self):

        self.reducer = ReductionApp()
        self.reducer.load_hidra_workspace(self._hydra_ws)
        
        self.reducer.reduce_data(sub_runs=None, instrument_file=None,
                                 calibration_file=None, mask=None, num_bins=720)

    def get_reduced_diffraction_data(self, sub_run):
        return self.reducer.get_diffraction_data(sub_run)

    def _get_diffraction_counts(self, sub_run):
        return self._hydra_ws.get_detector_counts(sub_run)

    def get_2D_diffraction_counts(self, sub_run):
        return self._get_diffraction_counts(sub_run).reshape(NUM_PIXEL_1D, NUM_PIXEL_1D)

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
