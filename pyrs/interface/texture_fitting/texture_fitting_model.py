import json
import traceback
from shutil import copyfile
import numpy as np
import os

# from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.summary_generator import SummaryGenerator
# from pyrs.interface.gui_helper import pop_message
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.core.polefigurecalculator import PoleFigureCalculator
from qtpy.QtWidgets import QTableWidgetItem  # type:ignore
from qtpy.QtCore import Signal, QObject  # type:ignore


class TextureFittingModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        super().__init__()
        self._peak_fit = peak_fit_core
        self.ws = None
        self.peak_fit_engine = None
        self._polefigureinterface = None
        self._run_number = None

    @property
    def runnumber(self):
        return self._run_number

    def load_hidra_project_file(self, filename):

        try:
            source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
            self.ws = HidraWorkspace(filename)
            self.ws.load_hidra_project(source_project, False, True)
            self.sub_runs = np.array(self.ws.get_sub_runs())

            for part in filename.split('/')[-1].replace('.h5', '').split('_'):
                try:
                    self._run_number = int(part)
                except ValueError:
                    pass

        except Exception as e:
            self.failureMsg.emit(f"Failed to load {filename}. Check that this is a Hidra Project File",
                                 str(e),
                                 traceback.format_exc())
            return None, dict()

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

    def fit_diff_peaks(self, min_tth, max_tth, peak_tag, _peak_function_name,
                       _background_function_name, out_of_plane_angle):
        _wavelength = self.ws.get_wavelength(True, True)

        fit_engine = PeakFitEngineFactory.getInstance(self.ws,
                                                      _peak_function_name, _background_function_name,
                                                      wavelength=_wavelength, out_of_plane_angle=out_of_plane_angle)

        fit_result = fit_engine.fit_multiple_peaks(peak_tag,
                                                   min_tth,
                                                   max_tth)

        return fit_result

    def export_fit_csv(self, out_file_name, peaks):
        sample_logs = self.ws._sample_logs

        generator = SummaryGenerator(out_file_name,
                                     log_list=sample_logs.keys())

        generator.setHeaderInformation(dict())
        generator.write_csv(sample_logs, peaks)

        return

    def save_fit_result(self, out_file_name='', fit_result=None):
        """Save the fit result, including a copy of the rest of the file if it does not exist at the specified path.

        If out_file_name is empty or if it matches the parent's current file, this updates the file.

        Otherwise, the parent's file is copied to out_file_name and
        then the updated peak fit data is written to the copy.

        :param out_file_name: string absolute fill path for the place to save the file

        """

        if fit_result is None:
            return

        if out_file_name is not None and self.parent._curr_file_name != out_file_name:
            copyfile(self.parent._curr_file_name, out_file_name)
            current_project_file = out_file_name
        else:
            current_project_file = self.parent._curr_file_name

        project_h5_file = HidraProjectFile(current_project_file, mode=HidraProjectFileMode.READWRITE)
        peakcollections = fit_result.peakcollections
        for peak in peakcollections:
            project_h5_file.write_peak_parameters(peak)
        project_h5_file.save(False)
        project_h5_file.close()

    def load_pole_data(self, peak_id, intensity, eta, peak_center, scan_index):
        if self._polefigureinterface is None:
            self._polefigureinterface = PoleFigureCalculator()

        logs_dict = {}
        for name in ['chi', 'phi', 'omega']:
            logs_dict[name] = self.ws.get_sample_log_values(name)

        log_dict = {}
        log_dict['chi'] = logs_dict['chi'][scan_index - 1]
        log_dict['phi'] = logs_dict['phi'][scan_index - 1]
        log_dict['omega'] = logs_dict['omega'][scan_index - 1]
        log_dict['eta'] = eta * np.ones_like(scan_index)
        log_dict['center'] = peak_center
        log_dict['intensity'] = intensity

        self._polefigureinterface.add_input_data_set(peak_id, log_dict)