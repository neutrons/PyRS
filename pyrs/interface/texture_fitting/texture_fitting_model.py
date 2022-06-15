import json
import traceback
from shutil import copyfile
import numpy as np

# from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from qtpy.QtCore import Signal, QObject  # type:ignore
# from pyrs.interface.gui_helper import pop_message
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore


class TextureFittingModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        super().__init__()
        self._peak_fit = peak_fit_core
        self.ws = None
        self.peak_fit_engine = None

    def set_workspaces(self, name, filenames):
        setattr(self, name, filenames)

    def get_default_csv_filename(self):
        runs = [[peak_collection.runnumber for peak_collection in getattr(self._stress, f'strain{d}').peak_collections]
                for d in ('11', '22', '33')]
        runnumbers = []
        for runs in runs:
            for runnumber in runs:
                if runnumber != -1:
                    runnumbers.append(str(runnumber))
        return "HB2B_{}_stress_grid_{}.csv".format('_'.join(runnumbers),
                                                   self.selectedPeak)

    def load_hidra_project_file(self, filename):
        try:
            source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
            self.ws = HidraWorkspace(filename)
            self.ws.load_hidra_project(source_project, False, True)
            self.sub_runs = np.array(self.ws.get_sub_runs())
        except Exception as e:
            self.failureMsg.emit(f"Failed to load {filename}. Check that this is a Hidra Project File",
                                 str(e),
                                 traceback.format_exc())
            return None, dict()

    def to_json(self, filename):
        try:
            json_output = dict()
            json_output['stress_case'] = self._stressCase
            json_output['filenames_11'] = self.get_filenames_for_direction('11')
            json_output['filenames_22'] = self.get_filenames_for_direction('22')
            json_output['filenames_33'] = self.get_filenames_for_direction('33')
            json_output['peak_tag'] = self.selectedPeak
            json_output['youngs_modulus'] = self._youngs_modulus
            json_output['poisson_ratio'] = self._poisson_ratio
            if self._d0 is None:
                json_output['d0'] = None
            elif len(self._d0) == 5:  # ScalarFieldSample case
                json_output['d0'] = {"d0": self._d0[0],
                                     "d0_error": self._d0[1],
                                     "vx": self._d0[2],
                                     "vy": self._d0[3],
                                     "vz": self._d0[4]}
            else:
                json_output['d0'] = {"d0": self._d0[0],
                                     "d0_error": self._d0[1]}

            with open(filename, 'w') as f:
                json.dump(json_output, f)
        except Exception as e:
            self.failureMsg.emit(f"Failed save json file to {filename}",
                                 str(e),
                                 traceback.format_exc())

    def from_json(self, filename):
        with open(filename) as f:
            data = json.load(f)

        self._selectedPeak = data["peak_tag"]

        for direction in ('11', '22', '33'):
            self.set_workspaces(f'e{direction}', data[f'filenames_{direction}'])
            if getattr(self, f'e{direction}'):
                self.create_strain(direction)

        d0_data = data["d0"]
        if d0_data is None:
            d0 = None
        elif len(d0_data) == 2:
            d0 = (d0_data['d0'], d0_data['d0_error'])
        else:
            d0 = (d0_data['d0'],
                  d0_data['d0_error'],
                  d0_data['vx'],
                  d0_data['vy'],
                  d0_data['vz'])

        self.calculate_stress(data["stress_case"],
                              data["youngs_modulus"],
                              data["poisson_ratio"],
                              d0)

        self.propertyUpdated.emit("modelUpdated")

    def fit_diff_peaks(self, min_tth, max_tth, peak_tag, _peak_function_name,
                       _background_function_name, out_of_plane_angle):
        _wavelength = self.ws.get_wavelength(True, True)

        fit_engine = PeakFitEngineFactory.getInstance(self.ws,
                                                      _peak_function_name, _background_function_name,
                                                      wavelength=_wavelength)

        fit_result = fit_engine.fit_multiple_peaks(peak_tag,
                                                   min_tth,
                                                   max_tth)

        return fit_result

    def save_fit_result(self, out_file_name=''):
        """Save the fit result, including a copy of the rest of the file if it does not exist at the specified path.

        If out_file_name is empty or if it matches the parent's current file, this updates the file.

        Otherwise, the parent's file is copied to out_file_name and
        then the updated peak fit data is written to the copy.

        :param out_file_name: string absolute fill path for the place to save the file

        """

        fit_result = self.parent.fit_result
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
