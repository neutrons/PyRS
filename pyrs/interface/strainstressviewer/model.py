import traceback
from pyrs.dataobjects.fields import StressField, StrainField, ScalarFieldSample
from pyrs.core.stress_facade import StressFacade
from pyrs.core.summary_generator_stress import SummaryGeneratorStress
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from qtpy.QtCore import Signal, QObject


class Model(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self):
        super().__init__()
        self._e11 = []
        self._e11_strain = None
        self._e22 = []
        self._e22_strain = None
        self._e33 = []
        self._e33_strain = None
        self._e11_peaks = []
        self._e22_peaks = []
        self._e33_peaks = []
        self._peakTags = []
        self._selectedPeak = None
        self._stress = None
        self._stress_facade = None
        self._stressCase = None
        self._youngs_modulus = None
        self._poisson_ratio = None
        self._d0 = None

    def set_workspaces(self, name, filenames):
        setattr(self, name, filenames)

    @property
    def e11(self):
        return self._e11

    @e11.setter
    def e11(self, filename):
        self._e11, self._e11_peaks = self.load_hidra_project_files(filename, '11')
        self._peakTags = list(self._e11_peaks[0].keys()) if len(self._e11_peaks) > 0 else []
        self.propertyUpdated.emit("peakTags")

    @property
    def e11_peaks(self):
        return self._e11_peaks

    @property
    def e22(self):
        return self._e22

    @e22.setter
    def e22(self, filename):
        self._e22, self._e22_peaks = self.load_hidra_project_files(filename, '22')
        if self._e22 and self.check_peak_for_direction('22'):
            self.create_strain('22')

    @property
    def e22_peaks(self):
        return self._e22_peaks

    @property
    def e33(self):
        return self._e33

    @e33.setter
    def e33(self, filename):
        self._e33, self._e33_peaks = self.load_hidra_project_files(filename, '33')
        if self._e33 and self.check_peak_for_direction('33'):
            self.create_strain('33')

    @property
    def e33_peaks(self):
        return self._e33_peaks

    @property
    def peakTags(self):
        return self._peakTags

    @property
    def selectedPeak(self):
        return self._selectedPeak

    @selectedPeak.setter
    def selectedPeak(self, tag):
        self._selectedPeak = tag
        for direction in ('11', '22', '33'):
            if getattr(self, f'e{direction}') and self.check_peak_for_direction(f'{direction}'):
                self.create_strain(direction)
        self.propertyUpdated.emit("selectedPeak")

    @property
    def d0(self):
        if self.stress_facade:
            try:
                return self.stress_facade.d_reference
            except Exception as e:
                self.failureMsg.emit("Reference d spacings are different on different directions",
                                     str(e), None)
                return None, dict()

    @property
    def stress(self):
        return self._stress

    @stress.setter
    def stress(self, stress):
        self._stress = stress
        if stress is None:
            self._stress_facade = None
        else:
            self._stress_facade = StressFacade(stress)

    @property
    def stress_facade(self):
        return self._stress_facade

    def create_strain(self, direction):
        strain_list = [StrainField(hidraworkspace=ws, peak_collection=peak[self.selectedPeak])
                       for ws, peak in zip(getattr(self, f'e{direction}'), getattr(self, f'e{direction}_peaks'))]
        if len(strain_list) == 1:
            setattr(self, f'_e{direction}_strain', strain_list[0])
        else:
            setattr(self, f'_e{direction}_strain', sum(strain_list[1:], strain_list[0]))

    def check_peak_for_direction(self, direction):
        for peak in getattr(self, f'e{direction}_peaks'):
            if self.selectedPeak not in peak:
                return False
        return True

    def validate_selection(self, direction):
        if not getattr(self, f'e{direction}'):
            return f"e{direction} file hasn't been loaded"

        if not self.e11:
            return "e11 is not loaded, the peak tags from this file will be used"

        if not self.e11_peaks:
            return "e11 contains no peaks, fit peaks first"

        if not self.check_peak_for_direction(f'{direction}'):
            return f"Peak {self.selectedPeak} is not in e{direction}"

    def get_parameter_field(self, plot_param, direction):
        if plot_param == 'strain':
            return getattr(self, f'_e{direction}_strain')
        elif plot_param == 'd-reference':
            return getattr(self, f'_e{direction}_strain').get_d_reference()
        elif plot_param == "dspacing-center":
            return getattr(self, f'_e{direction}_strain').get_dspacing_center()
        else:
            return getattr(self, f'_e{direction}_strain').get_effective_peak_parameter(plot_param)

    def get_field(self, direction, plot_param, stress_case):
        try:
            if plot_param == "stress":
                self.stress.select(direction)
                return self._stress
            elif plot_param == "strain" and direction == "33" and stress_case == "In-plane stress":
                return self.stress.strain33
            else:
                return self.get_parameter_field(plot_param, direction)
        except Exception as e:
            self.failureMsg.emit(f"Failed to generate field for parameter {plot_param} in direction {direction}",
                                 str(e),
                                 traceback.format_exc())

    def calculate_stress(self, stress_case, youngModulus, poissonsRatio, d0):
        build_stress = False
        if self.stress is None or stress_case != self._stressCase:
            build_stress = True

        if build_stress:
            self.stress = StressField(self._e11_strain,
                                      self._e22_strain,
                                      self._e33_strain if stress_case == "diagonal" else None,
                                      youngModulus, poissonsRatio, stress_case)
        else:
            if youngModulus != self._youngs_modulus:
                self._stress.youngs_modulus = youngModulus
            if poissonsRatio != self._poisson_ratio:
                self._stress.poisson_ratio = poissonsRatio

        if d0 is not None and (self._d0 is None or self._d0 != d0):
            if len(d0) == 5:  # ScalarFieldSample case
                self.stress_facade.d_reference = ScalarFieldSample('d0', *d0)
            else:
                self.stress_facade.d_reference = d0

        # cache values to compare if they are changed
        self._stressCase = stress_case
        self._youngs_modulus = youngModulus
        self._poisson_ratio = poissonsRatio
        self._d0 = d0

    def write_stress_to_csv(self, filename):
        try:
            stress_csv = SummaryGeneratorStress(filename, self._stress)
            stress_csv.write_summary_csv()
        except Exception as e:
            self.failureMsg.emit("Failed to write csv",
                                 str(e),
                                 traceback.format_exc())

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

    def load_hidra_project_file(self, filename, direction):
        try:
            source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
            ws = HidraWorkspace(direction)
            ws.load_hidra_project(source_project, False, False)
            peaks = dict()
            for peak in source_project.read_peak_tags():
                peaks[peak] = source_project.read_peak_parameters(peak)
            return ws, peaks
        except Exception as e:
            self.failureMsg.emit(f"Failed to load {filename}. Check that this is a Hidra Project File",
                                 str(e),
                                 traceback.format_exc())
            return None, dict()

    def load_hidra_project_files(self, filenames, direction):
        # remove existing stress as it will need to be recreated
        self.stress = None

        workspaces = []
        peaks = []
        if isinstance(filenames, str):
            filenames = [filenames]
        for filename in filenames:
            ws, p = self.load_hidra_project_file(filename, direction)
            if ws is None:
                return [], []
            workspaces.append(ws)
            if p:
                peaks.append(p)

        return workspaces, peaks
