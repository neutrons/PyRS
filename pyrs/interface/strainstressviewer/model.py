import numpy as np
from pyrs.dataobjects.fields import StrainField
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from qtpy.QtCore import Signal, QObject


class Model(QObject):
    propertyUpdated = Signal(str)

    def __init__(self):
        super().__init__()
        self._e11 = HidraWorkspace('11')
        self._e22 = HidraWorkspace('22')
        self._e33 = HidraWorkspace('33')
        self._e11_peaks = dict()
        self._e22_peaks = dict()
        self._e33_peaks = dict()
        self._peakTags = []
        self._selectedPeak = None

    def set_workspace(self, name, filename):
        setattr(self, name, filename)

    @property
    def e11(self):
        return self._e11

    @e11.setter
    def e11(self, filename):
        source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e11 = HidraWorkspace('11')
        self._e11.load_hidra_project(source_project, False, False)
        self._e11_peaks = dict()  # clear out existing peaks
        for peak in source_project.read_peak_tags():
            self._e11_peaks[peak] = source_project.read_peak_parameters(peak)
        self._peakTags = list(self._e11_peaks.keys())
        self.propertyUpdated.emit("peakTags")

    @property
    def e11_peaks(self):
        return self._e11_peaks

    @property
    def e22(self):
        return self._e22

    @e22.setter
    def e22(self, filename):
        source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e22 = HidraWorkspace('22')
        self._e22.load_hidra_project(source_project, False, False)
        self._e22_peaks = dict()  # clear out existing peaks
        for peak in source_project.read_peak_tags():
            self._e22_peaks[peak] = source_project.read_peak_parameters(peak)

    @property
    def e22_peaks(self):
        return self._e22_peaks

    @property
    def e33(self):
        return self._e33

    @e33.setter
    def e33(self, filename):
        source_project = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e33 = HidraWorkspace('33')
        self._e33.load_hidra_project(source_project, False, False)
        self._e33_peaks = dict()  # clear out existing peaks
        for peak in source_project.read_peak_tags():
            self._e33_peaks[peak] = source_project.read_peak_parameters(peak)

    @property
    def e33_peaks(self):
        return self._e33_peaks

    @property
    def subruns(self):
        return self.e11_peaks[self.selectedPeak].sub_runs

    @property
    def peakTags(self):
        return self._peakTags

    @property
    def selectedPeak(self):
        return self._selectedPeak

    @selectedPeak.setter
    def selectedPeak(self, tag):
        self._selectedPeak = tag
        self.propertyUpdated.emit("subruns")

    @property
    def d0(self):
        return self.e11_peaks[self.selectedPeak].get_d_reference()

    @d0.setter
    def d0(self, d0):
        for peaks in [self.e11_peaks, self.e22_peaks, self.e33_peaks]:
            if self.selectedPeak in peaks:
                peaks[self.selectedPeak].set_d_reference(np.array(d0))

    def get_field_md(self, direction='11', plot_param='strain'):
        return StrainField(hidraworkspace=getattr(self, f'e{direction}'),
                           peak_collection=getattr(self, f'e{direction}_peaks')[self.selectedPeak]
                           ).to_md_histo_workspace(f'e{direction} {plot_param}')
