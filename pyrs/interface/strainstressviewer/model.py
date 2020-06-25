import numpy as np
from pyrs.core.strain_stress_calculator import _to_md
from pyrs.dataobjects.sample_logs import DirectionExtents
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from qtpy.QtCore import Signal, QObject


def get_test_ws():
    wksp_name = 'test_simple'
    xyz = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    extents = tuple([DirectionExtents(coordinates) for coordinates in xyz])

    # we have one signal and one error for each of the 10 x 10 x 10 xyz coordinates
    signal, errors = np.random.normal(size=1000), np.zeros(1000, dtype=float)

    # create MDhisto
    ws = _to_md(wksp_name, extents, signal, errors, units='mm,mm,mm')
    return ws


class Model(QObject):
    propertyUpdated = Signal(str)

    def __init__(self):
        super().__init__()
        self._e11 = None
        self._e22 = None
        self._e33 = None
        self._e11_logs = None
        self._e22_logs = None
        self._e33_logs = None
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

    @property
    def e11_peaks(self):
        return self._e11_peaks

    @property
    def e11_logs(self):
        return self._e11_logs

    @e11.setter
    def e11(self, filename):
        self._e11 = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e11_logs = self._e11.read_sample_logs()
        self._e11_peaks = dict()  # clear out existing peaks
        for peak in self._e11.read_peak_tags():
            self._e11_peaks[peak] = self._e11.read_peak_parameters(peak)
        self._peakTags = list(self._e11_peaks.keys())
        self.propertyUpdated.emit("peakTags")

    @property
    def e22(self):
        return self._e22

    @property
    def e22_peaks(self):
        return self._e11_peaks

    @property
    def e22_logs(self):
        return self._e11_logs

    @e22.setter
    def e22(self, filename):
        self._e22 = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e22_logs = self._e22.read_sample_logs()
        self._e22_peaks = dict()  # clear out existing peaks
        for peak in self._e22.read_peak_tags():
            self._e22_peaks[peak] = self._e22.read_peak_parameters(peak)

    @property
    def e33(self):
        return self._e33

    @property
    def e33_peaks(self):
        return self._e11_peaks

    @property
    def e33_logs(self):
        return self._e11_logs

    @e33.setter
    def e33(self, filename):
        self._e33 = HidraProjectFile(filename, mode=HidraProjectFileMode.READONLY)
        self._e33_logs = self._e33.read_sample_logs()
        self._e33_peaks = dict()  # clear out existing peaks
        for peak in self._e33.read_peak_tags():
            self._e33_peaks[peak] = self._e33.read_peak_parameters(peak)

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
            if peaks is not None:
                peaks[self.selectedPeak].set_d_reference(np.array(d0))
