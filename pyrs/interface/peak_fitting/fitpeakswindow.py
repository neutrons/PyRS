"""
Backward-compatibility shim for the peak-fitting window.

The peak-fitting UI was refactored into a Model-View-Presenter structure
(:mod:`peak_fitting_model`, :mod:`peak_fitting_crtl`, :mod:`peak_fitting_viewer`).
This module preserves the historical ``FitPeaksWindow(parent, fit_peak_core=...)``
entry point used by existing tests and callers by constructing the MVP triple
internally.
"""

from pyrs.interface.peak_fitting.peak_fitting_crtl import PeakFittingCrtl
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel
from pyrs.interface.peak_fitting.peak_fitting_viewer import PeakFittingViewer


class FitPeaksWindow(PeakFittingViewer):
    """Historical entry point that wires up the model and presenter itself."""

    def __init__(self, parent, fit_peak_core=None):
        model = PeakFittingModel(fit_peak_core)
        ctrl = PeakFittingCrtl(model)
        super(FitPeaksWindow, self).__init__(model, ctrl, parent=parent)
