"""
Backward-compatibility shim for the manual-reduction window.

The manual-reduction UI was refactored into a Model-View-Presenter structure
(:mod:`manual_reduction_model`, :mod:`manual_reduction_crtl`,
:mod:`manual_reduction_viewer`). This module preserves the historical
``ManualReductionWindow(parent)`` entry point used by existing tests and callers
by constructing the MVP triple internally.
"""

from pyrs.interface.manual_reduction.manual_reduction_crtl import ManualReductionCrtl
from pyrs.interface.manual_reduction.manual_reduction_model import ManualReductionModel
from pyrs.interface.manual_reduction.manual_reduction_viewer import ManualReductionViewer


class ManualReductionWindow(ManualReductionViewer):
    """Historical entry point that wires up the model and presenter itself."""

    def __init__(self, parent):
        model = ManualReductionModel()
        ctrl = ManualReductionCrtl(model)
        super(ManualReductionWindow, self).__init__(model, ctrl, parent=parent)
