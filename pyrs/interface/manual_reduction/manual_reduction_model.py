"""
Model for the manual-reduction interface.

This is the "M" of the Model-View-Presenter structure shared with the
peak-fitting and texture-fitting UIs. It owns the data and business logic for
manually reducing HB2B NeXus files into Hidra project files / powder patterns.

The underlying reduction engine already lives in
:class:`pyrs.interface.manual_reduction.pyrs_api.ReductionController` (and the
module-level :func:`reduce_hidra_workflow`), which are also used directly by the
integration tests. This model composes a ``ReductionController`` and adds the Qt
signal plumbing expected of an MVP model, so the engine stays UI-agnostic and the
existing public API is untouched.
"""

from qtpy.QtCore import QObject, Signal  # type:ignore

from pyrs.interface.manual_reduction.pyrs_api import ReductionController


class ManualReductionModel(QObject):
    """Data and business logic for the manual-reduction window.

    Attributes:
        failureMsg: Emitted with ``(title, message, traceback)`` when an
            operation fails, so the view can surface it without the model
            depending on any UI helper.
    """

    failureMsg = Signal(str, str, str)

    def __init__(self):
        super().__init__()
        self._controller = ReductionController()

    @property
    def working_dir(self):
        """Working directory used for file dialogs."""
        return self._controller.working_dir

    @staticmethod
    def get_default_calibration_dir():
        """Default calibration directory on the analysis cluster."""
        return ReductionController.get_default_calibration_dir()

    @staticmethod
    def get_default_mask_dir():
        """Default mask directory on the analysis cluster."""
        return ReductionController.get_default_mask_dir()

    @staticmethod
    def get_default_nexus_dir(ipts_number=None):
        """Default NeXus directory, optionally for a specific IPTS."""
        return ReductionController.get_default_nexus_dir(ipts_number)

    def get_sub_runs(self):
        """Return the sub-runs of the currently loaded workspace."""
        return self._controller.get_sub_runs()

    def get_detector_counts(self, sub_run_number, output_matrix=True):
        """Return detector counts for a sub-run (2D when ``output_matrix``)."""
        return self._controller.get_detector_counts(sub_run_number, output_matrix)

    def get_powder_pattern(self, sub_run_number):
        """Return ``(vec_2theta, vec_intensity)`` for a sub-run."""
        return self._controller.get_powder_pattern(sub_run_number)

    def get_sample_log_value(self, log_name, sub_run_number):
        """Return a sample-log value for a sub-run."""
        return self._controller.get_sample_log_value(log_name, sub_run_number)

    def reduce_hidra_workflow(
        self,
        nexus,
        output_dir,
        progressbar,
        instrument=None,
        calibration=None,
        mask=None,
        vanadium_file=None,
        project_file_name=None,
    ):
        """Run the full NeXus-to-project reduction workflow.

        Returns:
            The reduced ``HidraWorkspace``.
        """
        return self._controller.reduce_hidra_workflow(
            nexus,
            output_dir,
            progressbar,
            instrument=instrument,
            calibration=calibration,
            mask=mask,
            vanadium_file=vanadium_file,
            project_file_name=project_file_name,
        )

    def save_project(self):
        """Save the current workspace to its project file."""
        return self._controller.save_project()
