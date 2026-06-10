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


def parse_run_numbers(text):
    """Parse a run-number specification into an explicit list of run numbers.

    Ranges use a dash and are **inclusive**; individual runs and ranges are
    separated by commas. For example ``"938-940,945"`` yields ``[938, 939, 940,
    945]``.

    Args:
        text: The run specification string.

    Returns:
        A list of integer run numbers, in the order given.

    Raises:
        ValueError: If a token is not an integer or a valid ``start-stop`` range.
    """
    runs = []
    for token in text.replace(" ", "").split(","):
        if token == "":
            continue
        if "-" in token:
            start, stop = token.split("-")
            runs.extend(range(int(start), int(stop) + 1))
        else:
            runs.append(int(token))
    return runs


def is_run_specification(text):
    """Return True if ``text`` looks like a run-number spec rather than a file path.

    A run spec contains only digits, dashes, commas and spaces, and has at least
    one digit (e.g. ``"938"`` or ``"938-940,945"``); anything else (such as a
    NeXus file path) is treated as a single file.
    """
    text = text.strip()
    return bool(text) and set(text) <= set("0123456789-, ") and any(ch.isdigit() for ch in text)


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
        # reduced workspaces from the most recent (possibly batch) reduction,
        # keyed by display label so the view can switch between them
        self._reduced = {}

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

    def reduce_runs(self, jobs, output_dir, progressbar, mask=None, calibration=None, vanadium_file=None):
        """Reduce one or more NeXus files and keep the results in memory.

        Args:
            jobs: List of ``(label, nexus_file)`` pairs to reduce in order.
            output_dir: Output directory for the reduced project files.
            progressbar: Qt progress-bar widget updated during reduction.
            mask: Optional mask file path.
            calibration: Optional calibration file path.
            vanadium_file: Optional vanadium file path.

        Returns:
            The list of labels actually reduced (in input order). The first
            label's workspace is left as the current workspace.
        """
        self._reduced = {}
        labels = []
        for label, nexus_file in jobs:
            hidra_ws = self._controller.reduce_hidra_workflow(
                nexus_file,
                output_dir,
                progressbar,
                mask=mask,
                calibration=calibration,
                vanadium_file=vanadium_file,
            )
            self._reduced[label] = hidra_ws
            labels.append(label)

        if labels:
            self.set_current_run(labels[0])

        return labels

    def set_current_run(self, label):
        """Make the reduced workspace for ``label`` the current workspace."""
        self._controller.set_current_workspace(self._reduced[label])

    def save_project(self):
        """Save the current workspace to its project file."""
        return self._controller.save_project()
