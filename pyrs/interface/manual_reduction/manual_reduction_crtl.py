"""
Presenter (controller) for the manual-reduction interface.

This is the "P" of the Model-View-Presenter structure shared with the
peak-fitting and texture-fitting UIs. It mediates between
:class:`ManualReductionViewer` (widgets) and :class:`ManualReductionModel`
(data/business logic).

Following the peak-fitting convention, the plotting methods receive the target
plot widget (``detector_view`` / ``diff_view``) as a parameter so the presenter
never touches ``self.ui``. Data-access failures are raised as ``RuntimeError`` for
the view to surface to the user.
"""

from pyrs.dataobjects.constants import HidraConstants  # type: ignore


class ManualReductionCrtl:
    """Presentation logic for the manual-reduction window."""

    def __init__(self, manual_reduction_model):
        """
        Args:
            manual_reduction_model: The :class:`ManualReductionModel` to mediate.
        """
        self._model = manual_reduction_model

    def plot_detector_counts(self, detector_view, sub_run):
        """Plot a sub-run's raw counts on the 2D detector view.

        Args:
            detector_view: The detector-view widget to draw on.
            sub_run: Sub-run number.

        Returns:
            An info string (sub-run + 2theta) for the view to display.

        Raises:
            RuntimeError: If the counts cannot be retrieved.
        """
        counts_matrix = self._model.get_detector_counts(sub_run, output_matrix=True)

        det_2theta = self._model.get_sample_log_value(HidraConstants.TWO_THETA, sub_run)
        info = "sub-run: {}, 2theta = {}".format(sub_run, det_2theta)

        detector_view.plot_detector_view(counts_matrix, (sub_run, None))

        return info

    def plot_powder_pattern(self, diff_view, sub_run):
        """Plot a sub-run's reduced powder pattern on the 1D diffraction view.

        Args:
            diff_view: The 1D diffraction-view widget to draw on.
            sub_run: Sub-run number.

        Raises:
            RuntimeError: If the powder pattern cannot be retrieved.
        """
        pattern = self._model.get_powder_pattern(sub_run)

        det_2theta = self._model.get_sample_log_value(HidraConstants.TWO_THETA, sub_run)
        info = "sub-run: {}, 2theta = {}".format(sub_run, det_2theta)

        diff_view.plot_diffraction(pattern[0], pattern[1], "2theta", "intensity", line_label=info, keep_prev=False)

    def reduce(self, nexus_file, output_dir, progressbar, mask=None, calibration=None, vanadium_file=None):
        """Reduce a NeXus file to a Hidra project file.

        Args:
            nexus_file: Path to the input NeXus file.
            output_dir: Output directory for the reduced project file.
            progressbar: Qt progress-bar widget updated during reduction.
            mask: Optional mask file path.
            calibration: Optional calibration file path.
            vanadium_file: Optional vanadium file path.

        Returns:
            The list of sub-runs in the reduced workspace.
        """
        hidra_ws = self._model.reduce_hidra_workflow(
            nexus_file,
            output_dir,
            progressbar,
            mask=mask,
            calibration=calibration,
            vanadium_file=vanadium_file,
        )

        return list(hidra_ws.get_sub_runs())
