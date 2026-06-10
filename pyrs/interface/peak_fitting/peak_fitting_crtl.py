"""
Presenter (controller) for the peak-fitting interface.

This is the "P" of the Model-View-Presenter structure shared with the
texture-fitting UI. It mediates between :class:`PeakFittingViewer` (widgets) and
:class:`PeakFittingModel` (data/business logic): it extracts plottable arrays from
the fit results and prepares plot data, but never touches Qt widgets directly.

Following the texture-fitting convention, the plotting methods receive the target
matplotlib widget (``ax_object`` / ``fit_setup_view``) as a parameter so the
presenter stays free of any ``self.ui`` reference.
"""

from typing import Optional, Tuple

import numpy as np

from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT
from pyrs.interface.peak_fitting.config import fit_dict as FIT_DICT
from pyrs.interface.utilities.plot_data_preparer import prepare_3d_plot_data


class PeakFittingCrtl:
    """Presentation logic for the peak-fitting window."""

    def __init__(self, peak_fit_model):
        """
        Args:
            peak_fit_model: The :class:`PeakFittingModel` to mediate.
        """
        self._model = peak_fit_model

    # ------------------------------------------------------------------
    # Data extraction (was peak_fitting/data_retriever.py)
    # ------------------------------------------------------------------
    def get_data(
        self, name: str = "Sub-runs", peak_index: int = 0, d_reference_list=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(values, error)`` arrays for a named axis.

        Args:
            name: Axis/quantity name (sample log, ``"Sub-runs"``, ``"d-spacing"``,
                ``"microstrain"``, or an effective fit parameter).
            peak_index: Zero-based index into the peak collections.
            d_reference_list: Per-peak d-reference values read from the peak-range
                table by the view; used only for the ``"d-spacing"`` axis.

        Raises:
            RuntimeError: If the quantity name is not recognised.
        """
        hidra_workspace = self._model.hidra_workspace
        fit_result = self._model.fit_result

        try:
            keep_list = np.array(
                [entry is False for entry in fit_result.peakcollections[peak_index].get_exclude_list()]
            )
        except AttributeError:
            keep_list = np.ones_like(hidra_workspace.get_sub_runs()) == 1

        if name == "Sub-runs":
            return (np.array(hidra_workspace.get_sub_runs())[keep_list], None)

        if name in LIST_AXIS_TO_PLOT["raw"].keys():  # type: ignore
            return (hidra_workspace._sample_logs[name][keep_list], None)

        if name == "d-spacing":
            peak_collection = fit_result.peakcollections[peak_index]
            _d_reference = np.float32(d_reference_list[peak_index])
            peak_collection.set_d_reference(values=_d_reference)
            values, error = peak_collection.get_dspacing_center()
            return (values[keep_list], error[keep_list])

        if name == "microstrain":
            peak_collection = fit_result.peakcollections[peak_index]
            values, error = peak_collection.get_strain(units="microstrain")
            return (values[keep_list], error[keep_list])

        if name in LIST_AXIS_TO_PLOT["fit"].keys():  # type: ignore
            return self.get_fitted_value(peak=fit_result.peakcollections[peak_index], value_to_display=name)

        raise RuntimeError('Do not know how to get values for "{}"'.format(name))

    def get_fitted_value(self, peak=None, value_to_display="Center"):
        """Return ``(values, error)`` of an effective fit parameter for a peak."""
        keep_list = np.array([entry is False for entry in peak.get_exclude_list()])

        value, error = peak.get_effective_params()

        mantid_value_to_display = FIT_DICT[value_to_display]
        value_selected = value[mantid_value_to_display]
        error_selected = error[mantid_value_to_display]
        return value_selected[keep_list], error_selected[keep_list]

    def get_function_parameter_data(self, param_name):
        """Return ``(sub_run_vec, values)`` for a raw fit parameter, or ``None``."""
        result = self._model.get_peak_fitting_result(return_format=dict, effective_parameter=False)
        if result is None:
            return None

        _param_names, param_data = result
        sub_run_vec = param_data[HidraConstants.SUB_RUNS]
        param_value_2darray = param_data[param_name]
        return sub_run_vec, param_value_2darray[:, 0]

    # ------------------------------------------------------------------
    # Diffraction-pattern plotting (was peak_fitting/plot.py)
    # ------------------------------------------------------------------
    def plot_diff_and_fitted_data(self, fit_setup_view, sub_run_number):
        """Plot one sub-run's experimental data and, if present, its fit."""
        diff_data_set = self._model.get_diffraction_data(sub_run=sub_run_number, mask=None)
        fit_setup_view.plot_experiment_data(
            diff_data_set=diff_data_set, data_reference="Scan {0}".format(sub_run_number)
        )

        fit_result = self._model.fit_result
        if fit_result:
            sub_run_index = int(fit_result.peakcollections[0].sub_runs.get_indices(sub_run_number)[0])

            x_array = fit_result.fitted.readX(sub_run_index)
            y_array = fit_result.fitted.readY(sub_run_index)
            fit_setup_view.plot_fitted_data(x_array, y_array)

            err_x_array = fit_result.difference.readX(sub_run_index)
            err_y_array = fit_result.difference.readY(sub_run_index)
            fit_setup_view.plot_fitting_diff_data(x_axis=err_x_array, y_axis=err_y_array)

    def plot_scan(self, fit_setup_view, scan_value, sub_run_list):
        """Plot the sub-run at ``scan_value`` (1-based) and return its number.

        The view uses the returned sub-run number to update its label.
        """
        sub_run = sub_run_list[scan_value - 1]

        try:
            fit_setup_view.reset_viewer()
            self.plot_diff_and_fitted_data(fit_setup_view, sub_run)
        except RuntimeError:
            pass

        return sub_run

    def plot_diff_data(self, fit_setup_view, scan_log_index_list, sub_run_list):
        """Plot one or more sub-runs of diffraction data.

        Args:
            fit_setup_view: The peak-fit setup plot widget to draw on.
            scan_log_index_list: Parsed list of sub-run indices to plot.
            sub_run_list: Full ordered list of sub-runs (for single-scan lookup).

        Returns:
            The plotted sub-run number when a single scan is requested, else
            ``None`` (the view updates its scan label only in the single case).
        """
        fit_setup_view.reset_viewer()

        if len(scan_log_index_list) == 1:
            return self.plot_scan(fit_setup_view, np.int32(scan_log_index_list[0]), sub_run_list)

        for scan_log_index in scan_log_index_list:
            try:
                self.plot_diff_and_fitted_data(fit_setup_view, scan_log_index)
            except RuntimeError:
                pass

        return None

    # ------------------------------------------------------------------
    # 1D / 2D parameter plotting (was peak_fitting/plot.py)
    # ------------------------------------------------------------------
    def parse_sub_run_list(self, sub_run_list, num_sub_runs):
        """Crop a list of user-entered sub-run indices to the valid range."""
        sub_run_list = np.array(sub_run_list)
        sub_run_index = (sub_run_list >= 1) == (sub_run_list < num_sub_runs)
        return sub_run_list[sub_run_index]

    def plot_1d(self, ax_object, x_axis_name, y_axis_name, peak_index, d_reference_list=None):
        """Plot a 1D scatter (with error bars) of one fit quantity against another."""
        ax_object.reset_viewer()

        axis_x_data, axis_x_error = self.get_data(
            name=x_axis_name, peak_index=peak_index, d_reference_list=d_reference_list
        )
        axis_y_data, axis_y_error = self.get_data(
            name=y_axis_name, peak_index=peak_index, d_reference_list=d_reference_list
        )

        ax_object.plot_scatter_with_errors(
            vec_x=axis_x_data,
            vec_y=axis_y_data,
            vec_x_error=axis_x_error,
            vec_y_error=axis_y_error,
            x_label=x_axis_name,
            y_label=y_axis_name,
        )

    def plot_2d(
        self,
        ax_object,
        x_axis_name,
        y_axis_name,
        z_axis_name,
        x_peak_index,
        y_peak_index,
        z_peak_index,
        mode="scatter",
        sub_run_list=None,
        d_reference_list=None,
    ):
        """Plot a 3D scatter/contour/line surface of three fit quantities.

        Args:
            ax_object: The 3D plot widget to draw on.
            x_axis_name, y_axis_name, z_axis_name: Axis quantity names.
            x_peak_index, y_peak_index, z_peak_index: Peak indices per axis.
            mode: ``"contour"``, ``"lines"`` or ``"scatter"``.
            sub_run_list: Optional list of sub-run indices to restrict the plot.
            d_reference_list: Per-peak d-reference values from the view.
        """
        try:
            axis_x_data, _ = self.get_data(
                name=x_axis_name, peak_index=x_peak_index, d_reference_list=d_reference_list
            )
            axis_y_data, _ = self.get_data(
                name=y_axis_name, peak_index=y_peak_index, d_reference_list=d_reference_list
            )
            axis_z_data, _ = self.get_data(
                name=z_axis_name, peak_index=z_peak_index, d_reference_list=d_reference_list
            )

            if sub_run_list is not None:
                sub_run_list = self.parse_sub_run_list(sub_run_list, len(axis_x_data))
                axis_x_data = axis_x_data[sub_run_list]
                axis_y_data = axis_y_data[sub_run_list]
                axis_z_data = axis_z_data[sub_run_list]

            vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(
                axis_x_data, axis_y_data, axis_z_data, mode=mode
            )

            ax_object.plot_3D_scatter(
                vec_x,
                vec_y,
                vec_z,
                plot_scatter,
                colors=colors,
                x_label=x_axis_name,
                y_label=y_axis_name,
                z_label=z_axis_name,
            )
        except RuntimeError:
            ax_object.reset_viewer()
