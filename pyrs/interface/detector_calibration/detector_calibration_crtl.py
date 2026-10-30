import numpy as np
from pyrs.interface.ui.mplconstants import MplBasicColors
from pyrs.interface.detector_calibration.detector_calibration_model import DEFAULT_STRIPS, METHOD_RING


class DetectorCalibrationCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model
        self._fits = None
        self._fitted_patterns = {}
        self._powders = np.array(["Ni", "Fe", "Mo"])
        self._sy = np.array([62, 12, -13])

    @property
    def rmse(self):
        return self._model.rmse

    @property
    def r_sum(self):
        return self._model.r_sum

    @staticmethod
    def validate_eta_tth(tth_bins, eta_bins):
        if tth_bins == "":
            tth_bins = 512
        elif type(tth_bins) is str:
            tth_bins = int(tth_bins)

        if eta_bins == "":
            eta_bins = 3
        elif type(eta_bins) is str:
            eta_bins = float(eta_bins)

        return tth_bins, eta_bins

    def export_calibration(self, folder, write_shared=True):
        """Write the calibration to ``folder`` and, with ``write_shared``, the shared HB2B copies; returns the files."""
        return self._model.export_calibration(folder, write_shared=write_shared)

    def set_method(self, method):
        self._model.set_method(method)

    def load_nexus(self, nexus_file, tth_bins, eta_bins):
        tth_bins, eta_bins = self.validate_eta_tth(tth_bins, eta_bins)

        self._model._init_calibration(nexus_file, tth_bins, eta_bins)

    def set_calibration_params(self, params):
        self._model.set_calibration_params(params)

    def set_reduction_param(self, tth_bins, eta_bins):
        tth_bins, eta_bins = self.validate_eta_tth(tth_bins, eta_bins)
        self._model.set_reduction_param(tth_bins, eta_bins)

    def set_refinement_params(self, method, max_nfev):
        self._model.set_refinement_params(method, max_nfev)

    def get_wavelength(self):
        return self._model.get_wavelength

    def fit_diffraction_peaks(self, keep_list, strips=None):
        """FitCalibration: fit the peaks; ring-based: extract the ring points in ``strips`` strips.

        Returns the RMS and the residual sum at the current calibration for both methods.
        """
        # self.check_eta_tth_bins(tth_bins, eta_bins)
        self._model.set_keep_subrun_list(keep_list)
        if self._model.method == METHOD_RING:
            _, _, rmse, r_sum = self._model.extract_rings(DEFAULT_STRIPS if strips is None else strips)
            return rmse, r_sum
        return self._model.fit_diffraction_peaks()

    def calibrate_detector(self, args):
        fit_recipe, keep_list = args[:]
        self._model.set_keep_subrun_list(keep_list)
        calibration, calibration_error, r_sum, rmse = self._model.calibrate_detector(fit_recipe)
        return calibration, calibration_error, r_sum, rmse

    def get_powders(self):
        return self._model.powders

    def update_diff_view(self, _ax, diff_plot_type, sub_run, exclude_list, ring_geometry=None, calibration_file=None):
        """Redraw the 2D (``diff_plot_type`` 0) or 1D view of ``sub_run``.

        On the 2D view, ``ring_geometry`` (one of the model's ``RING_GEOMETRIES``, or None for none)
        draws the predicted rings of that geometry; ``calibration_file`` is the JSON for "From file...".
        """
        if diff_plot_type == 0:
            counts_matrix = self._model.get_2D_diffraction_counts(sub_run)
            if counts_matrix is not None:
                _ax.plot_detector_view(counts_matrix, (sub_run, None))
                overlay = self._model.ring_overlay(sub_run, ring_geometry, calibration_file) if ring_geometry else None
                if overlay is None:
                    _ax.set_ring_overlay()
                else:
                    _ax.set_ring_overlay(*overlay)
        else:
            try:
                if exclude_list[sub_run - 1]:
                    for i_mask, mask in enumerate(self._model.reduction_masks):
                        tth, int_vec, error_vec = self._model.get_reduced_diffraction_data(sub_run, mask)
                        _ax.plot_diffraction(
                            tth[1:],
                            int_vec[1:],
                            r"2$\theta$ (degrees)",
                            "Intensity (ct.)",
                            line_label=mask,
                            color=MplBasicColors[i_mask],
                            keep_prev=i_mask != 0,
                        )

                    fitted_ws = self._model.get_fitted_diffraction_data(sub_run)

                    if fitted_ws is not None:
                        for fit_ws in fitted_ws:
                            for i_sub in range(len(self._model.reduction_masks)):
                                xvec = fit_ws.readX(int(i_sub))[1:]
                                yvec = fit_ws.readY(int(i_sub))[1:]
                                _ax.plot_diffraction(
                                    xvec,
                                    yvec,
                                    r"2$\theta$ (degrees)",
                                    "Intensity (ct.)",
                                    color=MplBasicColors[i_sub],
                                    line_style="--",
                                    keep_prev=True,
                                )
                else:
                    _ax.set_no_null_plot()

            except (AttributeError, TypeError, IndexError):
                # _ax.set_no_null_plot()
                pass

    def plot_2D_params(self, ax_obj, x_item, y_item, x_text, y_text):
        # reset object viewer
        ax_obj.reset_viewer()

        x_data, y_data = self._model.get_calibration_values(x_item, y_item)

        if x_data.size != y_data.size:
            x_data = np.arange(y_data.size)

        ax_obj.plot_scatter(x_data, y_data, x_text, y_text)

        return
