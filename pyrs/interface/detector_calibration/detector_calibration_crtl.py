import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# from matplotlib.cm import coolwarm


class DetectorCalibrationCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model
        self._fits = None
        self._fitted_patterns = {}
        self._powders = np.array(['Ni', 'Fe', 'Mo'])
        self._sy = np.array([62, 12, -13])

    @staticmethod
    def validate_eta_tth(tth_bins, eta_bins):
        if tth_bins == '':
            tth_bins = 512
        elif type(tth_bins) is str:
            tth_bins = int(tth_bins)

        if eta_bins == '':
            eta_bins = 3
        elif type(eta_bins) is str:
            eta_bins = float(eta_bins)

        return tth_bins, eta_bins

    def export_calibration(self, filename=None):
        self._model.export_calibration(filename=filename)

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

    def fit_diffraction_peaks(self, keep_list):
        # self.check_eta_tth_bins(tth_bins, eta_bins)
        self._model.set_keep_subrun_list(keep_list)
        return self._model.fit_diffraction_peaks()

    def calibrate_detector(self, fit_recipe, keep_list):
        self._model.set_keep_subrun_list(keep_list)
        calibration, calibration_error, r_sum, rmse = self._model.calibrate_detector(fit_recipe)
        return calibration, calibration_error, r_sum, rmse

    def get_powders(self):
        return self._model.powders

    def update_diffraction_view(self, ax, _parent, sub_run, two_d_data):

        if two_d_data:
            ax.imshow(self._model.get_2D_diffraction_counts(sub_run).T)
            ax.axis('off')
        else:
            ax.cla()
            for mask in self._model.reduction_masks:
                tth, int_vec, error_vec = self._model.get_reduced_diffraction_data(sub_run, mask)
                ax.plot(tth[1:], int_vec[1:], label=mask)

            fitted_ws = self._model.get_fitted_diffraction_data(sub_run)

            if fitted_ws is not None:
                for fit_ws in fitted_ws:
                    for i_sub in range(len(self._model.reduction_masks)):
                        ax.plot(fit_ws.readX(int(i_sub))[1:],
                                fit_ws.readY(int(i_sub))[1:], '--')

            ax.legend(frameon=False)
            ax.set_xlabel(r"2$\theta$ ($deg.$)")
            ax.set_ylabel("Intensity (ct.)")
            ax.set_ylabel("Diff (ct.)")

        return

    def plot_2D_params(self, ax, x_item, y_item):

        ax.cla()

        x, y = self._model.get_calibration_values(x_item, y_item)

        if x.size != y.size:
            x = np.arange(y.size)

        ax.plot(x, y, 'ko--')

        return
