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

    def load_nexus(self, nexus_file, tth_bins, eta_bins):
        tth_bins, eta_bins = self.validate_eta_tth(tth_bins, eta_bins)

        self._model._init_calibration(nexus_file, tth_bins, eta_bins)

    def set_reduction_param(self, tth_bins, eta_bins):
        tth_bins, eta_bins = self.validate_eta_tth(tth_bins, eta_bins)
        self._model.set_reduction_param(tth_bins, eta_bins)


    def fit_diffraction_peaks(self):
        self._model.fit_diffraction_peaks()

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

            ax.legend(frameon=False)
            ax.set_xlabel(r"2$\theta$ ($deg.$)")
            ax.set_ylabel("Intensity (ct.)")
            ax.set_ylabel("Diff (ct.)")
        return
