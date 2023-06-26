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

    def load_nexus(self, nexus_file):
        self._model._load_nexus_data(nexus_file)

    def get_powders(self):
        sy = self._model.sy

        powder = []
        for i_pos in range(sy.size):
            try:
                powder.append(self._powders[np.abs(self._sy - sy[i_pos]) < 2][0])
            except IndexError:
                pass

        return self._model.get_powders

    def update_diffraction_view(self, ax, _parent, sub_run, two_d_data):

        if two_d_data:
            ax.imshow(self._model.get_2D_diffraction_counts(sub_run).T)
            ax.axis('off')
        else:
            ax.cla()
            for mask in self._model._hydra_ws.reduction_masks:
                tth, int_vec, error_vec = self._model.get_reduced_diffraction_data(sub_run, mask)
                ax.plot(tth[1:], int_vec[1:], label=mask)

            ax.legend(frameon=False)
            ax.set_xlabel(r"2$\theta$ ($deg.$)")
            ax.set_ylabel("Intensity (ct.)")
            ax.set_ylabel("Diff (ct.)")
        return
