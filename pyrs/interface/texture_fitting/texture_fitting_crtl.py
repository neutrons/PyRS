class TextureFittingCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model

    def load_projectfile(self, filename):
        self._model.load_hidra_project_file(filename)

    def get_log_plot(self, xname, yname):
        if xname == "sub-runs":
            xdata = self._model.sub_runs
        else:
            xdata = self._model.ws.get_sample_log_values(xname)

        if yname == "sub-runs":
            ydata = self._model.sub_runs
        else:
            ydata = self._model.ws.get_sample_log_values(yname)

        return xdata, ydata

    def fit_peaks(self, min_tth, max_tth, peak_tag, peak_function_name, background_function_name,
                  out_of_plane_angle=False):

        fit_results = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                 background_function_name, out_of_plane_angle=out_of_plane_angle)

        return fit_results

    def save(self, filename):
        self._model.to_json(filename)

    def load(self, filename):
        self._model.from_json(filename)
