class TextureFittingCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model

    def load_projectfile(self, filename):
        self._model.load_hidra_project_file(filename)

    def get_log_plot(self, xname, yname, peak=1, zname=None, fit_object=None):
        def extract_data(name, fit_class, peak):
            param_entry = False

            if fit_class.fit_result is not None:
                if name in fit_class.clean_param_names:
                    param_entry = True

            if str(name) == str("sub-runs"):
                data = self._model.sub_runs
            elif param_entry:
                if str(name) == 'microstrain':
                    data = fit_class.fit_result.peakcollections[peak - 1].get_strain(units='microstrain')
                elif str(name) == 'd spacing':
                    data = fit_class.fit_result.peakcollections[peak - 1].get_dspacing_center()
                elif str(name) == 'chisq':
                    data = fit_class.fit_result.peakcollections[peak - 1].get_chisq()
                else:
                    values, error = fit_class.fit_result.peakcollections[peak - 1].get_effective_params()
                    data = [values[str(name)], error[str(name)]]
            else:
                data = self._model.ws.get_sample_log_values(name)

            return data

        xdata = extract_data(xname, fit_object, peak)
        ydata = extract_data(yname, fit_object, peak)

        if zname is None:
            return xdata, ydata
        else:
            zdata = extract_data(zname, fit_object, peak)
            return xdata, ydata, zdata

    def fit_peaks(self, min_tth, max_tth, peak_tag, peak_function_name, background_function_name,
                  out_of_plane_angle=False):

        fit_results = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                 background_function_name, out_of_plane_angle=out_of_plane_angle)

        return fit_results

    def save(self, filename):
        self._model.to_json(filename)

    def load(self, filename):
        self._model.from_json(filename)
