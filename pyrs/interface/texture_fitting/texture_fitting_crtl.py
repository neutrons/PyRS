import numpy as np


class TextureFittingCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model
        self._fits = None
        self._fitted_patterns = {}

    def load_projectfile(self, filename):
        self._model.load_hidra_project_file(filename)

    def get_fitted_data(self, sub_run, mask_id):

        fit_tth = self._fitted_patterns[mask_id][0][0][sub_run, :]
        fit_int = self._fitted_patterns[mask_id][0][1][sub_run, :]
        diff_tth = self._fitted_patterns[mask_id][1][0][sub_run, :]
        diff_int = self._fitted_patterns[mask_id][1][1][sub_run, :]

        return fit_tth, fit_int, diff_tth, diff_int

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
                  out_of_plane_angle=None):

        def _extract_fitted_data(fit_ws):
            diff_data = np.zeros((self._model.sub_runs.size, len(fit_ws.readX(int(self._model.sub_runs[0])))))
            tth_data = np.zeros_like(diff_data)
            for i_sub in range(self._model.sub_runs.size):
                tth_data[i_sub, :] = fit_ws.readX(int(i_sub))
                diff_data[i_sub, :] = fit_ws.readY(int(i_sub))

            return tth_data, diff_data

        fit_results = {}

        if len(self._model.ws.reduction_masks) == 2:
            fit_results[''] = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                         background_function_name, out_of_plane_angle=None)

            self._fitted_patterns[''] = [_extract_fitted_data(fit_results[''].fitted),
                                         _extract_fitted_data(fit_results[''].difference)]

        else:
            for mask_key in self._model.ws.reduction_masks:
                if '_var' not in mask_key:
                    fit_results[mask_key] = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                                       background_function_name,
                                                                       out_of_plane_angle=mask_key)

                    self._fitted_patterns[mask_key] = [_extract_fitted_data(fit_results[mask_key].fitted),
                                                       _extract_fitted_data(fit_results[mask_key].difference)]

        self._fits = fit_results

        return self._fits

    def get_reduced_diffraction_data(self, sub_run, mask_id=None):

        if mask_id not in self._model.ws.reduction_masks:
            mask_id = self._model.ws.reduction_masks[0]

        x, y, err = self._model.ws.get_reduced_diffraction_data(sub_run, mask_id=mask_id)

        return x, y

    def saveas(self, filename, fit_result):
        self._model.save_fit_result(filename, fit_result)

    def load(self, filename):
        self._model.from_json(filename)

    def save(self, filename, fit_result):
        self._model.save_fit_result(filename, fit_result)

    def save_fit_range(self, filename):
        self._model.to_json(filename)

    def load_fit_range(self, filename):
        self._model.to_json(filename)

    def export_peak_data(self, filename, fit_collection):
        pass
