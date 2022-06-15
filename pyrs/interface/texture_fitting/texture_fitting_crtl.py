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
                  out_of_plane_angle=None):

        if len(self._model.ws.reduction_masks) == 2:
            fit_results[''] = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                         background_function_name, out_of_plane_angle=None)
        else:
            for mask_key in self._model.ws.reduction_masks:
                if '_var' not in mask_key:
                    fit_results[mask_key] = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                                       background_function_name,
                                                                       out_of_plane_angle=mask_key)

        return fit_results

    def get_reduced_diffraction_data(self, sub_run, mask_id=None):

        if mask_id not in self._model.ws.reduction_masks:
            mask_id = self._model.ws.reduction_masks[0]

        x, y, err = self._model.ws.get_reduced_diffraction_data(sub_run, mask_id=mask_id)

        return x, y

    def save(self, filename):
        self._model.to_json(filename)

    def load(self, filename):
        self._model.from_json(filename)
