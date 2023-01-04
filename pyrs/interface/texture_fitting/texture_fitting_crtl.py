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

    def trim_data(self, xdata, ydata, zdata=None, include_list=[]):

        if len(include_list) > 1:
            keep_points = np.array([False for i in range(self._model.sub_runs.size)])
            keep_points[np.array(include_list)] = True
        else:
            keep_points = np.array([True for i in range(self._model.sub_runs.size)])

        def get_points_to_keep(data):
            if type(data[0]) is np.ndarray:
                keep_points = data[1] != 0
            else:
                keep_points = np.ones_like(data) == 1

            return keep_points

        def remove_points(data, keep_points):
            if type(data) is np.ndarray:
                data = data[keep_points]
            else:
                data = [data[0][keep_points],
                        data[1][keep_points]]
            return data

        keep_points *= get_points_to_keep(xdata)
        keep_points *= get_points_to_keep(ydata)

        if zdata is not None:
            keep_points *= get_points_to_keep(zdata)
            zdata = remove_points(zdata, keep_points)

        xdata = remove_points(xdata, keep_points)
        ydata = remove_points(ydata, keep_points)

        if zdata is not None:
            return xdata, ydata, zdata
        else:
            return xdata, ydata

    def get_log_plot(self, xname, yname, peak=1, zname=None, fit_object=None,
                     out_of_plane=None, include_list=[]):

        def extract_data(name, fit_class, peak):
            param_entry = True

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
                    try:
                        values, error = fit_class.fit_result.peakcollections[peak - 1].get_effective_params()
                        data = [values[str(name)], error[str(name)]]
                    except (ValueError, AttributeError):
                        data = self._model.ws.get_sample_log_values(name)
            else:
                data = None

            return data

        def parse_split_list(sub_run_list):

            if (sub_run_list == "") or (sub_run_list == []):
                subrun_list = []
            else:
                subrun_list = []
                for entry in sub_run_list.split(','):
                    if '-' in entry:
                        start, stop = entry.split('-')
                        subrun_list.extend(range(int(start), int(stop) + 1))
                    elif entry == '':
                        pass
                    else:
                        subrun_list.append(int(entry))

            return subrun_list

        xdata = extract_data(xname, fit_object, peak)
        ydata = extract_data(yname, fit_object, peak)

        if zname is None:
            return self.trim_data(xdata, ydata, include_list=parse_split_list(include_list))
        else:
            zdata = extract_data(zname, fit_object, peak)
            return self.trim_data(xdata, ydata, zdata, include_list=parse_split_list(include_list))

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

            if self.texture_run():
                self.parse_texture_fits(fit_results[''], 'eta_0.0', len(min_tth))

        else:
            for mask_key in self._model.ws.reduction_masks:
                if '_var' not in mask_key:
                    fit_results[mask_key] = self._model.fit_diff_peaks(min_tth, max_tth, peak_tag, peak_function_name,
                                                                       background_function_name,
                                                                       out_of_plane_angle=mask_key)

                    self._fitted_patterns[mask_key] = [_extract_fitted_data(fit_results[mask_key].fitted),
                                                       _extract_fitted_data(fit_results[mask_key].difference)]

                    self.parse_texture_fits(fit_results[mask_key], mask_key, len(min_tth))

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

    def parse_texture_fits(self, fit_obj, eta_mask, num_peaks):

        for i_peak in range(num_peaks):
            # peak params 'Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'
            sub_runs = self._model.sub_runs
            peak_fits, fit_errors = fit_obj.peakcollections[i_peak].get_effective_params()

            sub_runs, peak_center, peak_intensity = self.trim_data(sub_runs,
                                                                   [peak_fits['Center'], fit_errors['Center']],
                                                                   [peak_fits['Intensity'], fit_errors['Intensity']])

            self._model.load_pole_data(i_peak + 1, peak_intensity[0], float(eta_mask.split('_')[1]),
                                       peak_center[0], np.array(sub_runs))

        return

    def extract_polar_projection(self, peak_number):
        if self._model._polefigureinterface is not None:
            return self._model._polefigureinterface.get_polefigure_array(peak_id=int(peak_number))

        else:
            return None

    def export_polar_projection(self, output_folder, peak_id_list, peak_label_list):

        if self._model._polefigureinterface is not None:
            self._model._polefigureinterface.calculate_pole_figure()
            self._model._polefigureinterface.export_pole_figure(output_folder=output_folder,
                                                                peak_id_list=peak_id_list,
                                                                peak_name_list=peak_label_list,
                                                                run_number=self._model.runnumber)

    def texture_run(self):
        phi = np.unique(self._model.ws.get_sample_log_values("phi"))
        chi = np.unique(self._model.ws.get_sample_log_values("chi"))

        return not ((phi.size == 1) and (chi.size == 1))
