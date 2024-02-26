import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm


class TextureFittingCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model
        self._fits = None
        self._fitted_patterns = {}

    def load_projectfile(self, filename):
        self._model.load_hidra_project_file(filename)
        self._fits = None

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
            elif fit_class.fit_result is None:
                data = None
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

    def fit_peaks(self, fit_range_table, peak_function_name, background_function_name,
                  out_of_plane_angle=None):

        def _extract_fitted_data(fit_ws):
            diff_data = np.zeros((self._model.sub_runs.size, len(fit_ws.readX(int(self._model.sub_runs[0])))))
            tth_data = np.zeros_like(diff_data)
            for i_sub in range(self._model.sub_runs.size):
                tth_data[i_sub, :] = fit_ws.readX(int(i_sub))
                diff_data[i_sub, :] = fit_ws.readY(int(i_sub))

            return tth_data, diff_data

        peak_tag = []
        min_tth = []
        max_tth = []

        for peak_row in range(fit_range_table.rowCount()):
            if (fit_range_table.item(peak_row, 0) is not None and
                    fit_range_table.item(peak_row, 1) is not None):

                min_tth.append(float(fit_range_table.item(peak_row, 0).text()))
                max_tth.append(float(fit_range_table.item(peak_row, 1).text()))
                if fit_range_table.item(peak_row, 2) is None:
                    peak_tag.append('peak_{}'.format(peak_row + 1))
                else:
                    peak_tag.append(fit_range_table.item(peak_row, 2).text())

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

    def save_fit_range(self, filename, fit_range_table):
        self._model.to_json(filename, fit_range_table)

    def load_fit_range(self, filename, fit_range_table):
        self._model.from_json(filename, fit_range_table)

    def export_peak_data(self, out_folder):
        for key in self._fits.keys():
            if key == '':
                sep = ''
            else:
                sep = '_'

            self._model.export_fit_csv('{}/HB2B_{}{}{}.csv'.format(out_folder,
                                                                   self._model.runnumber,
                                                                   sep,
                                                                   key),
                                       self._fits[key].peakcollections)

        return

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

    def export_polar_projection(self, output_folder, fit_range_table):

        peak_label_list = []
        peak_id_list = []
        for peak_row in range(fit_range_table.rowCount()):
            if (fit_range_table.item(peak_row, 0) is not None and
                    fit_range_table.item(peak_row, 1) is not None):
                peak_id_list.append(peak_row + 1)
                if fit_range_table.item(peak_row, 2) is None:
                    peak_label_list.append('peak_{}'.format(peak_row + 1))
                else:
                    peak_label_list.append(fit_range_table.item(peak_row, 2).text())

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

    def plot_2D_params(self, ax_object, xlabel, ylabel, peak_number, fit_object, out_of_plane):

        if peak_number == "":
            peak_number = 1

        xdata, ydata = self.get_log_plot(xlabel, ylabel, peak=int(peak_number),
                                         fit_object=fit_object,
                                         out_of_plane=out_of_plane)

        if isinstance(ydata[0], np.ndarray):
            yerr = ydata[1]
            ydata = ydata[0]
        else:
            yerr = np.zeros_like(ydata)

        if isinstance(xdata[0], np.ndarray):
            xerr = xdata[1]
            xdata = xdata[0]
        else:
            xerr = np.zeros_like(xdata)

        ax_object.reset_viewer()
        ax_object.plot_scatter_with_errors(vec_x=xdata,
                                           vec_y=ydata,
                                           vec_x_error=xerr,
                                           vec_y_error=yerr,
                                           x_label=xlabel,
                                           y_label=ylabel)

        return

    def plot_3D_params(self, ax_object, VizSetup, x_label, y_label, z_label, peak_number, fit_object,
                       out_of_plane, include_list):

        def round_polar(vector, target):
            return np.round(vector / target, 0) * target

        if peak_number == "":
            peak_number = 1

        try:
            xdata, ydata, zdata = self.get_log_plot(x_label, y_label, zname=z_label, peak=int(peak_number),
                                                    fit_object=fit_object, out_of_plane=out_of_plane,
                                                    include_list=include_list)
    
            if isinstance(zdata[0], np.ndarray):
                zdata = zdata[0]
    
            if isinstance(xdata, list):
                xdata = np.array(xdata)
    
            if isinstance(ydata, list):
                ydata = np.array(ydata)
    
            if isinstance(zdata, list):
                zdata = np.array(zdata)
    
            plot_scatter = False
            colors = None
    
            if ((ydata.size == np.unique(ydata).size) or
                    (xdata.size == np.unique(xdata).size)):
    
                plot_scatter = True
    
            if (VizSetup.polar_bt.isChecked()):
                polar_data = self.extract_polar_projection(peak_number=int(peak_number))
    
                if polar_data is not None:
    
                    alpha = round_polar(polar_data[:, 0], 5)
                    beta = round_polar(polar_data[:, 1], 5)
    
                    R, P = np.meshgrid(np.unique(alpha), np.unique(beta))
                    vec_z = griddata(((alpha, beta)), polar_data[:, 2], (R, P), method='nearest')
    
                    if VizSetup.shift_bt.isChecked():
                        vec_x = (90 - R) * np.cos(np.deg2rad(P))
                        vec_y = (90 - R) * np.sin(np.deg2rad(P))
                    else:
                        vec_x = R * np.cos(np.deg2rad(P))
                        vec_y = R * np.sin(np.deg2rad(P))
    
                    x_label = r'$\alpha$'
                    y_label = r'$\beta$'
                    z_label = r'Intensity'
    
                    plot_scatter = False
                else:
                    plot_scatter = True
    
            if (VizSetup.contour_bt.isChecked()) and (not plot_scatter):
                vec_x, vec_y = np.meshgrid(np.unique(xdata), np.unique(ydata))
                vec_z = griddata(((xdata, ydata)), zdata, (vec_x, vec_y), method='nearest')
    
            elif (VizSetup.lines_bt.isChecked()) and (not plot_scatter):
                vec_x, vec_y = np.meshgrid(np.unique(xdata), np.unique(ydata))
                vec_z = griddata(((xdata, ydata)), zdata, (vec_x, vec_y), method='nearest')
    
                norm = plt.Normalize(vec_z.min(), vec_z.max())
                colors = coolwarm(norm(vec_z))
    
            elif (VizSetup.scatter_bt.isChecked()) or (plot_scatter):
                plot_scatter = True
    
                norm = plt.Normalize(zdata.min(), zdata.max())
                colors = coolwarm(norm(zdata))
    
                vec_x = np.copy(xdata)
                vec_y = np.copy(ydata)
                vec_z = np.copy(zdata)
    
            ax_object.plot_3D_scatter(vec_x, vec_y, vec_z, plot_scatter, colors=colors,
                                      x_label=x_label, y_label=y_label, z_label=z_label)
        except ValueError:
            ax_object.reset_viewer()

        return

    def update_diffraction_view(self, ax_object, fit_summary, sub_run):

        tth, int_vec = self.get_reduced_diffraction_data(sub_run, fit_summary.out_of_plan_angle)

        ax_object.plot_experiment_data(diff_data_set=[tth, int_vec],
                                       data_reference='Scan {0}'.format(sub_run))

        # plot fitted data
        if fit_summary.fit_table_operator.fit_result is not None:
            sub_run_index = int(np.where(self._model.sub_runs == sub_run)[0])

            fit_data = self.get_fitted_data(sub_run_index,
                                            fit_summary.out_of_plan_angle)

            fit_index = fit_data[1] > 0
            ax_object.plot_fitted_data(fit_data[0][fit_index], fit_data[1][fit_index])
            ax_object.plot_fitting_diff_data(x_axis=fit_data[2][fit_index], y_axis=fit_data[3][fit_index])

        return
