from .mplgraphicsview1d import MplGraphicsView1D
from .mplgraphicsview2d import MplGraphicsView2D
import numpy as np
from .mplfitplottingwidget import MplFitPlottingWidget


class DetectorView(MplGraphicsView2D):
    """
    Detector view
    """

    def __init__(self, parent):
        """
        init
        :param parent:
        """
        MplGraphicsView2D.__init__(self, parent)

        return

    def plot_counts(self, count_vec):

        linear_size = np.sqrt(count_vec.shape[0])

        image_array = count_vec.reshape((linear_size, linear_size))

        self.add_2d_plot(image_array, 0, linear_size, 0, linear_size, y_tick_label=None, plot_type='image')

        return

    def plot_detector_view(self, detector_counts, info_tuple):
        """Plot detector counts on 2D view

        Parameters
        ----------
        detector_counts: ndarray (N, )
            counts in 1D
        info_tuple

        Returns
        -------

        """
        sub_run_number, mask_id = info_tuple
        title = 'Sub run {}, Mask: {}'.format(sub_run_number, mask_id)
        self.set_title(title)

        # Resize detector count array
        if len(detector_counts.shape) == 1:
            # 1D array: reshape to 2D
            image_size = int(np.sqrt(detector_counts.shape[0]))
            if image_size * image_size != detector_counts.shape[0]:
                raise RuntimeError('Detector with {} counts cannot convert to a 2D view without further information'
                                   ''.format(detector_counts.shape))
            counts2d = detector_counts.reshape(image_size, image_size)
        else:
            # As Is
            counts2d = detector_counts
            image_size = counts2d.shape[0]

        self.add_2d_plot(np.rot90(counts2d), 0, image_size, 0, image_size)

        return


class GeneralDiffDataView(MplGraphicsView1D):
    """
    generalized diffraction view
    """

    def __init__(self, parent, three_d_fig=False):
        """ Initialization
        :param parent:
        """
        super(GeneralDiffDataView, self).__init__(parent, 1, 1,
                                                  three_d_fig=three_d_fig)

        # management
        self._line_reference_list = list()
        self._last_line_reference = None
        self._current_x_axis_name = None

    @property
    def current_x_name(self):
        """
        :return:
        """
        return self._current_x_axis_name

    def set_no_null_plot(self):
        self.set_ax_null()

    def set_3Dview(self):
        self.reset_view_3d()

    def plot_diffraction(self, vec_x, vec_y, x_label, y_label, color='red', line_style='-',
                         line_label=None, keep_prev=True):
        """ plot figure in scatter-style
        :param vec_x:
        :param vec_y:
        :param x_label:
        :param y_label:
        :return:
        """

        if not keep_prev:
            self.clear_all_lines()

        # plot data in a scattering plot with auto re-scale
        self.add_plot(vec_x, vec_y, line_style=line_style, marker=None,
                      color=color, x_label=x_label, y_label=y_label,
                      label=line_label)

    def plot_scatter_with_errors(self, vec_x=None, vec_y=None,
                                 vec_x_error=None, vec_y_error=None,
                                 x_label="", y_label=""):

        # plot data in a scattering plot with auto re-scale
        ref_id = self.add_plot(vec_x,
                               vec_y,
                               x_err=vec_x_error,
                               y_err=vec_y_error,
                               line_style='',
                               marker='*',
                               markersize=6,
                               color='black',
                               x_label=x_label,
                               y_label=y_label)

        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id
        self._current_x_axis_name = x_label

    def plot_scatter(self, vec_x, vec_y, x_label, y_label):
        """ plot figure in scatter-style
        :param vec_x:
        :param vec_y:
        :param x_label:
        :param y_label:
        :return:
        """

        # It is not allowed to plot 2 plot with different x-axis
        if self._last_line_reference is not None:
            if x_label != self.get_label_x():
                self.reset_viewer()

        # plot data in a scattering plot with auto re-scale
        ref_id = self.add_plot(vec_x,
                               vec_y,
                               line_style='',
                               marker='*',
                               markersize=6,
                               color='black',
                               x_label=x_label,
                               y_label=y_label)

        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id
        self._current_x_axis_name = x_label

    def plot_3D_scatter(self, vec_x, vec_y, vec_z, plot_scatter, colors=None,
                        x_label='', y_label='', z_label=''):

        # It is not allowed to plot 2 plot with different x-axis
        if self._last_line_reference is not None:
            self.reset_viewer()

        # plot data in a scattering plot with auto re-scale
        ref_id = self.add_3d_scatter(vec_x, vec_y, vec_z, plot_scatter, colors=colors,
                                     x_label=x_label, y_label=y_label, z_label=z_label)

        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id
        self._current_x_axis_name = x_label

    def reset_viewer(self):
        """
        reset current graphics view
        """

        # call to clean lines
        self.clear_all_lines()


class PeakFitSetupView(MplFitPlottingWidget):
    """
    Matplotlib graphics view to set up peak fitting
    """

    def __init__(self, parent):
        """
        Graphics view for peak fitting setup
        :param parent:
        """
        super(PeakFitSetupView, self).__init__(parent)

        # management
        self.parent = parent
        self._diff_reference_list = list()
        self._auto_color = True

    def plot_experiment_data(self, diff_data_set, data_reference):
        """
        plot a diffraction data
        :param diff_data_set: 2-tuple for vector X and vector Y
        :param data_reference: reference name for the data to plot for presentation purpose
        :return:
        """

        self.reset_viewer()

        # parse the data
        vec_x = diff_data_set[0][1:]
        vec_y = diff_data_set[1][1:]

        ref_id = self.plot_data(data_set=(vec_x, vec_y), line_label=data_reference, color='black')
        self.plot_data_fitting_ranges()
        # self._diff_reference_list.append(ref_id)
        self._last_diff_reference = ref_id

    def plot_fitted_data(self, x_array, y_array):
        self.plot_data(data_set=(x_array, y_array),
                       line_label='-',
                       color='black', peak_ranges=self.list_peak_ranges)

    def plot_model_data(self, diff_data_set, model_label, residual_set):
        """Plot model data from fitting

        Parameters
        ----------
        diff_data_set
        model_label
        residual_set

        Returns
        -------
        None
        """
        vec_x = diff_data_set[0]
        vec_y = diff_data_set[1]

        ref_id = self.plot_data(data_set=(vec_x, vec_y), color='red', line_label='model')
        self._last_model_reference = ref_id

        if residual_set is not None:
            # vec_x = residual_set[0]
            # diff_y = residual_set[1]
            self._myCanvas.add_plot_lower_axis(residual_set)

    def plot_diff_data(self, diff_data_set, data_reference):
        """
        plot a diffraction data
        :param diff_data_set:
        :param data_reference: reference name for the data to plot
        :return:
        """

        # parse the data
        vec_x = diff_data_set[0]
        vec_y = diff_data_set[1]

        self.plot_data(data_set=(vec_x, vec_y), color=self._get_next_color(), line_label=data_reference)

    def plot_fitting_diff_data(self, x_axis, y_axis):
        self._myCanvas.add_plot_lower_axis((x_axis, y_axis), self.list_peak_ranges)

    def plot_fit_diff(self, diff_data_set, model_data_set):
        """
        plot the difference between fitted diffraction data (model) and experimental data
        :param diff_data_set:
        :param model_data_set:
        :return:
        """
        # check input
        assert isinstance(diff_data_set, tuple) and len(diff_data_set) >= 2, 'Diffraction data set {} ' \
                                                                             'must be a 2-tuple but not a {}' \
                                                                             ''.format(diff_data_set,
                                                                                       type(diff_data_set))

        assert isinstance(model_data_set, tuple) and len(model_data_set) >= 2, \
            'Model data set {} must be a 2-tuple but not a {}'.format(model_data_set, type(model_data_set))

        # remove previous difference curve
        if self._last_fit_diff_reference is not None:
            self.remove_line(line_id=self._last_fit_diff_reference)
            self._last_fit_diff_reference = None

        # calculate
        fit_diff_vec = diff_data_set[1] - model_data_set[1]

        # plot
        self._myCanvas.add_plot_lower_axis(fit_diff_vec)

    def reset_viewer(self):
        """
        reset current graphics view
        :return:
        """

        # call to clean lines
        self.clear_canvas()
