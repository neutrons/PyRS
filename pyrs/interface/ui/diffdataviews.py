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
        # super(, self).__init__(parent)

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
        # TODO - #84 - Make it real!
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

    def __init__(self, parent):
        """ Initialization
        :param parent:
        """
        super(GeneralDiffDataView, self).__init__(parent, 1, 1)

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

    def plot_diffraction(self, vec_x, vec_y, x_label, y_label, line_label=None, keep_prev=True):
        """ plot figure in scatter-style
        :param vec_x:
        :param vec_y:
        :param x_label:
        :param y_label:
        :return:
        """
        # TODO Future: Need to write use cases.  Now it is for demo
        # It is not allowed to plot 2 plot with different x-axis
        if self._last_line_reference is not None:
            if x_label != self.get_label_x():
                self.reset_viewer()

        if not keep_prev and self._last_line_reference is not None:
            self.remove_line(0, 0, self._last_line_reference)

        # plot data in a scattering plot with auto re-scale
        ref_id = self.add_plot(vec_x, vec_y, line_style='-', marker=None,
                               color='red', x_label=x_label, y_label=y_label,
                               label=line_label)
        # TODO - 20181101 - Enable after auto_scale is fixed: self.auto_rescale()

        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id
        self._current_x_axis_name = x_label

    def plot_scatter_with_errors(self, vec_x=None, vec_y=None,
                                 vec_x_error=None, vec_y_error=None,
                                 x_label="", y_label=""):

        # # TODO Future: Need to write use cases.  Now it is for demo
        # # It is not allowed to plot 2 plot with different x-axis
        # if self._last_line_reference is not None:
        #     if x_label != self.get_label_x():
        #         self.reset_viewer()

        # plot data in a scattering plot with auto re-scale
        ref_id = self.add_plot(vec_x,
                               vec_y,
                               x_err=vec_x_error,
                               y_err=vec_y_error,
                               line_style='',
                               marker='*',
                               markersize=6,
                               color='red',
                               x_label=x_label,
                               y_label=y_label)

        # TODO - 20181101 - Enable after auto_scale is fixed: self.auto_rescale()
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
        # TODO Future: Need to write use cases.  Now it is for demo
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
                               color='red',
                               x_label=x_label,
                               y_label=y_label)

        # TODO - 20181101 - Enable after auto_scale is fixed: self.auto_rescale()
        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id
        self._current_x_axis_name = x_label

    def reset_viewer(self):
        """
        reset current graphics view
        """
        # reset all the management data structures
        self._last_line_reference = None
        self._line_reference_list = list()

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
        self._last_diff_reference = None  # last diffraction (raw) line ID
        self._last_model_reference = None  # last model diffraction (raw) line ID
        self._last_fit_diff_reference = None  # Last plot's reference for fitting residual
        self._auto_color = True

    def plot_experiment_data(self, diff_data_set, data_reference):
        """
        plot a diffraction data
        :param diff_data_set: 2-tuple for vector X and vector Y
        :param data_reference: reference name for the data to plot for presentation purpose
        :return:
        """
        # parse the data
        vec_x = diff_data_set[0]
        vec_y = diff_data_set[1]

        ref_id = self.plot_data(data_set=(vec_x, vec_y), line_label=data_reference)

        self._diff_reference_list.append(ref_id)
        self._last_diff_reference = ref_id

    def plot_fitted_data(self, x_array, y_array):
        self.plot_data(data_set=(x_array, y_array),
                       line_label='-',
                       color='black')

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

        # plot data
        # ref_id = self.add_plot(vec_x, vec_y, color=self._next_color(), x_label='$2\\theta (degree)$', marker=None,
        #                        show_legend=True, y_label=data_reference)

        ref_id = self.plot_data(data_set=(vec_x, vec_y), color=self._get_next_color(), line_label=data_reference)

        self._diff_reference_list.append(ref_id)
        self._last_diff_reference = ref_id

    def plot_fitting_diff_data(self, x_axis, y_axis):
        self._last_fit_diff_reference = self._myCanvas.add_plot_lower_axis((x_axis, y_axis))

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
            self.remove_line(row_index=0, col_index=0, line_id=self._last_fit_diff_reference)
            self._last_fit_diff_reference = None

        # calculate
        fit_diff_vec = diff_data_set[1] - model_data_set[1]

        # plot
        self._last_fit_diff_reference = self._myCanvas.add_plot_lower_axis(fit_diff_vec)

    def plot_model(self, model_data_set):
        """
        plot a model diffraction data
        :param model_data_set:
        :return:
        """
        # check condition
        if len(self._diff_reference_list) > 1:
            # very confusion to plot model
            print('There are more than 1 raw data plot.  It is very confusing to plot model.'
                  '\n FYI current diffraction data references: {0}'.format(self._diff_reference_list))
            raise RuntimeError('There are more than 1 raw data plot.  It is very confusing to plot model.')

        # remove previous model
        if self._last_model_reference is not None:
            print('[DB...BAT] About to remove last reference: {0}'.format(self._last_model_reference))
            self.remove_line(row_index=0, col_index=0, line_id=self._last_model_reference)
            self._last_model_reference = None

        # plot
        # TODO - TONIGHT - Merge this with FitPeak UI's Figure Canvas
        # self._last_model_reference = self.add_plot(model_data_set[0], model_data_set[1], color='red')

    def reset_viewer(self):
        """
        reset current graphics view
        :return:
        """
        # reset all the management data structures
        self._last_model_reference = None
        self._last_diff_reference = None
        self._last_fit_diff_reference = None
        self._diff_reference_list = list()

        # call to clean lines
        self.clear_canvas()
