from mplgraphicsview1d import MplGraphicsView1D
from mplgraphicsview2d import MplGraphicsView2D


class Diffraction2DPlot(MplGraphicsView2D):
    """
    General 2D plot view for diffraction data set
    """
    def __init__(self, parent):
        # TODO
        super(Diffraction2DPlot, self).__init__(parent)

        return


class DiffContourView(MplGraphicsView2D):
    def __init__(self, parent):
        super(DiffContourView, self).__init__(parent)


class GeneralDiffDataView(MplGraphicsView1D):
    """
    generalized diffraction view
    """
    def __init__(self, parent):
        """
        blabla
        :param parent:
        """
        super(GeneralDiffDataView, self).__init__(parent)

        # management
        self._line_reference_list = list()
        self._last_line_reference = None

        return

    def plot_scatter(self, vec_x, vec_y, x_label, y_label):
        """
        plot figure in scatter-style
        :param vec_x:
        :param vec_y:
        :return:
        """
        # TODO Future: Need to write use cases.  Now it is for demo
        if self._last_line_reference is not None:
            if x_label != self.get_label_x():
                self.reset_viewer()

        # plot data in a scattering plot
        ref_id = self.add_plot(vec_x, vec_y, line_style=None, color='red', x_label=x_label, y_label=y_label)
        self._line_reference_list.append(ref_id)
        self._last_line_reference = ref_id

        return

    def reset_viewer(self):
        """
        reset current graphics view
        :return:
        """
        # reset all the management data structures
        self._last_line_reference = None
        self._line_reference_list = list()

        # call to clean lines
        self.clear_all_lines(row_number=0, col_number=0)

        return


class PeakFitSetupView(MplGraphicsView1D):
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
        self._diff_reference_list = list()
        self._last_diff_reference = None  # last diffraction (raw) line ID
        self._last_model_reference = None  # last model diffraction (raw) line ID
        self._last_fit_diff_reference = None  # TODO

        #
        self._auto_color = True

        return

    def _next_color(self):
        # TODO Implement ASAP
        return 'blue'

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
        ref_id = self.add_plot(vec_x, vec_y, color=self._next_color(), x_label='$2\\theta (degree)$', marker=None,
                               show_legend=True, y_label=data_reference)

        self._diff_reference_list.append(ref_id)
        self._last_diff_reference = ref_id

        return

    def plot_fit_diff(self, diff_data_set, model_data_set):
        # TODO
        if self._last_fit_diff_reference is not None:
            self.remove_line(row_index=0, col_index=0, line_id=self._last_fit_diff_reference)
            self._last_fit_diff_reference = None

        # calculate
        fit_diff_vec = diff_data_set[1] - model_data_set[1]

        # plot
        self._last_fit_diff_reference = self.add_plot(diff_data_set[0], fit_diff_vec, color='green')

        return

    def plot_model(self, model_data_set):
        """
        plot a model diffraction data
        :param model_data_set:
        :return:
        """
        # check condition
        if len(self._diff_reference_list) > 1:
            # very confusion to plot model
            print ('There are more than 1 raw data plot.  It is very confusing to plot model.'
                   '\n FYI current diffraction data references: {0}'.format(self._diff_reference_list))
            raise RuntimeError('There are more than 1 raw data plot.  It is very confusing to plot model.')

        # remove previous model
        if self._last_model_reference is not None:

            print ('[DB...BAT] About to remove last reference: {0}'.format(self._last_model_reference))
            self.remove_line(row_index=0, col_index=0, line_id=self._last_model_reference)
            self._last_model_reference = None

        # plot
        print ('model X: {0}'.format(model_data_set[0]))
        print ('model Y: {0}'.format(model_data_set[1]))
        self._last_model_reference = self.add_plot(model_data_set[0], model_data_set[1], color='red')

        return

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
        self.clear_all_lines(row_number=0, col_number=0, include_main=True, include_right=False)

        return
