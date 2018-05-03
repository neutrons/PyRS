from mplgraphicsview1d import MplGraphicsView1D


class GeneralDiffDataView(MplGraphicsView1D):
    """
    blabla
    """
    def __init__(self, parent):
        """
        blabla
        :param parent:
        """
        super(GeneralDiffDataView, self).__init__(parent)

        return

    def plot_scatter(self, vec_x, vec_y, x_label, y_label):
        """
        plot figure in scatter-style
        :param vec_x:
        :param vec_y:
        :return:
        """
        # TODO
        self.add_plot(vec_x, vec_y, line_style=None, color='red', x_label=x_label, y_label=y_label)


class PeakFitSetupView(MplGraphicsView1D):
    """
    blabla
    """
    def __init__(self, parent):
        """
        blabla
        :param parent:
        """
        super(PeakFitSetupView, self).__init__(parent)

        return

    def plot_diff_data(self, diff_data_set):
        """

        :param diff_data_set:
        :return:
        """
        # TODO
        vec_x = diff_data_set[0]
        vec_y = diff_data_set[1]

        self.add_plot(vec_x, vec_y)
