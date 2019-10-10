import time
import matplotlib.image
from matplotlib.figure import Figure
import numpy as np
from qtpy.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas


class Mpl2DGraph(QWidget):
    """ A combined graphics view including matplotlib canvas and
    a navigation tool bar

    Note: Merged with HFIR_Powder_Reduction.MplFigureCAnvas
    """

    def __init__(self, parent):
        """ Initialization
        """
        # Initialize parent
        super(Mpl2DGraph, self).__init__(parent)

        # set up canvas
        self._myCanvas = Qt4Mpl2DCanvas(self)

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)

        return

    @property
    def canvas(self):
        """
        access to canvas
        :return:
        """
        return self._myCanvas

    def move_indicator(self, line_id, dx, dy):
        """
        Move the indicator line in horizontal
        :param line_id:
        :param dx:
        :return:
        """
        # Shift value
        self._myIndicatorsManager.shift(line_id, dx=dx, dy=dy)

        # apply to plot on canvas
        if self._myIndicatorsManager.get_line_type(line_id) < 2:
            # horizontal or vertical
            canvas_line_index = self._myIndicatorsManager.get_canvas_line_index(line_id)
            vec_x, vec_y = self._myIndicatorsManager.get_data(line_id)
            self._myCanvas.updateLine(ikey=canvas_line_index, vecx=vec_x, vecy=vec_y)
        else:
            # 2-way
            canvas_line_index_h, canvas_line_index_v = self._myIndicatorsManager.get_canvas_line_index(line_id)
            h_vec_set, v_vec_set = self._myIndicatorsManager.get_2way_data(line_id)

            self._myCanvas.updateLine(ikey=canvas_line_index_h, vecx=h_vec_set[0], vecy=h_vec_set[1])
            self._myCanvas.updateLine(ikey=canvas_line_index_v, vecx=v_vec_set[0], vecy=v_vec_set[1])

        return

    def remove_indicator(self, indicator_key):
        """ Remove indicator line
        :param indicator_key:
        :return:
        """
        #
        plot_id = self._myIndicatorsManager.get_canvas_line_index(indicator_key)
        self._myCanvas.remove_plot_1d(plot_id)
        self._myIndicatorsManager.delete(indicator_key)

        return

    def set_indicator_position(self, line_id, pos_x, pos_y):
        """ Set the indicator to new position
        :param line_id: indicator ID
        :param pos_x:
        :param pos_y:
        :return:
        """
        # Set value
        self._myIndicatorsManager.set_position(line_id, pos_x, pos_y)

        # apply to plot on canvas
        if self._myIndicatorsManager.get_line_type(line_id) < 2:
            # horizontal or vertical
            canvas_line_index = self._myIndicatorsManager.get_canvas_line_index(line_id)
            vec_x, vec_y = self._myIndicatorsManager.get_data(line_id)
            self._myCanvas.updateLine(ikey=canvas_line_index, vecx=vec_x, vecy=vec_y)
        else:
            # 2-way
            canvas_line_index_h, canvas_line_index_v = self._myIndicatorsManager.get_canvas_line_index(line_id)
            h_vec_set, v_vec_set = self._myIndicatorsManager.get_2way_data(line_id)

            self._myCanvas.updateLine(ikey=canvas_line_index_h, vecx=h_vec_set[0], vecy=h_vec_set[1])
            self._myCanvas.updateLine(ikey=canvas_line_index_v, vecx=v_vec_set[0], vecy=v_vec_set[1])

        return

    def update_indicator(self, i_key, color):
        """
        Update indicator with new color
        :param i_key:
        :param vec_x:
        :param vec_y:
        :param color:
        :return:
        """
        if self._myIndicatorsManager.get_line_type(i_key) < 2:
            # horizontal or vertical
            canvas_line_index = self._myIndicatorsManager.get_canvas_line_index(i_key)
            self._myCanvas.updateLine(ikey=canvas_line_index, vecx=None, vecy=None, linecolor=color)
        else:
            # 2-way
            canvas_line_index_h, canvas_line_index_v = self._myIndicatorsManager.get_canvas_line_index(i_key)
            # h_vec_set, v_vec_set = self._myIndicatorsManager.get_2way_data(i_key)

            self._myCanvas.updateLine(ikey=canvas_line_index_h, vecx=None, vecy=None, linecolor=color)
            self._myCanvas.updateLine(ikey=canvas_line_index_v, vecx=None, vecy=None, linecolor=color)

        return


class Qt4Mpl2DCanvas(FigureCanvas):
    """  A customized Qt widget for matplotlib figure.
    It can be used to replace GraphicsView of QtGui
    """

    def __init__(self, parent):
        """  Initialization
        """
        # Instantiating matplotlib Figure
        self.fig = Figure()
        self.fig.patch.set_facecolor('white')

        # initialization
        super(Qt4Mpl2DCanvas, self).__init__(self.fig)

        # set up axis/subplot (111) only for 2D
        self.axes = self.fig.add_subplot(111, polar=False)  # return: matplotlib.axes.AxesSubplot
        self.axes.set_aspect('auto')

        # plot management
        self._scatterPlot = None
        self._imagePlot = None

        # Initialize parent class and set parent
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Variables to manage all lines/subplot
        self._lineDict = {}
        self._lineIndex = 0

        # legend and color bar
        self._colorBar = None
        self._isLegendOn = False
        self._legendFontSize = 8

        return

    def add_contour_plot(self, vec_x, vec_y, matrix_z):
        """ add a contour plot
        Example: reduced data: vec_x: d-values, vec_y: run numbers, matrix_z, matrix for intensities
        :param vec_x: a list of a vector for X axis
        :param vec_y: a list of a vector for Y axis
        :param matrix_z:
        :return:
        """
        # check input
        assert isinstance(vec_x, list) or isinstance(vec_x, np.ndarray),\
            'Vector X {} must be either a list or a ndarray but not a {}'.format(vec_x, type(vec_x))
        assert isinstance(vec_y, list) or isinstance(vec_y, np.ndarray),\
            'Vector Y {} must be either a list or a ndarray but not a {}'.format(vec_y, type(vec_y))
        assert isinstance(matrix_z, np.ndarray),\
            'Vector Z {} must be a ndarray but not a {}'.format(matrix_z, type(matrix_z))

        # create mesh grid
        grid_x, grid_y = np.meshgrid(vec_x, vec_y)
        #
        # print '[DB...BAT] Grid X and Grid Y size: ', grid_x.shape, grid_y.shape

        # check size
        matrix_z = matrix_z.transpose()

        assert grid_x.shape == matrix_z.shape, 'Size of X ({}) and Y ({}) (X-Y mesh size {}) must match size of ' \
                                               'matrix Z ({}).'.format(vec_x.shape, vec_y.shape,
                                                                       grid_x.shape, matrix_z.shape)
        # # Release the current image
        # self.axes.hold(False)

        # Do plot: resolution on Z axis (color bar is set to 100)
        time_s = time.time()
        print('[DB...BAT] Contour Plot Starting time (float) = {}'.format(time_s))

        self.axes.clear()
        if False:
            contour_plot = self.axes.contourf(grid_x, grid_y, matrix_z, 100)
        else:
            contour_plot = self.axes.contourf(vec_x, vec_y, matrix_z, 50, cmap="RdBu_r")

        time_f = time.time()
        print('[DB...BAT] Stopping time (float) = {}.  Used {} second for plotting'.format(time_f,
                                                                                           time_f - time_s))

        labels = [item.get_text() for item in self.axes.get_yticklabels()]
        print '[DB...BAT] Number of Y labels = ', len(labels), ', Number of Y = ', len(vec_y)

        # TODO/ISSUE/NOW: how to make this part more flexible
        if len(labels) == 2*len(vec_y) - 1:
            new_labels = [''] * len(labels)
            for i in range(len(vec_y)):
                new_labels[i*2] = '%d' % int(vec_y[i])
            self.axes.set_yticklabels(new_labels)
        # END-IF

        # explicitly set aspect ratio of the image
        self.axes.set_aspect('equal')

        # Set color bar.  plt.colorbar() does not work!
        if self._colorBar is None:
            # set color map type
            contour_plot.set_cmap('spectral')
            self._colorBar = self.fig.colorbar(contour_plot)
        else:
            self._colorBar.update_bruteforce(contour_plot)

        # apply the change
        self._flush()

        return contour_plot

    def add_scatter(self, x, y):
        """
        Add scatter plot to canvas
        :param x:
        :param y:
        :return:
        """
        # TODO - 20180824 - shall be an option
        self.axes.plot(x, y, 'ko', ms=3)

        self._flush()

    def add_image_plot(self, array2d, xmin, xmax, ymin, ymax, yticklabels=None):
        """

        @param array2d:
        @param xmin:
        @param xmax:
        @param ymin:
        @param ymax:
        @param holdprev:
        @param yticklabels: list of string for y ticks
        @return:
        """
        # check
        assert isinstance(array2d, np.ndarray), '2D array {} must be a ndarray but not a {}' \
                                                ''.format(array2d, type(array2d))
        assert len(array2d.shape) == 2, 'Image {} must be given as a 2D array but not of shape {}' \
                                        ''.format(array2d, array2d.shape)

        # show image
        self._imagePlot = self.axes.imshow(array2d, extent=[xmin, xmax, ymin, ymax], interpolation='none')

        print(self._imagePlot, type(self._imagePlot))

        # set y ticks as an option:
        if yticklabels is not None:
            # it will always label the first N ticks even image is zoomed in
            print("[FIXME]: The way to set up the Y-axis ticks is wrong!")
            self.axes.set_yticklabels(yticklabels)

        # explicitly set aspect ratio of the image
        self.axes.set_aspect('auto')

        # set up color bar
        # # Set color bar.  plt.colorbar() does not work!
        # if self._colorBar is None:
        #     # set color map type
        #     imgplot.set_cmap('spectral')
        #     self._colorBar = self.fig.colorbar(imgplot)
        # else:
        #     self._colorBar.update_bruteforce(imgplot)

        # Flush...
        self._flush()

        return

    def add_image_file(self, imagefilename):
        """Add an image by file
        """
        # set aspect to auto mode
        self.axes.set_aspect('auto')

        img = matplotlib.image.imread(str(imagefilename))
        # lum_img = img[:,:,0]
        # FUTURE : refactor for image size, interpolation and origin
        imgplot = self.axes.imshow(img, extent=[0, 1000, 800, 0], interpolation='none', origin='lower')

        # Set color bar.  plt.colorbar() does not work!
        if self._colorBar is None:
            # set color map type
            imgplot.set_cmap('spectral')
            self._colorBar = self.fig.colorbar(imgplot)
        else:
            self._colorBar.update_bruteforce(imgplot)

        self._flush()

        return

    def add_scatter_plot(self, array2d):
        """
        add scatter plot
        @param array2d:
        @return:
        """
        # TODO FIXME - add_scatter()...  This method is not used! ...
        # check!
        # TODO - 20180801 - Make it work
        assert isinstance(array2d, np.ndarray), ''
        if array2d.shape[1] < 3:
            raise RuntimeError('blabla3')

        if False:
            array2d = np.ndarray(shape=(100, 3), dtype='float')
            array2d[0][0] = 0
            array2d[0][1] = 0
            array2d[0][2] = 1

            import random
            for index in range(1, 98):
                x = random.randint(1, 255)
                y = random.randint(1, 255)
                z = random.randint(1, 20000)
                array2d[index][0] = float(x)
                array2d[index][1] = float(y)
                array2d[index][2] = float(z)

            array2d[99][0] = 255
            array2d[99][1] = 255
            array2d[99][2] = 1

        self._scatterPlot = self.axes.scatter(array2d[:, 0], array2d[:, 1], s=80, c=array2d[:, 2],
                                              marker='s')

        return

    def clear_canvas(self):
        """ Clear data including lines and image from canvas
        """
        # clear the image for next operation
        # self.axes.hold(False)

        # clear image
        self.axes.cla()
        # Try to clear the color bar
        if len(self.fig.axes) > 1:
            self.fig.delaxes(self.fig.axes[1])
            self._colorBar = None
            # This clears the space claimed by color bar but destroys sub_plot too.
            self.fig.clear()
            # Re-create subplot
            self.axes = self.fig.add_subplot(111)
            self.fig.subplots_adjust(bottom=0.15)

        # flush/commit
        self._flush()

        return

    def decrease_legend_font_size(self):
        """
        reset the legend with the new font size
        Returns:

        """
        # minimum legend font size is 2! return if it already uses the smallest font size.
        if self._legendFontSize <= 2:
            return

        self._legendFontSize -= 1
        self._setup_legend(font_size=self._legendFontSize)

        self.draw()

        return

    def getLastPlotIndexKey(self):
        """ Get the index/key of the last added line
        """
        return self._lineIndex-1

    def getPlot(self):
        """ reture figure's axes to expose the matplotlib figure to PyQt client
        """
        return self.axes

    def getXLimit(self):
        """ Get limit of Y-axis
        """
        return self.axes.get_xlim()

    def getYLimit(self):
        """ Get limit of Y-axis
        """
        return self.axes.get_ylim()

    def has_plot(self, plot_type):
        if plot_type == 'image' and self._imagePlot is not None:
            return True

        return False

    @property
    def is_legend_on(self):
        """
        check whether the legend is shown or hide
        Returns:
        boolean
        """
        return self._isLegendOn

    def setXYLimit(self, xmin, xmax, ymin, ymax):
        """
        """
        # for X
        xlims = self.axes.get_xlim()
        xlims = list(xlims)
        if xmin is not None:
            xlims[0] = xmin
        if xmax is not None:
            xlims[1] = xmax
        self.axes.set_xlim(xlims)

        # for Y
        ylims = self.axes.get_ylim()
        ylims = list(ylims)
        if ymin is not None:
            ylims[0] = ymin
        if ymax is not None:
            ylims[1] = ymax
        self.axes.set_ylim(ylims)

        # try draw
        self.draw()

        return

    def set_title(self, title, color):
        """
        set the tile to an axis
        :param title:
        :param color
        :return:
        """
        # check input
        assert isinstance(title, str), 'Title must be a string but not a {0}.'.format(type(title))
        assert isinstance(color, str), 'Color must be a string but not a {0}.'.format(type(color))

        print '[DB...BAT] Set {0} in color {1} as the figure\'s title.'.format(title, color)
        self.setWindowTitle(title)
        self.fig.set_label(title)

        self.draw()

        return

    def set_xlabel(self, x_label):
        self.axes.set_xlabel(x_label)

    def update_image(self, array2d):
        """

        @return:
        """

        self._imagePlot.set_data(array2d)

        self._flush()

        return

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w+1, h)
        self.resize(w, h)

        return


class Mpl1DGraph(QWidget):
    """
    Simple matplotlib 1D plot
    """

    def __init__(self, parent):
        """Initialization
        :param parent:ns
        """
        # Initialize parent
        super(Mpl1DGraph, self).__init__(parent)

        # set up other variables
        # key = line ID, value = row, col, bool (is main axes???)
        self._lineSubplotMap = dict()

        # records for all the lines that are plot on the canvas
        # key = [row, col][line key], value = label, x-min, x-max, y-min and y-max
        self._myMainPlotDict = dict()
        self._myMainPlotDict[0, 0] = dict()  # init
        self._statMainPlotDict = dict()
        self._statMainPlotDict[0, 0] = None

        # auto line's maker+color list
        self._myLineMarkerColorList = []
        self._myLineMarkerColorIndex = 0

        # set up canvas
        row_size = 1
        col_size = 1
        self._myCanvas = Qt4MplCanvasMultiFigure(self, row_size, col_size)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)

        # instance to line

        return

# END-OF-CLASS (MplGraphicsView)


class Qt4MplCanvasMultiFigure(FigureCanvas):
    """  A customized Qt widget for matplotlib figure.
    It can be used to replace GraphicsView of QtGui
    """

    def __init__(self, parent, row_size=None, col_size=None, rotate=False):
        """Initialization
        :param parent:
        :param row_size:
        :param col_size:
        """
        # Instantiating matplotlib Figure. It is a requirement to initialize a figure canvas
        self.fig = Figure()
        self.fig.patch.set_facecolor('white')

        # Initialize parent class and set parent
        super(Qt4MplCanvasMultiFigure, self).__init__(self.fig)
        self.setParent(parent)

        # Variables to manage all lines/subplot:  key = integer line ID, value = reference to line
        # default to 1 subplot at (0, 0)
        self._mainLineDict = dict()
        self._mainLineDict[0, 0] = dict()

        # right axes
        self._rightLineDict = dict()

        # line index: single index for both main and right plot
        self._lineIndex = 0

        # legend and color bar
        self._legendStatusDict = dict()
        self._legendRightStatusDict = dict()
        self._legendFontSize = 8

        # data structure for sub plots
        self._numSubPlots = 0
        self.axes_main = dict()  # keys are 2-tuple, starting from (0, 0)
        self.axes_right = dict()

        # the subplots are not set up in the initialization
        self.set_subplots(row_size=1, col_size=1, rotate=True)

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self._is_rotated = rotate

        # prototype ...

        vec_x = np.arange(0, 150, 1)
        vec_y = np.sin(vec_x)  # * np.pi / 180.)

        # first of all, the base transformation of the data points is needed
        # base = pyplot.gca().transData
        if rotate:
            # base = self.axes_main[0, 0].transData
            # rot = transforms.Affine2D().rotate_deg(270)
            # output = self.axes_main[0, 0].plot(vec_x, vec_y, 'r--', transform=rot + base)
            output = self.axes_main[0, 0].plot(vec_y, vec_x)
        else:
            output = self.axes_main[0, 0].plot(vec_x, vec_y, 'r--')
        self.axes_main[0, 0].set_aspect('auto')

        self._curr_plot = output[0]

        #
        # print data
        # print vec_x
        # print vec_y
        #
        #
        #
        # # define transformed line
        # # line = pyplot.plot(data, 'r--', transform= rot + base)
        # # line = pyplot.plot(vec_x, vec_y)
        # line = pyplot.plot(vec_x, vec_y, 'r--', transform=rot + base)
        # # or alternatively, use:
        # # line.set_transform(rot + base)

        # pyplot.show()

        return

    def update_plot(self, vec_x, vec_y):
        """

        :param vec_x:
        :param vec_y:
        :return:
        """
        if self._is_rotated:
            self._curr_plot.set_xdata(vec_y)
            self._curr_plot.set_ydata(vec_x)
            self.axes_main[0, 0].set_ylim((vec_x.min(), vec_x.max()))
            self.axes_main[0, 0].set_xlim((vec_y.min(), vec_y.max()))
        else:
            # self.axes_main[0, 0].plot(vec_x, vec_y)
            # print vec_y.min(), vec_y.max()
            self._curr_plot.set_xdata(vec_x)
            self._curr_plot.set_ydata(vec_y)

            self.axes_main[0, 0].set_ylim((vec_y.min(), vec_y.max()))
        self.axes_main[0, 0].set_aspect('auto')

        self.draw()

        return

    def set_subplots(self, row_size, col_size, rotate):
        """
        set subplots
        :param row_size:
        :param col_size:
        :return:
        """
        # check input
        assert isinstance(row_size, int), 'Row size {0} must be an integer but not a {1}.' \
                                          ''.format(row_size, type(row_size))
        assert isinstance(col_size, int), 'Column size {0} must be an integer but not a {1}.' \
                                          ''.format(col_size, type(col_size))

        if row_size < 1:
            raise RuntimeError('Row size {0} must be larger than 0.'.format(row_size))
        if col_size < 1:
            raise RuntimeError('Column size {0} must be larger than 0.'.format(row_size))

        for row_index in range(row_size):
            for col_index in range(col_size):
                sub_plot_index = row_index * col_size + col_index + 1
                subplot_ref = self.fig.add_subplot(row_size, col_size, sub_plot_index)
                self.axes_main[row_index, col_index] = subplot_ref
                if rotate:
                    aspect_num = 2
                else:
                    aspect_num = 0.5
                self.axes_main[row_index, col_index].set_aspect(aspect=aspect_num)
                self._mainLineDict[row_index, col_index] = dict()
                self._legendStatusDict[row_index, col_index] = False
            # END-FOR
        # END-FOR
