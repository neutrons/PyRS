#pylint: disable=invalid-name,too-many-public-methods,too-many-arguments,non-parent-init-called,R0902,too-many-branches,C0302
import os
import numpy as np

try:
    from PyQt5.QtCore import pyqtSignal
    from PyQt5.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2
except ImportError:
    from PyQt4.QtGui import QWidget, QSizePolicy, QVBoxLayout
    from PyQt4.QtCore import pyqtSignal
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar2

from matplotlib.figure import Figure
import matplotlib.image

MplLineStyles = ['-', '--', '-.', ':', 'None', ' ', '']
MplLineMarkers = [
    ". (point         )",
    "* (star          )",
    "x (x             )",
    "o (circle        )",
    "s (square        )",
    "D (diamond       )",
    ", (pixel         )",
    "v (triangle_down )",
    "^ (triangle_up   )",
    "< (triangle_left )",
    "> (triangle_right)",
    "1 (tri_down      )",
    "2 (tri_up        )",
    "3 (tri_left      )",
    "4 (tri_right     )",
    "8 (octagon       )",
    "p (pentagon      )",
    "h (hexagon1      )",
    "H (hexagon2      )",
    "+ (plus          )",
    "d (thin_diamond  )",
    "| (vline         )",
    "_ (hline         )",
    "None (nothing    )"]

# Note: in colors, "white" is removed
MplBasicColors = [
    "black",
    "red",
    "blue",
    "green",
    "cyan",
    "magenta",
    "yellow"]


class MplGraphicsView2D(QWidget):
    """ A combined graphics view including matplotlib canvas and
    a navigation tool bar

    Note: Merged with HFIR_Powder_Reduction.MplFigureCAnvas
    """
    def __init__(self, parent):
        """ Initialization
        """
        # Initialize parent
        super(MplGraphicsView2D, self).__init__(parent)

        # set up canvas
        self._myCanvas = Qt4Mpl2DCanvas(self)
        self._myToolBar = MyNavigationToolbar(self, self._myCanvas)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myToolBar)

        self._arrowList = list()

        #
        self._hasImage = False

        return

    def add_arrow(self, start_x, start_y, stop_x, stop_y):
        """

        :param start_x:
        :param start_y:
        :param stop_x:
        :param stop_y:
        :return:
        """
        arrow = self._myCanvas.add_arrow(start_x, start_y, stop_x, stop_y)
        self._arrowList.append(arrow)

        return

    def add_image(self, image_file_name):
        """ Add an image by file
        """
        # check
        if os.path.exists(image_file_name) is False:
            raise NotImplementedError("Image file %s does not exist." % image_file_name)

        self._myCanvas.add_image_file(image_file_name)

        return

    def add_2d_plot(self, array2d, x_min, x_max, y_min, y_max, y_tick_label=None, plot_type='image'):
        """
        Add a 2D image to canvas
        :param array2d: numpy 2D array
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :param y_tick_label:
        :return:
        """
        # obsoleted: self._myCanvas.addPlot2D(array2d, x_min, x_max, y_min, y_max, hold_prev_image, y_tick_label)

        if plot_type == 'image':
            self._myCanvas.add_image_plot(array2d, x_min, x_max, y_min, y_max, yticklabels=y_tick_label)
        elif plot_type == 'image file':
            self._myCanvas.add_image_file()
        elif plot_type == 'scatter':
            blabla
        else:
            blabla

        self._hasImage = True

        return

    def has_image_on_canvas(self):
        """
        blabla
        @return:
        """
        # TODO/ASAP
        return self._hasImage

    def update_2d_plot(self):
        """

        @return:
        """
        pass

    def canvas(self):
        """ Get the canvas
        :return:
        """
        return self._myCanvas

    def clear_canvas(self):
        """ Clear canvas
        """
        # clear all the records
        # to-be-filled

        # about zoom
        # to-be-filled

        r = self._myCanvas.clear_canvas()

        return r

    def draw(self):
        """ Draw to commit the change
        """
        return self._myCanvas.draw()

    def evt_toolbar_home(self):
        """

        @return:
        """
        # turn off zoom mode
        self._isZoomed = False

        return

    def evt_view_updated(self):
        """ Event handling as canvas size updated
        :return:
        """
        # # update the indicator
        # new_x_range = self.getXLimit()
        # new_y_range = self.getYLimit()
        #
        # self._myIndicatorsManager.update_indicators_range(new_x_range, new_y_range)
        # for indicator_key in self._myIndicatorsManager.get_live_indicator_ids():
        #     canvas_line_id = self._myIndicatorsManager.get_canvas_line_index(indicator_key)
        #     data_x, data_y = self._myIndicatorsManager.get_data(indicator_key)
        #     self.updateLine(canvas_line_id, data_x, data_y)
        # # END-FOR

        return

    def evt_zoom_released(self):
        """ event for zoom is release
        @return:
        """
        # record home XY limit if it is never zoomed
        if self._isZoomed is False:
            self._homeXYLimit = list(self.getXLimit())
            self._homeXYLimit.extend(list(self.getYLimit()))
        # END-IF

        # set the state of being zoomed
        self._isZoomed = True

        return

    def getLastPlotIndexKey(self):
        """ Get ...
        """
        return self._myCanvas.getLastPlotIndexKey()

    def getXLimit(self):
        """ Get limit of Y-axis
        :return: 2-tuple as xmin, xmax
        """
        return self._myCanvas.getXLimit()

    def getYLimit(self):
        """ Get limit of Y-axis
        """
        return self._myCanvas.getYLimit()

    def get_y_min(self):
        """
        Get the minimum Y value of the plots on canvas
        :return:
        """
        if len(self._statDict) == 0:
            return 1E10

        line_id_list = self._statDict.keys()
        min_y = self._statDict[line_id_list[0]][2]
        for i_plot in range(1, len(line_id_list)):
            if self._statDict[line_id_list[i_plot]][2] < min_y:
                min_y = self._statDict[line_id_list[i_plot]][2]

        return min_y

    def get_y_max(self):
        """
        Get the maximum Y value of the plots on canvas
        :return:
        """
        if len(self._statDict) == 0:
            return -1E10

        line_id_list = self._statDict.keys()
        max_y = self._statDict[line_id_list[0]][3]
        for i_plot in range(1, len(line_id_list)):
            if self._statDict[line_id_list[i_plot]][3] > max_y:
                max_y = self._statDict[line_id_list[i_plot]][3]

        return max_y

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

    def remove_line(self, line_id):
        """ Remove a line
        :param line_id:
        :return:
        """
        # remove line
        self._myCanvas.remove_plot_1d(line_id)

        # remove the records
        if line_id in self._statDict:
            del self._statDict[line_id]
            del self._my1DPlotDict[line_id]
        else:
            del self._statRightPlotDict[line_id]

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

    def removePlot(self, ikey):
        """
        """
        return self._myCanvas.remove_plot_1d(ikey)

    def updateLine(self, ikey, vecx=None, vecy=None, linestyle=None, linecolor=None, marker=None, markercolor=None):
        """
        update a line's set up
        Parameters
        ----------
        ikey
        vecx
        vecy
        linestyle
        linecolor
        marker
        markercolor

        Returns
        -------

        """
        # check
        assert isinstance(ikey, int), 'Line key must be an integer.'
        assert ikey in self._my1DPlotDict, 'Line with ID %d is not on canvas. ' % ikey

        return self._myCanvas.updateLine(ikey, vecx, vecy, linestyle, linecolor, marker, markercolor)

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

    def get_canvas(self):
        """
        get canvas
        Returns:

        """
        return self._myCanvas

    def set_title(self, title, color='black'):
        """
        set title to canvas
        :param title:
        :param color:
        :return:
        """
        self._myCanvas.set_title(title, color)

        return

    def setXYLimit(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """ Set X-Y limit automatically
        """
        self._myCanvas.axes.set_xlim([xmin, xmax])
        self._myCanvas.axes.set_ylim([ymin, ymax])

        self._myCanvas.draw()

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

        # plot management
        self._scatterPlot = None
        self._imagePlot = None

        # Initialize parent class and set parent
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Variables to manage all lines/subplot
        self._lineDict = {}
        self._lineIndex = 0

        # legend and color bar
        self._colorBar = None
        self._isLegendOn = False
        self._legendFontSize = 8

        return

    def update_image(self, array2d):
        """

        @return:
        """

        self._imagePlot.set_data(array2d)

        self._flush()

        return

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

    def add_arrow(self, start_x, start_y, stop_x, stop_y):
        """
        Example: (0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
        @param start_x:
        @param start_y:
        @param stop_x:
        @param stop_y:
        @return:
        """
        head_width = 0.05
        head_length = 0.1
        fc = 'k'
        ec = 'k'

        arrow = self.axes.arrrow(start_x, start_y, stop_x, stop_y, head_width,
                                 head_length, fc, ec)

        return arrow

    def add_contour_plot(self, vec_x, vec_y, matrix_z):
        """ add a contour plot
        Example: reduced data: vec_x: d-values, vec_y: run numbers, matrix_z, matrix for intensities
        :param vec_x: a list of a vector for X axis
        :param vec_y: a list of a vector for Y axis
        :param matrix_z:
        :return:
        """
        # check input
        # TODO - labor
        assert isinstance(vec_x, list) or isinstance(vec_x, np.ndarray), 'blabla'
        assert isinstance(vec_y, list) or isinstance(vec_y, np.ndarray), 'blabla'
        assert isinstance(matrix_z, np.ndarray), 'blabla'

        # create mesh grid
        grid_x, grid_y = np.meshgrid(vec_x, vec_y)
        #
        # print '[DB...BAT] Grid X and Grid Y size: ', grid_x.shape, grid_y.shape

        # check size
        assert grid_x.shape == matrix_z.shape, 'Size of X (%d) and Y (%d) must match size of Z (%s).' \
                                               '' % (len(vec_x), len(vec_y), matrix_z.shape)

        # # Release the current image
        # self.axes.hold(False)

        # Do plot: resolution on Z axis (color bar is set to 100)
        self.axes.clear()
        contour_plot = self.axes.contourf(grid_x, grid_y, matrix_z, 100)

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
        self.axes.set_aspect('auto')

        # Set color bar.  plt.colorbar() does not work!
        if self._colorBar is None:
            # set color map type
            contour_plot.set_cmap('spectral')
            self._colorBar = self.fig.colorbar(contour_plot)
        else:
            self._colorBar.update_bruteforce(contour_plot)

        # Flush...
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
        assert isinstance(array2d, np.ndarray), 'blabla'
        assert len(array2d.shape) == 2, 'blabla'

        # show image
        self._imagePlot = self.axes.imshow(array2d, extent=[xmin, xmax, ymin, ymax], interpolation='none')

        print (self._imagePlot, type(self._imagePlot))

        # set y ticks as an option:
        if yticklabels is not None:
            # it will always label the first N ticks even image is zoomed in
            print ("[FIXME]: The way to set up the Y-axis ticks is wrong!")
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
        """ Add an image by file
        """
        #import matplotlib.image as mpimg

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
        # check!
        # TODO - 20180801 - Make it work
        assert isinstance(array2d, np.ndarray), 'blabla'
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

    def hide_legend(self):
        """
        hide the legend if it is not None
        Returns:

        """
        if self.axes.legend() is not None:
            # set visible to be False and re-draw
            self.axes.legend().set_visible(False)
            self.draw()

        self._isLegendOn = False

        return

    def increase_legend_font_size(self):
        """
        reset the legend with the new font size
        Returns:

        """
        self._legendFontSize += 1

        self._setup_legend(font_size=self._legendFontSize)

        self.draw()

        return

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

        self.draw()

        return

    def show_legend(self):
        """
        show the legend if the legend is not None
        Returns:

        """
        if self.axes.legend() is not None:
            # set visible to be True and re-draw
            # self.axes.legend().set_visible(True)
            self._setup_legend(font_size=self._legendFontSize)
            self.draw()

            # set flag on
            self._isLegendOn = True

        return

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w+1, h)
        self.resize(w, h)

        return

    def _setup_legend(self, location='best', font_size=10):
        """
        Set up legend
        self.axes.legend(): Handler is a Line2D object. Lable maps to the line object
        Args:
            location:
            font_size:

        Returns:

        """
        allowed_location_list = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center"]

        # Check legend location valid or not
        if location not in allowed_location_list:
            location = 'best'

        handles, labels = self.axes.get_legend_handles_labels()
        self.axes.legend(handles, labels, loc=location, fontsize=font_size)

        self._isLegendOn = True

        return

# END-OF-CLASS (MplGraphicsView)


class MyNavigationToolbar(NavigationToolbar2):
    """ A customized navigation tool bar attached to canvas
    Note:
    * home, left, right: will not disable zoom/pan mode
    * zoom and pan: will turn on/off both's mode

    Other methods
    * drag_pan(self, event): event handling method for dragging canvas in pan-mode
    """
    NAVIGATION_MODE_NONE = 0
    NAVIGATION_MODE_PAN = 1
    NAVIGATION_MODE_ZOOM = 2

    # This defines a signal called 'home_button_pressed' that takes 1 boolean
    # argument for being in zoomed state or not
    home_button_pressed = pyqtSignal()

    # This defines a signal called 'canvas_zoom_released'
    canvas_zoom_released = pyqtSignal()

    def __init__(self, parent, canvas):
        """ Initialization
        built-in methods
        - drag_zoom(self, event): triggered during holding the mouse and moving
        """
        NavigationToolbar2.__init__(self, canvas, canvas)

        # parent
        self._myParent = parent
        # tool bar mode
        self._myMode = MyNavigationToolbar.NAVIGATION_MODE_NONE

        # connect the events to parent
        self.home_button_pressed.connect(self._myParent.evt_toolbar_home)
        self.canvas_zoom_released.connect(self._myParent.evt_zoom_released)

        return

    @property
    def is_zoom_mode(self):
        """
        check whether the tool bar is in zoom mode
        Returns
        -------

        """
        return self._myMode == MyNavigationToolbar.NAVIGATION_MODE_ZOOM

    def get_mode(self):
        """
        :return: integer as none/pan/zoom mode
        """
        return self._myMode

    # Overriding base's methods
    def draw(self):
        """
        Canvas is drawn called by pan(), zoom()
        :return:
        """
        NavigationToolbar2.draw(self)

        self._myParent.evt_view_updated()

        return

    def home(self, *args):
        """

        Parameters
        ----------
        args

        Returns
        -------

        """
        # call super's home() method
        NavigationToolbar2.home(self, args)

        # send a signal to parent class for further operation
        self.home_button_pressed.emit()

        return

    def pan(self, *args):
        """

        :param args:
        :return:
        """
        NavigationToolbar2.pan(self, args)

        if self._myMode == MyNavigationToolbar.NAVIGATION_MODE_PAN:
            # out of pan mode
            self._myMode = MyNavigationToolbar.NAVIGATION_MODE_NONE
        else:
            # into pan mode
            self._myMode = MyNavigationToolbar.NAVIGATION_MODE_PAN

        print 'PANNED'

        return

    def zoom(self, *args):
        """
        Turn on/off zoom (zoom button)
        :param args:
        :return:
        """
        NavigationToolbar2.zoom(self, args)

        if self._myMode == MyNavigationToolbar.NAVIGATION_MODE_ZOOM:
            # out of zoom mode
            self._myMode = MyNavigationToolbar.NAVIGATION_MODE_NONE
        else:
            # into zoom mode
            self._myMode = MyNavigationToolbar.NAVIGATION_MODE_ZOOM

        return

    def release_zoom(self, event):
        """
        override zoom released method
        Parameters
        ----------
        event

        Returns
        -------

        """
        self.canvas_zoom_released.emit()

        NavigationToolbar2.release_zoom(self, event)

        return

    def _update_view(self):
        """
        view update called by home(), back() and forward()
        :return:
        """
        NavigationToolbar2._update_view(self)

        self._myParent.evt_view_updated()

        return
