"""
Graphics class with matplotlib backend specific for advanced 1D plot
"""
from matplotlib.pyplot import subplots, subplots_adjust
import numpy as np

from qtpy.QtWidgets import QWidget, QSizePolicy, QVBoxLayout  # type:ignore
from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas
from mantidqt.MPLwidgets import NavigationToolbar2QT as NavigationToolbar2
from pyrs.interface.ui.mplconstants import MplBasicColors, MplLineMarkers


class MplGraphicsView1D(QWidget):
    """ A combined graphics view including matplotlib canvas and a navigation tool bar
    1. specific for 1-D data
    """

    def __init__(self, parent, row_size=None, col_size=None, tool_bar=True):
        """Initialization
        :param parent:
        :param row_size: number of figures per column, i.e., number of rows
        :param col_size: number of figures per row, i.e., number of columns
        """
        # Initialize parent
        super(MplGraphicsView1D, self).__init__(parent)

        # set up other variables
        # key = line ID, value = row, col, bool (is main axes???)
        self._lineSubplotMap = dict()

        # records for all the lines that are plot on the canvas
        # key = [row, col][line key], value = label, x-min, x-max, y-min and y-max
        self._myMainPlotDict = dict()
        self._myMainPlotDict[0, 0] = dict()  # init
        self._statMainPlotDict = dict()
        self._statMainPlotDict[0, 0] = None

        # for right plot
        self._myRightPlotDict = dict()

        # auto line's maker+color list
        self._myLineMarkerColorList = list()
        self._myLineMarkerColorIndex = 0
        self.setAutoLineMarkerColorCombo()

        # set up canvas
        self._myCanvas = Qt4MplCanvasMultiFigure(self)

        if tool_bar:
            self._myToolBar = NavigationToolbar2(self._myCanvas, self)
        else:
            self._myToolBar = None

        self._myCanvas.mpl_connect('button_press_event', self.button_clicked_in_canvas)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myToolBar)

    def button_clicked_in_canvas(self, event):
        print("-> {} click: button={:d}, x={:d}, y={:d}, xdata={:f}, ydata={:f}".format(event.dblclick, event.button,
                                                                                        event.x, event.y, event.xdata,
                                                                                        event.ydata))

    def _update_plot_line_information(self, line_id, is_main, remove_line, vec_x=None,
                                      vec_y=None, label=None, ):
        """update the plot line information
        :param line_id:
        :param is_main: flag whether this is for main axes. Other wise it is for right axes
        :param remove_line:
        :param vec_x:
        :param vec_y:
        :param label:
        :return:
        """
        # get the row-index and column-index if not given
        if not (line_id in self._lineSubplotMap):
            raise RuntimeError('Line ID {0} is not recorded in line-subplot map'.format(line_id))

        # check inputs and others
        assert isinstance(line_id, int), 'Line ID {0} must be an integer but not a {1}.' \
                                         ''.format(line_id, type(line_id))

        plot_dict = self._myMainPlotDict

        # get range of x and y
        min_x = max_x = min_y = max_y = None
        if vec_x is not None:
            min_x = np.min(vec_x)
            max_x = np.max(vec_x)
        if vec_y is not None:
            min_y = np.max(vec_y)
            max_y = np.max(vec_y)

        # set information to plot dictionary
        plot_dict[line_id] = [label, min_x, max_x, min_y, max_y]

    def add_plot(self, vec_x, vec_y, x_err=None, y_err=None, is_right=False,
                 color=None, label='',
                 x_label=None, y_label=None, marker=None, markersize=2, line_style=None,
                 line_width=1, show_legend=True):

        # check whether the input is empty
        if len(vec_y) == 0:
            print('[WARNING] Input is an empty vector set')
            return False

        # plot at the main axis
        line_key = self._myCanvas.add_main_plot(vec_x, vec_y, x_err, y_err,
                                                color, label, x_label,
                                                y_label, marker, line_style,
                                                line_width, show_legend,
                                                markersize=markersize)

        # add line to dictionary
        self._lineSubplotMap[line_key] = line_key

        # update line information
        self._update_plot_line_information(line_key, is_main=not is_right,
                                           remove_line=False, label=label, vec_x=vec_x, vec_y=vec_y)

        return line_key

    def canvas(self):
        """ Get the canvas
        :return:
        """
        return self._myCanvas

    def clear_all_lines(self, row_number=None, col_number=None, include_main=True, include_right=True):
        """
        clear all the lines of all or not
        :param row_number:
        :param col_number:
        :param include_main:
        :param include_right:
        :return:
        """

        # about zoom
        self._isZoomed = False
        self._homeXYLimit = None

        self._myCanvas.clear_canvas()

    def draw(self):
        """ Draw to commit the change
        """
        return self._myCanvas.draw()

    def evt_toolbar_home(self):
        """
        """
        # turn off zoom mode
        self._isZoomed = False

    def evt_zoom_released(self, event):
        """
        Handling the event that is triggered by zoom-button in tool bar is released and a customized signal
        is thus emit.
        :param event: event instance
        :return:
        """
        assert event is not None

        # record home XY limit if it is never zoomed
        if self._isZoomed is False:
            self._homeXYLimit = list(self.get_x_limit())
            ylimit = self.get_y_limit()
            if not np.isnan(ylimit):
                self._homeXYLimit.extend(list(ylimit))

        # set the state of being zoomed
        self._isZoomed = True

    def get_label_x(self):
        """Get X-axis label
        :return: str
        """
        return self._myCanvas.axes_main.get_xlabel()

    def get_x_limit(self):
        """Get X-axis current limit
        Returns
        -------
        (float, float)
            x min, x max
        """
        return self._myCanvas.get_x_limits()

    def get_y_limit(self):
        """ Get limit of Y-axis
        """
        return self._myCanvas.getYLimit()

    def get_canvas(self):
        """
        get canvas
        Returns:

        """
        return self._myCanvas.axes_main

    def get_subplots_indexes(self):
        """
        get the indexes of all the subplots
        :return: a list of 2-tuples as (row-index, column-index)
        """
        return self._myCanvas.subplot_indexes

    def setAutoLineMarkerColorCombo(self):
        """ Set the default/auto line marker/color combination list
        """
        self._myLineMarkerColorList = list()
        for marker in MplLineMarkers:
            for color in MplBasicColors:
                self._myLineMarkerColorList.append((marker, color))


class Qt4MplCanvasMultiFigure(FigureCanvas):
    """  A customized Qt widget for matplotlib figure.
    It can be used to replace GraphicsView of QtGui
    """

    def __init__(self, parent):
        """Initialization
        :param parent:
        :param row_size:
        :param col_size:
        """
        # Instantiating matplotlib Figure. It is a requirement to initialize a figure canvas
        self.fig, self.axes_main = subplots(1, 1, sharex=True)
        self.fig.patch.set_facecolor('white')
        subplots_adjust(left=.15, bottom=.15, top=.9, right=.95)

        # Initialize parent class and set parent
        super(Qt4MplCanvasMultiFigure, self).__init__(self.fig)
        self.setParent(parent)

        # Variables to manage all lines/subplot:  key = integer line ID, value = reference to line
        # default to 1 subplot at (0, 0)
        self._mainLineDict = dict()

        # count of lines ever plot on the canvas. the newly added line's index is line_count - 1
        self._line_count = 0

        # legend and color bar
        self._legendStatusDict = dict()
        self._legendRightStatusDict = dict()
        self._legend_font_size = 8

        # the subplots are not set up in the initialization:
        self._is_initialized = True

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        return

    @property
    def subplot_indexes(self):
        """

        :return:  a list of 2 - tuples as (row - index, column - index)
        """
        return sorted(self.axes_main.keys())

    def add_main_plot(self, vec_x, vec_y,
                      x_err=None, y_err=None,
                      color=None, label='',
                      x_label=None, y_label=None,
                      marker=None, line_style=None,
                      line_width=1, show_legend=True, markersize=4,):
        """Add 1D plot on the main side (left)
        :param vec_x:
        :param vec_y:
        :param y_err:
        :param color:
        :param label:
        :param x_label:
        :param y_label:
        :param marker:
        :param line_style:
        :param line_width:
        :param show_legend:
        :return: line ID (i.e., new key)
        """

        if isinstance(vec_x, np.ndarray) is False or isinstance(vec_y, np.ndarray) is False:
            raise NotImplementedError('Input vec_x or vec_y for addPlot() must be numpy.array,'
                                      'but not {} and {}.'.format(type(vec_x), type(vec_y)))

        plot_errors = (y_err is not None) or (x_err is not None)

        if len(vec_x) != len(vec_y):
            raise NotImplementedError('Input vec_x (shape: {}) and vec_y (shape: {}) must have same size.'
                                      ''.format(vec_x.shape, vec_y.shape))

        if (y_err is not None) and (len(y_err) != len(vec_y)):
            raise NotImplementedError('Input vec_y and y_error must have same size.')

        if (x_err is not None) and (len(x_err) != len(vec_x)):
            raise NotImplementedError('Input vec_x and x_error must have same size.')

        # set x-axis and y-axis label
        if x_label is not None:
            self.axes_main.set_xlabel(x_label)  # or 20?
        if y_label is not None:
            self.axes_main.set_ylabel(y_label)

        # process inputs and defaults
        if color is None:
            color = (0, 1, 0, 1)
        if marker is None:
            marker = 'None'
        if line_style is None:
            line_style = '-'

        # self.clear_canvas()
        # self.axes_main.clear()

        # color must be RGBA (4-tuple)
        if plot_errors is False:
            # return: list of matplotlib.lines.Line2D object
            r = self.axes_main.plot(vec_x, vec_y, color=color,
                                    marker=marker, markersize=markersize,
                                    linestyle=line_style, label=label,
                                    linewidth=line_width)

            self.axes_main.autoscale()

        else:
            if y_err is None:
                r = self.axes_main.errorbar(vec_x, vec_y,
                                            xerr=x_err,
                                            color=color, marker=marker,
                                            linestyle=line_style, label=label,
                                            linewidth=line_width)
            elif x_err is None:
                r = self.axes_main.errorbar(vec_x, vec_y,
                                            yerr=y_err,
                                            color=color, marker=marker,
                                            linestyle=line_style, label=label,
                                            linewidth=line_width)
            else:
                # both error
                r = self.axes_main.errorbar(vec_x, vec_y,
                                            xerr=x_err, yerr=y_err,
                                            color=color, marker=marker,
                                            linestyle=line_style, label=label,
                                            linewidth=line_width)

        # set aspect ratio
        self.axes_main.set_aspect('auto')

        # just checking bounds, need to add tests for other situations (empty vec_x)
        if len(vec_x) == 1:
            delta_x = vec_x[0]
        else:
            delta_x = vec_x[1] - vec_x[0]

        x_left = vec_x[0] - delta_x

        x_right = vec_x[-1] + delta_x
        self.axes_main.set_xlim(x_left, x_right)

        # set/update legend
        if show_legend:
            self._setup_legend(is_main=True)

        # # Register
        line_key = self._line_count
        if len(r) == 1:
            # single line plot
            self._mainLineDict[line_key] = r[0]
            self._line_count += 1
        else:
            # line with error bars
            self._mainLineDict[line_key] = r
            self._line_count += 1
        # END-IF

        # Flush/commit
        # self.draw()

        return line_key

    def clear_canvas(self):
        """ Clear data including lines and image from canvas
        """
        # clear all lines
        self.axes_main.cla()

        # flush/commit
        self._flush()

        return

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w + 1, h)
        self.resize(w, h)

        return

    def get_axis(self):
        return self.axes_main

    def _setup_legend(self, location='best', is_main=True, font_size=10):
        """Set up legend
        self.axes.legend(): Handler is a Line2D object. Lable maps to the line object
        :param location:
        :param font_size:
        :return:
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

        # main axes on subplot
        handles, labels = self.axes_main.get_legend_handles_labels()
        self.axes_main.legend(handles, labels, loc=location, fontsize=font_size)
        self._legendStatusDict = True

        # END-IF

        return
