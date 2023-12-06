"""
Graphics class with matplotlib backend specific for advanced 1D plot
"""
from matplotlib.figure import Figure
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
        self._myCanvas = Qt4MplCanvasMultiFigure(self, row_size, col_size)
        # if row_size is not None and col_size is not None:
        #     self.set_subplots(row_size, col_size)

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

    def _update_plot_line_information(self, row_index, col_index, line_id, is_main, remove_line, vec_x=None,
                                      vec_y=None, label=None):
        """update the plot line information
        :param row_index:
        :param col_index:
        :param line_id:
        :param is_main: flag whether this is for main axes. Other wise it is for right axes
        :param remove_line:
        :param vec_x:
        :param vec_y:
        :param label:
        :return:
        """
        # get the row-index and column-index if not given
        if row_index is None or col_index is None:
            if not (line_id in self._lineSubplotMap):
                raise RuntimeError('Line ID {0} is not recorded in line-subplot map'.format(line_id))
            row_index, col_index, is_main = self._lineSubplotMap[line_id]

        # check inputs and others
        assert isinstance(line_id, int), 'Line ID {0} must be an integer but not a {1}.' \
                                         ''.format(line_id, type(line_id))
        assert isinstance(row_index, int), 'Row index {0} must be an integer but not a {1}.' \
                                           ''.format(row_index, type(row_index))
        assert isinstance(col_index, int), 'Column index {0} must be an integer but not a {1}.' \
                                           ''.format(col_index, type(col_index))

        plot_dict = self._myMainPlotDict

        # check
        if (row_index, col_index) not in plot_dict:
            raise RuntimeError('Subplot ({0}, {1}) does not exist in (main = {2}). Existing subplots are {3}.'
                               ''.format(row_index, col_index, is_main, plot_dict.keys()))

        # add a NEW line or update an existing line
        if line_id not in plot_dict[row_index, col_index]:
            # mode for add a new line
            assert isinstance(label, str), 'For adding a line (remove_line={0}), label {1} must be a string.' \
                                           'Plot dict keys: {2}'.format(remove_line, label,
                                                                        plot_dict[row_index, col_index].keys())
        elif label is None:
            # for update version, using current label if no new label is given
            label = plot_dict[row_index, col_index][line_id][0]

        # get range of x and y
        min_x = max_x = min_y = max_y = None
        if vec_x is not None:
            min_x = np.min(vec_x)
            max_x = np.max(vec_x)
        if vec_y is not None:
            min_y = np.max(vec_y)
            max_y = np.max(vec_y)

        # set information to plot dictionary
        plot_dict[row_index, col_index][line_id] = [label, min_x, max_x, min_y, max_y]

    def add_plot(self, vec_x, vec_y, x_err=None, y_err=None, row_index=0, col_index=0, is_right=False,
                 color=None, label='',
                 x_label=None, y_label=None, marker=None, markersize=2, line_style=None,
                 line_width=1, show_legend=True):

        # check whether the input is empty
        if len(vec_y) == 0:
            print('[WARNING] Input is an empty vector set')
            return False

        # plot at the main axis
        line_key = self._myCanvas.add_main_plot(row_index, col_index, vec_x, vec_y, x_err, y_err,
                                                color, label, x_label,
                                                y_label, marker, line_style,
                                                line_width, show_legend,
                                                markersize=markersize)

        # update line information
        self._update_plot_line_information(row_index, col_index, line_key, is_main=not is_right,
                                           remove_line=False, label=label, vec_x=vec_x, vec_y=vec_y)

        # add line to dictionary
        self._lineSubplotMap[line_key] = row_index, col_index, not is_right

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
        # treat left and right separately
        for row_index, col_index in self._myCanvas.subplot_indexes:
            # filter the condition to determine whether this subplot is to be cleared

            # remove line for main and right
            if include_main:
                # main axis
                self._myCanvas.clear_subplot_lines(row_index, col_index, True)
                self._myMainPlotDict[row_index, col_index].clear()
                # self._statMainPlotDict ???

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

    def get_label_x(self, row_index=0, col_index=0):
        """Get X-axis label
        :param row_index:
        :param col_index:
        :return:
        """
        return self._myCanvas.axes_main[row_index, col_index].get_xlabel()

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

    def __init__(self, parent, row_size=None, col_size=None):
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

        # count of lines ever plot on the canvas. the newly added line's index is line_count - 1
        self._line_count = 0

        # legend and color bar
        self._legendStatusDict = dict()
        self._legendRightStatusDict = dict()
        self._legend_font_size = 8

        # data structure for sub plots
        self._numSubPlots = 0
        self.axes_main = dict()  # keys are 2-tuple, starting from (0, 0)
        self.axes_right = dict()

        # the subplots are not set up in the initialization:
        self._is_initialized = False
        if row_size is not None and col_size is not None:
            self.set_subplots(row_size, col_size)

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

    def set_subplots(self, row_size, col_size):
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
                self._mainLineDict[row_index, col_index] = dict()
                self._legendStatusDict[row_index, col_index] = False
            # END-FOR
        # END-FOR

        self._is_initialized = True
        try:
            self.fig.tight_layout()
        except RuntimeError:
            pass

        return

    def add_main_plot(self, row_index, col_index,
                      vec_x, vec_y,
                      x_err=None, y_err=None,
                      color=None, label='',
                      x_label=None, y_label=None,
                      marker=None, line_style=None,
                      line_width=1, show_legend=True, markersize=4,):
        """Add 1D plot on the main side (left)
        :param row_index: numpy array X
        :param col_index: numpy array Y
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
        print('[DB...BAT] Add Main Y-label : {0}; Line-label : {1}'.format(y_label, label))

        # Check input
        self._check_subplot_index(row_index, col_index, is_main=True)

        if isinstance(vec_x, np.ndarray) is False or isinstance(vec_y, np.ndarray) is False:
            raise NotImplementedError('Input vec_x or vec_y for addPlot() must be numpy.array,'
                                      'but not {} and {}.'.format(type(vec_x), type(vec_y)))
        plot_error = (y_err is not None) or (x_err is not None)
        if plot_error is True:
            if isinstance(y_err, np.ndarray) is False:
                raise NotImplementedError('Input y_err must be either None or numpy.array.')

        if len(vec_x) != len(vec_y):
            raise NotImplementedError('Input vec_x (shape: {}) and vec_y (shape: {}) must have same size.'
                                      ''.format(vec_x.shape, vec_y.shape))
        if plot_error is True and len(y_err) != len(vec_x):
            raise NotImplementedError('Input vec_x, vec_y and y_error must have same size.')

        # set x-axis and y-axis label
        if x_label is not None:
            self.axes_main[row_index, col_index].set_xlabel(x_label, fontsize=16)  # or 20?
        if y_label is not None:
            self.axes_main[row_index, col_index].set_ylabel(y_label, fontsize=10)

        # process inputs and defaults
        if color is None:
            color = (0, 1, 0, 1)
        if marker is None:
            marker = 'None'
        if line_style is None:
            line_style = '-'

        self.clear_canvas()
        self.axes_main[row_index, col_index].clear()

        # color must be RGBA (4-tuple)
        if plot_error is False:
            # return: list of matplotlib.lines.Line2D object
            r = self.axes_main[row_index, col_index].plot(vec_x, vec_y, color=color,
                                                          marker=marker, markersize=markersize,
                                                          linestyle=line_style, label=label,
                                                          linewidth=line_width)

            self.axes_main[row_index, col_index].autoscale()

        else:
            if y_err is None:
                r = self.axes_main[row_index, col_index].errorbar(vec_x, vec_y,
                                                                  xerr=x_err,
                                                                  color=color, marker=marker,
                                                                  linestyle=line_style, label=label,
                                                                  linewidth=line_width)
            elif x_err is None:
                r = self.axes_main[row_index, col_index].errorbar(vec_x, vec_y,
                                                                  yerr=y_err,
                                                                  color=color, marker=marker,
                                                                  linestyle=line_style, label=label,
                                                                  linewidth=line_width)
            else:
                # both error
                r = self.axes_main[row_index, col_index].errorbar(vec_x, vec_y,
                                                                  xerr=x_err, yerr=y_err,
                                                                  color=color, marker=marker,
                                                                  linestyle=line_style, label=label,
                                                                  linewidth=line_width)

        # set aspect ratio
        self.axes_main[row_index, col_index].set_aspect('auto')

        # just checking bounds, need to add tests for other situations (empty vec_x)
        if len(vec_x) == 1:
            delta_x = vec_x[0]
        else:
            delta_x = vec_x[1] - vec_x[0]

        x_left = vec_x[0] - delta_x
        # TODO more bounds check
        x_right = vec_x[-1] + delta_x
        self.axes_main[row_index, col_index].set_xlim(x_left, x_right)

        # set/update legend
        if show_legend:
            self._setup_legend(row_index, col_index, is_main=True)

        # # Register
        line_key = self._line_count
        if len(r) == 1:
            # single line plot
            self._mainLineDict[row_index, col_index][line_key] = r[0]
            self._line_count += 1
        else:
            # line with error bars
            # TODO FIXME - need to find out the returned's data structure
            self._mainLineDict[row_index, col_index][line_key] = r
            self._line_count += 1
        # END-IF

        # Flush/commit
        # self.draw()

        return line_key

    def clear_subplot_lines(self, row_index, col_index, is_main):
        """
        Remove all lines from a subplot
        i.e., remove line and its record
        :param row_index:
        :param col_index:
        :param is_main:
        :return:
        """
        if is_main:
            # check
            self._check_subplot_index(row_index, col_index, is_main=is_main)

            for line_key in self._mainLineDict[row_index, col_index].keys():
                mpl_line = self._mainLineDict[row_index, col_index][line_key]
                if not mpl_line:
                    # removed
                    continue
                elif isinstance(mpl_line, tuple):
                    # with error bar and etc
                    mpl_line[0].remove()
                    for line in mpl_line[1]:
                        line.remove()
                    for line in mpl_line[2]:
                        line.remove()
                else:
                    # with single line
                    try:
                        self.axes_main[row_index, col_index].cla()
                    except ValueError as e:
                        print("[Error] Plot %s is not in axes_main.lines which has %d lines. Error message: %s" % (
                              str(line_key), len(self.axes_main[row_index, col_index].lines), str(e)))
                # END-IF-ELSE

                # remove record
                self._mainLineDict[row_index, col_index][line_key] = dict()
            # END-FOR

            # set up legend
            self._setup_legend(row_index, col_index, is_main=True)

        # END-IF-ELSE

        # draw
        self.draw()

        return

    def clear_canvas(self):
        """ Clear data including lines and image from canvas
        """
        # # clear the image for next operation
        # for subplot in self.axes_main.values():
        #     subplot.hold(False)
        # for subplot in self.axes_right.values():
        #     if subplot is not None:
        #         subplot.hold(False)

        # clear all lines
        for row_index, col_index in self.axes_main.keys():
            # main
            self.clear_subplot_lines(row_index, col_index, True)
            self.axes_main[row_index, col_index].cla()

            # right if it does exist
            if (row_index, col_index) in self.axes_right:
                self.clear_subplot_lines(row_index, col_index, False)
                self.axes_right[row_index, col_index].cla()
            # END-IF

        # flush/commit
        self._flush()

        return

    def get_axis(self, row_index, col_index, is_main):
        """
        return axis
        :param row_index:
        :param col_index:
        :param is_main:
        :return:
        """
        self._check_subplot_index(row_index, col_index, is_main)

        if is_main:
            axis = self.axes_main[row_index, col_index]
        else:
            axis = self.axes_right[row_index, col_index]

        return axis

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w + 1, h)
        self.resize(w, h)

        return

    def _setup_legend(self, row_index, col_index, location='best', is_main=True, font_size=10):
        """Set up legend
        self.axes.legend(): Handler is a Line2D object. Lable maps to the line object
        :param row_index:
        :param col_index:
        :param location:
        :param font_size:
        :return:
        """
        # check input
        self._check_subplot_index(row_index, col_index, is_main=is_main)

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

        if is_main:
            # main axes on subplot
            handles, labels = self.axes_main[row_index, col_index].get_legend_handles_labels()
            self.axes_main[row_index, col_index].legend(handles, labels, loc=location, fontsize=font_size)
            self._legendStatusDict[row_index, col_index] = True

        else:
            # right axes on subplot
            handles, labels = self.axes_right[row_index, col_index].get_legend_handles_labels()
            self.axes_right[row_index, col_index].legend(handles, labels, loc=location, fontsize=font_size)
            self._legendRightStatusDict[row_index, col_index] = True

        # END-IF

        return

    def _check_subplot_index(self, row_index, col_index, is_main):
        """check whether the subplot indexes are valid
        :param row_index:
        :param col_index:
        :return:
        """
        # check type
        assert isinstance(row_index, int), 'Subplot row index {0} must be an integer but not a {1}' \
            ''.format(row_index, type(row_index))
        assert isinstance(col_index, int), 'Subplot column index {0} must be an integer but not a {1}' \
            ''.format(col_index, type(col_index))

        if is_main and (row_index, col_index) not in list(self.axes_main.keys()):
            raise RuntimeError('Subplot index {0}, {1} does not exist. Keys are {2}. '
                               'Empty keys list indicates a bad init.'
                               ''.format(row_index, col_index, self.axes_main.keys()))
        elif is_main is False and (row_index, col_index) not in self.axes_right.keys():
            raise RuntimeError('Subplot index {0}, {1} does not exist. Keys are {2}. '
                               'Empty key list indicates a bad init'
                               ''.format(row_index, col_index, self.axes_main.keys()))

        return
