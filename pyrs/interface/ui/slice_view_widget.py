try:
    from PyQt5.QtCore import pyqtSignal
    from PyQt5 import QtCore, QtGui
    from PyQt5.QtWidgets import QWidget, QSizePolicy, QVBoxLayout, QGridLayout
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2
except ImportError:
    from PyQt4.QtGui import QWidget, QSizePolicy, QVBoxLayout, QGridLayout
    from PyQt4.QtCore import pyqtSignal
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar2

import numpy as np
import sliceviewwidgets


class SliceViewWidget(QWidget):
    """
    """
    # TODO - 20180830 - Doc!
    def __init__(self, Form):
        """Initialization
        :param parent:ns
        """
        # Initialize parent
        super(SliceViewWidget, self).__init__(Form)

        # self.sample=[Layer(thickness=np.inf),Layer(thickness=np.inf)]
        self.setLayout(QGridLayout())
        function_options=['NSLD_REAL','NSLD_IMAGINARY','MSLD_RHO','MSLD_THETA','MSLD_PHI','ROUGHNESS']
        # self.combo = QtWidgets.QComboBox(self)
        # for f in function_options:
        #     self.combo.addItem(f)
        self.function = 'NSLD_REAL'
        self.paintwidget=QWidget(self)
        self.paintwidget.setMinimumSize(800, 600)
        # self.layout().addWidget(self.paintwidget)
        self.layout().addWidget(self.paintwidget, 0, 0, 1, 1)
        self.canvas = sliceviewwidgets.Qt4Mpl2DCanvas(self.paintwidget)  #  QGraphicsView()   #  PlotCanvas(self.sample, self.function,self.paintwidget)
        ssizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ssizePolicy.setHorizontalStretch(1)
        ssizePolicy.setVerticalStretch(1)
        ssizePolicy.setHeightForWidth(self.paintwidget.sizePolicy().hasHeightForWidth())
        self.paintwidget.setSizePolicy(ssizePolicy)
        self.paintwidget.setObjectName("graphicsVerticalView")

        # vertical slice
        self.vertical_widget = QWidget(self)
        self.vertical_widget.setMinimumSize(200, 600)
        self.layout().addWidget(self.vertical_widget, 0, 1, 1, 1)
        self.vertical_canvas = sliceviewwidgets.Qt4MplCanvasMultiFigure(self.vertical_widget, rotate=1)
        vsizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        vsizePolicy.setHorizontalStretch(0)
        vsizePolicy.setVerticalStretch(1)
        vsizePolicy.setHeightForWidth(False)
        vsizePolicy.setWidthForHeight(False)
        self.vertical_widget.setSizePolicy(vsizePolicy)
        self.vertical_widget.setSizePolicy(vsizePolicy)
        self.vertical_widget.setObjectName("graphicsVerticalView")

        # horizontal slice
        self.horizontal_widget = QWidget(self)
        self.horizontal_widget.setMinimumSize(800, 150)
        self.layout().addWidget(self.horizontal_widget, 1, 0, 1, 1)
        self.horizontal_canvas = sliceviewwidgets.Qt4MplCanvasMultiFigure(self.horizontal_widget)
        hsizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        hsizePolicy.setHorizontalStretch(1)
        hsizePolicy.setVerticalStretch(0)
        hsizePolicy.setHeightForWidth(True)
        hsizePolicy.setWidthForHeight(True)
        self.horizontal_widget.setSizePolicy(hsizePolicy)
        self.horizontal_widget.setSizePolicy(hsizePolicy)
        self.horizontal_widget.setObjectName("graphicsHorizontalView")

        # connect the events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press_event)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release_event)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # indicator
        self._indicator = IndicatorManager()

        return

    def resizeEvent(self, event):
        self.horizontal_canvas.setGeometry(self.horizontal_widget.rect())
        self.vertical_canvas.setGeometry(self.vertical_widget.rect())
        self.canvas.setGeometry(self.paintwidget.rect())

        return

    def on_mouse_press_event(self, event):
        """ Handling when mouse button is pressed
        Designed feature:
        1. as left button is pressed, the 2-way indicator shall be moved to the mouse position
        2. sliced 1D plot shall be updated
        :param event:
        :return:
        """
        print ('[DB...BAT] Pressed {} at x = {}, y = {}'.format(event.button, event.xdata, event.ydata))

        # get current position
        pos_x = event.xdata
        pos_y = event.ydata

        if event.button == 1 and pos_x is not None and pos_y is not None:
            # left button
            x_min = -150
            x_max = 150
            y_min = 0
            y_max = 70
            self._indicator.plot_2way_indicator(self.canvas, pos_x, pos_y)

        else:
            # other buttons: do nothing
            pass

        #

        self.canvas._flush()

    def on_mouse_release_event(self, event):
        """ Handling when mouse button is released
        :param event:
        :return:
        """
        print 'Released {} at x = {}, y = {}'.format(event.button, event.xdata, event.ydata)

        return

    def on_mouse_move(self, event):
        """

        :param event:
        :return:
        """


    # NOTE: 1 = left, 3 = right.. xdata, ydata are coordiate with vec_x and vec_y


class IndicatorManager(object):
    """ Manager for all indicator lines

    Indicator's Type =
    - 0: horizontal.  moving along Y-direction. [x_min, x_max], [y, y];
    - 1: vertical. moving along X-direction. [x, x], [y_min, y_max];
    - 2: 2-way. moving in any direction. [x_min, x_max], [y, y], [x, x], [y_min, y_max].
    """
    def __init__(self):
        """
        :return:
        """
        # current indicator indexes
        self._horizontal_indicator = None
        self._vertical_indicator = None

        # Auto color index
        self._colorIndex = 0
        # Auto line ID
        self._autoLineID = 1

        self._lineManager = dict()  # key: indicator ID, value: 5-tuple about indicator's position and type
        self._canvasLineKeyDict = dict()  # key: indicator ID, value: line-key on the canvas
        self._indicatorTypeDict = dict()  # value: 0 (horizontal), 1 (vertical), 2 (2-way)

        return

    def _set_2way_indicators(self, horizontal_id, vertical_id):
        """
        add a 2-way indicator to manager
        :param horizontal_id:
        :param vertical_id:
        :return:
        """
        self._horizontal_indicator = horizontal_id
        self._vertical_indicator = vertical_id

        return

    # TODO - 20180831 - Doc and check
    def plot_2way_indicator(self, canvas, x_pos, y_pos):
        """

        :param canvas:
        :param x_pos:
        :param y_pos:
        :return:
        """
        x_min, x_max = canvas.getXLimit()
        y_min, y_max = canvas.getYLimit()
        vec_x = np.array([x_min, x_max])
        vec_y = np.array([y_pos, y_pos])

        if self._horizontal_indicator is None:
            self._horizontal_indicator = canvas.axes.plot(vec_x, vec_y, color='white')[0]
        else:
            self._horizontal_indicator.set_ydata(vec_y)

        print (type(self._horizontal_indicator), self._horizontal_indicator)

        vec_y = np.array([y_min, y_max])
        vec_x = np.array([x_pos, x_pos])
        canvas.axes.plot(vec_x, vec_y, color='white')

        return

    # def add_2way_indicator(self, x, x_min, x_max, y, y_min, y_max, color):
    #     """ add a 2-way indicator and store
    #     :param x:
    #     :param x_min:
    #     :param x_max:
    #     :param y:
    #     :param y_min:
    #     :param y_max:
    #     :param color:
    #     :return:
    #     """
    #     # Set up indicator ID
    #     this_id = str(self._autoLineID)
    #     self._autoLineID += 1
    #
    #     # Set up vectors
    #     vec_x_horizontal = np.array([x_min, x_max])
    #     vec_y_horizontal = np.array([y, y])
    #
    #     vec_x_vertical = np.array([x, x])
    #     vec_y_vertical = np.array([y_min, y_max])
    #
    #     #
    #     self._lineManager[this_id] = [vec_x_horizontal, vec_y_horizontal, vec_x_vertical, vec_y_vertical, color]
    #     self._indicatorTypeDict[this_id] = 2
    #
    #     return this_id
    #
    # def add_horizontal_indicator(self, y, x_min, x_max, color):
    #     """
    #     Add a horizontal indicator moving vertically
    #     :param y:
    #     :param x_min:
    #     :param x_max:
    #     :param color:
    #     :return:
    #     """
    #     # Get ID
    #     this_id = str(self._autoLineID)
    #     self._autoLineID += 1
    #
    #     #
    #     vec_x = np.array([x_min, x_max])
    #     vec_y = np.array([y, y])
    #
    #     #
    #     self._lineManager[this_id] = [vec_x, vec_y, color]
    #     self._indicatorTypeDict[this_id] = 0
    #
    #     return this_id
    #
    # def add_vertical_indicator(self, x, y_min, y_max, color):
    #     """
    #     Add a vertical indicator to data structure moving horizontally
    #     :return: indicator ID as an integer
    #     """
    #     # Get ID
    #     this_id = self._autoLineID
    #     self._autoLineID += 1
    #
    #     # form vec x and vec y
    #     vec_x = np.array([x, x])
    #     vec_y = np.array([y_min, y_max])
    #
    #     #
    #     self._lineManager[this_id] = [vec_x, vec_y, color]
    #     self._indicatorTypeDict[this_id] = 1
    #
    #     return this_id

    def delete(self, indicator_id):
        """
        Delete indicator
        """
        del self._lineManager[indicator_id]
        del self._canvasLineKeyDict[indicator_id]
        del self._indicatorTypeDict[indicator_id]

        return

    def get_canvas_line_index(self, indicator_id):
        """
        Get a line's ID (on canvas) from an indicator ID
        :param indicator_id:
        :return:
        """
        assert isinstance(indicator_id, int)

        if indicator_id not in self._canvasLineKeyDict:
            raise RuntimeError('Indicator ID %s cannot be found. Current keys are %s.' % (
                indicator_id, str(sorted(self._canvasLineKeyDict.keys()))
            ))
        return self._canvasLineKeyDict[indicator_id]

    def get_line_type(self, my_id):
        """

        :param my_id:
        :return:
        """
        if my_id not in self._indicatorTypeDict:
            raise KeyError('Input indicator ID {0} is not in IndicatorTypeDict. Current keys are {1}'
                           ''.format(my_id, self._indicatorTypeDict.keys()))

        return self._indicatorTypeDict[my_id]

    def get_2way_data(self, line_id):
        """
        get the indicator data for a 2-way indicator
        :param line_id:
        :return: list of (2) vectors.
        """
        assert line_id in self._indicatorTypeDict, 'Line ID {0} is not in the Indicator-Type-Dictionary. ' \
                                                   'Candidates are {1}.'.format(line_id, self._indicatorTypeDict)
        assert self._indicatorTypeDict[line_id] == 2, 'The type of the indicator must be 2 but not {0}.' \
                                                      ''.format(self._indicatorTypeDict[line_id])

        vec_set = [self._lineManager[line_id][0:2], self._lineManager[line_id][2:4]]

        return vec_set

    def get_data(self, line_id):
        """
        Get line's vector x and vector y
        :param line_id:
        :return: 2-tuple of numpy arrays
        """
        return self._lineManager[line_id][0], self._lineManager[line_id][1]

    def get_indicator_key(self, x, y):
        """ Get indicator's key with position
        :return:
        """
        if x is None and y is None:
            raise RuntimeError('It is not allowed to have both X and Y are none to get indicator key.')

        ret_key = None

        for line_key in self._lineManager.keys():

            if x is not None and y is not None:
                # 2 way
                raise NotImplementedError('ASAP')
            elif x is not None and self._indicatorTypeDict[line_key] == 1:
                # vertical indicator moving along X
                if abs(self._lineManager[line_key][0][0] - x) < 1.0E-2:
                    return line_key
            elif y is not None and self._indicatorTypeDict[line_key] == 0:
                # horizontal indicator moving along Y
                if abs(self._lineManager[line_key][1][0] - y) < 1.0E-2:
                    return line_key
        # END-FOR

        return ret_key

    @staticmethod
    def get_line_style(line_id=None):
        """

        :param line_id:
        :return:
        """
        if line_id is not None:
            style = '--'
        else:
            style = '--'

        return style

    def get_live_indicator_ids(self):
        """

        :return:
        """
        return sorted(self._lineManager.keys())

    @staticmethod
    def get_marker():
        """
        Get the marker a line
        :return:
        """
        return '.'

    def get_next_color(self):
        """
        Get next color by auto color index
        :return: string as color
        """
        next_color = MplBasicColors[self._colorIndex]

        # Advance and possibly reset color scheme
        self._colorIndex += 1
        if self._colorIndex == len(MplBasicColors):
            self._colorIndex = 0

        return next_color

    def set_canvas_line_index(self, my_id, canvas_line_index):
        """

        :param my_id:
        :param canvas_line_index:
        :return:
        """
        self._canvasLineKeyDict[my_id] = canvas_line_index

        return

    def set_position(self, my_id, pos_x, pos_y):
        """ Set the indicator to a new position
        :param line_id:
        :param pos_x:
        :param pos_y:
        :return:
        """
        if self._indicatorTypeDict[my_id] == 0:
            # horizontal
            self._lineManager[my_id][1][0] = pos_y
            self._lineManager[my_id][1][1] = pos_y

        elif self._indicatorTypeDict[my_id] == 1:
            # vertical
            self._lineManager[my_id][0][0] = pos_x
            self._lineManager[my_id][0][1] = pos_x

        elif self._indicatorTypeDict[my_id] == 2:
            # 2-way
            self._lineManager[my_id][0] = pos_x
            self._lineManager[my_id][1] = pos_y

        else:
            raise RuntimeError('Unsupported indicator of type %d' % self._indicatorTypeDict[my_id])

        self._lineManager[my_id][2] = 'black'

        return

    def shift(self, my_id, dx, dy):
        """

        :param my_id:
        :param dx:
        :param dy:
        :return:
        """
        if self._indicatorTypeDict[my_id] == 0:
            # horizontal
            self._lineManager[my_id][1] += dy

        elif self._indicatorTypeDict[my_id] == 1:
            # vertical
            self._lineManager[my_id][0] += dx

        elif self._indicatorTypeDict[my_id] == 2:
            # 2-way
            self._lineManager[my_id][2] += dx
            self._lineManager[my_id][1] += dy

        else:
            raise RuntimeError('Unsupported indicator of type %d' % self._indicatorTypeDict[my_id])

        return

    def update_indicators_range(self, x_range, y_range):
        """
        Update indicator's range
        :param x_range:
        :param y_range:
        :return:
        """
        for i_id in self._lineManager.keys():
            # NEXT - Need a new flag for direction of the indicating line, vertical or horizontal
            if True:
                self._lineManager[i_id][1][0] = y_range[0]
                self._lineManager[i_id][1][-1] = y_range[1]
            else:
                self._lineManager[i_id][0][0] = x_range[0]
                self._lineManager[i_id][0][-1] = x_range[1]

        return




