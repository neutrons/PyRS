from matplotlib.figure import Figure
import numpy as np

from qtpy.QtWidgets import QWidget, QSizePolicy, QVBoxLayout  # type:ignore
from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas
from mantidqt.MPLwidgets import NavigationToolbar2QT as NavigationToolbar2


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
        # self._myToolBar = MyNavigationToolbar(self, self._myCanvas)
        self._myToolBar = NavigationToolbar2(self._myCanvas, self)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myToolBar)

        #
        self._hasImage = False

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
            raise NotImplementedError('plot_type="scatter" has not been implemented')
        else:
            raise RuntimeError('Do not know how to add_2d_plot(..., plot_type="{}")'.format(plot_type))

        self._hasImage = True

    def set_title(self, title, color='black'):
        """
        set title to canvas
        :param title:
        :param color:
        :return:
        """
        self._myCanvas.set_title(title, color)

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

        # set y ticks as an option:
        if yticklabels is not None:
            # it will always label the first N ticks even image is zoomed in
            print("[FIXME]: The way to set up the Y-axis ticks is wrong!")
            self.axes.set_yticklabels(yticklabels)

        # explicitly set aspect ratio of the image
        self.axes.set_aspect('equal')
        self.axes.axis('off')

        self._flush()

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

        self.setWindowTitle(title)

        self.draw()

        return

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w + 1, h)
        self.resize(w, h)

        return

# END-OF-CLASS (MplGraphicsView)
