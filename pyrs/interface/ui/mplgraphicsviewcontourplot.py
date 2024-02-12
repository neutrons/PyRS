from matplotlib.figure import Figure

from qtpy.QtWidgets import QWidget, QVBoxLayout  # type:ignore
from mantidqt.MPLwidgets import FigureCanvasQTAgg as FigureCanvas
from mantidqt.MPLwidgets import NavigationToolbar2QT as NavigationToolbar2


class MplGraphicsViewContourPlot(QWidget):
    """ A combined graphics view including matplotlib canvas and
    a navigation tool bar

    Note: Merged with HFIR_Powder_Reduction.MplFigureCAnvas
    """

    def __init__(self, parent):
        """ Initialization
        """
        # Initialize parent
        super(MplGraphicsViewContourPlot, self).__init__(parent)

        # set up canvas
        self.figure = Figure()
        self.colorbar = None
        self._myCanvas = FigureCanvas(self.figure)
        self._myToolBar = NavigationToolbar2(self._myCanvas, self)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        self.ax = self.figure.add_subplot(111)

        # set up layout
        self._vBox = QVBoxLayout(self)
        # self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myCanvas)
        self._vBox.addWidget(self._myToolBar)

        self._arrowList = list()

        self._hasImage = False
