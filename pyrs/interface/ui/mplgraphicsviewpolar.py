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


class MplGraphicsPolarView(QWidget):
    """ A single graphics view for polar projection.
    the tool bar is not added as a default

    Note: Merged with HFIR_Powder_Reduction.MplFigureCAnvas
    """
    def __init__(self, parent):
        """ Initialization
        """
        # Initialize parent
        super(MplGraphicsPolarView, self).__init__(parent)

        # set up canvas
        self._myCanvas = Qt4MplPolarCanvas(self)

        # state of operation
        self._isZoomed = False
        # X and Y limit with home button
        self._homeXYLimit = None

        # set up layout
        self._vBox = QVBoxLayout(self)
        self._vBox.addWidget(self._myCanvas)

        self._hasImage = False

        return

    def clear_image(self):
        """

        :return:
        """
        self._myCanvas.axes.cla()
        self._myCanvas._flush()

        return

# END-CLASS


class Qt4MplPolarCanvas(FigureCanvas):
    """  A customized Qt widget for matplotlib figure.
    It can be used to replace GraphicsView of QtGui
    """
    def __init__(self, parent):
        """  Initialization
        """
        # Instantiating matplotlib Figure
        self.fig = Figure()
        self.fig.patch.set_facecolor('white')

        # Initialize parent class and set parent
        super(Qt4MplPolarCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # set up axis/subplot (111) only for 2D
        self.axes = self.fig.add_subplot(111, polar=True)  # return: matplotlib.axes.AxesSubplot

        # general canvas setup
        self.fig.subplots_adjust(bottom=0.15)

        # plot management
        self._contourPlot = None

        # legend and color bar
        self._colorBar = None
        self._isLegendOn = False
        self._legendFontSize = 8

        return

    def _flush(self):
        """ A dirty hack to flush the image
        """
        w, h = self.get_width_height()
        self.resize(w+1, h)
        self.resize(w, h)

        return

    def plot_polar_xy(self, vec_theta, vec_r):
        self.axes.plot(vec_theta, vec_r, 'ko', ms=3)

    def plot_contour(self, vec_theta, vec_r, vec_values, max_r, r_resolution, theta_resolution, init_value=0):
        """
        plot contour
        :param vec_theta: 1D vector
        :param vec_r: 1D vector
        :param vec_values: 1D vector
        :param max_r:
        :param r_resolution:
        :param theta_resolution:
        :param init_value: init value of the plot to be interpolated
        :return:
        """
        # check inputs
        check_1D_array(vec_theta)
        check_1D_array(vec_r)
        check_1D_array(vec_values)
        if not (vec_values.shape[0] == vec_theta.shape[0] and vec_values.shape[0] == vec_r.shape[0]):
            raise RuntimeError('Input vector of theta ({}), r ({}) and values ({}) are of different '
                               'sizes.'.format(vec_theta.shape, vec_r.shape, vec_values.shape))
        check_float('Maximum R', max_r, 0, None)
        check_float('R resolution', r_resolution, 0, None)
        check_float('Theta resolution', theta_resolution, 0, 90.)

        # create the mesh grid for contour plot
        # create 1D arrays for theta and r: beta/2
        azimuths = np.radians(np.linspace(-theta_resolution/2., 360+theta_resolution/2., 360/theta_resolution+1))  # degree
        # Chris change to non-linear spacing from alpha:  zeniths = np.arange(0, max_r+r_resolution, r_resolution)  # radius
        zeniths = np.tan(np.pi / 360. * np.arange(-r_resolution/2., max_r+r_resolution, r_resolution))  # radius

        # convert to meshgrid
        mesh_r, mesh_theta = np.meshgrid(zeniths, azimuths)
        # mesh_values = np.zeros(mesh_r.shape, dtype=float, order='C')
        mesh_values = np.empty(mesh_r.shape)
        mesh_values[:] = init_value
        # TODO - remove me: example a = numpy.empty((3, 3)), a[:] = numpy.nan

        # set up the data points on mesh grid, i.e., mapping from vector to 2D map
        num_pts = vec_theta.shape[0]
        r_ref_vec = mesh_r[0]
        theta_ref_vec = mesh_theta[:, 0]

        print ('[DB...BAT] Plot pole figure:  Number of data points = {}, '
               'Maximum intensity = {}'.format(num_pts, np.max(vec_values)))

        for i in range(num_pts):
            r_i = vec_r[i]
            theta_i = vec_theta[i] * np.pi / 180.  # convert from degree to rad
            value_i = vec_values[i]

            # locate for theta
            index_theta = np.searchsorted(theta_ref_vec, theta_i, side='left')
            if index_theta == 0:
                # before first one
                index_theta = 0
            elif index_theta >= len(theta_ref_vec):
                # out of boundary. it is not likely to happen
                print ('[DB...BAT] Find an out-of-boundary theta {0} exceeding {1}'
                       ''.format(theta_i, theta_ref_vec[-1]))
                index_theta -= 1
            else:
                # theta is between two valid values: use the closer one
                left_theta = theta_ref_vec[index_theta-1]
                right_theta = theta_ref_vec[index_theta]
                if theta_i - left_theta < right_theta - theta_i:
                    index_theta -= 1
            # END-IF-ELSE

            # locate for r
            index_r = np.searchsorted(r_ref_vec, r_i, side='left')
            if index_r == 0:
                # smaller than first item
                index_r = 0
            elif index_r >= len(r_ref_vec):
                # out of upper boundary. it is not likely to happen
                print ('[DB...BAT] Find an out-of-boundary r {0} exceeding {1}'
                       ''.format(r_i, r_ref_vec[-1]))
                index_r -= 1
            else:
                # r is between two valid values: use the closer one
                left_r = r_ref_vec[index_r-1]
                right_r = r_ref_vec[index_r]
                if r_i - left_r < right_r - r_i:
                    index_r -= 1
            # END-IF-ELSE

            # set value
            if np.isnan(mesh_values[index_theta, index_r]):
                mesh_values[index_theta, index_r] = value_i
            else:
                mesh_values[index_theta, index_r] += value_i
        # END-FOR

        # plot
        self.axes.contourf(mesh_theta, mesh_r, mesh_values)

        # flush
        self._flush()

        # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        # ax.contourf(theta, r, values)

        return
    #

def check_1D_array(vector):
    """
    check 1D array
    :param vector: 
    :return: 
    """
    assert isinstance(vector, np.ndarray), 'Input {0} must be a numpy ndarray but not a {1}.' \
                                           ''.format(vector, type(vector))

    if len(vector.shape) != 1:
        raise RuntimeError('Input vector {0} must be a 1D array but not of shape {1}'
                           ''.format(vector, vector.shape))

    return


def check_float(value_name, value, min_value=None, max_value=None):
    """
    check whether a value is a float or at least an integer
    :param value_name
    :param value:
    :param min_value:
    :param max_value:
    :return:
    """
    assert isinstance(value, float) or isinstance(value, int), '{0} {1} must be a float but not a {2}.' \
                                                               ''.format(value_name, value, type(value))

    # check boundary
    if min_value is not None and value <= min_value:
        raise ValueError('Variable {0}\'s value {1} is below specified minimum value {2}'
                         ''.format(value_name, value, min_value))

    if max_value is not None and value > max_value:
        raise ValueError('Variable {0}\'s value {1} is larger than specified maximum value {2}'
                         ''.format(value_name, value, max_value))

    return
