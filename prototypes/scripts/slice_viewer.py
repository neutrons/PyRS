try:
    from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    from PyQt4.QtGui import QDialog, QMainWindow, QApplication
    from PyQt4.QtCore import pyqtSignal

import sys
import ui_sliceviewer
import h5py
import numpy as np
import matplotlib.tri as tri


class SliceViewer(QMainWindow):
    """
    prototype of slicer viewer
    """
    def __init__(self, parent):
        """
        slice viewer
        :param parent:
        """
        super(SliceViewer, self).__init__(parent)

        # set up UI
        self.ui = ui_sliceviewer.Ui_MainWindow()
        self.ui.setupUi(self)

        return

    def plot_contour(self, vec_x, vec_y, vec_z):

        # Create grid values first.
        contour_resolution = 1000
        ngridx = contour_resolution
        ngridy = contour_resolution

        xi = np.linspace(vec_x.min(), vec_x.max(), ngridx)
        yi = np.linspace(vec_y.min(), vec_y.max(), ngridy)

        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        self.ui.graphicsView_2DSlice.canvas.add_contour_plot(xi, yi, zi)

# END-CLASS


def import_h5_array(file_name):
    """
    """
    h5file = h5py.File(file_name, 'r')
    for entry in h5file:
        print entry
        if entry.startswith('Slice'):
            data_entry = h5file[entry]
            data_set = data_entry.value
            x = data_set[:, 0]
            y = data_set[:, 1]
            z = data_set[:, 2]


    h5file.close()

    return x, y, z


def main(argv):
    """
    """
    if QApplication.instance():
        _app = QApplication.instance()
    else:
        _app = QApplication(sys.argv)
    return _app


if __name__ == '__main__':
    # Main application
    app = main(sys.argv)

    main_window = SliceViewer(None)  # .FourCircleMainWindow()
    main_window.show()

    # Import data
    x, y, z = import_h5_array('/tmp/pyrs_test_ss/test.hdf5')
    print x.min(), x.max()
    print y.min(), y.max()
    print z.min(), z.max(), z.mean()

    main_window.plot_contour(x, y, z)

    app.exec_()


