try:
    from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication, QVBoxLayout
    from PyQt5.QtCore import pyqtSignal
    from PyQt5 import QtGui
except ImportError:
    from PyQt4.QtGui import QDialog, QMainWindow, QApplication
    from PyQt4.QtCore import pyqtSignal

import sys
from pyrs.interface.ui import uitest_test_slice_view
import h5py
import numpy as np
import matplotlib.tri as tri
from pyrs.interface.ui.slice_view_widget import SliceViewWidget


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
        self.ui = uitest_test_slice_view.Ui_MainWindow()
        self.ui.setupUi(self)

        # tweak
        # _layout = QVBoxLayout()
        # _layout.addWidget(SliceViewWidget(self))
        # self.ui.widget_sliceView.setLayout(_layout)
        # self.ui.widget_sliceView.addWidget(SliceViewWidget)

        return

    def plot_contour(self, vec_x, vec_y, vec_z):
        """
        plot contour from vec X, vec Y
        :param vec_x:
        :param vec_y:
        :param vec_z:
        :return:
        """
        contour_resolution = 1000

        self.ui.widget.plot_contour_interpolate(vec_x, vec_y, vec_z, contour_resolution, False)
        self.ui.widget.plot_scatter(vec_x, vec_y, flush=True)

        #
        # # Create grid values first.
        # contour_resolution = 1000
        # ngridx = contour_resolution
        # ngridy = contour_resolution
        #
        # xi = np.linspace(vec_x.min(), vec_x.max(), ngridx)
        # yi = np.linspace(vec_y.min(), vec_y.max(), ngridy)
        #
        # # Perform linear interpolation of the data (x,y)
        # # on a grid defined by (xi,yi)
        # triang = tri.Triangulation(x, y)
        # interpolator = tri.LinearTriInterpolator(triang, z)
        # Xi, Yi = np.meshgrid(xi, yi)
        # zi = interpolator(Xi, Yi)
        #
        # self.ui.widget.plot_contour(xi, yi, zi, flush=False)
        # self.ui.widget.plot_scatter(vec_x, vec_y, flush=True)
        #
        # # self.ui.widget.main_canvas.add_contour_plot(xi, yi, zi)
        # # self.ui.widget.main_canvas.add_scatter(vec_x, vec_y)
        # # self.ui.widget.main_canvas._flush()
        #
        # # plot vertically
        # self.ui.widget.vertical_canvas

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

    # mport data
    x, y, z = import_h5_array('./tests/testdata/sliced_peak_positions.hdf5')
    print x.min(), x.max()
    print y.min(), y.max()
    print z.min(), z.max(), z.mean()

    main_window.plot_contour(x, y, z)

    app.exec_()


