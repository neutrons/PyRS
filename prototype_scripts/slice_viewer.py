try:
    from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    from PyQt4.QtGui import QDialog, QMainWindow, QApplication
    from PyQt4.QtCore import pyqtSignal

import sys
import ui_sliceviewer


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

    def plot_contour(self):

        self.ui.graphicsView_2DSlice.canvas.add_contour_plot()

# END-CLASS



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

    app.exec_()


