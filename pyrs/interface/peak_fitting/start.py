import sys
import vtk.qt
from pyrs.core import pyrscore

# https://stackoverflow.com/questions/51357630/vtk-rendering-not-working-as-expected-inside-pyqt
vtk.qt.QVTKRWIBase = "QGLWidget"  # noqa: E402
from qtpy.QtWidgets import QApplication  # type:ignore  # noqa: E402
from pyrs.interface.peak_fitting.peak_fitting_viewer import PeakFittingViewer  # noqa: E402
from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel  # noqa: E402
from pyrs.interface.peak_fitting.peak_fitting_crtl import PeakFittingCrtl  # noqa: E402


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = PeakFittingModel(pyrscore.PyRsCore())
        self.ctrl = PeakFittingCrtl(self.model)
        self.view = PeakFittingViewer(self.model, self.ctrl)
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
