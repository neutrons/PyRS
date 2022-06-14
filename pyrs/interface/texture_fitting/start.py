import sys
import vtk.qt
from pyrs.core import pyrscore
# https://stackoverflow.com/questions/51357630/vtk-rendering-not-working-as-expected-inside-pyqt
vtk.qt.QVTKRWIBase = "QGLWidget"  # noqa: E402
from qtpy.QtWidgets import QApplication  # type:ignore  # noqa: E402
from pyrs.interface.texture_fitting.texture_fitting_viewer import TextureFittingViewer  # noqa: E402
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel  # noqa: E402
from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl  # noqa: E402


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = TextureFittingModel(pyrscore.PyRsCore())
        self.ctrl = TextureFittingCrtl(self.model)
        self.view = TextureFittingViewer(self.model, self.ctrl)
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
