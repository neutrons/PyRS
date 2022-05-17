import sys
import vtk.qt
# https://stackoverflow.com/questions/51357630/vtk-rendering-not-working-as-expected-inside-pyqt
vtk.qt.QVTKRWIBase = "QGLWidget"  # noqa: E402
from qtpy.QtWidgets import QApplication  # type:ignore
from pyrs.interface.strainstressviewer.strain_stress_view import StrainStressViewer  # noqa: E402
from pyrs.interface.strainstressviewer.model import Model  # noqa: E402
from pyrs.interface.strainstressviewer.controller import Controller  # noqa: E402


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = Model()
        self.ctrl = Controller(self.model)
        self.view = StrainStressViewer(self.model, self.ctrl)
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
