from qtpy.QtWidgets import QApplication
from pyrs.interface.strainstressviewer.strain_stress_view import StrainStressViewer
from pyrs.interface.strainstressviewer.model import get_test_ws, Model
from pyrs.interface.strainstressviewer.controller import Controller
import sys


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = Model()
        self.ctrl = Controller(self.model)
        self.view = StrainStressViewer(self.model, self.ctrl)
        self.view.viz_tab.set_ws(get_test_ws())
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
